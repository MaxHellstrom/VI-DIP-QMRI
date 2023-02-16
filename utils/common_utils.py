import numpy as np
import torch
from tqdm import tqdm

from utils.misc import _Parameters
from utils.physics_utils import (
    diffusion_signal,
    simulate_complex_noise,
    spgr_signal,
    spin_echo_signal,
)
from utils.statistics_utils import (
    ADC_multi_b_NLLS_estimate,
    T1_VFA_NLLS_estimator,
    T2_multi_echo_NLLS_estimator,
    kramer_rao_ADC_multi_b,
    kramer_rao_T1_VFA,
    kramer_rao_T2_multi_echo,
)


def _load_acqusition_specifics_to_param(parameter_dict, data):

    device = parameter_dict["general"]["device"]

    # load general parameters
    parameter_dict["data"]["method"] = str(data["method"])
    parameter_dict["data"]["image_height"] = int(data["image_height"])
    parameter_dict["data"]["image_width"] = int(data["image_width"])

    # load torch parameters
    for parameter_name in ["mask", "y"]:

        parameter_dict["data"][parameter_name] = torch.from_numpy(
            data[parameter_name]
        ).to(device)

    # randomize noise mask for multirun eval
    if parameter_dict["training"]["randomize_signal_noise"]:
        print("randomizing signal")

        y = simulate_complex_noise(image=data["y_clean"], noise_std=data["noise_std"])

        parameter_dict["data"]["y"] = torch.from_numpy(y).to(device)

        np.save(file=parameter_dict["path"]["results"] + "y.npy", arr=y)

    # load method specific parameters
    if data["method"] == "T1_est":

        for parameter_name in ["B1_corr", "TR", "FA"]:
            parameter_dict["physics"][parameter_name] = torch.from_numpy(
                data[parameter_name]
            ).to(device)

    elif data["method"] == "T2_est":
        for parameter_name in ["TE"]:
            parameter_dict["physics"][parameter_name] = torch.from_numpy(
                data[parameter_name]
            ).to(device)

    elif data["method"] == "ADC_est":
        for parameter_name in ["b_values"]:
            parameter_dict["physics"][parameter_name] = torch.from_numpy(
                data[parameter_name]
            ).to(device)

    return _Parameters(parameter_dict)


def calculate_denoised_magnitude_signal(P_est, data):

    if data["method"] == "T1_est":

        y_denoised = spgr_signal(
            S0=P_est[0, :, :],
            T1=P_est[1, :, :],
            FA_values=data["FA"],
            TR=data["TR"],
            mask=data["mask"],
            B1_corr=data["B1_corr"],
            mode="numpy",
        )

    elif data["method"] == "T2_est":
        y_denoised = spin_echo_signal(
            S0=P_est[0, :, :],
            T2=P_est[1, :, :],
            TE_values=data["TE"],
            mask=data["mask"],
            mode="numpy",
        )

    elif data["method"] == "ADC_est":
        y_denoised = diffusion_signal(
            S0=P_est[0, :, :],
            ADC=P_est[1, :, :],
            b_values=data["b_values"],
            mask=data["mask"],
            mode="numpy",
        )

    return y_denoised


def calculate_NLLS_estimate(data, param):

    y = param.data.y.cpu().detach().numpy()

    if data["method"] == "T1_est":

        P_NLLS = T1_VFA_NLLS_estimator(
            y=y,
            FA_values=data["FA"],
            TR=data["TR"],
            B1_corr=data["B1_corr"],
            mask=data["mask"],
            bounds=data["bounds"],
            init_guess=data["init_guess"],
        )

    elif data["method"] == "T2_est":

        P_NLLS = T2_multi_echo_NLLS_estimator(
            y=y,
            TE_values=data["TE"],
            mask=data["mask"],
            bounds=data["bounds"],
            init_guess=data["init_guess"],
        )

    elif data["method"] == "ADC_est":

        P_NLLS = ADC_multi_b_NLLS_estimate(
            y=y,
            b_values=data["b_values"],
            mask=data["mask"],
            bounds=data["bounds"],
            init_guess=data["init_guess"],
        )

    else:
        raise NotImplementedError

    return P_NLLS


def calculate_kramer_rao_variance(P_NLLS, data, param):

    y = param.data.y.cpu().detach().numpy()

    if param.general.method == "T1_est":

        P_NLLS_var = kramer_rao_T1_VFA(
            y=y,
            NLLS_params=P_NLLS,
            FA=data["FA"],
            TR=data["TR"],
            mask=data["mask"],
            B1_corr=data["B1_corr"],
        )

    elif data["method"] == "T2_est":

        P_NLLS_var = kramer_rao_T2_multi_echo(
            y=y,
            NLLS_params=P_NLLS,
            TE=data["TE"],
            mask=data["mask"],
        )

    elif data["method"] == "ADC_est":
        P_NLLS_var = kramer_rao_ADC_multi_b(
            y=y,
            NLLS_params=P_NLLS,
            b_values=data["b_values"],
            mask=data["mask"],
        )

    return P_NLLS_var


def calculate_qmri_loss(nw_output, param):

    alpha = nw_output[0, 0, :, :] * param.data.mask
    beta = nw_output[0, 1, :, :] * param.data.mask

    # prevent nan losses
    if param.training.clamp_alpha:
        alpha = torch.clamp(alpha, min=0)

    if param.training.clamp_beta:
        beta = torch.clamp(beta, min=-20, max=+20)

    if param.general.method == "T1_est":

        s = spgr_signal(
            S0=alpha,
            T1=torch.exp(-beta),
            FA_values=param.physics.FA,
            TR=param.physics.TR,
            mask=param.data.mask,
            B1_corr=param.data.B1_corr,
            mode="torch",
            device=param.general.device,
        )

    elif param.general.method == "T2_est":

        s = spin_echo_signal(
            S0=alpha,
            T2=torch.exp(-beta),
            TE_values=param.physics.TE,
            mask=param.data.mask,
            mode="torch",
            device=param.general.device,
        )

    elif param.general.method == "ADC_est":
        s = diffusion_signal(
            S0=alpha,
            ADC=torch.exp(-beta),
            b_values=param.physics.b_values,
            mask=param.data.mask,
            mode="torch",
            device=param.general.device,
        )

    loss = torch.pow(param.data.y - s, 2)

    loss = loss[param.data.mask.expand(loss.size())]

    return torch.mean(loss)


def get_params(opt_over, net, net_input, downsampler=None):
    """Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    """
    opt_over_list = opt_over.split(",")
    params = []

    for opt in opt_over_list:

        if opt == "net":
            params += [x for x in net.parameters()]
        elif opt == "down":
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == "input":
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, "what is it?"

    return params


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == "u":
        x.uniform_()
    elif noise_type == "n":
        x.normal_()
    else:
        assert False


def get_noise(input_depth, method, spatial_size, noise_type="u", var=1.0 / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x
    `spatial_size[1]`) initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard
        deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == "noise":
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == "meshgrid":
        assert input_depth == 2
        X, Y = np.meshgrid(
            np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
            np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1),
        )
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input


def np_to_torch(img_np):
    """Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var, drop_c0=True):
    """Converts an image in torch.Tensor format to np.array."""
    if drop_c0:
        return img_var.detach().cpu().numpy()[0]

    return img_var.detach().cpu().numpy()


def optimize(optimizer, closure, num_iter):

    for j in range(num_iter):
        optimizer.zero_grad()
        closure()
        optimizer.step()
