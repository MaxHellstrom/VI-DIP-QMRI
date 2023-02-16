import numpy as np
import torch
import torch.optim
import time

from models import get_net
from utils.common_utils import (
    _load_acqusition_specifics_to_param,
    calculate_denoised_magnitude_signal,
    calculate_kramer_rao_variance,
    calculate_NLLS_estimate,
    calculate_qmri_loss,
    get_noise,
    get_params,
    optimize,
    torch_to_np,
)
from utils.misc import (
    _copy_dataset_to_results_path,
    _create_results_folders,
    _Parameters,
    _turn_of_bayesian_settings,
)
from utils.plot_utils import plot_estimation_progress

parameter_dict = dict(
    data=dict(),
    general=dict(

        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ),
    path=dict(
        data="data/T1/0204111315_9.2_1_snr50_unclipped/",
        results="results/T1/",
        pretrained_network="",
    ),
    physics=dict(),
    training=dict(
        calculate_NLLS_est=False,
        randomize_signal_noise=True,
        num_burnin=0,
        num_iter=50000,  # 110000,
        sampling_interval=100,
        num_mc=128,
        LR=0.0003,
        reg_noise_std=0.05,
        load_pretrained_network=False,
        clamp_alpha=False,
        clamp_beta=True,
        weight_decay=1e-4,
        # removes bayesian approximation by modifying
        # network and training settings
        disable_bayesian_approximation=False,
    ),
    network=dict(
        input_depth=32,
        net_type="skip",
        pad="reflection",
        upsample_mode="bilinear",
        n_channels=2,
        act_fun="ReLU",
        skip_n33d=128,
        skip_n33u=128,
        skip_n11=4,
        num_scales=5,
        downsample_mode="stride",
        bayes=False,
        dropout_mode_down="2d",
        dropout_p_down=0.1,
        dropout_mode_up="2d",
        dropout_p_up=0.1,
        dropout_mode_skip="None",
        dropout_p_skip=0.1,
        dropout_mode_output="None",
        dropout_p_output=0.1,
        need_output_act=False,
    ),
)
print(parameter_dict["general"]["device"])

# create results folders and update path parameters
parameter_dict = _create_results_folders(parameter_dict, timestamp=True)

# disable bayesian approximation if needed
if parameter_dict["training"]["disable_bayesian_approximation"]:
    parameter_dict = _turn_of_bayesian_settings(parameter_dict)

# create parameter class
param = _Parameters(parameter_dict)

net = get_net(
    input_depth=param.network.input_depth,
    NET_TYPE=param.network.net_type,
    pad=param.network.pad,
    upsample_mode=param.network.upsample_mode,
    n_channels=param.network.n_channels,
    act_fun=param.network.act_fun,
    skip_n33d=param.network.skip_n33d,
    skip_n33u=param.network.skip_n33u,
    skip_n11=param.network.skip_n11,
    num_scales=param.network.num_scales,
    downsample_mode=param.network.downsample_mode,
    bayes=param.network.bayes,
    dropout_mode_down=param.network.dropout_mode_down,
    dropout_p_down=param.network.dropout_p_down,
    dropout_mode_up=param.network.dropout_mode_up,
    dropout_p_up=param.network.dropout_p_up,
    dropout_mode_skip=param.network.dropout_mode_skip,
    dropout_p_skip=param.network.dropout_p_skip,
    dropout_mode_output=param.network.dropout_mode_output,
    dropout_p_output=param.network.dropout_p_output,
    need_output_act=param.network.need_output_act,
).to(param.general.device)

path_data = param.path.data

# if pretrained network
if param.training.load_pretrained_network:
    print("loading pretrained net")

    loss_list = np.load(param.path.pretrained_network + "loss_list.npy")
    loss_list = loss_list.tolist()
    # set data path to pretrained network path
    path_data = param.path.pretrained_network

    # apply state of pretrained network
    net.load_state_dict(torch.load(path_data + "network_state"))

    # load the net input of pretrained network
    net_input = torch.load(path_data + "network_input")
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

# copy dataset to results path
_copy_dataset_to_results_path(path_data=path_data, path_results=param.path.results)

# load dataset
data = np.load(path_data + "dataset.npz")

param = _load_acqusition_specifics_to_param(parameter_dict, data)


if param.training.calculate_NLLS_est:

    print("calculating NLLS...")
    P_NLLS = calculate_NLLS_estimate(data=data, param=param)
    P_NLLS_var = calculate_kramer_rao_variance(P_NLLS, data, param)

    np.save(file=param.path.results + "P_NLLS.npy", arr=P_NLLS)
    np.save(file=param.path.results + "P_NLLS_var.npy", arr=P_NLLS_var)

    print("NLLS calculation completed")


if not param.training.load_pretrained_network:
    print("training network from scratch...")

    # create list to store losses
    loss_list = []

    net_input = (
        get_noise(
            param.network.input_depth,
            "noise",
            (param.data.image_height, param.data.image_width),
        )
        .to(param.general.device)
        .detach()
    )
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

# save parameter to .txt file
with open(param.path.results + "params.txt", "w") as f:
    print(parameter_dict, file=f)

i = 0


actions = np.zeros(param.training.num_iter)
actions[param.training.num_burnin :: param.training.sampling_interval] = 1
actions[-1] = 1
actions[0] = 0

def closure():

    global i, out_avg, last_net, net_input

    if param.training.reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * param.training.reg_noise_std)

    # forward pass
    out = net(net_input)

    # calculate training loss
    _loss = calculate_qmri_loss(nw_output=out, param=param)

    # backprop
    _loss.backward()
    loss_list.append(_loss.item())

    if actions[i] == 1:

        print(f"Iteration: {i} MSE loss: {_loss.item():.4f}")
        # if we are at the last iteration...
        if i == param.training.num_iter - 1:
            print("final iteration")
            print(f"saving network")

            # save network state dict
            torch.save(net.state_dict(), param.path.results + f"network_state")

            # Save network input
            torch.save(net_input, param.path.results + f"network_input")

        # create list to store samples
        P_samples = []

        # monte carlo dropout
        with torch.no_grad():
            net_input = net_input_saved + (
                noise.normal_() * param.training.reg_noise_std
            )

            # for each MC sample
            for _ in range(param.training.num_mc):

                # forward pass
                out = net(net_input)

                out[0, 1, :, :] = torch.exp(-out[0, 1, :, :])
                P_samples.append(torch_to_np(out))

            P_samples = np.asarray(P_samples)
            P_est = np.mean(P_samples, axis=0)

            # calculate denoised signal from estimated tissue params
            y_denoised = calculate_denoised_magnitude_signal(P_est=P_est, data=data)

            # save estimated tissue params for post-training plotting
            np.save(file=f"{param.path.sliders}P_est_{i}", arr=P_est)

            # save progressplot
            plot_estimation_progress(
                P_ref=data["P_ref"],
                P_NLLS=data["P_NLLS"],
                P_est=P_est,
                P_epi_std=np.std(P_samples, axis=0),
                y=data["y"],
                y_denoised=y_denoised,
                mask=data["mask"],
                iter=i,
                loss_list=loss_list,
                save_path=param.path.progress_plots,
                param_names=data["param_names"],
                param_units=data["param_units"],
                image_type="png",
                suptitle=str(data["method"]),
                noise_std=data["noise_std"].item(),
                LR=param.training.LR,
            )

                # save samples at last iteration
            if i == param.training.num_iter - 1:
                # saving samples at last iter
                print(f"saving MC sample stack and loss at iter={i}")
                np.save(file=param.path.results + "P_samples.npy", arr=P_samples)
                np.save(file=param.path.results + "y_denoised.npy", arr=y_denoised)
                np.save(file=param.path.results + "loss_list.npy", arr=loss_list)

    i += 1

    return _loss


optimizer = torch.optim.AdamW(
    get_params(opt_over="net", net=net, net_input=net_input),
    lr=param.training.LR,
    weight_decay=param.training.weight_decay,
)
t = time.time()
optimize(optimizer=optimizer, closure=closure, num_iter=param.training.num_iter)
elapsed = time.time() - t

print(elapsed/60)
print("done")
