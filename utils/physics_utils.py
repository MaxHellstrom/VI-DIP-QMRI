import numpy as np
import torch
from scipy.optimize import least_squares


def spgr_signal(
    S0, T1, FA_values, TR, mask=None, B1_corr=None, mode="numpy", device=None
):

    # for a single voxel
    if mode == "voxel":

        exp_term = np.exp(-TR / T1)
        s = (
            S0
            * np.sin(B1_corr * FA_values)
            * (1 - exp_term)
            / (1 - np.cos(B1_corr * FA_values) * exp_term)
        )

        return s

    # for torch tensors
    if mode == "torch":

        s = torch.zeros(1, len(FA_values), mask.size()[0], mask.size()[1]).to(device)

        for FA_value in range(len(FA_values)):
            tmp = (
                S0[mask]
                * torch.sin(B1_corr[mask] * FA_values[FA_value])
                * (1 - torch.exp(-TR / T1[mask]))
                / (
                    1
                    - torch.cos(B1_corr[mask] * FA_values[FA_value])
                    * torch.exp(-TR / T1[mask])
                )
            )

            s[0, FA_value, :, :][mask] = tmp.float()

        return s

    # for numpy arrays
    if mode == "numpy":

        s = np.zeros((len(FA_values), mask.shape[0], mask.shape[1]))

        exp_term = np.exp(-TR / T1[mask])

        for FA_value in range(len(FA_values)):

            tmp = (
                S0[mask]
                * np.sin(B1_corr[mask] * FA_values[FA_value])
                * (1 - exp_term)
                / (1 - np.cos(B1_corr[mask] * FA_values[FA_value]) * exp_term)
            )

            s[FA_value, :, :][mask] = tmp

    return s


def spin_echo_signal(S0, T2, TE_values, mask=None, mode="numpy", device=None):

    # for a single voxel
    if mode == "voxel":
        s = S0 * np.exp(-TE_values / T2)

        return s

    # for torch tensors
    if mode == "torch":

        s = torch.zeros((1, len(TE_values), mask.shape[0], mask.shape[1])).to(device)

        for i in range(len(TE_values)):
            s[0, i, :, :][mask] = S0[mask] * torch.exp(-TE_values[i] / T2[mask])

        return s

    # for numpy arrays
    if mode == "numpy":

        if mask is None:
            mask = np.ones(shape=S0.shape, dtype=bool)

        s = np.zeros((len(TE_values), mask.shape[0], mask.shape[1]))

        for i in range(len(TE_values)):
            s[i, :, :][mask] = S0[mask] * np.exp(-TE_values[i] / T2[mask])

        return s


def diffusion_signal(S0, ADC, b_values, mask=None, mode="numpy", device=None):

    # for a single voxel
    if mode == "voxel":
        s = S0 * np.exp(-b_values * ADC)

        return s

    # for torch tensors
    if mode == "torch":

        s = torch.zeros((1, len(b_values), mask.shape[0], mask.shape[1])).to(device)

        for i in range(len(b_values)):
            s[0, i, :, :][mask] = S0[mask] * torch.exp(-b_values[i] * ADC[mask])

        return s

    # for numpy arrays
    if mode == "numpy":

        if mask is None:
            mask = np.ones(shape=S0.shape, dtype=bool)

        s = np.zeros((len(b_values), mask.shape[0], mask.shape[1]))

        for i in range(len(b_values)):
            s[i, :, :][mask] = S0[mask] * np.exp(-b_values[i] * ADC[mask])

        return s


def simulate_complex_noise(image, noise_std):
    """Adds complex noise to 2D image

    Parameters
    ----------
    img :
        2D magnitude image
    noise_std :
        standard deviation of noise
    """

    if len(image.shape) == 2:

        nx = image.shape[0]
        ny = image.shape[1]

        r1 = np.random.randn(nx, ny)
        r2 = np.random.randn(nx, ny)

        res = np.abs(image + noise_std * r1 + 1j * noise_std * r2)

        return res

    if len(image.shape) == 3:

        nc = image.shape[0]
        nx = image.shape[1]
        ny = image.shape[2]

        r1 = np.random.randn(nc, nx, ny)
        r2 = np.random.randn(nc, nx, ny)

        res = np.zeros((nc, nx, ny))

        for channel in range(nc):
            tmp = image[channel, :, :]
            res[channel, :, :] = np.abs(
                tmp + noise_std * r1[channel, :, :] + 1j * noise_std * r2[channel, :, :]
            )

        return res
