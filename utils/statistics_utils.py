from distutils.command.build_scripts import first_line_re

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

from utils.physics_utils import diffusion_signal, spgr_signal, spin_echo_signal


def log_normalize_uncertainty_std_map(P_map, P_std, mask):
    # scale_map = np.zeros(P_map.shape)
    scale_map_log = np.zeros(P_map.shape)

    scale_map = P_map.copy()
    std_map = P_std.copy()

    # P_map[np.isinf(P_map)] = 0

    scale_map_log[mask] = np.log(scale_map[mask])
    scale_map_log[mask] -= scale_map_log[mask].min()

    std_map[mask] /= scale_map_log[mask]
    return std_map


def kramer_rao_T1_VFA(y, NLLS_params, FA, TR, mask, B1_corr):
    """Calculate Kramer Rao variance from NLLS estimated tissue params in T1 mapping.

    Parameters
    ----------
    y : np.ndarray
        magnitude signal data, of shape (nFA, nx, ny)
    NLLS_params : np.ndarray
        NLLS estimated tissue parameters of shape (2, nx, ny).
        (0, :, :) = estimated S0 map
        (1, :, :) = estimated T1 map
    FA : np.ndarray
        Array of flip angles in radians.
    TR : float
        Repetition time in seconds
    mask : np.ndarray (bool)
        mask of shape (nx, ny)
    B1_corr : np.ndarray
        Array of B1 corrections

    Returns
    -------
    list
        list of the four elements of the Covariance matrix (Kramer Rao bound), where
        each element contains the entire map (as an np.array).
        0: beta_1 x beta_1
        1: beta_1 x beta_2
        2: beta_2 x beta_1
        3: beta_2 x beta_2
    """
    S0 = NLLS_params[0, :, :][mask]
    T1 = NLLS_params[1, :, :][mask]
    y, B1_corr = y[:, mask], B1_corr[mask]
    KRLB11, KRLB12, KRLB21, KRLB22 = (
        np.zeros(mask.sum()),
        np.zeros(mask.sum()),
        np.zeros(mask.sum()),
        np.zeros(mask.sum()),
    )
    for voxel_index in range(mask.sum()):

        if voxel_index == 1068:
            a = 1
        # residuals
        r = y[:, voxel_index] - spgr_signal(
            S0=S0[voxel_index],
            T1=T1[voxel_index],
            FA_values=FA,
            B1_corr=B1_corr[voxel_index],
            TR=TR,
            mode="voxel",
        )

        # number of data points
        M = len(FA)
        # residual standare error
        noise_var = (1 / (M - 2)) * np.sum(r**2)
        # noise_var = 0.0001
        # derivatives
        ds_db1 = dSPGR_dS0(T1=T1[voxel_index], FA=FA, TR=TR)
        ds_db2 = dSPGR_dT1(S0=S0[voxel_index], T1=T1[voxel_index], FA=FA, TR=TR)

        f11 = np.sum(ds_db1 * ds_db1) / noise_var
        f12 = np.sum(ds_db1 * ds_db2) / noise_var
        f21 = np.sum(ds_db2 * ds_db1) / noise_var
        f22 = np.sum(ds_db2 * ds_db2) / noise_var

        if f11 == 0:
            f11 += np.finfo(float).eps

        if f12 == 0:
            f12 += np.finfo(float).eps

        if f21 == 0:
            f21 += np.finfo(float).eps

        if f22 == 0:
            f22 += np.finfo(float).eps

        fisher = np.array([[f11, f12], [f21, f22]])
        KRLB = np.linalg.inv(fisher)

        KRLB11[voxel_index] = KRLB[0, 0]
        KRLB12[voxel_index] = KRLB[0, 1]
        KRLB21[voxel_index] = KRLB[1, 0]
        KRLB22[voxel_index] = KRLB[1, 1]

    KRLB_map_b0, KRLB_map_b0_b1, KRLB_map_b1_b0, KRLB_map_b1 = (
        np.zeros(mask.shape),
        np.zeros(mask.shape),
        np.zeros(mask.shape),
        np.zeros(mask.shape),
    )

    KRLB_map_b0[mask], KRLB_map_b0_b1[mask], KRLB_map_b1_b0[mask], KRLB_map_b1[mask] = (
        KRLB11,
        KRLB12,
        KRLB21,
        KRLB22,
    )

    return np.stack((KRLB_map_b0, KRLB_map_b0_b1, KRLB_map_b1_b0, KRLB_map_b1), axis=0)


def kramer_rao_T2_multi_echo(y, NLLS_params, TE, mask):

    S0 = NLLS_params[0, :, :][mask]
    T2 = NLLS_params[1, :, :][mask]
    y = y[:, mask]

    KRLB11, KRLB12, KRLB21, KRLB22 = (
        np.zeros(mask.sum()),
        np.zeros(mask.sum()),
        np.zeros(mask.sum()),
        np.zeros(mask.sum()),
    )

    for voxel_index in range(mask.sum()):

        # residuals
        r = y[:, voxel_index] - spin_echo_signal(
            S0=S0[voxel_index],
            T2=T2[voxel_index],
            TE_values=TE,
            mode="voxel",
        )

        # number of data points
        M = len(TE)
        # residual standare error
        noise_var = (1 / (M - 2)) * np.sum(r**2)
        # derivatives
        ds_db1 = dSE_dS0(TE=TE, T2=T2[voxel_index])
        ds_db2 = dSE_dT2(S0=S0[voxel_index], T2=T2[voxel_index], TE=TE)

        f11 = np.sum(ds_db1 * ds_db1) / noise_var
        f12 = np.sum(ds_db1 * ds_db2) / noise_var
        f21 = np.sum(ds_db2 * ds_db1) / noise_var
        f22 = np.sum(ds_db2 * ds_db2) / noise_var

        # f11 = np.sum(np.exp(-2 * TE / T2[voxel_index])) / noise_var
        # f12 = np.sum(
        #    S0[voxel_index]
        #    * TE
        #    * np.exp(-2 * TE / T2[voxel_index])
        #    / (T2[voxel_index] ** 2)
        # ) / (noise_var)
        # f21 = f12
        # f22 = (
        #    np.sum(
        #        ((S0[voxel_index] ** 2 * TE**2) / (T2[voxel_index] ** 4))
        #        * np.exp(-2 * TE / T2[voxel_index])
        #    )
        #    / noise_var
        # )

        # populate fisher matrix and calculate KRLB
        fisher = np.array([[f11, f12], [f21, f22]])
        KRLB = np.linalg.inv(fisher)

        KRLB11[voxel_index] = KRLB[0, 0]
        KRLB12[voxel_index] = KRLB[0, 1]
        KRLB21[voxel_index] = KRLB[1, 0]
        KRLB22[voxel_index] = KRLB[1, 1]

    KRLB_map_b1, KRLB_map_b1_b2, KRLB_map_b2_b1, KRLB_map_b2 = (
        np.zeros(mask.shape),
        np.zeros(mask.shape),
        np.zeros(mask.shape),
        np.zeros(mask.shape),
    )

    KRLB_map_b1[mask], KRLB_map_b1_b2[mask], KRLB_map_b2_b1[mask], KRLB_map_b2[mask] = (
        KRLB11,
        KRLB12,
        KRLB21,
        KRLB22,
    )

    return np.stack((KRLB_map_b1, KRLB_map_b1_b2, KRLB_map_b2_b1, KRLB_map_b2), axis=0)


def kramer_rao_ADC_multi_b(y, NLLS_params, b_values, mask):

    S0 = NLLS_params[0, :, :][mask]
    ADC = NLLS_params[1, :, :][mask]
    y = y[:, mask]

    KRLB11, KRLB12, KRLB21, KRLB22 = (
        np.zeros(mask.sum()),
        np.zeros(mask.sum()),
        np.zeros(mask.sum()),
        np.zeros(mask.sum()),
    )

    for voxel_index in range(mask.sum()):

        # residuals
        r = y[:, voxel_index] - diffusion_signal(
            S0=S0[voxel_index],
            ADC=ADC[voxel_index],
            b_values=b_values,
            mode="voxel",
        )

        # number of data points
        M = len(b_values)
        # residual standare error
        noise_var = (1 / (M - 2)) * np.sum(r**2)
        # derivatives
        ds_db1 = dDWI_dS0(b_values=b_values, ADC=ADC[voxel_index])
        ds_db2 = dDWI_dADC(S0=S0[voxel_index], b_values=b_values, ADC=ADC[voxel_index])

        f11 = np.sum(ds_db1 * ds_db1) / noise_var
        f12 = np.sum(ds_db1 * ds_db2) / noise_var
        f21 = np.sum(ds_db2 * ds_db1) / noise_var
        f22 = np.sum(ds_db2 * ds_db2) / noise_var

        # populate fisher matrix and calculate KRLB
        fisher = np.array([[f11, f12], [f21, f22]])
        KRLB = np.linalg.inv(fisher)

        KRLB11[voxel_index] = KRLB[0, 0]
        KRLB12[voxel_index] = KRLB[0, 1]
        KRLB21[voxel_index] = KRLB[1, 0]
        KRLB22[voxel_index] = KRLB[1, 1]

    KRLB_map_b1, KRLB_map_b1_b2, KRLB_map_b2_b1, KRLB_map_b2 = (
        np.zeros(mask.shape),
        np.zeros(mask.shape),
        np.zeros(mask.shape),
        np.zeros(mask.shape),
    )

    KRLB_map_b1[mask], KRLB_map_b1_b2[mask], KRLB_map_b2_b1[mask], KRLB_map_b2[mask] = (
        KRLB11,
        KRLB12,
        KRLB21,
        KRLB22,
    )

    return np.stack((KRLB_map_b1, KRLB_map_b1_b2, KRLB_map_b2_b1, KRLB_map_b2), axis=0)


def T1_VFA_NLLS_estimator(y, FA_values, TR, B1_corr, mask, bounds, init_guess=[0, 0]):
    class Optimizer:
        def __init__(self, FA_values, TR, B1_corr, y):
            self.y = y
            self.FA_values = FA_values
            self.TR = TR
            self.B1_corr = B1_corr

        def calculate_voxelwise_VFA_signal(self, S0, T1):
            exp_term = np.exp(-TR / T1)

            return (
                S0
                * np.sin(self.B1_corr * self.FA_values)
                * (1 - exp_term)
                / (1 - np.cos(self.B1_corr * self.FA_values) * exp_term)
            )

        def calc_obj(self, tissue_params):
            S0 = tissue_params[0]
            T1 = tissue_params[1]
            diff = self.y - self.calculate_voxelwise_VFA_signal(S0, T1)

            return diff

    S0, T1 = np.zeros(mask.sum()), np.zeros(mask.sum())
    y, B1_corr = y[:, mask], B1_corr[mask]

    for voxel_index in range(mask.sum()):
        obj = Optimizer(
            FA_values=FA_values,
            TR=TR,
            B1_corr=B1_corr[voxel_index],
            y=y[:, voxel_index],
        )

        sol = least_squares(
            fun=obj.calc_obj, x0=init_guess, bounds=(bounds[0], bounds[1])
        )

        S0[voxel_index], T1[voxel_index] = sol.x[0], sol.x[1]

    # reshape results to mask
    S0_NLLS, T1_NLLS = np.zeros(mask.shape), np.zeros(mask.shape)
    S0_NLLS[mask], T1_NLLS[mask] = S0, T1

    return np.stack((S0_NLLS, T1_NLLS), axis=0)


def T2_multi_echo_NLLS_estimator(y, TE_values, mask, bounds, init_guess=[0, 0]):
    class Optimizer:
        def __init__(self, TE_values, y):
            self.y = y
            self.TE_values = TE_values

        def calculate_voxelwise_spin_echo_signal(self, S0, T2):
            exp_term = np.exp(-self.TE_values / T2)

            return S0 * exp_term

        def calc_obj(self, tissue_params):
            S0 = tissue_params[0]
            T2 = tissue_params[1]
            diff = self.y - self.calculate_voxelwise_spin_echo_signal(S0, T2)

            return diff

    S0, T2 = np.zeros(mask.sum()), np.zeros(mask.sum())
    y = y[:, mask]

    for voxel_index in range(mask.sum()):
        obj = Optimizer(
            TE_values=TE_values,
            y=y[:, voxel_index],
        )

        sol = least_squares(
            fun=obj.calc_obj, x0=init_guess, bounds=(bounds[0], bounds[1])
        )

        S0[voxel_index], T2[voxel_index] = sol.x[0], sol.x[1]

    # reshape results to mask
    S0_NLLS, T2_NLLS = np.zeros(mask.shape), np.zeros(mask.shape)
    S0_NLLS[mask], T2_NLLS[mask] = S0, T2

    return np.stack((S0_NLLS, T2_NLLS), axis=0)


def ADC_multi_b_NLLS_estimate(y, b_values, mask, bounds, init_guess=[0, 0]):
    class Optimizer:
        def __init__(self, b_values, y):
            self.y = y
            self.b_values = b_values

        def calculate_voxelwise_ADC_signal(self, S0, ADC):

            return S0 * np.exp(-self.b_values * ADC)

        def calc_obj(self, tissue_params):

            S0 = tissue_params[0]
            ADC = tissue_params[1]

            diff = self.y - self.calculate_voxelwise_ADC_signal(S0, ADC)

            return diff

    S0, ADC = np.zeros(mask.sum()), np.zeros(mask.sum())
    y = y[:, mask]

    for voxel_index in range(mask.sum()):
        obj = Optimizer(y=y[:, voxel_index], b_values=b_values)

        sol = least_squares(
            fun=obj.calc_obj, x0=init_guess, bounds=(bounds[0], bounds[1])
        )

        S0[voxel_index], ADC[voxel_index] = sol.x[0], sol.x[1]

    # reshape results to mask
    S0_NLLS, ADC_NLLS = np.zeros(mask.shape), np.zeros(mask.shape)
    S0_NLLS[mask], ADC_NLLS[mask] = S0, ADC

    return np.stack((S0_NLLS, ADC_NLLS), axis=0)


def dSE_dS0(TE, T2):

    res = np.exp(-TE / T2)

    return res


def dSE_dT2(S0, T2, TE):

    res = S0 * ((TE * np.exp(-TE / T2)) / (T2**2))
    return res


def dSE_dS0_dT2(TE, T2):

    res = (TE * np.exp(-TE / T2)) / (T2**2)
    return res


def dSE_dT2_dT2(S0, TE, T2):

    res = (S0 * TE * np.exp(-TE / T2) * (TE - 2 * T2)) / (T2**4)
    return res


def dSPGR_dS0(T1, FA, TR):

    sin = np.sin(FA)
    cos = np.cos(FA)
    exp = np.exp(-TR / T1)

    res = sin * (1 - exp) / (1 - cos * exp)
    return res


def dSPGR_dT1(S0, T1, FA, TR):

    sin = np.sin(FA)
    cos = np.cos(FA)
    exp = np.exp(-TR / T1)

    res = ((S0 * TR * sin) / (T1**2)) * (((cos - 1) * exp) / (1 - cos * exp) ** 2)
    return res


def dDWI_dS0(b_values, ADC):

    res = np.exp(-b_values * ADC)
    return res


def dDWI_dADC(S0, b_values, ADC):

    res = -S0 * b_values * np.exp(-b_values * ADC)
    return res


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n
