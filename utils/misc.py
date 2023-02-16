import os
import time
from collections import OrderedDict
from pathlib import Path
from shutil import copyfile
from typing import Dict


class _Parameters:
    # Function that parses parameter dict to parameter class
    def __init__(self, parameter_dict: Dict):
        for parameter_name in parameter_dict.keys():
            if isinstance(parameter_dict[parameter_name], dict):
                setattr(
                    _Parameters,
                    parameter_name,
                    _Parameter_subclass(parameter_dict[parameter_name]),
                )


class _Parameter_subclass:
    def __init__(self, parameter_subdict):
        for parameter_name in parameter_subdict.keys():

            setattr(
                _Parameter_subclass,
                parameter_name,
                parameter_subdict[parameter_name],
            )


def _copy_dataset_to_results_path(path_data, path_results):

    for file_name in os.listdir(path_data):
        if ".npy" in file_name or ".npz" in file_name:
            copyfile(path_data + file_name, path_results + file_name)

    return


def rename_modules(sequential, string, number):
    _modules = OrderedDict()
    for key in sequential._modules.keys():
        module = sequential._modules[key]
        if len(key) == 1:
            _module_name = "{}_{}_{}".format(module._get_name(), string, number)
            # now it's only equipped for two same modules
            if _module_name not in _modules.keys():
                _modules[_module_name] = module
            else:
                _modules[_module_name + "_1"] = module
        else:
            _modules[key] = module
    sequential._modules = _modules
    return number + 1


def _turn_of_bayesian_settings(param_dict):

    print("disabling bayesian approximation...")

    for probability in [
        "dropout_p_down",
        "dropout_p_up",
        "dropout_p_skip",
        "dropout_p_output",
    ]:

        print(f"setting {probability} to 0")
        param_dict["network"][probability] = 0

    for dropout_mode in ["dropout_mode_down", "dropout_mode_up"]:
        print(f"setting {dropout_mode} to None")
        param_dict["network"][dropout_mode] = "None"

    param_dict["training"]["weight_decay"] = 0
    print("setting weight_decay to 0")

    for n_samples in ["num_mc_iter", "num_mc_iter_final"]:
        # set to 2 samples instead of just one so the rest of the code works
        # (mean values etc)
        param_dict["training"][n_samples] = 2
        print(f"setting {n_samples} to 2")

    return param_dict


def _create_results_folders(parameter_dict, timestamp=False):

    # create results folder if required
    Path(parameter_dict["path"]["results"]).mkdir(parents=True, exist_ok=True)

    # create dir to store results
    if timestamp:
        parameter_dict["path"]["results"] = (
            parameter_dict["path"]["results"] + str(int(time.time())) + "/"
        )
        os.mkdir(parameter_dict["path"]["results"])

    # make subdir to store progress plots
    parameter_dict["path"]["progress_plots"] = (
        parameter_dict["path"]["results"] + "progress_plots/"
    )
    os.mkdir(parameter_dict["path"]["progress_plots"])

    # make subdir to store data for interactive slider plots
    parameter_dict["path"]["sliders"] = (
        parameter_dict["path"]["results"] + "data_for_sliders/"
    )
    os.mkdir(parameter_dict["path"]["sliders"])

    return parameter_dict
