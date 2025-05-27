import os
import pickle
import numpy as np
import scipy.io as sio


def open_file(sim="emode", simulation_name=None):
    """
    Opens an EMode simulation file with either .eph or .mat extension.
    """
    sim = simulation_name or sim
    if not isinstance(sim, str):
        raise TypeError("input parameter 'simulation_name' must be a string")

    ext = ".eph"
    mat = ".mat"
    found = False
    for file in os.listdir():
        if (file == sim + ext) or ((file == sim) and (sim.endswith(ext))):
            found = "eph"
            with open(file, "rb") as fl:
                f = pickle.load(fl)
        elif (file == sim + mat) or ((file == sim) and (sim.endswith(mat))):
            found = "mat"
            f = loadmat(sim + mat)

    if not found:
        raise FileNotFoundError(
            f"no file: {sim}, {sim + ext}, or {sim + mat} was found"
        )

    return f


def get(variable, sim="emode", simulation_name=None):
    """
    Return data from simulation file.
    """
    if not isinstance(variable, str):
        raise TypeError("input parameter 'variable' must be a string")

    sim = simulation_name or sim
    if not isinstance(sim, str):
        raise TypeError("input parameter 'simulation_name' must be a string")

    f = open_file(sim=sim)

    if variable in list(f.keys()):
        data = f[variable]
    else:
        print("Data does not exist.")
        return

    return data


def inspect(sim="emode", simulation_name=None):
    """
    Return list of keys from available data in simulation file.
    """
    sim = simulation_name or sim
    if not isinstance(sim, str):
        raise TypeError("input parameter 'simulation_name' must be a string")

    f = open_file(sim=sim)

    fkeys = list(f.keys())
    fkeys.remove("EMode_simulation_file")
    return fkeys


def loadmat(filename):
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(data):
    force_1D = ["effective_index", "TE_indices", "TM_indices"]
    force_3D = ["Fx", "Fy", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    for key in data:
        if isinstance(data[key], sio.matlab.mio5_params.mat_struct):  # type: ignore
            data[key] = _todict(data[key])

        if key in force_1D:
            if (not isinstance(data[key], list)) and (
                not isinstance(data[key], np.ndarray)
            ):
                data[key] = np.expand_dims(data[key], axis=0)

        if key in force_3D:
            if len(np.shape(data[key])) < 3:
                data[key] = np.expand_dims(data[key], axis=0)

    return data


def _todict(matobj):
    ddict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):  # type: ignore
            ddict[strg] = _todict(elem)
        if isinstance(elem, str):
            ddict[strg] = elem.strip()
        else:
            ddict[strg] = elem
    return ddict
