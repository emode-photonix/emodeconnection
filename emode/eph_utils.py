import os
import pickle

from .types import FileError, reconstruct

MAT_REMOVED = (
    "simulation .mat files are no longer supported as of EMode 1.0.9. Save and open "
    "simulations as .eph files, and read results as numpy arrays with get(). To convert "
    "an old .mat file, open it with EMode 1.0.8 and save it as .eph."
)


def open_file(sim="emode", simulation_name=None):
    """
    Opens an EMode simulation file with the .eph extension.
    """
    sim = simulation_name or sim
    if not isinstance(sim, str):
        raise TypeError("input parameter 'simulation_name' must be a string")

    ext = ".eph"
    if sim.lower().endswith(".mat"):
        raise FileError(MAT_REMOVED, sim)

    name = sim if sim.endswith(ext) else sim + ext
    if not os.path.isfile(name):
        if os.path.isfile(sim + ".mat"):
            raise FileError(MAT_REMOVED, sim + ".mat")
        raise FileNotFoundError(f"no file: {sim} or {sim + ext} was found")

    with open(name, "rb") as fl:
        return pickle.load(fl)


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

    # Simulation files store registered wire types (e.g. SMatrix) as tagged
    # dicts; rebuild them the same way the live socket client does on recv().
    data = reconstruct(data)
    if variable == "sweep_data" and isinstance(data, dict) and "key" in data and "values" in data:
        # the file keeps the sweep's plain dict, which plots as the live result does
        from .sweep import SweepResult

        data = SweepResult(data=data)
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
