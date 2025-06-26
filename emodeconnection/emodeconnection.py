import atexit
from datetime import datetime, timezone
from logging import getLogger
from logging.config import dictConfig
import os
from pathlib import Path
import platform
from subprocess import Popen, PIPE
from threading import Thread
import time
from typing import Optional, Literal, Union
from .emodeclient import EModeClient
from .file_utils import Cache
from .types import EModeError

logger = getLogger(__name__)


def _forward_stdout(pipe):
    try:
        for line in iter(pipe.readline, ""):
            # print() => IPython captures it into the cell output
            print(line, end="", flush=True)
    finally:
        pipe.close()

class EMode:
    def __init__(
        self,
        sim: Optional[str] = None,
        simulation_name: Optional[str] = "emode",
        license_type: Literal["2d", "3d", "default"] = "default",
        save_path: Union[str, Path] = ".",
        verbose: bool = False,
        roaming: bool = False,
        open_existing: bool = False,
        new_name: bool = False,
        priority: Literal["pH", "pAN", "pBN", "pI", "pN"] = "pN",
        emode_cmd: Optional[list[str]] = None,
    ):
        """
        Initialize defaults and create an EMode session.

        parameters:
        -----
        simulation_name: str
            The name of the default simulation to load.

        sim: str
            Alias for simulation_name.  simulation_name takes precedence.

        license_type: Literal['2d','3d','default'] = 'default'
            The type of license you wish to check out for this session.

        save_path: str | Path = '.'
            The path to save results.

        verbose: bool = False
            Verbose output from EMode.

        roaming: bool = False
            Enable roaming mode

        open_existing: bool = False
            open an existing simulation.  If False, and <save_path>/<simulation_name>.eph
            exists, <save_path>/<simulation_name>_0.eph will be created.

        new_name: bool = False
            I'm not sure what this does?

        priority: Literal['pH', 'pAN', 'pN', 'pBN', 'pI'] = 'pN'
            The EMode process priority:
                'pH': High priority
                'pAN': Above Normal priority
                'pN': Normal priority
                'pBN': Below Normal priority
                'pI': Idle priority

        emode_cmd: Optional[list[str]] = None
            The command to use to invoke EMode as a command list.  This shouldn't need to be
            modified if you installed EMode normally.
        """
        self.setup_logging()

        if sim:
            logger.warning("The `sim` argument in the `EMode` class is depreciated, use `simulation_name` instead.")
        
        simulation_name = simulation_name or sim

        if not isinstance(simulation_name, str):
            raise TypeError("parameter 'simulation_name' must be a string")

        if not isinstance(save_path, (str, Path)):
            raise TypeError("parameter 'save_path' must be a string or pathlib.Path")

        if license_type not in ["2d", "3d", "default"]:
            raise ValueError(
                "parameter 'license_type' must be one of ['2d','3d','default']"
            )

        if priority not in ["pH", "pAN", "pN", "pBN", "pI"]:
            raise ValueError(
                "parameter 'priority' must be one of ['pH','pAN','pN','pBN','pI']"
            )

        self.dsim = simulation_name
        self.priority = priority
        self.verbose = verbose
        self.license_type = license_type
        self.roaming = roaming
        self.ext = ".eph"
        self.port_file_label = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        self.cache = Cache(self.port_file_label)

        if self.in_ipython():
            self.proc = Popen(
                self.build_cmd_list(emode_cmd),
                stdout=PIPE,
                stderr=None,
                text=True,
                bufsize=1,
            )
            self.setup_print_thread()
        else:
            self.proc = Popen(
                self.build_cmd_list(emode_cmd),
                stdout=None,
                stderr=None,
                # stderr=None,
            )
        self.client = EModeClient(self.cache)
        atexit.register(self.close_atexit)

        if open_existing:
            RV = self.call(
                "EM_open", simulation_name=simulation_name, save_path=save_path, new_simulation_name=new_name
            )
        else:
            RV = self.call("EM_init", simulation_name=simulation_name, save_path=save_path)

        if RV == "failed":
            raise EModeError("EMode failed to launch.")
        self.dsim = RV[len("sim:") :]

    def build_cmd_list(self, emode_cmd):
        if platform.system() == "Windows":
            cmd = emode_cmd or ["EMode.exe"]
        else:
            cmd = emode_cmd or ["emode"]

        cmd += ["run", self.port_file_label]
        if self.license_type != "default":
            cmd += [f"-{self.license_type}"]
        if self.verbose:
            cmd += ["-v"]
        if self.priority != "pN":
            cmd += [self.priority]
        if self.roaming:
            cmd += ["-r"]

        logger.info(f"emode command list: {cmd}")
        self.cmd = cmd
        return cmd

    def setup_logging(self):
        if log_level := os.getenv("EMODE_LOGGING", "WARNING"):
            dictConfig(
                {
                    "version": 1,
                    "disable_existing_loggers": False,
                    "formatters": {
                        "std": {
                            "format": "%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(funcName)s %(message)s"
                        }
                    },
                    "handlers": {
                        "console": {
                            "class": "logging.StreamHandler",
                            "formatter": "std",
                            "level": log_level,
                        }
                    },
                    "loggers": {
                        "": {"handlers": ["console"], "level": log_level},
                        "matplotlib": {"level": "WARNING"},
                        "PIL": {"level": "WARNING"},
                    },
                }
            )

    def setup_print_thread(self):
        # self.print_thread = Thread(
        #     target=lambda: shutil.copyfileobj(
        #         self.proc.stdout, sys.stdout.buffer, length=1
        #     ),
        #     daemon=True,
        # )
        self.print_thread = Thread(
            name="print EMode output",
            target=_forward_stdout,
            args=(self.proc.stdout,),
            daemon=True,
        )
        self.print_thread.start()

    def in_ipython(self):
        try:
            _ = get_ipython()  # type: ignore
            self.ipython = True
        except NameError:
            self.ipython = False

        return self.ipython

    def call(self, function: str, **kwargs):
        logger.debug(f"calling '{function}' with args: {kwargs}")
        if not isinstance(function, str):
            raise TypeError("parameter 'function' must be of type 'str'")

        sendset = kwargs
        sendset.update({"function": function})

        if "sim" not in sendset and "simulation_name" not in sendset:
            sendset["simulation_name"] = self.dsim

        self.client.send(sendset)

        try:
            return self.client.recv()
        except ConnectionResetError:
            logger.debug("connection reset from EMode, shutting down")
            self.client.close()
            return "failed"

    def close(self, **kwargs):
        logger.debug(f"closing connection with kwargs {kwargs}")
        try:
            self.call("EM_close", **kwargs)
            self.client.close()

            while True:
                time.sleep(0.01)
                if self.proc.poll() is None or self.proc.poll() is self.proc.returncode:
                    break
                time.sleep(0.01)

        except Exception:
            logger.exception("got exception closing client")

        return

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            if args:
                kwargs["key"] = args[0]
                if len(args) > 1:
                    raise ValueError("Please pass all arguments as kwargs")
            return self.call("EM_" + name, **kwargs)

        return wrapper

    def close_atexit(self):
        if self.client.connected:
            self.close()

        return
