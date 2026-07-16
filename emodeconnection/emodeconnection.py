import atexit
from datetime import datetime, timezone
from logging import getLogger
from logging.config import dictConfig
import os
from pathlib import Path
import platform
import signal
from subprocess import Popen, PIPE
from threading import Thread
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
        clear: Literal["none", "all", "latest", "oldest", "mine", "others"] = "none",
        save_path: Union[str, Path] = ".",
        verbose: bool = False,
        roaming: bool = False,
        open_existing: bool = False,
        new_name: Union[bool, str] = False,
        priority: Literal["pH", "pAN", "pBN", "pI", "pN"] = "pN",
        emode_cmd: Optional[list[str]] = None,
        force_open: bool = False,
        save: bool = True,
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

        clear: Literal['none', 'all', 'latest', 'oldest', 'mine', 'others'] = 'none'
            Whether to clear old EMode sessions (thus kill previous sessions).
            Clearing is only supported for sessions launched from the same computer.
                'none': Do not clear any old sessions.
                'all': Clear all non-roaming sessions.
                'latest': Clear the latest non-roaming session.
                'oldest': Clear the oldest non-roaming session.
                'mine': Clear only sessions launched by the current user.
                'others': Clear sessions owned by the current user but launched by other users.

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

        if clear not in ["none", "all", "latest", "oldest", "mine", "others"]:
            raise ValueError(
                "parameter 'clear' must be one of ['none', 'all', 'latest', 'oldest', 'mine', 'others']"
            )

        if priority not in ["pH", "pAN", "pN", "pBN", "pI"]:
            raise ValueError(
                "parameter 'priority' must be one of ['pH','pAN','pN','pBN','pI']"
            )

        self._dsim = simulation_name
        self._priority = priority
        self._verbose = verbose
        self._license_type = license_type
        self._clear = clear
        self._roaming = roaming
        self._ext = ".eph"
        self._port_file_label = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        self._cache = Cache(self._port_file_label)
        self._save = save
        self.running = True
        self._in_flight = False

        if self.in_ipython():
            self._proc = Popen(
                self.build_cmd_list(emode_cmd),
                stdout=PIPE,
                stderr=None,
                text=True,
                bufsize=1,
            )
            self.setup_print_thread()
        else:
            self._proc = Popen(
                self.build_cmd_list(emode_cmd),
                stdout=None,
                stderr=None,
                # stderr=None,
            )
        self._client = EModeClient(self._cache)
        atexit.register(self.close_atexit)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self._signal_handler)

        try:
            if open_existing:
                RV = self.call(
                    "EM_open", simulation_name=simulation_name, save_path=save_path, new_simulation_name=new_name, force=force_open
                )
            else:
                RV = self.call("EM_init", simulation_name=simulation_name, save_path=save_path)
        except ConnectionError:
            raise EModeError("EMode failed to launch.")

        self._dsim = RV[len("sim:") :]  # type: ignore

    def build_cmd_list(self, emode_cmd):
        if platform.system() == "Windows":
            cmd = emode_cmd or ["EMode.exe"]
        else:
            cmd = emode_cmd or ["emode"]

        cmd += ["run", self._port_file_label]
        if self._license_type != "default":
            cmd += [f"-{self._license_type}"]
        if self._clear != "none":
            cmd += [f"-{self._clear}"]
        if self._verbose:
            cmd += ["-v"]
        if self._priority != "pN":
            cmd += [self._priority]
        if self._roaming:
            cmd += ["-r"]

        logger.info(f"emode command list: {cmd}")
        self._cmd = cmd
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
        self._print_thread = Thread(
            name="print EMode output",
            target=_forward_stdout,
            args=(self._proc.stdout,),
            daemon=True,
        )
        self._print_thread.start()

    def in_ipython(self):
        try:
            _ = get_ipython()  # type: ignore
            self._ipython = True
        except NameError:
            self._ipython = False

        return self._ipython

    def call(self, function: str, **kwargs) -> Union[dict, Literal['failed'], float, str, list]:
        logger.debug(f"calling '{function}' with args: {kwargs}")
        if not isinstance(function, str):
            raise TypeError("parameter 'function' must be of type 'str'")

        sendset = kwargs
        sendset.update({"function": function})

        if "sim" not in sendset and "simulation_name" not in sendset:
            sendset["simulation_name"] = self._dsim

        self._client.send(sendset)
        self._in_flight = True

        try:
            rv = self._client.recv()
        except EModeError:
            # an error reply is still a complete reply
            self._in_flight = False
            raise
        except ConnectionError:
            logger.debug("connection closed by EMode, shutting down")
            self._client.close()
            self.running = False
            raise

        self._in_flight = False
        return rv

    def close(self, **kwargs):
        logger.debug(f"closing connection with kwargs {kwargs}")
        try:
            kwargs.setdefault('save', self._save)
            if self._in_flight:
                # A call was interrupted mid-flight (e.g. by Ctrl-C): the server
                # answers it before it can see EM_close, so drain that reply first.
                try:
                    self._client.recv()
                except EModeError:
                    pass
                self._in_flight = False
            self.call("EM_close", **kwargs)
            self._client.close()
            self._proc.wait(timeout=60)

        except Exception:
            logger.exception("got exception closing client")

        self.running = False

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            if args:
                kwargs["key"] = args[0]
                if len(args) > 1:
                    raise ValueError("Please pass all arguments as kwargs")
            return self.call("EM_" + name, **kwargs)

        return wrapper

    def _signal_handler(self, signum, frame):
        self.close_atexit()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def close_atexit(self):
        if self._client.connected and self.running:
            self.close(save=self._save)
