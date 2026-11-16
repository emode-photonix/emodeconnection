"""Drive a headless Chrome over its DevTools pipe: load a page at a size, run script, take a shot.

The client needs no browser library for this. Chrome reads NUL-terminated JSON commands on
file descriptor 3 and answers on 4 (``--remote-debugging-pipe``).
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from typing import Any, Dict, Optional


class Page:
    def __init__(self, chrome: str, url: str, width: int, height: int) -> None:
        to_chrome_r, to_chrome_w = os.pipe()
        from_chrome_r, from_chrome_w = os.pipe()

        def child() -> None:
            os.dup2(to_chrome_r, 3)
            os.dup2(from_chrome_w, 4)

        self.proc = subprocess.Popen(
            [
                chrome,
                "--headless",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=1",
                "--remote-debugging-pipe",
                "--no-sandbox",
                "about:blank",
            ],
            preexec_fn=child,
            close_fds=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.close(to_chrome_r)
        os.close(from_chrome_w)
        self._w = os.fdopen(to_chrome_w, "wb", buffering=0)
        self._r = os.fdopen(from_chrome_r, "rb", buffering=0)
        self._buf = b""
        self._id = 0
        target = self._send("Target.createTarget", {"url": "about:blank"})["targetId"]
        self.session = self._send(
            "Target.attachToTarget", {"targetId": target, "flatten": True}
        )["sessionId"]
        self.cmd(
            "Emulation.setDeviceMetricsOverride",
            {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False},
        )
        self.cmd("Page.navigate", {"url": url})

    def _send(
        self, method: str, params: Dict[str, Any], session: Optional[str] = None
    ) -> Any:
        self._id += 1
        msg: Dict[str, Any] = {"id": self._id, "method": method, "params": params}
        if session:
            msg["sessionId"] = session
        self._w.write(json.dumps(msg).encode() + b"\0")
        while True:
            while b"\0" not in self._buf:
                chunk = self._r.read(65536)
                if not chunk:
                    raise RuntimeError("Chrome closed its pipe")
                self._buf += chunk
            raw, self._buf = self._buf.split(b"\0", 1)
            reply = json.loads(raw)
            if reply.get("id") == self._id:
                if "error" in reply:
                    raise RuntimeError(reply["error"])
                return reply.get("result")

    def cmd(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._send(method, params or {}, self.session)

    def eval(self, expression: str) -> Any:
        got = self.cmd(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        return got["result"].get("value")

    def wait_for(self, expression: str, seconds: float = 20.0) -> None:
        end = time.time() + seconds
        while time.time() < end:
            if self.eval(expression):
                return
            time.sleep(0.2)
        raise TimeoutError(expression)

    def screenshot(self, path: str) -> None:
        data = self.cmd("Page.captureScreenshot", {"format": "png"})["data"]
        with open(path, "wb") as handle:
            handle.write(base64.b64decode(data))

    def close(self) -> None:
        self.proc.kill()
        self.proc.wait()
        self._w.close()
        self._r.close()
