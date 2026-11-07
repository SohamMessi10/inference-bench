"""Starting and stopping a model server process, for cold starts and experiments."""

from __future__ import annotations

import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


class ServerStartError(RuntimeError):
    """The server process exited or never answered."""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ServerProcess:
    """`with ServerProcess("--model", "digits-small", ...) as server:` starts a server and waits for it.

    The server's error output goes to a temporary file, not a pipe: nobody reads a pipe while the
    server runs, so a talkative server could fill it and freeze. `startup_s` is the time from
    starting the process to its first /health answer.
    """

    def __init__(self, *serve_args: str, models_dir: Path | None = None, timeout: float = 60.0):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.args = [sys.executable, "-m", "inferbench.cli", "serve", "--port", str(self.port), *serve_args]
        if models_dir is not None:
            self.args += ["--models-dir", str(models_dir)]
        self.timeout = timeout
        self.startup_s = 0.0

    def __enter__(self) -> ServerProcess:
        self.log = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        started = time.perf_counter()
        self.proc = subprocess.Popen(self.args, stdout=subprocess.DEVNULL, stderr=self.log)
        try:
            while True:
                try:
                    urllib.request.urlopen(f"{self.url}/health", timeout=1).close()
                    break
                except OSError:
                    if self.proc.poll() is not None:
                        raise ServerStartError(f"the server exited: {self.errors()[-500:]}") from None
                    if time.perf_counter() - started > self.timeout:
                        raise ServerStartError(f"the server didn't answer within {self.timeout:g} s") from None
                    time.sleep(0.005)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        self.startup_s = time.perf_counter() - started
        return self

    def errors(self) -> str:
        self.log.seek(0)
        return self.log.read().strip()

    def __exit__(self, *exc) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.log.close()
