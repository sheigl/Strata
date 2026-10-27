"""serve/rpc_workers.py - the supervisor's local RPC worker process.

The first RPC feature has exactly one worker: the prefix stage.  This module resolves and starts that worker before
the controller engine, authenticates the per-run handshake, keeps the process registered so it cannot outlive the
server, and always reaps and unlinks it when the server stops.  A worker is intentionally never restarted after a
conversation has begun: restarting would leave the controller's KV state and the worker's layer state inconsistent.
"""
from __future__ import annotations

import errno
import hashlib
import os
import secrets
import socket
import struct
import subprocess
import time
from pathlib import Path

from strata_control import engine_component, private_files, rpc_config, runtime as runtime_mod
from serve import lifeline

MAGIC = 0x43505253
VERSION = 1
HEADER_SIZE = 40
MAX_CONTROL_PAYLOAD = 16 * 1024
OP_HELLO = 1
OP_HELLO_ACK = 2
OP_SHUTDOWN = 16
HEADER_STRUCT = struct.Struct("<IHHQIIQQ")
TOKEN_ENV = "STRATA_RPC_TOKEN"
SOCKET_ENV = "STRATA_RPC_SOCKET"


class RpcWorkerError(RuntimeError):
    """A worker could not be prepared, reached, or ended cleanly."""


def _encode(opcode: int, request_id: int, payload: bytes = b"", stage_id: int = 0) -> bytes:
    return HEADER_STRUCT.pack(MAGIC, VERSION, opcode, request_id, stage_id, 0, len(payload), 0) + payload


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks = []
    left = count
    while left:
        chunk = sock.recv(left)
        if not chunk:
            raise ConnectionError("the RPC worker closed the control socket")
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)


def _recv_frame(sock: socket.socket) -> tuple[int, int, bytes]:
    header = _recv_exact(sock, HEADER_SIZE)
    magic, version, opcode, request_id, _stage, _flags, payload_len, bulk = HEADER_STRUCT.unpack(header)
    if magic != MAGIC or version != VERSION or bulk:
        raise ConnectionError("the RPC worker used a protocol this supervisor does not understand")
    if payload_len > MAX_CONTROL_PAYLOAD:
        raise ConnectionError("the RPC worker sent an oversized control payload")
    return opcode, request_id, _recv_exact(sock, payload_len)


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


class RpcWorkerSupervisor:
    """Starts one prefix worker and its Unix-domain control channel.

    The worker owns the listening socket; the supervisor connects to it after the process brings it up.  This order
    mirrors ``LocalTransport`` in the engine: ``listen`` is the worker, ``connect`` is the controller.
    """

    def __init__(self, plan: dict, *, base_engine_args: list[str] | None = None,
                 log_path: str | None = None, ready_timeout_s: float = 900.0,
                 env: dict[str, str] | None = None, cwd: str | Path | None = None,
                 n_layers: int | None = None):
        if not isinstance(plan, dict) or "worker" not in plan or "boundary" not in plan:
            raise RpcWorkerError("the RPC worker needs a validated rpc_split plan")
        if os.name != "posix" or not hasattr(socket, "AF_UNIX"):
            raise RpcWorkerError("rpc_split workers are currently supported on Linux and macOS POSIX hosts")
        self.plan = plan
        self.base_engine_args = [str(x) for x in (base_engine_args or [])]
        self.log_path = log_path
        self.ready_timeout_s = max(0.05, float(ready_timeout_s))
        self.env = dict(os.environ if env is None else env)
        self.cwd = str(cwd) if cwd is not None else None
        self.n_layers = n_layers
        self.token = secrets.token_hex(32)
        self.process: subprocess.Popen | None = None
        self.exit_code: int | None = None
        self.argv: list[str] = []
        self.socket_path = self._make_socket_path()
        self._sock: socket.socket | None = None
        self._request_id = 0
        self._log_file = None
        self._released = False
        self._handed_off = False

    def __enter__(self) -> "RpcWorkerSupervisor":
        self.start()
        return self

    def __exit__(self, *_exc) -> bool:
        self.shutdown()
        return False

    def controller_args(self, n_layers: int | None = None) -> list[str]:
        """The controller arguments the supervisor generated for its worker (token stays out of argv)."""
        layers = n_layers if n_layers is not None else self.n_layers
        _, args = rpc_config.engine_args_for(self.plan, socket_path=str(self.socket_path),
                                             worker_id=self.plan["worker"].get("id"), n_layers=layers)
        return args

    def start(self) -> None:
        """Launch the worker, wait for its socket, authenticate the handshake, and keep the socket open."""
        if self.process is not None:
            raise RpcWorkerError("the RPC worker supervisor can start its one worker only once")
        try:
            exe = self._resolve_executable()
            self._unlink_old_socket()
            worker = self.plan["worker"]
            boundary = int(self.plan["boundary"])
            self.argv = [exe, "--serve", "--rpc-listen", str(self.socket_path),
                         "--rpc-layer-range", f"0:{boundary}", *self.base_engine_args]
            child_env = self._child_env()
            self._log_file = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
            self.process = lifeline.popen(
                self.argv, role="rpc-worker", cwd=self.cwd, env=child_env,
                stdin=subprocess.DEVNULL, stdout=self._log_file, stderr=subprocess.STDOUT,
                close_fds=True, start_new_session=True,
            )
            self._wait_and_handshake()
        except Exception:
            self.shutdown(grace_s=0.2)
            raise

    def poll(self) -> int | None:
        return None if self.process is None else self.process.poll()

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None and (self._sock is not None or self._handed_off)

    def raise_if_dead(self, why: str = "the RPC worker process ended") -> None:
        """Fail a later controller action before it sends work into a lost stage (no restart)."""
        rc = self.poll()
        if rc is not None:
            raise RpcWorkerError(f"{why} (exit {rc}); a worker is not restarted into an existing conversation")

    def hand_off_socket(self) -> int:
        """Give the authenticated control socket to the controller engine, which continues its request sequence.

        The returned descriptor is inheritable and must be passed to the controller's Popen.  This supervisor no longer
        writes protocol frames on the connection: two writers would share one request sequence and corrupt it.  It
        keeps process supervision and terminates the worker if the controller disappears.
        """
        if self._handed_off:
            raise RpcWorkerError("the RPC worker control socket was already handed to the controller")
        if self._sock is None:
            raise RpcWorkerError("the RPC worker control socket is not connected")
        try:
            fd = os.dup(self._sock.fileno())
            os.set_inheritable(fd, True)
        except OSError as exc:
            raise RpcWorkerError(f"cannot pass the RPC control socket to the controller: {exc}") from exc
        try:
            self._sock.close()
        finally:
            self._sock = None
            self._handed_off = True
        return fd

    def shutdown(self, grace_s: float = 5.0) -> None:
        """Close the control socket (if not handed off), then end and reap the worker."""
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                self._request_id += 1
                sock.sendall(_encode(OP_SHUTDOWN, self._request_id))
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        proc, self.process = self.process, None
        if proc is not None:
            try:
                proc.wait(timeout=max(0.0, float(grace_s)))
            except subprocess.TimeoutExpired:
                try:
                    proc.terminate()
                    proc.wait(timeout=2.0)
                except (subprocess.TimeoutExpired, OSError):
                    try:
                        proc.kill()
                        proc.wait(timeout=2.0)
                    except (subprocess.TimeoutExpired, OSError):
                        pass
            try:
                self.exit_code = proc.poll()
            except OSError:
                self.exit_code = None
            if not self._released:
                lifeline.release(proc)
                self._released = True
        if self._log_file is not None and self._log_file is not subprocess.DEVNULL:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None
        _unlink(self.socket_path)

    stop = shutdown

    def _make_socket_path(self) -> Path:
        base = runtime_mod.runtime_dir() / "rpc"
        private_files.private_mkdir(base, parents=True, exist_ok=True)
        digest = hashlib.sha256(f"{self.plan['worker'].get('id', 'prefix')}:{self.token}".encode()).hexdigest()[:16]
        path = base / f"strata-rpc-{digest}.sock"
        if len(os.fsencode(path)) > 100:
            raise RpcWorkerError("the RPC runtime directory does not leave enough room for a Unix socket path")
        return path

    def _unlink_old_socket(self) -> None:
        try:
            if self.socket_path.exists():
                raise RpcWorkerError(f"the RPC socket path already exists: {self.socket_path}")
        except OSError as exc:
            raise RpcWorkerError(f"cannot check the RPC socket path: {exc}") from exc

    def _resolve_executable(self) -> str:
        explicit = self.plan["worker"].get("engine")
        try:
            if explicit:
                return str(engine_component.expand(explicit, {"backend": self.plan["worker"]["backend"]}, full=True))
            return engine_component.expand("component:strata-engine",
                                           {"backend": self.plan["worker"]["backend"]}, full=True)
        except engine_component.EngineMissing as exc:
            raise RpcWorkerError(f"the {self.plan['worker']['backend']} engine component for RPC worker "
                                 f"{self.plan['worker'].get('id', 'prefix')!r} is not installed ({exc})") from exc
        except (OSError, ValueError) as exc:
            raise RpcWorkerError(f"cannot use the engine file for RPC worker "
                                 f"{self.plan['worker'].get('id', 'prefix')!r}: {exc}") from exc

    def _child_env(self) -> dict[str, str]:
        env = dict(self.env)
        env.update(self.plan["worker"].get("env") or {})
        env[TOKEN_ENV] = self.token
        env[SOCKET_ENV] = str(self.socket_path)
        # A worker is a GPU stage, not an API client.  It gets the run token, not model or serving secrets.
        for name in ("STRATA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            env.pop(name, None)
        return env

    def _wait_and_handshake(self) -> None:
        assert self.process is not None
        deadline = time.monotonic() + self.ready_timeout_s
        last_error = "the worker did not create its socket"
        while True:
            rc = self.process.poll()
            if rc is not None:
                raise RpcWorkerError(f"the RPC worker ended before it was ready (exit {rc})")
            if time.monotonic() > deadline:
                raise RpcWorkerError(f"the RPC worker was not ready in {self.ready_timeout_s:g} s: {last_error}")
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(max(0.05, deadline - time.monotonic()))
                sock.connect(str(self.socket_path))
                self._sock = sock
                break
            except OSError as exc:
                try:
                    sock.close()
                except (OSError, NameError):
                    pass
                if exc.errno not in (errno.ENOENT, errno.ECONNREFUSED, errno.EAGAIN, errno.EWOULDBLOCK,
                                     errno.ENAMETOOLONG):
                    raise RpcWorkerError(f"cannot connect to the RPC worker socket: {exc}") from exc
                last_error = str(exc)
                time.sleep(0.05)

        assert self._sock is not None
        self._request_id += 1
        self._sock.sendall(_encode(OP_HELLO, self._request_id, self.token.encode()))
        try:
            opcode, request_id, payload = _recv_frame(self._sock)
        except (ConnectionError, OSError) as exc:
            raise RpcWorkerError(f"the RPC worker did not complete the handshake: {exc}") from exc
        if opcode != OP_HELLO_ACK or request_id != self._request_id or payload != self.token.encode():
            raise RpcWorkerError("the RPC worker did not present this run's token")
