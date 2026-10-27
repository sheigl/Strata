"""Lifecycle tests for one RPC worker process and its Unix-socket handshake (no GPU or engine binary)."""
from __future__ import annotations

import os
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path

from strata_control import rpc_config as R
from serve import rpc_workers

FAKE_WORKER = r'''#!/usr/bin/env python3
import os, socket, struct, sys, time

args = sys.argv[1:]
listen = args[args.index("--rpc-listen") + 1]
mode = os.environ.get("FAKE_RPC_MODE", "success")
marker = os.environ.get("FAKE_RPC_MARKER")
token = os.environ["STRATA_RPC_TOKEN"].encode()

def note(value):
    if marker:
        with open(marker, "a", encoding="utf-8") as f:
            f.write(value + "\n")

note("started")
if mode == "crash":
    raise SystemExit(3)
if mode == "never":
    time.sleep(10)
    raise SystemExit(0)
if mode == "slow":
    time.sleep(0.25)

srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(listen)
srv.listen(1)
conn, _addr = srv.accept()

def recvn(count):
    out = bytearray()
    while len(out) < count:
        chunk = conn.recv(count - len(out))
        if not chunk:
            raise EOFError
        out += chunk
    return bytes(out)

magic, version, opcode, request_id, _stage, _flags, payload_len, bulk = struct.unpack("<IHHQIIQQ", recvn(40))
if magic != 0x43505253 or version != 1 or opcode != 1 or bulk:
    raise SystemExit(4)
if recvn(payload_len) != token:
    raise SystemExit(5)
note("ready")
conn.sendall(struct.pack("<IHHQIIQQ", 0x43505253, 1, 2, request_id, 0, 0, len(token), 0) + token)
while True:
    try:
        if not conn.recv(4096):
            break
    except OSError:
        break
conn.close()
srv.close()
try:
    os.unlink(listen)
except FileNotFoundError:
    pass
'''


class RpcWorkerSupervisorTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("STRATA_RUNTIME_DIR", "FAKE_RPC_MODE", "FAKE_RPC_MARKER")}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def supervisor(self, tmp: Path, mode: str, ready_timeout_s: float = 2.0):
        script = tmp / "fake_worker.py"
        script.write_text("#!/usr/bin/env python3\n" + FAKE_WORKER, encoding="utf-8")
        script.chmod(0o700)
        os.environ["STRATA_RUNTIME_DIR"] = str(tmp / "runtime")
        os.environ["FAKE_RPC_MODE"] = mode
        marker = tmp / "marker"
        os.environ["FAKE_RPC_MARKER"] = str(marker)
        cfg = {
            "backend": "cuda", "gpu": 0, "args": ["--model", "model.gguf"],
            "rpc_split": {"boundary": 4, "workers": [{
                "id": "fake-prefix", "backend": "sycl", "device": 0, "engine": str(script),
            }]},
        }
        plan = R.validate_rpc_split(cfg)
        return rpc_workers.RpcWorkerSupervisor(plan, base_engine_args=[],
                                               log_path=str(tmp / "worker.log"),
                                               ready_timeout_s=ready_timeout_s), marker

    def cmdline(self, pid: int) -> bytes:
        path = Path(f"/proc/{pid}/cmdline")
        return path.read_bytes() if path.exists() else b""

    def test_start_waits_for_handshake_then_shutdown_reaps_it(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmpdir:
            tmp = Path(tmpdir)
            sup, marker = self.supervisor(tmp, "success")
            try:
                started = time.monotonic()
                sup.start()
                self.assertGreaterEqual(time.monotonic() - started, 0.0)
                self.assertIn("ready", marker.read_text())
                self.assertTrue(sup.socket_path.exists())
                self.assertTrue(sup.alive())
                self.assertNotIn(sup.token, sup.argv)
                self.assertNotIn(sup.token.encode(), self.cmdline(sup.process.pid))
                controller = sup.controller_args(n_layers=8)
                self.assertEqual(controller[:2], ["--rpc-connect", str(sup.socket_path)])
                self.assertEqual(controller[3], "4:8")
            finally:
                sup.shutdown()
            self.assertEqual(sup.exit_code, 0)
            self.assertFalse(sup.alive())
            self.assertFalse(sup.socket_path.exists())
            log = (tmp / "worker.log").read_text()
            self.assertNotIn(sup.token, log)

    def test_authenticated_socket_is_dup_for_the_controller_then_supervision_continues(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmpdir:
            tmp = Path(tmpdir)
            sup, _marker = self.supervisor(tmp, "success")
            try:
                sup.start()
                fd = sup.hand_off_socket()
                try:
                    self.assertTrue(os.fstat(fd).st_mode)
                    self.assertTrue(sup._handed_off)
                    self.assertIsNone(sup._sock)
                    self.assertTrue(sup.alive())
                finally:
                    os.close(fd)
            finally:
                sup.shutdown(grace_s=1.0)

    def test_startup_order_waits_for_a_slow_worker(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmpdir:
            sup, marker = self.supervisor(Path(tmpdir), "slow")
            try:
                started = time.monotonic()
                sup.start()
                elapsed = time.monotonic() - started
                self.assertGreaterEqual(elapsed, 0.20)
                self.assertIn("ready", marker.read_text())
            finally:
                sup.shutdown()

    def test_ready_timeout_ends_the_worker_without_orphans(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmpdir:
            sup, marker = self.supervisor(Path(tmpdir), "never", ready_timeout_s=0.20)
            with self.assertRaisesRegex(rpc_workers.RpcWorkerError, "was not ready"):
                sup.start()
            self.assertIn("started", marker.read_text())
            self.assertFalse(sup.alive())
            self.assertFalse(sup.socket_path.exists())
            self.assertEqual(sup.exit_code, -15)     # terminated by the supervisor, not left running

    def test_startup_crash_is_reported_and_the_socket_is_cleaned(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmpdir:
            sup, _marker = self.supervisor(Path(tmpdir), "crash")
            with self.assertRaisesRegex(rpc_workers.RpcWorkerError, "ended before it was ready"):
                sup.start()
            self.assertEqual(sup.exit_code, 3)
            self.assertFalse(sup.socket_path.exists())

    def test_worker_is_never_restarted_into_an_existing_conversation(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmpdir:
            sup, _marker = self.supervisor(Path(tmpdir), "success")
            try:
                sup.start()
                sup.process.terminate()
                deadline = time.monotonic() + 2.0
                while sup.process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                with self.assertRaisesRegex(rpc_workers.RpcWorkerError, "not restarted"):
                    sup.raise_if_dead()
                with self.assertRaisesRegex(rpc_workers.RpcWorkerError, "start its one worker only once"):
                    sup.start()
            finally:
                sup.shutdown()


if __name__ == "__main__":
    unittest.main()
