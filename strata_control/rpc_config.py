"""strata_control/rpc_config.py - validation for the opt-in "rpc_split" run-config section.

The feature is deliberately narrow: one local prefix worker and the configured controller.  This module checks the
requested split before minutes-long engine loading starts.  It does not start processes: serve/rpc_workers.py does
that once the protocol and worker runtime exist.  When "rpc_split" is absent, callers leave the configuration alone.
"""
from __future__ import annotations

import re

ALLOWED_WORKER_BACKENDS = ("cuda", "sycl")

# Only variables whose purpose is backend/device selection.  A worker inherits the supervisor's environment, but an
# arbitrary config key must not smuggle library paths, model paths, or credential material into a second process.
ALLOWED_WORKER_ENV_KEYS = frozenset({
    "ONEAPI_DEVICE_SELECTOR",
    "ZE_AFFINITY_MASK",
    "CUDA_VISIBLE_DEVICES",
    "STRATA_SYCL_BIN",
})

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_SOCKET_COMPONENT = 64
_RPC_CONFLICT_FLAGS = (
    "--layer-split",
    "--peer-device",
    "--batch",
    "--slots",
    "--vision",
    "--mmproj",
    "--session-save-reclaim",
)
_ALLOWED_KEYS = frozenset({"boundary", "workers"})
_WORKER_KEYS = frozenset({"id", "backend", "device", "engine", "env"})


class RpcConfigError(ValueError):
    """A plain, user-facing configuration error.  Callers present it as a startup failure; nothing is started."""


def _plain(value) -> str:
    return " ".join(str(value).split())


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_env(raw: object, where: str) -> dict[str, str]:
    if raw is None or raw == {}:
        return {}
    if not isinstance(raw, dict):
        raise RpcConfigError(f'{where}: "env" must be an object of allowed backend/device variables')
    env: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or key not in ALLOWED_WORKER_ENV_KEYS:
            allowed = ", ".join(sorted(ALLOWED_WORKER_ENV_KEYS))
            raise RpcConfigError(f'{where}: environment variable {key!r} is not allowed; use only {allowed}')
        if value is None or isinstance(value, bool) or not isinstance(value, (str, int)):
            raise RpcConfigError(f"{where}: {key} must be a backend/device selection string")
        text = str(value)
        if not text or len(text) > 512 or "\0" in text or any(c in text for c in "\r\n"):
            raise RpcConfigError(f"{where}: {key} must be one plain value no longer than 512 characters")
        env[key] = text
    return env


def _validate_worker(raw: object, index: int) -> dict:
    where = f'rpc_split workers[{index}]'
    if not isinstance(raw, dict):
        raise RpcConfigError(f"{where}: each worker must be an object with id, backend, and device")
    unknown = sorted(set(raw) - _WORKER_KEYS)
    if unknown:
        raise RpcConfigError(f"{where}: {' and '.join(unknown)} cannot be set for a worker")

    worker_id = raw.get("id")
    if not isinstance(worker_id, str) or not _ID_RE.fullmatch(worker_id):
        raise RpcConfigError(f"{where}: id must use letters, digits, '.', '_' or '-' and be no longer than 128 characters")

    backend = raw.get("backend")
    if not isinstance(backend, str) or backend not in ALLOWED_WORKER_BACKENDS:
        raise RpcConfigError(f"{where}: backend must be one of {' or '.join(ALLOWED_WORKER_BACKENDS)}")

    device = raw.get("device")
    if not _is_int(device) or device < 0 or device > 127:
        raise RpcConfigError(f"{where}: device must be a whole graphics-card number from 0 to 127")

    engine = raw.get("engine")
    if engine is not None:
        if not isinstance(engine, str) or not engine.strip() or "\0" in engine:
            raise RpcConfigError(f"{where}: engine must name a development engine file, or be omitted to use the installed component")
        engine = engine.strip()

    return {
        "id": worker_id,
        "backend": backend,
        "device": int(device),
        "engine": engine,
        "env": _validate_env(raw.get("env"), where),
    }


def validate_rpc_split(cfg: dict, *, n_layers: int | None = None) -> dict | None:
    """The validated internal plan for `cfg["rpc_split"]`, or None when RPC mode is not requested.

    `n_layers` is optional because setup can validate a saved profile before it opens the model.  When it is known,
    the controller range's end is checked too.  ValueError names the unsupported combination in plain words.
    """
    if not isinstance(cfg, dict):
        raise RpcConfigError("the run config must be a JSON object")
    raw = cfg.get("rpc_split")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RpcConfigError('"rpc_split" must be an object with "boundary" and exactly one worker')

    unknown = sorted(set(raw) - _ALLOWED_KEYS)
    if unknown:
        raise RpcConfigError(f'rpc_split: {" and ".join(unknown)} cannot be set; use "boundary" and "workers"')

    boundary = raw.get("boundary")
    if not _is_int(boundary) or boundary < 3:
        raise RpcConfigError('rpc_split: "boundary" must be a whole number of layers at least 3 (kMinStageLayers)')
    if n_layers is not None:
        if not _is_int(n_layers) or n_layers <= boundary:
            raise RpcConfigError(f'rpc_split: boundary {boundary} must leave the controller at least one layer below {n_layers}')

    workers_raw = raw.get("workers")
    if not isinstance(workers_raw, list) or len(workers_raw) != 1:
        raise RpcConfigError('rpc_split: "workers" must contain exactly one prefix worker in the first version')
    worker = _validate_worker(workers_raw[0], 0)

    controller_backend = cfg.get("backend")
    if not isinstance(controller_backend, str) or controller_backend not in ALLOWED_WORKER_BACKENDS:
        raise RpcConfigError('rpc_split: the controller engine backend must be "cuda" or "sycl"')

    args = cfg.get("args") or []
    if not isinstance(args, list):
        raise RpcConfigError('the run config\'s "args" must be a list')
    flags = {str(a) for a in args}
    conflicts = sorted(flags.intersection(_RPC_CONFLICT_FLAGS))
    if conflicts:
        raise RpcConfigError("rpc_split is opt-in and cannot be combined with " + ", ".join(conflicts))
    if "layer_split" in cfg and cfg.get("layer_split") not in (None, "", "auto"):
        raise RpcConfigError('rpc_split cannot be combined with "layer_split"')

    # Images and multiple concurrent requests cross the new stage boundary in ways that are not implemented yet.
    if cfg.get("vision") is not None:
        raise RpcConfigError('rpc_split cannot be combined with "vision" (image input)')
    if cfg.get("parallel") is not None and cfg.get("parallel") is not False:
        raise RpcConfigError('rpc_split cannot be combined with "parallel" (batch)')
    if cfg.get("replicas") is not None and cfg.get("replicas") is not False:
        raise RpcConfigError('rpc_split cannot be combined with "replicas"')

    # The controller owns the suffix.  More than one controller-visible GPU implies the old in-process split or a
    # peer expert tier, and card roles have not been adapted to the process boundary.
    gpu = cfg.get("gpu")
    cards = gpu if isinstance(gpu, (list, tuple)) else str(gpu).split(",") if gpu not in (None, "") else []
    try:
        card_count = len([int(str(x).strip()) for x in cards if str(x).strip() != ""])
    except ValueError:
        raise RpcConfigError('rpc_split: "gpu" must name one controller card')
    if card_count != 1:
        raise RpcConfigError("rpc_split: the controller must name exactly one graphics card; the prefix worker names its own device")

    session_restore_keys = [key for key in ("session_restore", "restore_after_load") if cfg.get(key) is not None]
    if session_restore_keys:
        raise RpcConfigError("rpc_split cannot be combined with " + ", ".join(session_restore_keys))

    if n_layers is None:
        controller_range = f"{boundary}:?"
    else:
        controller_range = f"{boundary}:{int(n_layers)}"
    return {
        "boundary": int(boundary),
        "worker": worker,
        "worker_args": {
            "rpc-listen": "worker.sock",
            "rpc-layer-range": f"0:{int(boundary)}",
        },
        "controller_args": {
            "rpc-connect": "worker.sock",
            "rpc-layer-range": controller_range,
        },
    }


def engine_args_for(plan: dict, *, socket_path: str, worker_id: str | None = None,
                    n_layers: int | None = None) -> tuple[list[str], list[str]]:
    """The generated worker and controller engine arguments, with the supervisor's actual socket path inserted.

    The token is never part of these arguments; Task 4 passes it through the protected environment/fd channel.
    """
    if not isinstance(plan, dict) or "worker_args" not in plan or "controller_args" not in plan:
        raise RpcConfigError("an RPC split plan must be validated before engine arguments can be generated")
    if not isinstance(socket_path, str) or not socket_path or "\0" in socket_path:
        raise RpcConfigError("the RPC socket path is not valid")
    if len(socket_path.rsplit("/", 1)[-1]) > _MAX_SOCKET_COMPONENT:
        raise RpcConfigError(f"the RPC socket file name must be no longer than {_MAX_SOCKET_COMPONENT} characters")
    if n_layers is not None:
        if not _is_int(n_layers) or n_layers <= plan["boundary"]:
            raise RpcConfigError("the model's layer count must be a whole number above the RPC boundary")

    worker = plan["worker"]
    if worker_id is not None and worker_id != worker.get("id"):
        raise RpcConfigError("the worker id does not match the validated RPC split plan")
    boundary = plan["boundary"]
    worker = [str(x) for key, value in (("rpc-listen", socket_path), ("rpc-layer-range", f"0:{boundary}"))
              for x in (f"--{key}", str(value))]
    controller_end = int(n_layers) if n_layers is not None else "?"
    controller = [str(x) for key, value in (("rpc-connect", socket_path),
                                           ("rpc-layer-range", f"{boundary}:{controller_end}"))
                  for x in (f"--{key}", str(value))]
    return worker, controller


def socket_component(plan: dict, worker_id: str | None = None) -> str:
    """A bounded, collision-resistant file name component for one worker's local socket."""
    worker = plan.get("worker") or {}
    chosen = worker_id or worker.get("id") or "prefix"
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", str(chosen))[:_MAX_SOCKET_COMPONENT]
    return f"strata-rpc-{safe}"
