#!/usr/bin/env bash
# strata-sycl-host.sh: run the SYCL-built engine natively (no Docker) as a drop-in `exe` for serve/server.py
# (--engine strata), for a machine with oneAPI installed on the host. This mirrors sycl/serve/strata-sycl.sh's
# environment handling but execs the binary directly instead of in a container, so the config's args use HOST
# paths (no /work mount) and STRATA_SYCL_ROOT is unused.
#
#   STRATA_SYCL_BIN    the engine binary, relative to the repo   (default: build-sycl-aot/strata)
#   STRATA_ONEAPI      the oneAPI root to source setvars.sh from (default: /opt/intel/oneapi)
# Passed through to the engine: every variable of this script's environment that starts with STRATA_ (the config's
# "env" block arrives that way: the server puts it in the engine's environment), ONEAPI_ (level_zero:0,1 for a
# two-card split), UR_, IGC_, SYCL_ or ZES_. The port's own defaults (below) are overridden the same way,
# e.g. "env": {"STRATA_VERIFY_NO_HOST": "0"} in the config.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)                 # the repo

# oneAPI on the host. setvars.sh reads OCL_ICD_FILENAMES under `set -u`, so source it with -u off (docs/INTEL_ARC.md).
set +u
# shellcheck disable=SC1091
source "${STRATA_ONEAPI:-/opt/intel/oneapi}/setvars.sh" >/dev/null 2>&1 || true
set -u
# the persistent JIT cache segfaults on Xe2 during the first compile (docs/INTEL_ARC.md)
export SYCL_CACHE_PERSISTENT=0

# the port's run-time switches: the device-built verify plan, and no host handshakes in it (xe driver: the
# host<->GPU flag stores are not visible there; an i915 card runs without NO_HOST - docs/INTEL.md)
declare -A setting=([STRATA_VERIFY_DEVICE_PLAN]=1 [STRATA_VERIFY_NO_HOST]=1 [STRATA_STAGER_THREADS]=12)
while IFS= read -r k; do
    case "$k" in
        STRATA_SYCL_*|STRATA_ONEAPI) ;;                   # this script's own settings
        STRATA_*|ONEAPI_*|UR_*|IGC_*|SYCL_*|ZES_*|NEOReadDebugKeys|OverrideDefaultFP64Settings) setting[$k]=${!k} ;;
    esac
done < <(compgen -e)
for k in "${!setting[@]}"; do
    # the engine tests the switches by presence: a value of 0 (or empty) means "not set", so it is not passed on
    case "${setting[$k]}" in 0|"") ;; *) export "$k=${setting[$k]}" ;; esac
done

exec "$here/${STRATA_SYCL_BIN:-build-sycl-aot/strata}" "$@"
