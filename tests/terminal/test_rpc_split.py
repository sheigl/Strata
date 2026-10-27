"""Terminal-side validation of the opt-in "rpc_split" config.  No GPU or engine binary is used."""

from __future__ import annotations

import unittest

from strata_control import rpc_config as R


def valid_cfg(**changes):
    cfg = {
        "backend": "cuda",
        "exe": "component:strata-engine",
        "gpu": 0,
        "args": ["--model", "qwen.gguf"],
        "rpc_split": {
            "boundary": 24,
            "workers": [{
                "id": "arc-prefix",
                "backend": "sycl",
                "device": 0,
                "env": {"ONEAPI_DEVICE_SELECTOR": "level_zero:0"},
            }],
        },
    }
    cfg.update(changes)
    return cfg


class RpcConfig(unittest.TestCase):
    def test_absent_is_no_plan_and_changes_nothing(self):
        self.assertIsNone(R.validate_rpc_split(valid_cfg(rpc_split=None)))

    def test_valid_cuda_controller_with_sycl_prefix(self):
        plan = R.validate_rpc_split(valid_cfg(), n_layers=48)
        self.assertEqual(plan["boundary"], 24)
        self.assertEqual(plan["worker"]["backend"], "sycl")
        worker, controller = R.engine_args_for(plan, socket_path="/run/user/1000/strata-rpc.sock",
                                               worker_id="arc-prefix", n_layers=48)
        self.assertEqual(worker, ["--rpc-listen", "/run/user/1000/strata-rpc.sock",
                                  "--rpc-layer-range", "0:24"])
        self.assertEqual(controller, ["--rpc-connect", "/run/user/1000/strata-rpc.sock",
                                      "--rpc-layer-range", "24:48"])

    def test_unknown_or_multiple_workers_are_refused(self):
        for workers in ([], [valid_cfg()["rpc_split"]["workers"][0],
                             valid_cfg()["rpc_split"]["workers"][0]]):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                R.validate_rpc_split(valid_cfg(rpc_split={"boundary": 24, "workers": workers}))

    def test_bad_worker_fields_are_plain_errors(self):
        base = valid_cfg()["rpc_split"]["workers"][0]
        cases = [
            ({**base, "backend": "hip"}, "backend"),
            ({**base, "device": -1}, "device"),
            ({**base, "id": "arc prefix"}, "id"),
            ({**base, "engine": ""}, "engine"),
            ({**base, "env": {"PATH": "/tmp"}}, "PATH"),
        ]
        for worker, why in cases:
            with self.subTest(worker=why):
                with self.assertRaisesRegex(ValueError, why):
                    R.validate_rpc_split(valid_cfg(rpc_split={"boundary": 24, "workers": [worker]}))

    def test_bad_boundary_is_refused(self):
        for boundary in (0, 2, "24", True, None):
            with self.subTest(boundary=boundary):
                with self.assertRaisesRegex(ValueError, "boundary"):
                    R.validate_rpc_split(valid_cfg(rpc_split={"boundary": boundary,
                                                              "workers": valid_cfg()["rpc_split"]["workers"]}))

    def test_boundary_that_leaves_no_controller_layers_is_refused_when_layers_are_known(self):
        with self.assertRaisesRegex(ValueError, "controller"):
            R.validate_rpc_split(valid_cfg(), n_layers=24)

    def test_combination_with_the_old_split_or_peer_or_batch_or_images_is_refused(self):
        cases = [
            ({"layer_split": "24"}, "layer_split"),
            ({"args": ["--layer-split", "24"]}, "--layer-split"),
            ({"args": ["--peer-device", "1"]}, "--peer-device"),
            ({"parallel": 2}, "parallel"),
            ({"args": ["--batch", "2"]}, "--batch"),
            ({"vision": {"exe": "encoder"}}, "vision"),
            ({"args": ["--vision"]}, "--vision"),
            ({"replicas": 2}, "replicas"),
            ({"gpu": [0, 1]}, "controller"),
            ({"session_restore": True}, "session_restore"),
        ]
        for change, why in cases:
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, why):
                    R.validate_rpc_split(valid_cfg(**change))


if __name__ == "__main__":
    unittest.main()
