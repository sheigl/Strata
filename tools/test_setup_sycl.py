"""Tests for setup.py --backend sycl, the experimental Intel Arc engine (sycl/, PR #423): it warns, stops with a clear
message on Windows, and on Linux hands the run to sycl/setup_intel.py without its --backend flag. The CUDA / HIP
paths never see it. No GPU, no network.

    python -m unittest tools.test_setup_sycl
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def run(argv, win):
    out = io.StringIO()
    with mock.patch.object(setup, "WIN", win), mock.patch.object(setup.subprocess, "call", return_value=0) as call, \
            contextlib.redirect_stdout(out):
        try:
            rc = setup.sycl_setup(argv)
        except SystemExit as e:
            rc = ("exit", e.code)
    return rc, out.getvalue(), call


class SyclBackend(unittest.TestCase):
    def test_windows_stops_with_the_docs_pointer(self):
        rc, out, call = run(["--backend", "sycl"], win=True)
        self.assertEqual(rc, ("exit", 1))
        self.assertIn("supported since 0.1.40.2", out)
        self.assertIn("docs/INTEL_ARC.md", out)
        call.assert_not_called()

    def test_linux_hands_over_to_setup_intel_without_the_backend_flag(self):
        rc, out, call = run(["--model", "IQ2_XS", "--backend", "sycl", "--port", "8085"], win=False)
        self.assertEqual(rc, 0)
        self.assertIn("supported since 0.1.40.2", out)
        self.assertIn("built from source", out)
        cmd = call.call_args[0][0]
        self.assertTrue(cmd[1].endswith(str(Path("sycl") / "setup_intel.py")))
        self.assertEqual(cmd[2:], ["--model", "IQ2_XS", "--port", "8085"])

    def test_backend_equals_form_is_dropped_too(self):
        _, _, call = run(["--backend=sycl", "--check"], win=False)
        self.assertEqual(call.call_args[0][0][2:], ["--check"])

    def test_the_parser_accepts_sycl(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertRegex(src, r'"--backend", choices=\["cuda", "hip", "sycl"\]')

    def test_setup_intel_finds_the_setup_functions_it_replaces(self):
        """sycl/setup_intel.py swaps these steps; setup.py must still have them (it stops with a message otherwise)."""
        src = (ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8")
        names = re.search(r'for name in \(([^)]*)\):', src).group(1)
        for name in re.findall(r'"(\w+)"', names):
            self.assertTrue(callable(getattr(setup, name, None)), name)

    def test_setup_intel_replacements_take_every_argument_of_the_function_they_replace(self):
        """#870: setup.py grew write_run_script's fourth argument and the port's replacement still took three, so
        setup died with a TypeError after the download.  Every `S.<name> = <replacement>` in setup_intel.py must accept
        all the parameters of setup.<name> (or take *args / **kwargs)."""
        import ast
        import inspect
        tree = ast.parse((ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8"))
        defs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        checked = 0
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute)
                    and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == "S"):
                continue
            name = n.targets[0].attr
            fn = n.value if isinstance(n.value, ast.Lambda) else defs.get(getattr(n.value, "id", ""))
            orig = getattr(setup, name, None)
            if fn is None or not callable(orig):
                continue
            a = fn.args
            if a.vararg is not None and a.kwarg is not None:
                checked += 1
                continue
            have = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
            for param in inspect.signature(orig).parameters.values():
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue
                self.assertIn(param.name, have, f"setup_intel's {name} lacks setup.{name}'s `{param.name}`")
            checked += 1
        self.assertGreaterEqual(checked, 8)

    def test_arc_pro_b60_is_known_under_both_pci_ids(self):
        src = (ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8")
        self.assertIn('"e211": ("Arc Pro B60", 24.0)', src)
        self.assertIn('"e221": ("Arc Pro B60", 24.0)', src)


class CMakeOption(unittest.TestCase):
    def test_sycl_is_off_by_default_and_exclusive(self):
        src = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig")
        self.assertIn('option(STRATA_ENABLE_SYCL "EXPERIMENTAL', src)
        self.assertRegex(src, r'option\(STRATA_ENABLE_SYCL "[^"]*" OFF\)')
        self.assertIn("add_subdirectory(sycl)", src)


class SyclVersion(unittest.TestCase):
    def setUp(self):
        from sycl import setup_intel
        self.mod = setup_intel

    def test_version_matches_the_source_and_satisfies_native_models(self):
        version = self.mod.sycl_version()
        self.assertNotEqual(version, "0")
        self.assertEqual(version, setup.source_version())
        numbers = tuple(int(x) for x in version.split("."))
        for model in ("UD-IQ4_XS", "UD-Q4_K_XL"):
            with self.subTest(model=model):
                self.assertGreaterEqual(numbers, setup.MODELS[model].get("engine", setup.UNSLOTH_ENGINE))

    def test_version_follows_a_source_hotfix_with_a_variable_sycl_project(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CMakeLists.txt").write_text("project(strata VERSION 0.1.41.2 LANGUAGES CXX)\n", encoding="utf-8")
            (root / "sycl").mkdir()
            (root / "sycl" / "CMakeLists.txt").write_text(
                "set(STRATA_SYCL_VERSION 0.1.40.2)\n"
                "project(strata_sycl VERSION ${STRATA_SYCL_VERSION} LANGUAGES C CXX)\n", encoding="utf-8")
            with mock.patch.object(self.mod, "ROOT", root), mock.patch.object(setup, "ROOT", root):
                self.assertEqual(self.mod.sycl_version(), "0.1.41.2")


class IntelIds(unittest.TestCase):
    """sycl/setup_intel.py names a card by its PCI device id. The Arc Pro B60s measured on 2x B60 (`lspci -nn` 8086:e211,
    bmg-g21) must not fall through to "Intel GPU e211 (xe)" with a guessed VRAM size."""

    @staticmethod
    def table():
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_intel", ROOT / "sycl" / "setup_intel.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.INTEL_ARC

    def test_b60_e211_is_named_with_its_vram(self):
        self.assertEqual(self.table()["e211"], ("Arc Pro B60", 24.0))

    def test_the_ids_that_were_listed_before_are_kept(self):
        t = self.table()
        self.assertEqual(t["e223"][0], "Arc Pro B70")
        self.assertEqual(t["e221"][0], "Arc Pro B60")


class ToSycl(unittest.TestCase):
    """The config sycl/setup_intel.py writes: an explicit --vram-reserve-mib is kept, the B-series streams its experts,
    the A-series (i915) cannot (a single pinned host allocation above a few GB fails there) and loads them into RAM."""

    @staticmethod
    def mod():
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_intel", ROOT / "sycl" / "setup_intel.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def cfg(self, m, extra=()):
        base = m.MOUNT
        return {"exe": "x", "cwd": "y", "args": ["--pack", str(base / "d" / "pack"), "--native", str(base / "d" / "a.gguf"),
                                                 "--expert-cache", "auto", "--prefill", "auto", "--max-context", "32768",
                                                 "--kv", "int8", *extra]}

    def test_b_series_streams_and_defaults_the_reserve(self):
        m = self.mod()
        c = m.to_sycl(self.cfg(m), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertIn("--stream-experts", c["args"])
        self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "1024")
        self.assertNotIn("STRATA_VERIFY_NO_HOST", c.get("env", {}))

    def test_xe_ranks_the_experts_with_the_sycl_profile_and_i915_keeps_the_shipped_one(self):
        m = self.mod()
        shipped = str(m.MOUNT / "data" / "expert-profile.bin")
        other = str(m.MOUNT / "mine" / "learned.bin")
        c = m.to_sycl(self.cfg(m, ["--expert-profile", shipped]), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertTrue(c["args"][c["args"].index("--expert-profile") + 1].endswith("expert-profile-sycl.bin"))
        c = m.to_sycl(self.cfg(m, ["--expert-profile", other]), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertTrue(c["args"][c["args"].index("--expert-profile") + 1].endswith("learned.bin"))   # the user's own file stays
        c = m.to_sycl(self.cfg(m, ["--expert-profile", shipped]), m.ROOT / "build-sycl" / "strata", 64.0, {}, 8.0, "i915")
        self.assertTrue(c["args"][c["args"].index("--expert-profile") + 1].endswith("expert-profile.bin"))   # the CPU computes the misses there

    def test_an_explicit_reserve_is_kept_on_both_drivers(self):
        m = self.mod()
        for vram, drv in ((32.0, "xe"), (8.0, "i915")):
            c = m.to_sycl(self.cfg(m, ["--vram-reserve-mib", "500"]), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, vram, drv)
            self.assertEqual(c["args"].count("--vram-reserve-mib"), 1)
            self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "500")

    def test_a_series_loads_experts_into_ram_and_leaves_no_host_off(self):
        m = self.mod()
        c = m.to_sycl(self.cfg(m), m.ROOT / "build-sycl" / "strata", 64.0, {}, 8.0, "i915")
        self.assertNotIn("--stream-experts", c["args"])
        self.assertEqual(c["args"][c["args"].index("--ple-io") + 1], "ram")
        self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "300")
        self.assertEqual(c["env"]["STRATA_VERIFY_NO_HOST"], "0")
        self.assertEqual(c["env"]["STRATA_SYCL_BIN"], "build-sycl/strata")

    def test_small_card_rule(self):
        m = self.mod()
        self.assertTrue(m.small_card(8.0, "i915"))
        self.assertTrue(m.small_card(24.0, "i915"))
        self.assertTrue(m.small_card(10.0, "xe"))
        self.assertFalse(m.small_card(32.0, "xe"))

    def test_no_images_keeps_the_section_out_of_the_config(self):
        """A setup run that asked for nothing writes no vision block: the config is what it was before the SYCL port
        read pictures at all."""
        m = self.mod()
        c = m.to_sycl(self.cfg(m), m.ROOT / "build-sycl" / "strata", 64.0, {}, 32.0, "xe")
        self.assertNotIn("vision", c)

    def test_a_vision_section_keeps_everything_but_its_encoder_path(self):
        """setup writes the block for its CUDA/HIP engine, whose encoder sits beside that engine; the SYCL config points
        at the one beside the SYCL engine.  The mmproj and the model shard stay host files - the encoder runs on the
        host, and the engine only ever sees the embeddings file."""
        m = self.mod()
        cfg = self.cfg(m)
        cfg["vision"] = {"exe": str(m.ROOT / "build-hip" / "strata-vision"),
                         "mmproj": "/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
                         "model": "/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
                         "gpu": False, "max_tokens": 300, "threads": 8}
        c = m.to_sycl(cfg, m.ROOT / "build-sycl" / "strata", 64.0, {}, 32.0, "xe")
        self.assertEqual(c["vision"]["exe"], str(m.ROOT / "build-sycl" / "strata-vision"))
        for key in ("mmproj", "model", "gpu", "max_tokens", "threads"):
            self.assertEqual(c["vision"][key], cfg["vision"][key], key)


class SplitEnv(unittest.TestCase):
    """The environment a layer split gets (sycl/setup_intel.py, split_env): the dense weights are trimmed to one copy
    (#1980) and each later stage's misses are mirrored in pinned host memory instead of being REFUSED (STRATA_VERIFY_NO_HOST
    is 1 on xe, so the verify path never falls back to the GGUF).  A 2-card Q4_K_M split cannot start without the
    second one; a split that misses nothing builds no mirror and pays nothing for it."""

    @staticmethod
    def mod():
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_intel", ROOT / "sycl" / "setup_intel.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_a_split_mirrors_its_misses_and_trims_the_dense_weights(self):
        m = self.mod()
        env = m.split_env({})
        self.assertEqual(env["STRATA_STAGE_TRIM"], "1")
        self.assertEqual(env["STRATA_SPLIT_MIRROR"], "1")

    def test_the_mirror_cap_is_left_to_the_engine(self):
        """No STRATA_MIRROR_MIB: the engine's default (MemAvailable less 4 GiB) is the machine's own answer, and a cap
        smaller than the misses is a REFUSED, not a fallback to the SSD."""
        m = self.mod()
        self.assertNotIn("STRATA_MIRROR_MIB", m.split_env({}))

    def test_what_earlier_in_the_run_set_is_kept(self):
        m = self.mod()
        cfg = {"env": {"STRATA_STAGE_TRIM": "0", "STRATA_SYCL_BIN": "build-sycl/strata"}}
        env = m.split_env(cfg)
        self.assertEqual(env["STRATA_STAGE_TRIM"], "0")             # an answer already there wins
        self.assertEqual(env["STRATA_SYCL_BIN"], "build-sycl/strata")
        self.assertEqual(env["STRATA_SPLIT_MIRROR"], "1")

    def test_write_run_script_calls_it_for_a_split_and_only_there(self):
        """The config write_run_script writes is what the server runs, so the split's env has to be added there (and a
        one-card config must stay exactly what it was)."""
        src = (ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8")
        self.assertRegex(src, r"if split:\n\s+split_env\(cfg\)\n")
        self.assertEqual(src.count("split_env("), 2)                # the def and that one call


class VisionEncoder(unittest.TestCase):
    """The image encoder on the SYCL port: the CPU one when it is asked for, off for anything else (no SYCL build of
    the encoder exists yet) - and off when nothing was asked for, which is what the port did before."""

    @staticmethod
    def mod():
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_intel", ROOT / "sycl" / "setup_intel.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_only_cpu_turns_images_on(self):
        m = self.mod()
        self.assertEqual(m.sycl_vision_answer(None), "none")
        for asked in ("no", "none", "yes", "gpu", ""):
            self.assertEqual(m.sycl_vision_answer(asked), "none", asked)
        self.assertEqual(m.sycl_vision_answer("cpu"), "cpu")

    def test_the_encoder_sits_beside_the_engine_it_serves(self):
        m = self.mod()
        self.assertEqual(m.host_vision_exe(m.ROOT / "build-sycl" / "strata"),
                         m.ROOT / "build-sycl" / setup.VEXE)

    def test_q4_k_m_gets_its_own_mmproj_name_and_repository(self):
        """The plain Swift 1.5 release Q4_K_M comes from names its encoder after itself and ships it with its shards;
        every other model still gets its family's file (the default path is unchanged)."""
        self.assertEqual(setup.model_mmproj(setup.FAMILIES["swift"], "Q4_K_M"),
                         ("mmproj-Swift-1.5-Qwen3.8-Flash-Next-BF16.gguf",
                          setup.hf("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF")))
        for model, family in (("IQ3_S", "qwen"), ("UD-Q4_K_XL", "unsloth"), ("IQ1_M", "coder")):
            with self.subTest(model=model):
                self.assertEqual(setup.model_mmproj(setup.FAMILIES[family], model),
                                 (setup.FAMILIES[family]["mmproj"], setup.FAMILIES[family]["mmproj_hf"]))
        self.assertIsNot(setup.MODELS["Q4_K_M"].get("vision"), False)


if __name__ == "__main__":
    unittest.main()
