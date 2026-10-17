#!/usr/bin/env python3
"""Validate the GDN C ABI against ggml fixtures and paired native producers."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time

import numpy as np


class Config(ct.Structure):
    _fields_ = [(name, ct.c_int) for name in ["tokens", "key_heads", "value_heads", "key_dim", "value_dim", "chunk_size", "cpu_threads"]] + [
        ("core_mask", ct.c_uint32), ("timeout_ms", ct.c_uint32), ("precision", ct.c_int), ("head_mapping", ct.c_int)]


class Timings(ct.Structure):
    _fields_ = [(name, ct.c_double) for name in ["total_us", "prepare_us", "coefficients_us", "solve_us", "collect_us",
        "hardware_us", "gram_hardware_us", "predict_hardware_us", "update_hardware_us", "cpu_user_us", "cpu_system_us"]]


P = ct.POINTER(ct.c_float)


def pointer(array):
    assert array.dtype == np.float32 and array.flags.c_contiguous
    return array.ctypes.data_as(P)


def compare(actual, expected):
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(diff)))
    relative = float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(expected.astype(np.float64).ravel()), 1e-30))
    limit = 1e-5 + 1e-4 * float(np.max(np.abs(expected)))
    return dict(max_abs=maximum, relative_l2=relative, finite=bool(np.isfinite(actual).all()),
                passed=bool(np.isfinite(actual).all() and maximum <= limit and relative <= 1e-4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--reference-prefill", type=Path, required=True)
    parser.add_argument("--conv-assets", type=Path)
    parser.add_argument("--reference-producer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert bool(args.conv_assets) == bool(args.reference_producer)
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    for file in [Path(__file__), root / "include/rk_npu_gdn.h", *sorted((root / "src").glob("rk_npu_gdn*")),
                 root / "src/rk_npu_float_backend.h", root / "src/rk_npu_matmul_f16.cpp", root / "src/rk_npu_core.cpp"]:
        shutil.copy2(file, args.output / (file.name + ".snapshot"))
    lib = ct.CDLL(str(args.library.resolve()))
    lib.rk_npu_open.argtypes = [ct.c_char_p]; lib.rk_npu_open.restype = ct.c_void_p
    lib.rk_npu_close.argtypes = [ct.c_void_p]
    lib.rk_npu_iommu_domain_create.argtypes = [ct.c_void_p, ct.c_uint32]; lib.rk_npu_iommu_domain_create.restype = ct.c_void_p
    lib.rk_npu_iommu_domain_free.argtypes = [ct.c_void_p]
    lib.rk_npu_gdn_config_init.argtypes = [ct.POINTER(Config), ct.c_int]
    lib.rk_npu_gdn_prepare.argtypes = [ct.c_void_p, ct.POINTER(Config), ct.POINTER(ct.c_void_p)]
    lib.rk_npu_gdn_workspace_free.argtypes = [ct.c_void_p]
    lib.rk_npu_gdn_run.argtypes = [ct.c_void_p] + [P] * 8 + [ct.POINTER(Timings)]
    lib.rk_npu_gdn_run_projected.argtypes = [ct.c_void_p] + [P] * 9 + [ct.POINTER(Timings)]
    ctx = lib.rk_npu_open(None); assert ctx
    domain = lib.rk_npu_iommu_domain_create(ctx, 0); assert domain
    lock = os.environ.get("LLM_INFER_LOCK_FD"); assert lock, "run under the shared board runner"
    report = dict(status="running", benchmark=args.benchmark, records=[], tensor_retention="metrics_only",
                  scope="normalized recurrence or projected Conv+SiLU+L2+GDN; one layer, state import/export included")

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def invoke(command, log):
        with log.open("w") as output:
            subprocess.run([str(x) for x in command], check=True, stdout=output, stderr=subprocess.STDOUT,
                           pass_fds=(int(lock),), timeout=120)

    def check_case(name, arrays, tokens, chunk, precision, expected=None, projected=False):
        config = Config(); lib.rk_npu_gdn_config_init(ct.byref(config), tokens)
        config.chunk_size = chunk; config.precision = precision
        record = dict(name=name, tokens=tokens, chunk=chunk, precision=precision,
                      kind="projected" if projected else "normalized", runs=[])
        report["records"].append(record); save()
        with tempfile.TemporaryDirectory(prefix="tensors-", dir=args.output) as temp:
            temp = Path(temp)
            packed = temp / "input.f32"
            with packed.open("wb") as output:
                for value in arrays:
                    output.write(value.tobytes())
            reference = args.reference_producer if projected else args.reference_prefill
            reference_data = None
            modes = ["before", "after", "after", "before"] * 2 if args.benchmark else ["before", "after"]
            for position, version in enumerate(modes):
                stem = f"{name}-t{tokens}-c{chunk}-p{precision}-{position}-{version}"
                output_path = temp / "output.f32"
                if version == "before":
                    log = args.output / (stem + ".log")
                    command = [reference, packed, output_path, tokens, chunk, "split_tf32" if precision else "tf32",
                               9 if args.benchmark else 1]
                    if projected:
                        command += ["native"]
                    invoke(command, log)
                    actual = np.fromfile(output_path, dtype=np.float32)
                    if reference_data is None:
                        reference_data = actual.copy()
                    else:
                        assert np.array_equal(actual, reference_data)
                    rows = []
                    for line in log.read_text().splitlines():
                        if line.split(",", 1)[0].isdigit():
                            row = [float(value) for value in line.split(",")[1:]]
                            # total is the first field in both reference runners.
                            rows.append(dict(total_us=row[0]))
                    if args.benchmark:
                        import csv
                        text = log.read_text().splitlines()
                        header = next(line for line in text if line.startswith("sample,"))
                        selected = [line for line in text if line.split(",", 1)[0].isdigit()]
                        parsed = list(csv.DictReader([header, *selected]))[2:]
                        rows = [dict(total_us=float(row["total_us"]), cpu_user_us=float(row["cpu_user_us"]),
                                     cpu_system_us=float(row["cpu_system_us"])) for row in parsed]
                else:
                    workspace = ct.c_void_p()
                    assert lib.rk_npu_gdn_prepare(domain, ct.byref(config), ct.byref(workspace)) == 0
                    try:
                        output = np.empty((tokens, 32, 128), dtype=np.float32)
                        state = np.empty((32, 128, 128), dtype=np.float32)
                        history = np.empty((8192, 3), dtype=np.float32) if projected else None
                        timing = Timings()
                        if projected:
                            weights, old_history, inputs, g, beta, initial = arrays
                            parameters = [workspace, *map(pointer, [inputs, weights, old_history, g, beta, initial, output, state, history]), ct.byref(timing)]
                            function = lib.rk_npu_gdn_run_projected
                        else:
                            parameters = [workspace, *map(pointer, [*arrays, output, state]), ct.byref(timing)]
                            function = lib.rk_npu_gdn_run
                        rows = []
                        for _ in range(9 if args.benchmark else 2):
                            start = time.perf_counter_ns()
                            assert function(*parameters) == 0
                            wall = (time.perf_counter_ns() - start) / 1000
                            row = {name: getattr(timing, name) for name, _ in timing._fields_}
                            row["wall_us"] = wall; rows.append(row)
                        rows = rows[2:] if args.benchmark else rows
                        actual = np.concatenate([output.ravel(), state.ravel(), *([history.ravel()] if projected else [])])
                        assert np.array_equal(actual, reference_data), (name, chunk, precision, compare(actual, reference_data))
                        if projected:
                            expected_history = np.concatenate([old_history.T, inputs], axis=0)[-3:].T
                            assert np.array_equal(history, expected_history)
                    finally:
                        lib.rk_npu_gdn_workspace_free(workspace)
                record["runs"].append(dict(version=version, position=position, samples=rows))
                save()
            record["bitwise_equal_to_probe"] = True
            if expected is not None:
                split = tokens * 4096
                record.update(output=compare(reference_data[:split].reshape(expected[0].shape), expected[0]),
                              state=compare(reference_data[split:].reshape(expected[1].shape), expected[1]))
                if precision:
                    assert record["output"]["passed"] and record["state"]["passed"]
                else:
                    assert record["output"]["finite"] and record["state"]["finite"]
            if args.benchmark:
                for version in ["before", "after"]:
                    rows = [row for run in record["runs"] if run["version"] == version for row in run["samples"]]
                    assert len(rows) == 28
                    record[version + "_mean_us"] = {key: statistics.mean(row[key] for row in rows) for key in rows[0]}
            save(); print(name, tokens, chunk, precision, "PASS exact", flush=True)

    save()
    try:
        for chunk in ([64] if args.benchmark or args.smoke else [64, 16, 32, 128]):
            names = (["tokens-0128", "tokens-0512"] if args.benchmark else ["tokens-0007"] if args.smoke else
                     [f"tokens-{t:04d}" for t in ([1, 2, 4, 7, 8, 32, 128, 512] if chunk == 64 else [1, 7, 128, 512])] +
                     (["stress-0001", "stress-0007", "stress-0128"] if chunk == 64 else ["stress-0128"]))
            for name in names:
                fixture = np.load(args.fixtures / (name + ".npz"))
                arrays = [np.ascontiguousarray(fixture[key], dtype=np.float32) for key in ["q", "k", "v", "g", "beta", "state_initial"]]
                for precision in ([0] if args.benchmark else [0, 1]):
                    check_case(name, arrays, arrays[0].shape[0], chunk, precision, (fixture["output"], fixture["state_final"]))
        if args.conv_assets:
            metadata = json.loads((args.conv_assets / "input-metadata.json").read_text())
            layers = [entry["layer"] for entry in metadata["layers"]]
            weights = np.fromfile(args.conv_assets / "weights.f32", dtype=np.float32).reshape(24, 8192, 4)
            trace = np.fromfile(args.conv_assets / "trace.f32", dtype=np.float32).reshape(24, 24, 8192)
            fixture = np.load(args.fixtures / "tokens-0512.npz")
            stress = np.load(args.fixtures / "stress-0128.npz")
            shapes = ([(layer, tokens, 64, "uniform") for layer in [0, 12, 30] for tokens in [128, 512]] if args.benchmark else
                      [(0, 7, 64, "stress")] if args.smoke else
                      [(layer, 7, 64, "stress") for layer in layers] +
                      [(0, tokens, chunk, "uniform") for tokens in [1, 65, 129] for chunk in [16, 32, 64, 128]] +
                      [(layer, tokens, 64, "uniform") for layer in [0, 12, 30] for tokens in [128, 512]] +
                      [(layer, 512, 64, "headwise") for layer in [0, 12, 30]])
            for layer, tokens, chunk, gate in shapes:
                i = layers.index(layer)
                gates = stress if gate == "stress" else fixture
                g = gates["g"][np.arange(tokens) % gates["g"].shape[0]].copy()
                beta = gates["beta"][np.arange(tokens) % gates["beta"].shape[0]].copy()
                if gate == "headwise":
                    rates = np.geomspace(.002, .14, 32).astype(np.float32)
                    np.random.default_rng(20261017).shuffle(rates); g[:] = -rates
                arrays = [weights[i].copy(), trace[i, :3].T.copy(), trace[i, 3 + np.arange(tokens) % 21].copy(),
                          g, beta, fixture["state_initial"].copy()]
                for precision in ([0] if args.benchmark else [0, 1]):
                    check_case(f"conv-l{layer}-{gate}", arrays, tokens, chunk, precision, projected=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=repr(error)); raise
    finally:
        save(); lib.rk_npu_iommu_domain_free(domain); lib.rk_npu_close(ctx)


if __name__ == "__main__":
    main()
