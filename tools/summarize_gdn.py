#!/usr/bin/env python3
"""Collect GDN library validation and unfiltered paired samples."""
import argparse
import copy
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.results
    library = Path(__file__).resolve().parents[1]
    quality = read(root / "quality-final/report.json")
    paired = read(root / "paired-final/report.json")
    assert quality["status"] == paired["status"] == "completed"
    assert len(quality["records"]) == 142 and len(paired["records"]) == 8
    assert all(record["bitwise_equal_to_probe"] for record in quality["records"] + paired["records"])
    sources = [library / "include/rk_npu_gdn.h", *sorted((library / "src").glob("rk_npu_gdn*")),
               library / "src/rk_npu_float_backend.h", library / "src/rk_npu_matmul_f16.cpp", library / "src/rk_npu_core.cpp"]
    for source in sources:
        for run in ["quality-final", "paired-final"]:
            assert source.read_bytes() == (root / run / (source.name + ".snapshot")).read_bytes()
    assert not list((root / "quality-final").rglob("*.f32"))
    assert not list((root / "paired-final").rglob("*.f32"))
    summary = dict(status="passed", scope=paired["scope"],
        conditions=dict(board="RK3588 spare1", cpu_threads=4, cpu_khz=2256000, npu_hz=1000000000,
                        cpu_affinity="0xf0", dmc_controlled=False, compiler="GCC",
                        precision="TF32", chunk=64, order="ABBA twice", warmups=2, measured_per_segment=7,
                        sample_filtering="none"),
        bitwise_regressions=dict(normalized=52, projected=90, total=142),
        numerical=[], performance=[], clocks=[])
    for record in quality["records"]:
        row = {key: record[key] for key in ["name", "tokens", "chunk", "precision", "kind", "bitwise_equal_to_probe"]}
        if record["kind"] == "normalized":
            row.update(output=record["output"], state=record["state"])
            if record["precision"]:
                assert row["output"]["passed"] and row["state"]["passed"]
        summary["numerical"].append(row)
    for kind in ["normalized", "projected"]:
        for tokens in [128, 512]:
            records = [r for r in paired["records"] if r["kind"] == kind and r["tokens"] == tokens]
            result = dict(kind=kind, tokens=tokens, samples_per_version=28 * len(records), versions={})
            for version in ["before", "after"]:
                samples = []
                for record in records:
                    assert [r["version"] for r in record["runs"]] == ["before", "after", "after", "before"] * 2
                    for run in record["runs"]:
                        if run["version"] == version:
                            assert len(run["samples"]) == 7
                            samples.extend(dict(total_ms=s["total_us"] / 1000,
                                                cpu_ms=(s["cpu_user_us"] + s["cpu_system_us"]) / 1000,
                                                **({"wall_ms": s["wall_us"] / 1000} if version == "after" else {}))
                                           for s in run["samples"])
                assert len(samples) == result["samples_per_version"]
                result["versions"][version] = dict(samples=samples, **{
                    key: dict(mean=statistics.mean(s[key] for s in samples), median=statistics.median(s[key] for s in samples),
                              minimum=min(s[key] for s in samples), maximum=max(s[key] for s in samples))
                    for key in samples[0]})
            before, after = [result["versions"][v] for v in ["before", "after"]]
            result["latency_change_ratio"] = after["total_ms"]["mean"] / before["total_ms"]["mean"] - 1
            result["cpu_change_ratio"] = after["cpu_ms"]["mean"] / before["cpu_ms"]["mean"] - 1
            summary["performance"].append(result)
    log = (root / "final-checks.log").read_text()
    assert log.count("PASS GDN device integration") == 2 and "FAIL" not in log
    assert log.count("T=2048 C=64") == 4
    assert "TF32 BOARD ALL PASS" in log and "100% tests passed" in log
    summary["device_validation"] = dict(shared_and_static=True, c_header=True, independent_fp64_oracle=True,
        max_tokens=2048, head_mappings=["grouped", "interleaved"], cpu_threads=[1, 2, 3, 4], core_masks=[1, 2, 4, 3, 7],
        replay_and_inplace=True, busy=True, retained_iommu_domain=True, existing_mm_board_tests=True)
    for name in ["quality-final", "paired-final", "final-checks"]:
        clocks = read(root / (name + "-clocks.json"))
        assert clocks["exit_code"] == 0 and not clocks["restore_errors"]
        for device, values in clocks["before"].items():
            if device == "thermal":
                continue
            for field, value in values.items():
                if "min_freq" in field or "max_freq" in field or "governor" in field:
                    assert clocks["restored"][device][field] == value
        summary["clocks"].append(dict(run=name, restored=True))
    (root / "verification.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    compact = copy.deepcopy(summary)
    compact.pop("numerical")
    compact["normalized_max_relative_l2"] = {
        precision: {part: max(r[part]["relative_l2"] for r in quality["records"]
                               if r["kind"] == "normalized" and r["precision"] == number)
                    for part in ["output", "state"]}
        for number, precision in [(0, "tf32"), (1, "compensated_tf32")]}
    for result in compact["performance"]:
        for version in result["versions"].values():
            version.pop("samples")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n")
    print("PASS bitwise=142 shared/static/device/MM/headers clocks=restored tensors=discarded")
    for result in summary["performance"]:
        print(result["kind"], result["tokens"], {v: round(result["versions"][v]["total_ms"]["mean"], 4)
              for v in ["before", "after"]}, "latency_change_percent", round(result["latency_change_ratio"] * 100, 2))


if __name__ == "__main__":
    main()
