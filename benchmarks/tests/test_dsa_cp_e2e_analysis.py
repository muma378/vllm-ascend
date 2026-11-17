# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "dsa_cp_e2e_analysis", Path(__file__).parents[1] / "scripts" / "analyze_dsa_cp_e2e.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def cells():
    return {
        ("fused", length, concurrency, output): dict(
            rounds=3,
            matches_baseline_outputs=True,
            output_length_ok=True,
            ttft_improvement_percent=5.0,
            tpot_improvement_percent=1.0,
        )
        for length in (1024, 4096, 8192)
        for concurrency in (1, 8)
        for output in (1, 256)
    }


def test_complete_correct_matrix_with_gain_passes():
    assert _module.evaluate_acceptance(cells(), "fused")["passed"]


def test_regression_in_one_scenario_rejects_a_large_gain_elsewhere():
    data = cells()
    data[("fused", 8192, 8, 256)]["tpot_improvement_percent"] = -2.01
    result = _module.evaluate_acceptance(data, "fused")
    assert not result["passed"]
    assert result["regressed_more_than_2_percent"] == [[8192, 8, 256]]


def test_missing_rounds_and_output_mismatch_do_not_pass():
    data = cells()
    data[("fused", 1024, 1, 1)]["rounds"] = 2
    assert not _module.evaluate_acceptance(data, "fused")["passed"]
    data = cells()
    data[("fused", 1024, 1, 1)]["matches_baseline_outputs"] = False
    assert not _module.evaluate_acceptance(data, "fused")["passed"]


def test_two_percent_boundary_tolerates_float_rounding():
    data = cells()
    data[("fused", 8192, 8, 256)]["tpot_improvement_percent"] = (1 - 1.02) * 100
    assert _module.evaluate_acceptance(data, "fused")["passed"]


def test_a_gain_below_five_percent_is_not_accepted():
    data = cells()
    for row in data.values():
        row["ttft_improvement_percent"] = 4.99
    result = _module.evaluate_acceptance(data, "fused")
    assert not result["passed"]
    assert not result["improved_at_least_5_percent"]
