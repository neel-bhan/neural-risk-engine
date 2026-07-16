import json
import math
import unittest
from pathlib import Path


class PortfolioBenchmarkSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[2]
        cls.report = json.loads((root / "benchmarks/m10-portfolio-v1.json").read_text())

    def test_workload_and_routing_accounting(self) -> None:
        report = self.report
        self.assertEqual(report["result_version"], "nre.portfolio_benchmark.v1")
        workload = report["workload"]
        routing = report["routing"]
        self.assertEqual(workload["position_count"], 18)
        self.assertEqual(workload["scenario_count"], 9)
        self.assertEqual(workload["repricing_count"], 162)
        self.assertEqual(
            routing["neural_accepted_count"] + routing["monte_carlo_fallback_count"],
            workload["repricing_count"],
        )
        self.assertEqual(
            sum(routing["fallback_reason_counts"].values()),
            routing["monte_carlo_fallback_count"],
        )
        self.assertEqual(routing["fallback_reason_counts"]["input_domain"], 36)

    def test_matched_tolerance_and_timing_schema(self) -> None:
        matched = self.report["matched_tolerance"]
        tolerance = matched["guarded_route_tolerance"]
        selected = matched["selected_monte_carlo_metrics"]
        for metric in (
            "median_normalized_price_error",
            "p99_normalized_price_error",
            "delta_rmse",
        ):
            self.assertLessEqual(selected[metric], tolerance[metric])
            self.assertTrue(math.isfinite(selected[metric]))
        passing = [
            candidate
            for candidate in matched["candidate_measurements"]
            if all(
                candidate["metrics"][metric] <= tolerance[metric]
                for metric in (
                    "median_normalized_price_error",
                    "p99_normalized_price_error",
                    "delta_rmse",
                )
            )
        ]
        self.assertEqual(passing[0]["effective_paths"], matched["selected_effective_paths"])

        timing = self.report["timing"]
        for scope in (
            "raw_neural_price_delta_batch",
            "guarded_route_including_checks_and_fallback",
            "trusted_fallback_subset_only",
            "matched_tolerance_all_monte_carlo",
        ):
            value = timing[scope]
            self.assertGreater(value["warmups"], 0)
            self.assertGreater(value["repetitions"], 0)
            self.assertGreater(value["median_milliseconds"], 0.0)
            self.assertGreaterEqual(
                value["empirical_p99_milliseconds"], value["median_milliseconds"]
            )
            self.assertTrue(math.isfinite(value["result_checksum"]))

    def test_slice_and_worst_example_schema(self) -> None:
        for key, count in (
            ("full_routed_style_type_slices", 6),
            ("full_routed_scenario_slices", 9),
        ):
            slices = self.report[key]
            self.assertEqual(len(slices), count)
            for value in slices:
                self.assertEqual(
                    value["neural_accepted_count"] + value["fallback_count"],
                    value["metrics"]["count"],
                )
                self.assertEqual(
                    sum(value["fallback_reason_counts"].values()), value["fallback_count"]
                )
        worst = self.report["worst_full_routed_examples"]
        self.assertEqual(len(worst), 5)
        self.assertEqual(
            [row["normalized_price_error"] for row in worst],
            sorted(
                (row["normalized_price_error"] for row in worst), reverse=True
            ),
        )


if __name__ == "__main__":
    unittest.main()
