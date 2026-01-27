import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Runs on both H200 and B200 via nightly-8-gpu-common suite
register_cuda_ci(est_time=900, suite="per-commit-8-gpu-h200-tgl")

QWEN3_VL_30B_MODEL_PATH = "Qwen/Qwen3-VL-32B-Instruct"


class TestQwen3VL30BUnified(unittest.TestCase):
    """Unified test class for Qwen3-VL-30B performance and accuracy.

    Single variant with simple TP=8 configuration.
    Runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with mgsm_en)
    """

    def test_qwen3_30b(self):
        """Run performance and accuracy for Qwen3-VL-30B."""
        base_args = [
            "--tp=8",
            "--trust-remote-code",
            "--enable-multimodal",
            "--tool-call-parser=qwen",
            "--context-length=262144",
            "--served-model-name=Qwen/Qwen3-VL-32B-Instruct",
        ]

        variants = [
            ModelLaunchSettings(
                QWEN3_VL_30B_MODEL_PATH,
                tp_size=2,
                extra_args=base_args,
                variant="TP2",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Qwen3-VL-30B Unified",
            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.95),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_qwen3_vl_30b",
            ),
        )


if __name__ == "__main__":
    unittest.main()
