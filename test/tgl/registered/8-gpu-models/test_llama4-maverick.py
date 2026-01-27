import unittest
from types import SimpleNamespace

from sglang.private.test.utils import run_bench_serving
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_cuda_ci(est_time=900, suite="per-commit-8-gpu-h200-tgl")

LLAMA4_MAVERICK_MODEL_PATH = "togethercomputer/Llama-4-Maverick-17B-128E-Instruct-FP8"


class TestLlama4Maverick(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA4_MAVERICK_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "8",
            "--context-length",
            "1048576",
            "--enable-multimodal",
            "--attention-backend",
            "fa3",
            "--kv-cache-dtype",
            "auto",
            "--tool-call-parser",
            "llama3",
            "--enable-lora",
            "--max-lora-rank",
            "64",
            "--lora-target-modules",
            "all",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3000,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=1319,
            parallel=1319,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v3-fp4 cursor)\n"
                f'{metrics["accuracy"]=:.3f}\n'
            )
        print(f"{metrics['accuracy']=:.3f}")
        self.assertGreater(metrics["accuracy"], 0.925)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (llama4-maverick)\n" f"{speed=:.2f} token/s\n"
            )
        print(f"{speed=:.2f}")
        self.assertGreater(speed, 148)

    def test_benchserving(self):
        res = run_bench_serving(
            num_prompts=100,
            request_rate=2,
            random_input_len=2048,
            random_output_len=1024,
            dataset_name="random",
            need_warmup=False,
            disable_ignore_eos=True,
            seed=42,
            disable_tqdm=True,
        )
        if is_in_ci():
            write_github_step_summary(
                f"### test_pp_long_context_latency_prefill\n"
                f"input_throughput: {res['input_throughput']:.2f} tokens/s\n"
                f"output_throughput: {res['output_throughput']:.2f} tokens/s\n"
            )
        self.assertGreater(res["input_throughput"], 1800)  # at least 1800 tokens/s
        self.assertGreater(res["output_throughput"], 760)  # at least 760 tokens/s


if __name__ == "__main__":
    unittest.main()
