import argparse
import glob
import os
import sys
from typing import List

import tabulate

from sglang.test.ci.ci_register import CIRegistry, HWBackend, collect_tests
from sglang.test.ci.ci_utils import run_unittest_files

HW_MAPPING = {
    "cuda": HWBackend.CUDA,
}

# Per-commit test suites (run on every PR)
PER_COMMIT_SUITES = {
    HWBackend.CUDA: [
        "per-commit-4-gpu-b200-tgl",
        "per-commit-8-gpu-h200-tgl",
    ],
    HWBackend.NPU: [],
}

# Nightly test suites (run nightly, organized by GPU configuration)
NIGHTLY_SUITES = {
    HWBackend.CUDA: [],
}


def filter_tests(
    ci_tests: List[CIRegistry], hw: HWBackend, suite: str, nightly: bool = False
) -> List[CIRegistry]:
    ci_tests = [
        t
        for t in ci_tests
        if t.backend == hw and t.suite == suite and t.nightly == nightly
    ]

    valid_suites = (
        NIGHTLY_SUITES.get(hw, []) if nightly else PER_COMMIT_SUITES.get(hw, [])
    )

    if suite not in valid_suites:
        print(
            f"Warning: Unknown suite {suite} for backend {hw.name}, nightly={nightly}"
        )

    enabled_tests = [t for t in ci_tests if t.disabled is None]
    skipped_tests = [t for t in ci_tests if t.disabled is not None]

    return enabled_tests, skipped_tests


def _normalize_ci_path(path: str) -> str:
    """Normalize user-provided paths to match CI registry filenames."""
    if not path:
        return ""
    abs_path = os.path.abspath(path)
    cwd = os.getcwd()
    try:
        rel_path = os.path.relpath(abs_path, cwd)
    except ValueError:
        rel_path = path
    norm_path = os.path.normpath(rel_path)
    prefix = f"test{os.sep}"
    if norm_path.startswith(prefix):
        norm_path = norm_path[len(prefix) :]
    return norm_path


def apply_manual_skips(
    ci_tests: List[CIRegistry],
    skipped_tests: List[CIRegistry],
    skip_args: List[str],
) -> tuple[List[CIRegistry], List[CIRegistry]]:
    if not skip_args:
        return ci_tests, skipped_tests

    default_reason = "manually skipped via --skip-file"
    skip_map: dict[str, str] = {}
    for raw in skip_args:
        if "=" in raw:
            path, reason = raw.split("=", 1)
            reason = reason.strip() or default_reason
        else:
            path, reason = raw, default_reason
        norm_path = _normalize_ci_path(path.strip())
        skip_map[norm_path] = reason

    filtered_tests: List[CIRegistry] = []
    matched: set[str] = set()
    for test in ci_tests:
        filename_norm = _normalize_ci_path(test.filename)
        reason = skip_map.get(filename_norm)
        if reason:
            matched.add(filename_norm)
            skipped_tests.append(
                CIRegistry(
                    backend=test.backend,
                    filename=test.filename,
                    est_time=test.est_time,
                    suite=test.suite,
                    nightly=test.nightly,
                    disabled=reason,
                )
            )
            continue
        filtered_tests.append(test)

    unmatched = set(skip_map.keys()) - matched
    for missing in sorted(unmatched):
        print(f"Warning: --skip-file target '{missing}' not found in selected tests.")

    return filtered_tests, skipped_tests


def auto_partition(files: List[CIRegistry], rank, size):
    """
    Partition files into size sublists with approximately equal sums of estimated times
    using a greedy algorithm (LPT heuristic), and return the partition for the specified rank.
    """
    if not files or size <= 0:
        return []

    # Sort files by estimated_time in descending order (LPT heuristic)
    sorted_files = sorted(files, key=lambda f: f.est_time, reverse=True)

    partitions = [[] for _ in range(size)]
    partition_sums = [0.0] * size

    # Greedily assign each file to the partition with the smallest current total time
    for file in sorted_files:
        min_sum_idx = min(range(size), key=partition_sums.__getitem__)
        partitions[min_sum_idx].append(file)
        partition_sums[min_sum_idx] += file.est_time

    if rank < size:
        return partitions[rank]
    return []


def pretty_print_tests(
    args, ci_tests: List[CIRegistry], skipped_tests: List[CIRegistry]
):
    hw = HW_MAPPING[args.hw]
    suite = args.suite
    nightly = args.nightly
    if args.auto_partition_size:
        partition_info = (
            f"{args.auto_partition_id + 1}/{args.auto_partition_size} "
            f"(0-based id={args.auto_partition_id})"
        )
    else:
        partition_info = "full"

    headers = ["Hardware", "Suite", "Nightly", "Partition"]
    rows = [[hw.name, suite, str(nightly), partition_info]]
    msg = tabulate.tabulate(rows, headers=headers, tablefmt="psql") + "\n"

    if skipped_tests:
        msg += f"⚠️  Skipped {len(skipped_tests)} test(s):\n"
        for t in skipped_tests:
            reason = t.disabled or "disabled"
            msg += f"  - {t.filename} (reason: {reason})\n"
        msg += "\n"

    if len(ci_tests) == 0:
        msg += f"No tests found for hw={hw.name}, suite={suite}, nightly={nightly}\n"
        msg += "This is expected during incremental migration. Skipping.\n"
    else:
        total_est_time = sum(t.est_time for t in ci_tests)
        msg += (
            f"✅ Enabled {len(ci_tests)} test(s) (est total {total_est_time:.1f}s):\n"
        )
        for t in ci_tests:
            msg += f"  - {t.filename} (est_time={t.est_time})\n"

    print(msg, flush=True)


def run_a_suite(args):
    hw = HW_MAPPING[args.hw]
    suite = args.suite
    nightly = args.nightly
    auto_partition_id = args.auto_partition_id
    auto_partition_size = args.auto_partition_size

    # All tests (per-commit and nightly) are now in registered/
    files = glob.glob("registered/**/*.py", recursive=True)

    # Strict: all registered files must have proper registration
    sanity_check = True

    all_tests = collect_tests(files, sanity_check=sanity_check)
    ci_tests, skipped_tests = filter_tests(all_tests, hw, suite, nightly)
    ci_tests, skipped_tests = apply_manual_skips(
        ci_tests, skipped_tests, args.skip_file
    )

    if auto_partition_size:
        ci_tests = auto_partition(ci_tests, auto_partition_id, auto_partition_size)

    pretty_print_tests(args, ci_tests, skipped_tests)

    # Add extra timeout when retry is enabled
    timeout = args.timeout_per_file
    if args.enable_retry:
        timeout += args.retry_timeout_increase

    return run_unittest_files(
        ci_tests,
        timeout_per_file=timeout,
        continue_on_error=args.continue_on_error,
        enable_retry=args.enable_retry,
        max_attempts=args.max_attempts,
        retry_wait_seconds=args.retry_wait_seconds,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run CI test suites from test/registered/"
    )
    parser.add_argument(
        "--hw",
        type=str,
        choices=HW_MAPPING.keys(),
        required=True,
        help="Hardware backend to run tests on.",
    )
    parser.add_argument("--suite", type=str, required=True, help="Test suite to run.")
    parser.add_argument(
        "--nightly",
        action="store_true",
        help="Run nightly tests instead of per-commit tests.",
    )
    parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1200,
        help="The time limit for running one file in seconds (default: 1200).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue running remaining tests even if one fails (default: False, useful for nightly tests).",
    )
    parser.add_argument(
        "--auto-partition-id",
        type=int,
        help="Use auto load balancing. The part id.",
    )
    parser.add_argument(
        "--auto-partition-size",
        type=int,
        help="Use auto load balancing. The number of parts.",
    )
    parser.add_argument(
        "--enable-retry",
        action="store_true",
        default=False,
        help="Enable smart retry for accuracy/performance assertion failures (not code errors)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum number of attempts per file including initial run (default: 2)",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=int,
        default=60,
        help="Seconds to wait between retries (default: 60)",
    )
    parser.add_argument(
        "--retry-timeout-increase",
        type=int,
        default=600,
        help="Additional timeout in seconds when retry is enabled (default: 600)",
    )
    parser.add_argument(
        "--skip-file",
        action="append",
        default=[],
        metavar="PATH[=REASON]",
        help=(
            "Skip a registered test file (relative path as shown in listings). "
            "Can be provided multiple times. Optionally specify a reason via PATH=REASON."
        ),
    )
    args = parser.parse_args()

    # Validate auto-partition arguments
    if (args.auto_partition_id is not None) != (args.auto_partition_size is not None):
        parser.error(
            "--auto-partition-id and --auto-partition-size must be specified together."
        )
    if args.auto_partition_size is not None:
        if args.auto_partition_size <= 0:
            parser.error("--auto-partition-size must be positive.")
        if not 0 <= args.auto_partition_id < args.auto_partition_size:
            parser.error(
                f"--auto-partition-id must be in range [0, {args.auto_partition_size}), "
                f"but got {args.auto_partition_id}"
            )

    exit_code = run_a_suite(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
