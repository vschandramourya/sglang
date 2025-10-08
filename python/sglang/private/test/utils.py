from typing import List, Optional

from pydantic import BaseModel


class ProfileLinks(BaseModel):
    """Pydantic model for profile trace links."""

    extend: Optional[str] = None
    decode: Optional[str] = None


class BenchmarkResult(BaseModel):
    """Pydantic model for benchmark results table data, for a single isl and osl"""

    model_path: str
    run_name: str
    batch_size: int
    input_len: int
    output_len: int
    latency: float
    ttft: float
    input_throughput: float
    output_throughput: float
    overall_throughput: float
    last_gen_throughput: float
    acc_length: Optional[float] = None
    profile_links: Optional[ProfileLinks] = None

    @staticmethod
    def help_str() -> str:
        return f"""
Note: To view the traces through perfetto-ui, please:
    1. open with Google Chrome
    2. allow popup
"""

    def to_markdown_row(
        self, trace_dir, base_url: str = "", relay_base: str = ""
    ) -> str:
        """Convert this benchmark result to a markdown table row."""
        # Calculate costs (assuming H100 pricing for now)
        hourly_cost_per_gpu = 2  # $2/hour for one H100
        hourly_cost = hourly_cost_per_gpu * 1  # Assuming tp_size = 1 for simplicity
        input_util = 0.7
        accept_length = (
            round(self.acc_length, 2) if self.acc_length is not None else "n/a"
        )
        itl = 1 / (self.output_throughput / self.batch_size) * 1000
        input_cost = 1e6 / (self.input_throughput * input_util) / 3600 * hourly_cost
        output_cost = 1e6 / self.output_throughput / 3600 * hourly_cost

        def get_perfetto_relay_link_from_trace_file(trace_file: str):
            import os
            from urllib.parse import quote

            rel_path = os.path.relpath(trace_file, trace_dir)
            raw_file_link = f"{base_url}/{rel_path}"
            relay_link = (
                f"{relay_base}?src={quote(raw_file_link, safe='')}"
                if relay_base and quote
                else raw_file_link
            )
            return relay_link

        # Handle profile links
        profile_link = "NA | NA"
        if self.profile_links:
            if self.profile_links.extend or self.profile_links.decode:
                # Create a combined link or use the first available one
                trace_files = [self.profile_links.extend, self.profile_links.decode]
                trace_files_relay_links = [
                    f"[trace]({get_perfetto_relay_link_from_trace_file(trace_file)})"
                    for trace_file in trace_files
                ]

                profile_link = " | ".join(trace_files_relay_links)

        # Build the row
        return f"| {self.batch_size} | {self.input_len} | {self.latency:.2f} | {self.input_throughput:.2f} | {self.output_throughput:.2f} | {accept_length} | {itl:.2f} | {input_cost:.2f} | {output_cost:.2f} | {profile_link} |\n"

    @classmethod
    def generate_markdown_report(
        cls, trace_dir, results: List["BenchmarkResult"]
    ) -> str:
        """Generate a markdown report from a list of BenchmarkResult object from a single run."""
        import os

        summary = f"### {results[0].model_path}\n"

        # summary += (
        #     f"Input lens: {result.input_len}. Output lens: {result.output_len}.\n"
        # )
        summary += "| batch size | input len | latency (s) | input throughput (tok/s)  | output throughput (tok/s) | acc length | ITL (ms) | input cost ($/1M) | output cost ($/1M) | profile (extend) | profile (decode)|\n"
        summary += "| ---------- | --------- | ----------- | ------------------------- | ------------------------- | ---------- | -------- | ----------------- | ------------------ | --------------- | -------------- |\n"

        # all results should share the same isl & osl
        for result in results:
            base_url = os.getenv("TRACE_BASE_URL", "").rstrip("/")
            relay_base = os.getenv("PERFETTO_RELAY_URL", "").rstrip("/")
            summary += result.to_markdown_row(trace_dir, base_url, relay_base)
        return summary
