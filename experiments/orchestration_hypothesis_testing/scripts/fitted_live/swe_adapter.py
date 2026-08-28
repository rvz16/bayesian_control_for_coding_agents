"""SWE-Bench Lite/Verified adapter for live fitted-controller runs."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import Candidate, CriticResult, VerifyResult, feedback_block


def _truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n\n...[truncated for context length]...\n\n"
    keep = max(0, max_chars - len(marker))
    if keep <= 0:
        return text[:max_chars]
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def _cap_oracle_files(oracle_files: dict[str, str], budget: int) -> dict[str, str]:
    if budget <= 0 or not oracle_files:
        return {}
    total = sum(len(content) for content in oracle_files.values())
    if total <= budget:
        return dict(oracle_files)
    per_file = max(2000, budget // max(1, len(oracle_files)))
    return {
        path: _truncate_middle(content, per_file)
        for path, content in oracle_files.items()
    }


@dataclass
class SWEAdapter:
    benchmark: str
    dataset_name: str
    n_instances: int
    seed: int
    output_dir: Path
    harness_workers: int = 1
    _oracle_cache: dict[str, dict[str, str]] = field(default_factory=dict)

    def load_instances(self) -> list[dict]:
        import spot_check_generators as scg

        return scg.sample_instances(
            seed=self.seed,
            n=self.n_instances,
            dataset_name=self.dataset_name,
        )

    def instance_id(self, instance: dict) -> str:
        return str(instance["instance_id"])

    def _oracle_files(self, instance: dict) -> dict[str, str]:
        import spot_check_generators as scg

        inst_id = self.instance_id(instance)
        if inst_id not in self._oracle_cache:
            files = scg.get_changed_files_from_patch(instance.get("patch", "") or "")
            self._oracle_cache[inst_id] = scg.fetch_oracle_files(
                instance["repo"],
                instance["base_commit"],
                files,
            )
        return self._oracle_cache[inst_id]

    def build_prompt(
        self,
        instance: dict,
        previous: Candidate | None,
        action_log: list[dict[str, Any]],
    ) -> str:
        import spot_check_generators as scg

        max_prompt_chars = int(os.getenv("SWE_AGENT_PROMPT_MAX_CHARS", "45000"))
        if max_prompt_chars <= 0:
            return scg.make_prompt(instance, self._oracle_files(instance)) + feedback_block(
                previous,
                action_log,
            )

        prompt_instance = dict(instance)
        problem_cap = int(os.getenv("SWE_AGENT_PROBLEM_MAX_CHARS", "16000"))
        hints_cap = int(os.getenv("SWE_AGENT_HINTS_MAX_CHARS", "3000"))
        prompt_instance["problem_statement"] = _truncate_middle(
            str(prompt_instance.get("problem_statement", "")),
            problem_cap,
        )
        if prompt_instance.get("hints_text"):
            prompt_instance["hints_text"] = _truncate_middle(
                str(prompt_instance.get("hints_text", "")),
                hints_cap,
            )

        non_file_chars = (
            len(str(prompt_instance.get("problem_statement", "")))
            + len(str(prompt_instance.get("hints_text", "")))
            + 8000
        )
        file_budget = max(8000, max_prompt_chars - non_file_chars)
        oracle_files = _cap_oracle_files(self._oracle_files(instance), file_budget)
        return scg.make_prompt(prompt_instance, oracle_files) + feedback_block(previous, action_log)

    def extract_candidate(self, instance: dict, response_text: str) -> Candidate:
        import spot_check_generators as scg

        oracle = self._oracle_files(instance)
        diff, extraction_path, n_blocks = scg._extract_diff_from_response(response_text, oracle)
        return Candidate(
            payload=diff,
            raw_text=response_text,
            kind="diff",
            metadata={"extraction_path": extraction_path, "n_blocks": n_blocks},
        )

    def run_critic(self, critic: str, instance: dict, candidate: Candidate, reviewer_client) -> CriticResult:
        if critic == "L2":
            # SWE-Bench has no public-test critic in this pipeline. The fitted
            # likelihood tables for SWE should omit L2, so controllers should
            # not normally request it.
            return CriticResult(None, detail="unsupported_for_swe")

        import calibrate_from_spotcheck as cfs

        diff = candidate.payload or ""
        if not diff.strip():
            return CriticResult(False, detail="empty_diff")
        if critic == "L3":
            ok, cost = cfs.critic_L3_llm_review(
                self.instance_id(instance),
                instance.get("problem_statement", ""),
                diff,
                reviewer_client,
            )
            return CriticResult(bool(ok), api_cost_usd=float(cost))

        modified = cfs._modified_file_contents(diff, self._oracle_files(instance))
        if modified is None:
            return CriticResult(False, detail="diff_apply_failed")
        if critic == "L0":
            return CriticResult(bool(cfs.critic_L0_syntax(modified)))
        if critic == "L1":
            return CriticResult(bool(cfs.critic_L1_lint(modified)))
        raise ValueError(f"unknown critic: {critic}")

    def verify(self, instance: dict, candidate: Candidate, run_id: str) -> VerifyResult:
        import spot_check_generators as scg

        work_dir = (self.output_dir / "swe_harness").resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        pred_path = work_dir / f"{run_id}.jsonl"
        pred = {
            "instance_id": self.instance_id(instance),
            "model_name_or_path": run_id,
            "model_patch": candidate.payload or "",
        }
        pred_path.write_text(json.dumps(pred) + "\n")
        try:
            report_path = scg.run_swebench_eval(
                predictions_path=pred_path,
                run_id=run_id,
                max_workers=self.harness_workers,
                work_dir=work_dir,
                dataset_name=self.dataset_name,
            )
            resolved = scg.parse_resolved(scg.load_report(report_path))
            ok = self.instance_id(instance) in resolved
            return VerifyResult(ok, detail=str(report_path))
        except Exception as exc:
            return VerifyResult(False, detail=f"harness_error: {type(exc).__name__}: {exc}")


def make_swe_adapter(
    benchmark: str,
    n_instances: int,
    seed: int,
    output_dir: Path,
    harness_workers: int,
) -> SWEAdapter:
    if benchmark == "swebench_lite":
        dataset = "princeton-nlp/SWE-bench_Lite"
    elif benchmark == "swebench_verified":
        dataset = "princeton-nlp/SWE-bench_Verified"
    else:
        raise ValueError(f"not a SWE benchmark: {benchmark}")
    return SWEAdapter(
        benchmark=benchmark,
        dataset_name=dataset,
        n_instances=n_instances,
        seed=seed,
        output_dir=output_dir,
        harness_workers=harness_workers,
    )
