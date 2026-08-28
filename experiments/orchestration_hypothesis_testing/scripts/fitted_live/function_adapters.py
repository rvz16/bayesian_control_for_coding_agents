"""Function-level adapters for LCB, MBPP+, and HumanEval+."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from .common import Candidate, CriticResult, VerifyResult, feedback_block


@dataclass
class LCBAdapter:
    benchmark: str
    difficulty: str
    n_instances: int
    seed: int
    platform: str = "leetcode"
    lcb_version: str = "all"
    private_test_cap: int = 12

    def load_instances(self) -> list[dict]:
        from lcb_calibrate import load_lcb

        problems = load_lcb(
            difficulty=self.difficulty,
            platform=self.platform,
            lcb_version=self.lcb_version,
        )
        random.Random(self.seed).shuffle(problems)
        return problems[: self.n_instances] if self.n_instances else problems

    def instance_id(self, instance: dict) -> str:
        return str(instance["question_id"])

    def build_prompt(
        self,
        instance: dict,
        previous: Candidate | None,
        action_log: list[dict[str, Any]],
    ) -> str:
        from lcb_calibrate import build_prompt

        return build_prompt(instance) + feedback_block(previous, action_log)

    def extract_candidate(self, instance: dict, response_text: str) -> Candidate:
        from lcb_calibrate import extract_code

        return Candidate(payload=extract_code(response_text), raw_text=response_text, kind="code")

    def run_critic(self, critic: str, instance: dict, candidate: Candidate, reviewer_client) -> CriticResult:
        from lcb_calibrate import (
            check_tests,
            critic_L0_syntax,
            critic_L1_lint,
            critic_L3_review,
        )

        code = candidate.payload
        if critic == "L0":
            return CriticResult(bool(critic_L0_syntax(code)))
        if critic == "L1":
            return CriticResult(bool(critic_L1_lint(code)))
        if critic == "L2":
            tests = instance.get("public_test_cases") or []
            if isinstance(tests, str):
                try:
                    tests = json.loads(tests)
                except Exception:
                    tests = []
            passed, total = check_tests(code, tests, starter_code=instance.get("starter_code", "") or "")
            return CriticResult((passed == total) and total > 0, detail=f"{passed}/{total}")
        if critic == "L3":
            ok, cost = critic_L3_review(instance.get("question_content", "")[:3000], code, reviewer_client)
            detail = "PASS" if ok is True else "FAIL" if ok is False else "review_error_or_unparseable"
            return CriticResult(ok, detail=detail, api_cost_usd=float(cost))
        raise ValueError(f"unknown critic: {critic}")

    def verify(self, instance: dict, candidate: Candidate, run_id: str) -> VerifyResult:
        from lcb_calibrate import MAX_PRIVATE_TESTS, check_tests, decode_private_tests

        tests = decode_private_tests(instance.get("private_test_cases", "") or "")
        cap = self.private_test_cap if self.private_test_cap is not None else MAX_PRIVATE_TESTS
        if cap > 0:
            tests = tests[:cap]
        passed, total = check_tests(
            candidate.payload,
            tests,
            starter_code=instance.get("starter_code", "") or "",
        )
        return VerifyResult((passed == total) and total > 0, detail=f"{passed}/{total}")


@dataclass
class MBPPAdapter:
    benchmark: str
    n_instances: int
    seed: int

    def load_instances(self) -> list[dict]:
        from mbpp_calibrate import load_mbpp_plus

        return load_mbpp_plus(self.n_instances, seed=self.seed)

    def instance_id(self, instance: dict) -> str:
        return str(instance["task_id"])

    def build_prompt(
        self,
        instance: dict,
        previous: Candidate | None,
        action_log: list[dict[str, Any]],
    ) -> str:
        from mbpp_calibrate import build_prompt

        return build_prompt(instance) + feedback_block(previous, action_log)

    def extract_candidate(self, instance: dict, response_text: str) -> Candidate:
        from lcb_calibrate import extract_code

        return Candidate(payload=extract_code(response_text), raw_text=response_text, kind="code")

    def run_critic(self, critic: str, instance: dict, candidate: Candidate, reviewer_client) -> CriticResult:
        from lcb_calibrate import critic_L0_syntax, critic_L1_lint, critic_L3_review
        from mbpp_calibrate import run_assertions

        code = candidate.payload
        if critic == "L0":
            return CriticResult(bool(critic_L0_syntax(code)))
        if critic == "L1":
            return CriticResult(bool(critic_L1_lint(code)))
        if critic == "L2":
            passed, total = run_assertions(code, instance.get("test_list") or [])
            return CriticResult((passed == total) and total > 0, detail=f"{passed}/{total}")
        if critic == "L3":
            ok, cost = critic_L3_review(instance.get("prompt", "")[:3000], code, reviewer_client)
            detail = "PASS" if ok is True else "FAIL" if ok is False else "review_error_or_unparseable"
            return CriticResult(ok, detail=detail, api_cost_usd=float(cost))
        raise ValueError(f"unknown critic: {critic}")

    def verify(self, instance: dict, candidate: Candidate, run_id: str) -> VerifyResult:
        from mbpp_calibrate import run_full_test

        ok = run_full_test(candidate.payload, instance.get("test") or "", instance.get("entry_point"))
        return VerifyResult(bool(ok))


@dataclass
class HumanEvalPlusAdapter:
    benchmark: str
    n_instances: int
    seed: int
    plus_input_cap: int = 200

    def load_instances(self) -> list[dict]:
        from humaneval_calibrate import load_humaneval_plus

        return load_humaneval_plus(self.n_instances, self.plus_input_cap, seed=self.seed)

    def instance_id(self, instance: dict) -> str:
        return str(instance["task_id"])

    def build_prompt(
        self,
        instance: dict,
        previous: Candidate | None,
        action_log: list[dict[str, Any]],
    ) -> str:
        from humaneval_calibrate import build_prompt

        return build_prompt(instance) + feedback_block(previous, action_log)

    def extract_candidate(self, instance: dict, response_text: str) -> Candidate:
        from lcb_calibrate import extract_code

        return Candidate(payload=extract_code(response_text), raw_text=response_text, kind="code")

    def run_critic(self, critic: str, instance: dict, candidate: Candidate, reviewer_client) -> CriticResult:
        from humaneval_calibrate import run_test_inputs
        from lcb_calibrate import critic_L0_syntax, critic_L1_lint, critic_L3_review

        code = candidate.payload
        if critic == "L0":
            return CriticResult(bool(critic_L0_syntax(code)))
        if critic == "L1":
            return CriticResult(bool(critic_L1_lint(code)))
        if critic == "L2":
            passed, total = run_test_inputs(
                code,
                instance["entry_point"],
                instance["canonical_full"],
                instance.get("base_input") or [],
                instance.get("atol") or 0,
            )
            return CriticResult((passed == total) and total > 0, detail=f"{passed}/{total}")
        if critic == "L3":
            ok, cost = critic_L3_review(instance.get("prompt", "")[:3000], code, reviewer_client)
            detail = "PASS" if ok is True else "FAIL" if ok is False else "review_error_or_unparseable"
            return CriticResult(ok, detail=detail, api_cost_usd=float(cost))
        raise ValueError(f"unknown critic: {critic}")

    def verify(self, instance: dict, candidate: Candidate, run_id: str) -> VerifyResult:
        from humaneval_calibrate import run_test_inputs

        full_inputs = list(instance.get("base_input") or []) + list(instance.get("plus_input") or [])
        passed, total = run_test_inputs(
            candidate.payload,
            instance["entry_point"],
            instance["canonical_full"],
            full_inputs,
            instance.get("atol") or 0,
        )
        return VerifyResult((passed == total) and total > 0, detail=f"{passed}/{total}")


@dataclass
class HumanEvalFixAdapter:
    benchmark: str
    n_instances: int
    seed: int

    def load_instances(self) -> list[dict]:
        from humanevalfix_calibrate import load_humanevalfix

        return load_humanevalfix(self.n_instances, seed=self.seed)

    def instance_id(self, instance: dict) -> str:
        return str(instance["task_id"])

    def build_prompt(
        self,
        instance: dict,
        previous: Candidate | None,
        action_log: list[dict[str, Any]],
    ) -> str:
        from humanevalfix_calibrate import build_prompt

        return build_prompt(instance) + feedback_block(previous, action_log)

    def extract_candidate(self, instance: dict, response_text: str) -> Candidate:
        from lcb_calibrate import extract_code

        return Candidate(payload=extract_code(response_text), raw_text=response_text, kind="code")

    def run_critic(self, critic: str, instance: dict, candidate: Candidate, reviewer_client) -> CriticResult:
        from humanevalfix_calibrate import _eval_with_block
        from lcb_calibrate import critic_L0_syntax, critic_L1_lint, critic_L3_review

        code = candidate.payload
        if critic == "L0":
            return CriticResult(bool(critic_L0_syntax(code)))
        if critic == "L1":
            return CriticResult(bool(critic_L1_lint(code)))
        if critic == "L2":
            ok = _eval_with_block(
                code,
                instance.get("example_test") or "",
                instance.get("entry_point"),
                instance.get("test_setup") or "",
                timeout=10,
            )
            return CriticResult(bool(ok))
        if critic == "L3":
            ok, cost = critic_L3_review(instance.get("prompt", "")[:3000], code, reviewer_client)
            detail = "PASS" if ok is True else "FAIL" if ok is False else "review_error_or_unparseable"
            return CriticResult(ok, detail=detail, api_cost_usd=float(cost))
        raise ValueError(f"unknown critic: {critic}")

    def verify(self, instance: dict, candidate: Candidate, run_id: str) -> VerifyResult:
        from humanevalfix_calibrate import _eval_with_block

        ok = _eval_with_block(
            candidate.payload,
            instance.get("test") or "",
            instance.get("entry_point"),
            instance.get("test_setup") or "",
            timeout=15,
        )
        return VerifyResult(bool(ok))


@dataclass
class CodeContestsAdapter:
    benchmark: str
    n_instances: int
    seed: int

    def load_instances(self) -> list[dict]:
        from codecontests_calibrate import load_codecontests

        return load_codecontests(self.n_instances, seed=self.seed)

    def instance_id(self, instance: dict) -> str:
        return str(instance["name"])

    def build_prompt(
        self,
        instance: dict,
        previous: Candidate | None,
        action_log: list[dict[str, Any]],
    ) -> str:
        from codecontests_calibrate import build_prompt

        return build_prompt(instance) + feedback_block(previous, action_log)

    def extract_candidate(self, instance: dict, response_text: str) -> Candidate:
        from lcb_calibrate import extract_code

        return Candidate(payload=extract_code(response_text), raw_text=response_text, kind="code")

    def _public_tests(self, instance: dict) -> tuple[list, list]:
        from codecontests_calibrate import PUBLIC_TEST_CAP

        tests = instance.get("public_tests") or {}
        return (
            list(tests.get("input", []))[:PUBLIC_TEST_CAP],
            list(tests.get("output", []))[:PUBLIC_TEST_CAP],
        )

    def _oracle_tests(self, instance: dict) -> tuple[list, list]:
        from codecontests_calibrate import ORACLE_TEST_CAP

        public_in, public_out = self._public_tests(instance)
        private = instance.get("private_tests") or {}
        generated = instance.get("generated_tests") or {}
        inputs = list(private.get("input", [])) + list(generated.get("input", [])) + public_in
        outputs = list(private.get("output", [])) + list(generated.get("output", [])) + public_out
        return inputs[:ORACLE_TEST_CAP], outputs[:ORACLE_TEST_CAP]

    def run_critic(self, critic: str, instance: dict, candidate: Candidate, reviewer_client) -> CriticResult:
        from codecontests_calibrate import run_stdio_tests
        from lcb_calibrate import critic_L0_syntax, critic_L1_lint, critic_L3_review

        code = candidate.payload
        if critic == "L0":
            return CriticResult(bool(critic_L0_syntax(code)))
        if critic == "L1":
            return CriticResult(bool(critic_L1_lint(code)))
        if critic == "L2":
            inputs, outputs = self._public_tests(instance)
            passed, total = run_stdio_tests(code, inputs, outputs)
            return CriticResult((passed == total) and total > 0, detail=f"{passed}/{total}")
        if critic == "L3":
            ok, cost = critic_L3_review(instance.get("description", "")[:3000], code, reviewer_client)
            detail = "PASS" if ok is True else "FAIL" if ok is False else "review_error_or_unparseable"
            return CriticResult(ok, detail=detail, api_cost_usd=float(cost))
        raise ValueError(f"unknown critic: {critic}")

    def verify(self, instance: dict, candidate: Candidate, run_id: str) -> VerifyResult:
        from codecontests_calibrate import run_stdio_tests

        inputs, outputs = self._oracle_tests(instance)
        passed, total = run_stdio_tests(candidate.payload, inputs, outputs)
        return VerifyResult((passed == total) and total > 0, detail=f"{passed}/{total}")


def make_function_adapter(
    benchmark: str,
    n_instances: int,
    seed: int,
    lcb_version: str,
    plus_input_cap: int,
    lcb_private_test_cap: int = 12,
    platform: str = "leetcode",
):
    if benchmark.startswith("lcb_"):
        difficulty = benchmark.split("_", 1)[1]
        return LCBAdapter(
            benchmark=benchmark,
            difficulty={"med": "medium"}.get(difficulty, difficulty),
            n_instances=n_instances,
            seed=seed,
            platform=platform,
            lcb_version=lcb_version,
            private_test_cap=lcb_private_test_cap,
        )
    if benchmark == "mbpp":
        return MBPPAdapter(benchmark=benchmark, n_instances=n_instances, seed=seed)
    if benchmark == "humaneval":
        return HumanEvalPlusAdapter(
            benchmark=benchmark,
            n_instances=n_instances,
            seed=seed,
            plus_input_cap=plus_input_cap,
        )
    if benchmark == "humanevalfix":
        return HumanEvalFixAdapter(benchmark=benchmark, n_instances=n_instances, seed=seed)
    if benchmark == "codecontests":
        return CodeContestsAdapter(benchmark=benchmark, n_instances=n_instances, seed=seed)
    raise ValueError(f"not a function benchmark: {benchmark}")
