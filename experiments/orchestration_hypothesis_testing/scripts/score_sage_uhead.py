#!/usr/bin/env python3
"""Post-hoc UHead scores for saved SAGE code-generation trajectories."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import multiprocessing
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from different_agents.v4.lcb_llm_tool_agent import _adapter_log, make_adapter  # noqa: E402
from experiments.orchestration_hypothesis_testing.scripts.fitted_live.common import (  # noqa: E402
    Candidate,
)

MODELS = {
    "gpt_oss_20b_local": "openai/gpt-oss-20b",
    "gpt_oss_120b_local": "openai/gpt-oss-120b",
    "qwen25_32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
}
DEFAULT_GPT_UHEAD = (
    "ArtemVazhentsev21/uhead_math500_gpt_oss_120b_claim_pro_fixed_repaired_l21_cap30"
)
O200K_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_harmony_vocab(model: str) -> None:
    """Make vLLM's native GPT-OSS Harmony formatter work offline."""
    base = Path(
        os.environ.get(
            "TIKTOKEN_ENCODINGS_BASE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "harmony",
        )
    )
    target = base / "o200k_base.tiktoken"
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == O200K_SHA256:
        os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(base)
        return

    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    from transformers.utils.hub import cached_file

    tokenizer_path = cached_file(model, "tokenizer.json")
    vocab = json.loads(Path(tokenizer_path).read_text())["model"]["vocab"]
    inverse_bytes = {char: byte for byte, char in bytes_to_unicode().items()}
    rows = sorted(
        (rank, bytes(inverse_bytes[char] for char in token)) for token, rank in vocab.items()
    )
    payload = b"".join(
        base64.b64encode(token) + b" " + str(rank).encode() + b"\n" for rank, token in rows
    )
    if hashlib.sha256(payload).hexdigest() != O200K_SHA256:
        raise ValueError("GPT-OSS tokenizer does not match the o200k_base vocabulary")
    base.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, target)
    os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(base)


def infer_harmony_start_date(run_root: Path) -> str | None:
    for part in reversed(run_root.parts):
        match = re.search(r"(?:^|_)(\d{4})(\d{2})(\d{2})_\d{6}$", part)
        if match:
            return "-".join(match.groups())
    return None


def token_ids_by_bytes(tokenizer: Any) -> dict[bytes, int]:
    cached = getattr(tokenizer, "_sage_uhead_token_ids_by_bytes", None)
    if cached is not None:
        return cached

    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    inverse_bytes = {char: byte for byte, char in bytes_to_unicode().items()}
    mapping = {
        bytes(inverse_bytes[char] for char in token): int(token_id)
        for token, token_id in tokenizer.get_vocab().items()
        if all(char in inverse_bytes for char in token)
    }
    tokenizer._sage_uhead_token_ids_by_bytes = mapping
    return mapping


def completion_ids(tokenizer: Any, row: dict[str, Any]) -> list[int]:
    content = (row.get("logprobs") or {}).get("content") or []
    if not content or any(not isinstance(item, dict) for item in content):
        raise ValueError("missing completion tokens in logprobs.content")
    mapping = token_ids_by_bytes(tokenizer)
    try:
        ids = [mapping[bytes(item["bytes"])] for item in content]
    except (KeyError, TypeError) as exc:
        raise ValueError("saved token bytes are absent from the tokenizer vocabulary") from exc
    expected = int(row.get("completion_tokens") or 0)
    if expected and len(ids) != expected:
        raise ValueError(f"completion token mismatch: reconstructed={len(ids)}, logged={expected}")
    return [int(token_id) for token_id in ids]


def prompt_ids(
    tokenizer: Any,
    prompt: str,
    *,
    harmony: bool,
    harmony_start_date: str | None,
) -> list[int]:
    if harmony:
        from vllm.entrypoints.harmony_utils import (
            get_developer_message,
            get_system_message,
            parse_input_to_harmony_message,
            render_for_completion,
        )

        messages = [
            get_system_message(
                reasoning_effort=None,
                start_date=harmony_start_date,
                browser_description=None,
                python_description=None,
                with_custom_tools=False,
            ),
            get_developer_message(tools=None),
        ]
        messages.extend(parse_input_to_harmony_message({"role": "user", "content": prompt}))
        return [int(token_id) for token_id in render_for_completion(messages)]

    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    return [int(token_id) for token_id in ids]


def iter_inputs(args: argparse.Namespace, tokenizer: Any):
    stem = f"{args.benchmark}__{args.generator}"
    results = {
        str(row["instance_id"]): row
        for row in read_jsonl(args.run_root / f"{stem}.jsonl")
        if row.get("split") == "test"
    }
    adapter_args = argparse.Namespace(
        benchmark=args.benchmark,
        n_instances=0,
        seed=args.seed,
        output=args.run_root / f"{stem}.jsonl",
        swe_harness_workers=args.swe_harness_workers,
        lcb_version=args.lcb_version,
        plus_input_cap=args.plus_input_cap,
        private_test_cap=args.private_test_cap,
        platform=args.platform,
    )
    adapter = make_adapter(adapter_args)
    instances = {adapter.instance_id(instance): instance for instance in adapter.load_instances()}
    kind = "diff" if args.benchmark.startswith("swebench_") else "code"
    generate_actions = {
        iid: [
            (pos, action)
            for pos, action in enumerate(result.get("trajectory") or [])
            if action.get("action") == "generate" and action.get("skipped") is not True
        ]
        for iid, result in results.items()
    }
    previous: dict[str, Candidate] = {}
    path = args.run_root / f"{stem}.generation_logprobs.jsonl"

    # The log is append-only across resumed attempts. Index lightweight offsets
    # first, then select the latest complete attempt matching the final trajectory.
    segments: dict[str, list[dict[int, tuple[int, dict[str, Any]]]]] = {}
    current: dict[str, dict[int, tuple[int, dict[str, Any]]]] = {}
    with path.open() as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            iid = str(row.get("instance_id"))
            if iid not in results or iid not in instances:
                continue
            gen_idx = int(row.get("generation_index", 0))
            if gen_idx == 0 or iid not in current:
                current[iid] = {}
                segments.setdefault(iid, []).append(current[iid])
            current[iid][gen_idx] = (
                offset,
                {
                    "step": row.get("step"),
                    "prompt_tokens": row.get("prompt_tokens"),
                    "completion_tokens": row.get("completion_tokens"),
                },
            )

    selected: list[tuple[int, str, int, int]] = []
    for iid, actions in generate_actions.items():
        if not actions:
            continue
        expected = set(range(len(actions)))
        chosen = None
        for segment in reversed(segments.get(iid, [])):
            if set(segment) != expected:
                continue
            if all(
                int(segment[idx][1].get("step", -1)) == int(action.get("step", -2))
                and int(segment[idx][1].get("prompt_tokens") or 0)
                == int(action.get("prompt_tokens") or 0)
                and int(segment[idx][1].get("completion_tokens") or 0)
                == int(action.get("completion_tokens") or 0)
                for idx, (_, action) in enumerate(actions)
            ):
                chosen = segment
                break
        if chosen is None:
            raise ValueError(f"{iid}: no saved generation attempt matches final trajectory")
        selected.extend((chosen[idx][0], iid, idx, pos) for idx, (pos, _) in enumerate(actions))

    with path.open() as f:
        for offset, iid, gen_idx, pos in sorted(selected):
            f.seek(offset)
            row = json.loads(f.readline())
            result = results[iid]
            instance = instances[iid]
            if gen_idx == 0:
                previous.pop(iid, None)
            trajectory = result.get("trajectory") or []
            prompt = adapter.build_prompt(
                instance, previous.get(iid), _adapter_log(trajectory[:pos])
            )
            p_ids = prompt_ids(
                tokenizer,
                prompt,
                harmony=args.generator.startswith("gpt_oss_"),
                harmony_start_date=args.harmony_start_date,
            )
            c_ids = completion_ids(tokenizer, row)
            logged_prompt = int(row.get("prompt_tokens") or 0)
            if logged_prompt and len(p_ids) != logged_prompt:
                raise ValueError(
                    f"{iid} generation {gen_idx}: prompt token mismatch: "
                    f"reconstructed={len(p_ids)}, logged={logged_prompt}"
                )
            yield {
                "benchmark": args.benchmark,
                "instance_id": iid,
                "generation_index": gen_idx,
                "action_step": int(row.get("step", 0) or 0),
                "prompt_ids": p_ids,
                "completion_ids": c_ids,
            }
            payload = str(row.get("code") or "")
            if payload:
                previous[iid] = Candidate(
                    payload=payload,
                    raw_text=str(row.get("raw_text") or ""),
                    kind=kind,
                )
            else:
                previous.pop(iid, None)


class UHeadScorer:
    def __init__(self, args: argparse.Namespace):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method("spawn")

        import numpy as np
        import torch
        from lm_polygraph.stat_calculators.extract_claims import Claim
        from luh import AutoUncertaintyHead
        from luh.calculator_apply_uq_head import CalculatorApplyUQHead
        from luh.luh_claim_estimator_dummy import LuhClaimEstimatorDummy
        from luh.vllm.vllm_uhead_features import VLLMUncertaintyHeadFeatures
        from vllm import LLM, SamplingParams, TokensPrompt

        self.np = np
        self.torch = torch
        self.Claim = Claim
        self.TokensPrompt = TokensPrompt
        self.capture_params = SamplingParams(temperature=0, max_tokens=1)
        self.llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
            enable_prefix_caching=False,
            worker_extension_cls="utils.hook_hs_extension.HookHiddenStatesExtension",
        )
        head = AutoUncertaintyHead.from_pretrained(args.uhead, base_model=self.llm)
        self.features = VLLMUncertaintyHeadFeatures(head, model_path=args.model)
        self.apply_head = CalculatorApplyUQHead(head, device=args.uhead_device)
        self.estimator = LuhClaimEstimatorDummy()
        self.layer_ids = self.features.vllm_with_uncertainty_arguments()["hs_layer_ids"]
        self.engine = self.llm.llm_engine.engine_core
        self.engine.collective_rpc("_setup_hidden_states_capture", args=(self.layer_ids,))

    def score(self, p_ids: list[int], c_ids: list[int]) -> tuple[float, float]:
        full_ids = p_ids + c_ids
        if len(full_ids) >= self.llm.llm_engine.model_config.max_model_len:
            raise ValueError(
                f"sequence has {len(full_ids)} tokens, max_model_len is "
                f"{self.llm.llm_engine.model_config.max_model_len}"
            )
        self.engine.collective_rpc("_reset_capture")
        outputs = self.llm.generate(
            [self.TokensPrompt(prompt_token_ids=full_ids)],
            sampling_params=self.capture_params,
        )
        per_rank = self.engine.collective_rpc("_get_captured_states")
        self.engine.collective_rpc("_get_capture_metadata")
        captured = next((rank for rank in per_rank if rank), {})
        req_id = outputs[0].request_id
        layers = []
        for layer_id in self.layer_ids:
            by_request = captured.get(layer_id, {})
            payload = by_request.get(req_id)
            if payload is None and len(by_request) == 1:
                payload = next(iter(by_request.values()))
            if payload is None:
                raise RuntimeError(f"missing hidden states for layer={layer_id}, request={req_id}")
            tensor = self.torch.from_numpy(pickle.loads(payload))
            if tensor.shape[0] < len(full_ids):
                raise RuntimeError(
                    f"captured {tensor.shape[0]} of {len(full_ids)} tokens at layer {layer_id}"
                )
            layers.append(tensor[: len(full_ids)])

        deps: dict[str, Any] = {
            "vllm_hidden_states_output": {"hidden_states": layers},
            "token_ids": c_ids,
            "context_lengths": [len(p_ids)],
        }
        deps = self.features(deps, texts=[""], model=self.llm, max_new_tokens=len(c_ids))
        deps["claims"] = [[self.Claim(None, None, list(range(len(c_ids))))]]
        deps = self.apply_head(deps, texts=[""], model=self.llm, max_new_tokens=len(c_ids))
        uncertainty = float(self.np.asarray(self.estimator(deps)).reshape(-1)[0])
        return uncertainty, 1.0 / (1.0 + uncertainty)

    def close(self) -> None:
        self.engine.collective_rpc("_setup_hidden_states_capture", args=([],))
        self.engine.collective_rpc("_reset_capture")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--generator", default="gpt_oss_20b_local", choices=sorted(MODELS))
    p.add_argument("--model", default=None)
    p.add_argument("--uhead", default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=65536)
    p.add_argument("--uhead-device", default="cuda:0")
    p.add_argument("--harmony-start-date", default=None)
    p.add_argument("--platform", default="leetcode")
    p.add_argument("--lcb-version", default="all")
    p.add_argument("--private-test-cap", type=int, default=12)
    p.add_argument("--plus-input-cap", type=int, default=200)
    p.add_argument("--swe-harness-workers", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.model = args.model or MODELS[args.generator]
    if args.uhead is None:
        if not args.generator.startswith("gpt_oss_"):
            raise SystemExit(
                "--uhead is required for Qwen; use a Qwen2.5-Coder-32B-compatible head"
            )
        args.uhead = DEFAULT_GPT_UHEAD
    if args.generator.startswith("gpt_oss_"):
        prepare_harmony_vocab(args.model)
        args.harmony_start_date = args.harmony_start_date or infer_harmony_start_date(args.run_root)
    stem = f"{args.benchmark}__{args.generator}"
    args.output = args.output or args.run_root / f"{stem}.uhead.jsonl"

    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        count = 0
        for _ in iter_inputs(args, tokenizer):
            count += 1
            if args.limit > 0 and count >= args.limit:
                break
        print(f"validated {count} generations; no model was loaded")
        return

    scorer = UHeadScorer(args)
    done = (
        {
            (str(row.get("instance_id")), int(row.get("generation_index", 0)))
            for row in read_jsonl(args.output)
            if args.resume and row.get("uhead_confidence") is not None
        }
        if args.resume and args.output.exists()
        else set()
    )
    if not args.resume:
        args.output.unlink(missing_ok=True)
    try:
        for i, row in enumerate(iter_inputs(args, scorer.llm.get_tokenizer()), 1):
            if args.limit > 0 and i > args.limit:
                break
            key = (row["instance_id"], row["generation_index"])
            if key in done:
                continue
            out = {k: v for k, v in row.items() if not k.endswith("_ids")}
            try:
                uncertainty, confidence = scorer.score(row["prompt_ids"], row["completion_ids"])
                out.update(
                    {
                        "uhead_uncertainty": uncertainty,
                        "uhead_confidence": confidence,
                        "uhead_model": args.uhead,
                        "uhead_scope": "full_generation",
                        "prompt_tokens": len(row["prompt_ids"]),
                        "completion_tokens": len(row["completion_ids"]),
                        "error": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                out["error"] = f"{type(exc).__name__}: {exc}"
                if args.strict:
                    raise
            append_jsonl(args.output, out)
            print(
                f"[{i}] {row['instance_id']} g{row['generation_index']} "
                f"confidence={out.get('uhead_confidence')} error={out.get('error')}",
                flush=True,
            )
    finally:
        scorer.close()
    print(args.output)


if __name__ == "__main__":
    main()
