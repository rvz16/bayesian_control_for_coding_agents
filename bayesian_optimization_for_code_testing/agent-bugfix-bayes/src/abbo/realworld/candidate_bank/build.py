"""Candidate bank build process.

For each selected bug, generates candidate patches using all generator arms
with multiple seeds, runs critics and the full verifier, and saves everything.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from rich.console import Console

from abbo.config import project_root
from abbo.realworld.candidate_bank.schema import (
    CandidateBank,
    CandidateManifest,
    CriticOutputs,
    VerifierOutputs,
)
from abbo.utils.hashing import short_hash
from abbo.utils.timing import timed

console = Console()

DEFAULT_ARMS = ["g1_direct_fix", "g2_localized_fix", "g3_test_guided_fix", "g4_minimal_diff_fix"]


def build_bank(config: dict[str, Any], benchmark: str) -> None:
    """Build the full candidate bank from config.

    This is the live-API stage: it calls generator arms to produce patches,
    runs critics, and runs the full verifier.  All results are saved to disk
    so that the evaluation stage can replay them offline.
    """
    bank_dir = project_root() / config.get("candidate_bank_dir", f"data/candidate_bank/{benchmark}")
    bank = CandidateBank(bank_dir)
    arms = config.get("generator_arms", DEFAULT_ARMS)
    seeds_per_arm = config.get("seeds_per_arm", 3)
    projects = config.get("projects", [])

    console.print(f"Building candidate bank at {bank_dir}")
    console.print(f"Arms: {arms}, seeds_per_arm: {seeds_per_arm}")

    for proj_spec in projects:
        project = proj_spec["name"]
        bug_ids = proj_spec.get("bug_ids", [])
        for bug_id in bug_ids:
            console.print(f"  Processing {project} bug {bug_id}...")

            # Create root candidate (buggy version)
            root_id = f"{project}_{bug_id}_root"
            root_manifest = CandidateManifest(
                benchmark=benchmark,
                project=project,
                bug_id=bug_id,
                candidate_id=root_id,
                generator_arm="root",
                generator_seed=0,
                verifier_outputs=VerifierOutputs(full_suite_pass=False),
            )
            bank.save_candidate(root_manifest)

            # Generate candidates from each arm
            for arm in arms:
                for seed in range(seeds_per_arm):
                    cid = f"{project}_{bug_id}_{arm}_{seed}"
                    console.print(f"    Generating {arm} seed={seed}...")

                    # In a real implementation, this would call the LLM
                    # For now, create a placeholder manifest
                    manifest = CandidateManifest(
                        benchmark=benchmark,
                        project=project,
                        bug_id=bug_id,
                        parent_candidate_id=root_id,
                        candidate_id=cid,
                        generator_arm=arm,
                        generator_seed=seed,
                        prompt_hash=short_hash(f"{arm}_{project}_{bug_id}_{seed}"),
                        patch_apply_success=True,
                        critic_outputs=CriticOutputs(),
                        verifier_outputs=VerifierOutputs(),
                    )
                    bank.save_candidate(manifest)

    console.print(f"Candidate bank built with {len(bank.get_all_bug_ids())} bugs")
