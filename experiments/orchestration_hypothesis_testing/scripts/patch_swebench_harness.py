#!/usr/bin/env python3
"""Patch the installed SWE-Bench harness for rootless podman 3.4.4 compatibility.

Why this is needed
==================

The standard SWE-Bench harness (`swebench` PyPI package, tested against v4.1.0)
talks to the Docker daemon via the docker-py SDK. On a host where:

  * Docker is restricted (no docker group membership) so we use rootless podman
    via its docker-compat socket (DOCKER_HOST=unix:///run/user/$UID/podman/podman.sock),
  * podman is v3.4.4 (Ubuntu 22.04 default),

two failure modes appear:

  1. `client.api.build()` against podman 3.4.4 ignores the pull=False kwarg and
     tries to pull the unqualified base image `sweb.base.py.x86_64:latest` from
     docker.io. docker.io has no such image (because SWE-Bench expects the base
     image to live in local podman storage as `localhost/sweb.base.py.x86_64`),
     so the build fails immediately. This breaks every env-image build, which
     in turn breaks every instance build with `--namespace none`.

  2. `client.containers.create(platform=...)` fails with
     `docker.errors.InvalidVersion: platform is not supported for API version
     < 1.41`. podman 3.4.4 reports API v1.40.

This script patches the installed `swebench/harness/docker_build.py` to:

  1. Replace `client.api.build(...)` with a `subprocess.run(["podman", "build",
     "--pull=false", ...])` call. The podman CLI's `--pull=false` is honored
     (unlike the SDK kwarg), so local images are used as the FROM base.

  2. Remove the `platform=test_spec.platform` kwarg from the `containers.create`
     call inside `build_container`. The platform value is still passed through
     to `build_image` (which builds with `--platform=linux/x86_64` via the
     Dockerfile FROM directive), so the container inherits the right arch.

Idempotent — running twice is a no-op.

The complementary script-level change (adding `--namespace none` to the
harness command in `spot_check_generators.run_swebench_eval()`) is in this
repo already.

Usage
-----

    python experiments/orchestration_hypothesis_testing/scripts/patch_swebench_harness.py

Also requires (set in your pipeline shell script before invoking the harness):

    mkdir -p $HOME/buildah-tmp   # or somewhere outside the quota'd partition
    export TMPDIR=$HOME/buildah-tmp
    export BUILDAH_TMPDIR=$HOME/buildah-tmp

Without TMPDIR redirect, buildah layer commits scribble into /var/tmp and may
blow through a tight root-partition disk quota mid-build.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import sys
from pathlib import Path


PATCH_MARKER_BUILD = "# PATCH(podman-3.4.4): use podman CLI with --pull=false"
PATCH_MARKER_PLATFORM = "# PATCH(podman-3.4.4): platform kwarg disabled"
PATCH_MARKER_TAR = "# PATCH(rootless-podman): TAR_OPTIONS skips chown failures"
PATCH_MARKER_PIP = "# PATCH(rootless-podman): conditional pip<23.1 in testbed env (v2)"
# Old marker from the v1 sed-rewrite implementation of patch 4.
# Detected so re-applying the patcher on top of a v1-patched python.py
# strips the v1 block before adding the v2 (current) block.
PATCH_MARKER_PIP_V1 = "# PATCH(rootless-podman): pip compat for setup_repo.sh"


def locate_docker_build() -> Path:
    """Find docker_build.py inside the installed swebench package."""
    spec = importlib.util.find_spec("swebench")
    if spec is None or spec.origin is None:
        raise SystemExit(
            "swebench not importable in this Python. Activate the env that runs "
            "the harness, then re-run."
        )
    root = Path(spec.origin).parent
    candidate = root / "harness" / "docker_build.py"
    if not candidate.exists():
        raise SystemExit(f"docker_build.py not found at {candidate}")
    return candidate


def locate_dockerfile_python() -> Path:
    """Find dockerfiles/python.py inside the installed swebench package."""
    spec = importlib.util.find_spec("swebench")
    if spec is None or spec.origin is None:
        raise SystemExit(
            "swebench not importable in this Python. Activate the env that runs "
            "the harness, then re-run."
        )
    root = Path(spec.origin).parent
    candidate = root / "harness" / "dockerfiles" / "python.py"
    if not candidate.exists():
        raise SystemExit(f"dockerfiles/python.py not found at {candidate}")
    return candidate


def patch_build_call(src: str) -> tuple[str, bool]:
    """Replace `response = client.api.build(...)` + its streaming for-loop with
    a `subprocess.run(["podman", "build", "--pull=false", ...])` block.

    Returns (new_src, changed_bool).
    """
    if PATCH_MARKER_BUILD in src:
        return src, False
    # Fallback: detect patched state by the patched output's signature (older
    # patcher runs applied the rewrite without leaving a marker comment).
    if "podman_cmd" in src and "subprocess.run(podman_cmd" in src:
        return src, False

    lines = src.splitlines(keepends=True)

    # Locate `response = client.api.build(`
    i_start = next(
        (i for i, ln in enumerate(lines) if "response = client.api.build(" in ln),
        None,
    )
    if i_start is None:
        raise SystemExit(
            "Could not locate 'response = client.api.build(' line. The swebench "
            "version may have changed — inspect docker_build.py manually."
        )
    indent = " " * (len(lines[i_start]) - len(lines[i_start].lstrip()))

    # Find matching close paren (bare `)` at same indent)
    i_end = None
    for j in range(i_start + 1, len(lines)):
        if lines[j].strip() == ")":
            i_end = j
            break
    if i_end is None:
        raise SystemExit("Could not find closing ')' for client.api.build call")

    # Find `for chunk in response:` and the end of its block
    i_for = next(
        (j for j in range(i_end + 1, len(lines)) if "for chunk in response:" in lines[j]),
        None,
    )
    if i_for is None:
        raise SystemExit("Could not find 'for chunk in response:' loop")

    # End of for-loop = first non-empty line that dedents below the loop body
    inner_indent = None
    i_for_end = None
    for j in range(i_for + 1, len(lines)):
        s = lines[j]
        if not s.strip():
            continue
        cur = len(s) - len(s.lstrip())
        if inner_indent is None:
            inner_indent = cur
            continue
        if cur < inner_indent:
            i_for_end = j
            break
    if i_for_end is None:
        i_for_end = len(lines)

    replacement = (
        f"{indent}{PATCH_MARKER_BUILD} so unqualified FROM image names\n"
        f"{indent}# (e.g. 'sweb.base.py.x86_64:latest') resolve against the local\n"
        f"{indent}# image store instead of docker.io. The Docker SDK's\n"
        f"{indent}# client.api.build() ignores pull=False on podman 3.4.4.\n"
        f"{indent}import subprocess\n"
        f"{indent}podman_cmd = [\n"
        f'{indent}    "podman", "build",\n'
        f'{indent}    "--pull=false",\n'
        f'{indent}    "-t", image_name,\n'
        f'{indent}    "-f", str(Path(build_dir) / "Dockerfile"),\n'
        f"{indent}    str(build_dir),\n"
        f"{indent}]\n"
        f"{indent}if nocache:\n"
        f'{indent}    podman_cmd.insert(2, "--no-cache")\n'
        f"{indent}logger.info(f\"build cmd: {{' '.join(podman_cmd)}}\")\n"
        f"{indent}proc = subprocess.run(podman_cmd, capture_output=True, text=True)\n"
        f"{indent}buildlog = proc.stdout + proc.stderr\n"
        f"{indent}for line in buildlog.splitlines():\n"
        f"{indent}    logger.info(ansi_escape(line))\n"
        f"{indent}if proc.returncode != 0:\n"
        f'{indent}    raise docker.errors.BuildError(reason=f"podman build exited {{proc.returncode}}", build_log=buildlog)\n'
    )

    new_lines = lines[:i_start] + [replacement] + lines[i_for_end:]
    new_src = "".join(new_lines)

    # Ensure `from pathlib import Path` is imported
    if "from pathlib import Path" not in new_src:
        new_src = new_src.replace(
            "import os\n", "import os\nfrom pathlib import Path\n", 1
        )

    return new_src, True


def patch_platform_kwarg(src: str) -> tuple[str, bool]:
    """Comment out `platform=test_spec.platform,` inside the build_container()
    call to containers.create(). Leave the SAME kwarg inside build_image()
    calls untouched (it's a positional arg there).

    Uses a paren-depth scanner rather than a regex because the call body
    contains nested parens (e.g. `name=test_spec.get_instance_container_name(run_id)`).
    """
    if PATCH_MARKER_PLATFORM in src:
        return src, False
    # Fallback: detect patched state by the commented kwarg (older patcher
    # runs applied the rewrite without leaving a marker comment).
    if "# platform=test_spec.platform," in src:
        return src, False

    open_idx = src.find("client.containers.create(")
    if open_idx == -1:
        raise SystemExit(
            "Could not locate 'client.containers.create(' in docker_build.py. "
            "The swebench version may have changed — inspect it manually."
        )
    paren_idx = src.index("(", open_idx)
    depth = 1
    i = paren_idx + 1
    while i < len(src) and depth > 0:
        ch = src[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        i += 1
    if depth != 0:
        raise SystemExit(
            "Could not find matching ')' for client.containers.create(. "
            "The swebench version may have changed — inspect it manually."
        )
    close_idx = i  # one past the matching ')'

    call_span = src[open_idx:close_idx]
    needle = "platform=test_spec.platform,"
    rel = call_span.find(needle)
    if rel == -1:
        raise SystemExit(
            "Could not find 'platform=test_spec.platform,' inside the matched "
            "containers.create(...) call. The swebench version may have changed."
        )

    line_start = src.rfind("\n", 0, open_idx + rel) + 1
    indent = src[line_start:open_idx + rel]
    if indent.strip():
        raise SystemExit(
            "Unexpected formatting: 'platform=...' is not at the start of its "
            "own indent block — inspect docker_build.py manually."
        )

    replacement = (
        f"{indent}{PATCH_MARKER_PLATFORM}\n"
        f"{indent}# platform=test_spec.platform,"
    )
    line_end = src.index("\n", open_idx + rel)
    new_src = src[:line_start] + replacement + src[line_end:]
    return new_src, True


def patch_dockerfile_tar_options(src: str) -> tuple[str, bool]:
    """Inject `ENV TAR_OPTIONS="--no-same-owner"` into the python instance
    Dockerfile template (_DOCKERFILE_INSTANCE_PY in dockerfiles/python.py).

    Without this, setup_repo.sh's `tar -xvzf` calls (e.g. matplotlib's qhull
    extract) fail because the tarball preserves UIDs/GIDs that fall outside
    rootless podman's id-mapping range, so chown returns EINVAL and tar exits
    non-zero, breaking the whole RUN step. `TAR_OPTIONS=--no-same-owner` tells
    GNU tar to skip the chown step entirely; files end up owned by root
    (the user inside the container), which is what setup scripts expect.
    """
    if PATCH_MARKER_TAR in src:
        return src, False

    anchor = "\nCOPY ./setup_repo.sh /root/\n"
    if anchor not in src:
        raise SystemExit(
            "Could not find 'COPY ./setup_repo.sh /root/' in dockerfiles/python.py — "
            "the swebench template may have changed; inspect it manually."
        )
    replacement = (
        f"\n{PATCH_MARKER_TAR}\n"
        f'ENV TAR_OPTIONS="--no-same-owner"\n'
        f"{anchor}"
    )
    return src.replace(anchor, replacement, 1), True


def patch_dockerfile_pip_compat(src: str) -> tuple[str, bool]:
    """Inject a pip-downgrade RUN step into the python instance Dockerfile
    template (_DOCKERFILE_INSTANCE_PY in dockerfiles/python.py).

    Modern pip (>=23.1) requires PEP 660 hooks for editable installs and
    rejected the legacy `--no-use-pep517` flag. Both patterns appear in
    SWE-Bench setup_repo.sh content:

      - pylint instances run `pip install -e .` with a build backend that
        has zero PEP 660 hooks. Even `--config-settings editable_mode=compat`
        is rejected because pip checks for `build_editable` before falling
        back. Old pip (<23.1) skips the check and uses the legacy egg-info
        editable path that pylint's backend does support.
      - scikit-learn instances run `pip install --no-use-pep517 -e .`. Old
        pip recognises the flag and succeeds.
      - django instances with an already-old pip in the env image are
        unaffected (the downgrade is a no-op when pip is already <23.1).

    The downgrade targets the testbed conda env created by setup_env.sh
    (already present at instance-image build time). Failure is silenced
    with `|| true` so env images without the expected testbed layout fall
    through to the original behaviour rather than breaking outright.

    This replaces a prior sed-rewrite approach that injected
    `--config-settings editable_mode=compat`. That sed both failed for
    pylint (pip rejects before compat fallback) AND regressed django-12470
    (its already-old pip didn't recognise `--config-settings`).
    """
    if PATCH_MARKER_PIP in src:
        return src, False

    # If v1 (sed-rewrite) is already applied, strip the v1 block first so
    # the v2 (this) block can replace it cleanly. The v1 block is the v1
    # marker + the next non-empty line (the `RUN sed -i ...` that was
    # written alongside the marker).
    if PATCH_MARKER_PIP_V1 in src:
        lines = src.splitlines(keepends=True)
        out_lines = []
        i = 0
        while i < len(lines):
            if lines[i].strip() == PATCH_MARKER_PIP_V1:
                # Skip the marker line and the following RUN line.
                # The v1 block is exactly two lines: marker + RUN-sed.
                if i + 1 < len(lines) and lines[i + 1].lstrip().startswith("RUN sed"):
                    i += 2
                    continue
                # Fall back: only skip the marker if next line is unrelated.
                i += 1
                continue
            out_lines.append(lines[i])
            i += 1
        src = "".join(out_lines)

    anchor = "RUN /bin/bash /root/setup_repo.sh\n"
    if anchor not in src:
        raise SystemExit(
            "Could not find 'RUN /bin/bash /root/setup_repo.sh' in dockerfiles/python.py "
            "— the swebench template may have changed; inspect it manually."
        )
    # The shell snippet downgrades pip ONLY if the current pip is >=23.1.
    # Old pip (21.x/22.x) is preserved (avoids accidentally upgrading legacy
    # envs to pip 23.0.1, which would still differ from their original).
    # The whole thing trails `|| true` so env images without the testbed
    # layout fall through to the original setup_repo.sh behaviour.
    py = "/opt/miniconda3/envs/testbed/bin/python"
    needs_downgrade = (
        f'{py} -c "import pip,sys; v=tuple(int(x) for x in pip.__version__.split(\\".\\")[:2]); '
        f'sys.exit(0 if v >= (23,1) else 1)" 2>/dev/null'
    )
    pip_install = (
        f'{py} -m pip install "pip<23.1" --quiet --disable-pip-version-check '
        f"> /dev/null 2>&1"
    )
    replacement = (
        f"{PATCH_MARKER_PIP}\n"
        f"RUN {needs_downgrade} && {pip_install} || true\n"
        f"{anchor}"
    )
    return src.replace(anchor, replacement, 1), True


def _apply_and_write(target: Path, patches: list, dry_run: bool) -> int:
    """Apply a list of (name, fn) patches to target, syntax-check, write."""
    src = target.read_text()
    original = src
    n_changes = 0
    for name, fn in patches:
        src, changed = fn(src)
        if changed:
            n_changes += 1
            print(f"  applied: {name}")
        else:
            print(f"  skipped: {name} (already present)")
    if n_changes == 0:
        return 0
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"ERROR: patched {target.name} has SyntaxError: {e}", file=sys.stderr)
        return 1
    if dry_run:
        print(f"--dry-run: would write {len(src)} chars to {target}")
        return 0
    backup = target.with_suffix(".py.preharnesspatch")
    if not backup.exists():
        backup.write_text(original)
        print(f"backup: {backup}")
    target.write_text(src)
    print(f"wrote: {target}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing the file.",
    )
    args = parser.parse_args()

    targets = [
        (
            locate_docker_build(),
            [
                ("podman CLI build (1/4)", patch_build_call),
                ("platform kwarg disabled in containers.create (2/4)", patch_platform_kwarg),
            ],
        ),
        (
            locate_dockerfile_python(),
            [
                ("TAR_OPTIONS=--no-same-owner in instance Dockerfile (3/4)",
                 patch_dockerfile_tar_options),
                ("pip downgrade to <23.1 in testbed env (4/4)",
                 patch_dockerfile_pip_compat),
            ],
        ),
    ]

    any_change = False
    for target, patches in targets:
        print(f"target: {target}")
        before = target.read_text()
        rc = _apply_and_write(target, patches, args.dry_run)
        if rc != 0:
            return rc
        if target.read_text() != before or args.dry_run:
            any_change = True
    if not any_change:
        print("Already patched. Nothing to do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
