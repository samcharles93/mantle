#!/usr/bin/env python3
"""RLVR loop: external LLM proposes code changes, we verify them.

This is the piece you actually wanted.

It runs a tight loop:
  1) Load current kernel code (and nearby tests) as context.
  2) Ask an OpenAI‑compatible model to propose a unified diff patch.
  3) Apply patch in a throwaway working copy.
  4) Verify: gofmt, go test, benchmarks.
  5) If correct and faster, promote the change into the main working tree.

This script is intentionally dependency‑free (stdlib only). It talks to the
OpenAI REST API directly.

Config
------

API key and base URL can be set via flags or env vars:
  - OPENAI_API_KEY
  - OPENAI_BASE_URL  (default: https://api.openai.com)

Endpoints
---------

The OpenAI platform supports both:
  - POST /v1/chat/completions

Examples
--------

Baseline + one iteration against GEMM:

  python scripts/rlvr_loop.py iterate \
    --target internal/tensor/gemm.go \
    --bench ./internal/tensor:BenchmarkGemmPar \
    --model gpt-5.2

Notes
-----

* This does not attempt to be a full "agent framework". It is a ruthless
  verifier loop. The model proposes; the harness accepts/rejects.
* Benchmarks are noisy. Default uses median over multiple runs and requires
  a minimum improvement.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def die(msg: str, code: int = 2) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "go.mod").exists():
            return parent
    die("could not find go.mod (run from within the repo)")


def run(cmd: List[str], cwd: Path, timeout: int = 900, env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout


@dataclasses.dataclass
class BenchResult:
    name: str
    ns_per_op: float
    allocs_per_op: Optional[float] = None
    bytes_per_op: Optional[float] = None


BENCH_LINE_RE = re.compile(
    r"^(Benchmark\S+)\s+\d+\s+(?P<ns>[0-9.]+)\s+ns/op(?:\s+(?P<b>[0-9.]+)\s+B/op)?(?:\s+(?P<a>[0-9.]+)\s+allocs/op)?\s*$"
)


def parse_go_bench(output: str) -> List[BenchResult]:
    out: List[BenchResult] = []
    for line in output.splitlines():
        line = line.strip()
        m = BENCH_LINE_RE.match(line)
        if not m:
            continue
        out.append(
            BenchResult(
                name=m.group(1),
                ns_per_op=float(m.group("ns")),
                bytes_per_op=float(m.group("b")) if m.group("b") else None,
                allocs_per_op=float(m.group("a")) if m.group("a") else None,
            )
        )
    return out


def median(vals: List[float]) -> float:
    s = sorted(vals)
    if not s:
        die("median() on empty list")
    mid = len(s) // 2
    if len(s) % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def read_text(path: Path, limit_bytes: int = 80_000) -> str:
    data = path.read_text(encoding="utf-8")
    if len(data.encode("utf-8")) > limit_bytes:
        # Keep the tail too; hot code often near the bottom.
        head = data[: limit_bytes // 2]
        tail = data[-limit_bytes // 2 :]
        return head + "\n\n/* ... truncated ... */\n\n" + tail
    return data


def collect_context(root: Path, target: Path) -> Dict[str, str]:
    ctx: Dict[str, str] = {}
    plan = root / "PLAN.md"
    if plan.exists():
        ctx["PLAN.md"] = read_text(plan, limit_bytes=60_000)

    # Always include the target.
    ctx[str(target)] = read_text(root / target)

    # Heuristic: include *_test.go in same dir.
    for test in (root / target.parent).glob("*_test.go"):
        ctx[str(test.relative_to(root))] = read_text(test)

    # Include sibling file if it looks relevant (same package, likely helpers).
    sibs = list((root / target.parent).glob("*.go"))
    for sib in sibs:
        rel = sib.relative_to(root)
        if rel == target:
            continue
        # Don't explode context. Keep max 2 extra.
        if len(ctx) >= 1 + 1 + 3:  # plan + target + up to 2 tests/extra
            break
        if sib.name.endswith("_test.go"):
            continue
        ctx[str(rel)] = read_text(sib)

    return ctx


def build_prompt(context: Dict[str, str], target: str, bench_spec: str) -> str:
    parts: List[str] = []
    parts.append(
        "You are improving a pure-Go inference kernel. You MUST propose a patch that compiles, passes tests, and is faster.\n"
        "Constraints:\n"
        "- Pure Go only (no CGO).\n"
        "- No stubs, no TODOs, no unimplemented functions.\n"
        "- Keep API surface stable unless tests/bench require adjustment.\n"
        "- Prefer zero allocations in hot paths.\n"
        "- Keep changes small and local.\n"
        "- Output MUST be a unified diff patch ONLY (no commentary), using paths relative to repo root.\n"
        "\n"
        f"Target file: {target}\n"
        f"Benchmark gate: {bench_spec} (must be faster)\n"
    )
    parts.append("\n--- CONTEXT START ---\n")
    for name, text in context.items():
        parts.append(f"\n### FILE: {name}\n")
        parts.append(text)
    parts.append("\n--- CONTEXT END ---\n")
    parts.append(
        "\nNow produce a unified diff patch. If you need to adjust tests/benchmarks, include them in the diff."
    )
    return "\n".join(parts)


def http_post_json(url: str, api_key: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        die(f"HTTP {e.code} calling {url}: {body}")
    except urllib.error.URLError as e:
        die(f"network error calling {url}: {e}")


def call_model(base_url: str, api_key: str, model: str, endpoint: str, prompt: str, temperature: float, max_tokens: int) -> str:
    base = base_url.rstrip("/")
    if endpoint == "chat":
        url = f"{base}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a senior performance engineer. Output only unified diffs."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        resp = http_post_json(url, api_key, payload)
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception:
            die(f"unexpected response shape from chat completions: {json.dumps(resp)[:1000]}")

def extract_unified_diff(text: str) -> str:
    # If the model wrapped diff in code fences, unwrap.
    m = re.search(r"```diff\n(.*?)```", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    # Basic sanity.
    if "*** Begin Patch" in text and "*** End Patch" in text:
        die("model returned apply_patch format; require unified diff (git-style)")
    if "diff --git" not in text:
        die("model did not return a unified diff (missing 'diff --git')")
    return text.strip() + "\n"

def ensure_git_repo(path: Path) -> None:
    if (path / ".git").exists():
        return
    rc, out = run(["git", "init"], cwd=path)
    if rc != 0:
        die(f"git init failed: {out}")
    # Make git apply happy with clean index.
    run(["git", "add", "-A"], cwd=path)
    run(["git", "commit", "-m", "init", "--allow-empty"], cwd=path)

def apply_patch_git(workdir: Path, patch_text: str) -> List[str]:
    ensure_git_repo(workdir)
    patch_file = workdir / "_candidate.patch"
    patch_file.write_text(patch_text, encoding="utf-8")
    rc, out = run(["git", "apply", "--check", str(patch_file)], cwd=workdir)
    if rc != 0:
        die(f"patch does not apply cleanly:\n{out}")
    rc, out = run(["git", "apply", str(patch_file)], cwd=workdir)
    if rc != 0:
        die(f"patch apply failed:\n{out}")

    # List changed files.
    rc, out = run(["git", "diff", "--name-only"], cwd=workdir)
    if rc != 0:
        die(f"git diff failed: {out}")
    changed = [line.strip() for line in out.splitlines() if line.strip()]
    return changed

def gofmt_files(workdir: Path, files: List[str]) -> None:
    go_files = [f for f in files if f.endswith(".go")]
    if not go_files:
        return
    rc, out = run(["gofmt", "-w", *go_files], cwd=workdir)
    if rc != 0:
        die(f"gofmt failed:\n{out}")

def go_test_all(workdir: Path) -> None:
    rc, out = run(["go", "test", "./..."], cwd=workdir)
    if rc != 0:
        die(f"go test failed:\n{out}")

def run_bench(workdir: Path, bench_spec: str, count: int) -> Dict[str, float]:
    """Run benchmark spec in the form "./path:BenchmarkName".

    Returns mapping benchmarkName->median(ns/op) across `count` invocations.
    """
    if ":" not in bench_spec:
        die("--bench must be like ./internal/tensor:BenchmarkGemmPar")
    pkg, bench = bench_spec.split(":", 1)

    samples: Dict[str, List[float]] = {}
    for i in range(count):
        rc, out = run(
            [
                "go",
                "test",
                pkg,
                "-run",
                "^$",
                "-bench",
                f"^{bench}$",
                "-benchmem",
                "-count",
                "1",
            ],
            cwd=workdir,
        )
        if rc != 0:
            die(f"go test -bench failed:\n{out}")
        res = parse_go_bench(out)
        if not res:
            die(f"no benchmark lines parsed. output was:\n{out}")
        for r in res:
            samples.setdefault(r.name, []).append(r.ns_per_op)
        # Small pause to reduce thermal / scheduler weirdness.
        time.sleep(0.05)

    return {name: median(vs) for name, vs in samples.items()}


def copy_repo(src: Path, dst: Path) -> None:
    ignore = shutil.ignore_patterns(
        ".git",
        "work",
        "*.zip",
        "_candidate.patch",
        "__pycache__",
        "*.pyc",
    )
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)


@dataclasses.dataclass
class Baseline:
    bench_ns: Dict[str, float]


def load_baseline(path: Path) -> Optional[Baseline]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Baseline(bench_ns={k: float(v) for k, v in data.get("bench_ns", {}).items()})


def save_baseline(path: Path, baseline: Baseline) -> None:
    path.write_text(json.dumps(dataclasses.asdict(baseline), indent=2) + "\n", encoding="utf-8")


def improved(candidate: Dict[str, float], baseline: Dict[str, float], min_improve: float) -> Tuple[bool, str]:
    # Require all benchmarks present and improved by threshold.
    msgs = []
    ok = True
    for name, base_ns in baseline.items():
        cand_ns = candidate.get(name)
        if cand_ns is None:
            ok = False
            msgs.append(f"missing benchmark {name}")
            continue
        delta = (base_ns - cand_ns) / base_ns
        msgs.append(f"{name}: {base_ns:.1f} -> {cand_ns:.1f} ns/op ({delta*100:.2f}%)")
        if delta < min_improve:
            ok = False
    return ok, "\n".join(msgs)


def promote_changes(src_root: Path, cand_root: Path, changed_files: List[str]) -> None:
    for rel in changed_files:
        src = cand_root / rel
        dst = src_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def cmd_iterate(args: argparse.Namespace) -> None:
    root = repo_root(Path.cwd())
    work = root / "work"
    work.mkdir(exist_ok=True)
    baseline_path = work / "baseline.json"

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        die("missing API key (set OPENAI_API_KEY or pass --api-key)")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com"

    target = Path(args.target)
    if not (root / target).exists():
        die(f"target not found: {target}")

    # Compute or load baseline.
    base = load_baseline(baseline_path)
    if base is None or args.rebaseline:
        print("== computing baseline ==")
        go_test_all(root)
        bench_ns = run_bench(root, args.bench, args.bench_runs)
        base = Baseline(bench_ns=bench_ns)
        save_baseline(baseline_path, base)
        for k, v in bench_ns.items():
            print(f"baseline {k}: {v:.1f} ns/op")

    # Iteration loop.
    wins_dir = work / "wins"
    wins_dir.mkdir(exist_ok=True)

    for it in range(1, args.iterations + 1):
        print(f"\n== iteration {it}/{args.iterations} ==")
        ctx = collect_context(root, target)
        prompt = build_prompt(ctx, str(target), args.bench)

        print("calling model...")
        raw = call_model(
            base_url=base_url,
            api_key=api_key,
            model=args.model,
            endpoint=args.api_endpoint,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        patch = extract_unified_diff(raw)

        # Create candidate copy.
        with tempfile.TemporaryDirectory(prefix="rlvr_cand_") as td:
            cand_root = Path(td)
            copy_repo(root, cand_root)

            # Apply patch.
            changed = apply_patch_git(cand_root, patch)
            gofmt_files(cand_root, changed)

            # Verify.
            try:
                go_test_all(cand_root)
                cand_bench = run_bench(cand_root, args.bench, args.bench_runs)
            except SystemExit:
                print("candidate rejected (tests/bench failed)")
                if args.keep_rejects:
                    rej = work / f"reject_{int(time.time())}"
                    shutil.copytree(cand_root, rej, dirs_exist_ok=True)
                continue

            ok, msg = improved(cand_bench, base.bench_ns, args.min_improvement)
            print(msg)
            if not ok:
                print("candidate rejected (not faster)")
                if args.keep_rejects:
                    rej = work / f"reject_{int(time.time())}"
                    shutil.copytree(cand_root, rej, dirs_exist_ok=True)
                continue

            print("candidate ACCEPTED")
            stamp = time.strftime("%Y%m%d_%H%M%S")
            (wins_dir / f"{stamp}.patch").write_text(patch, encoding="utf-8")
            promote_changes(root, cand_root, changed)

            # Update baseline to the new best.
            base = Baseline(bench_ns=cand_bench)
            save_baseline(baseline_path, base)

            if args.stop_on_win:
                print("stopping on first win")
                return


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rlvr_loop")
    sub = p.add_subparsers(dest="cmd", required=True)

    it = sub.add_parser("iterate", help="Run RLVR propose->verify loop")
    it.add_argument("--target", required=True, help="Kernel file to optimise (relative to repo root)")
    it.add_argument("--bench", required=True, help="Benchmark gate: ./pkg:path (e.g. ./internal/tensor:BenchmarkGemmPar)")
    it.add_argument("--bench-runs", type=int, default=9, help="Benchmark repetitions; median is used")
    it.add_argument("--min-improvement", type=float, default=0.02, help="Minimum fractional improvement (0.02 = 2%%)")
    it.add_argument("--iterations", type=int, default=10, help="How many model proposals to attempt")
    it.add_argument("--stop-on-win", action="store_true", help="Stop after first accepted improvement")
    it.add_argument("--keep-rejects", action="store_true", help="Keep failed candidates under work/")
    it.add_argument("--rebaseline", action="store_true", help="Recompute baseline before iterating")

    it.add_argument("--model", default="gpt-5.2", help="OpenAI model name")
    it.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    it.add_argument("--base-url", default=None, help="API base URL (or set OPENAI_BASE_URL)")
    it.add_argument("--temperature", type=float, default=0.2, help="Model temperature")
    it.add_argument("--max-tokens", type=int, default=900, help="Max completion tokens")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.cmd == "iterate":
        cmd_iterate(args)
        return
    die("unknown command")


if __name__ == "__main__":
    main()
