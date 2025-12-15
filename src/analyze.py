#!/usr/bin/env python3
import argparse
import ast
import dataclasses
import json
import os
import sys
import textwrap
import subprocess
import re
from typing import Optional, Tuple

from google import genai  # assume installed
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL_NAME = "gemini-2.5-flash"

TEST_FILE_TOO_LARGE_LINES = 2000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TestPairResult:
    mode: str  # "tvm_torch" or "torch_tvm"
    converted_file: str
    original_file: Optional[str]

    original_exists: bool
    num_tests_original: Optional[int]
    num_tests_converted: Optional[int]
    coverage_pct: Optional[float]

    pytest_original_ok: Optional[bool]
    pytest_converted_ok: Optional[bool]
    pytest_original_exit_code: Optional[int]
    pytest_converted_exit_code: Optional[int]

    # Filled only if Gemini thinks something is wrong / missing for this pair
    gemini_summary: Optional[str] = None


# ---------------------------------------------------------------------------
# Test counting
# ---------------------------------------------------------------------------

def find_test_functions(path: str) -> int:
    """
    Count pytest-style test functions: any FunctionDef/AsyncFunctionDef
    whose name starts with 'test_', anywhere in the file (top-level or
    as a method in a class).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
    except OSError:
        return 0

    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return 0

    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                count += 1
    return count


def count_total_original_tests(mode: str, original_root: str) -> int:
    """
    Count total test functions in the *entire* original test tree.

    This mirrors the earlier simple script:
      - For tvm_torch: repos/tvm/tests/python
      - For torch_tvm: repos/pytorch/test
    """
    if mode == "tvm_torch":
        tests_root = os.path.join(original_root, "tests", "python")
    else:  # torch_tvm
        tests_root = os.path.join(original_root, "test")

    total = 0
    for root, _, files in os.walk(tests_root):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            if not (fname.startswith("test_") or fname.endswith("_test.py")):
                continue
            path = os.path.join(root, fname)
            total += find_test_functions(path)
    return total


# ---------------------------------------------------------------------------
# Path mapping helpers
# ---------------------------------------------------------------------------

def map_tvm_torch(
    converted_root: str, converted_file: str, tvm_root: str
) -> str:
    """
    Map converted_tests/tvm_torch/... to repos/tvm/...
    """
    rel = os.path.relpath(converted_file, converted_root)
    return os.path.join(tvm_root, rel)


def map_torch_tvm(
    converted_root: str, converted_file: str, pytorch_root: str
) -> str:
    """
    Map converted_tests/torch_tvm/... to repos/pytorch/test/...
    """
    rel = os.path.relpath(converted_file, converted_root)
    return os.path.join(pytorch_root, rel)


# ---------------------------------------------------------------------------
# Pytest summary parsing and runners
# ---------------------------------------------------------------------------

_SUMMARY_RE = re.compile(r"=+\s*(.+?)\s*in\s*[\d\.]+s\s*=+")


def parse_pytest_summary(output: str) -> Tuple[int, int]:
    """
    Parse pytest's summary line.

    Returns (tests_run, tests_failed).
    """
    tests_run = 0
    tests_failed = 0

    for line in output.splitlines():
        m = _SUMMARY_RE.search(line)
        if not m:
            continue
        summary = m.group(1)
        parts = [p.strip() for p in summary.split(",")]

        for p in parts:
            tokens = p.split()
            if len(tokens) < 2:
                continue
            try:
                n = int(tokens[0])
            except ValueError:
                continue
            word = tokens[1].lower()

            if word in ("failed", "error", "errors"):
                tests_failed += n
                tests_run += n
            elif word in ("passed", "skipped", "xfailed", "xpassed"):
                tests_run += n

    return tests_run, tests_failed


def run_pytest(test_path: str) -> Tuple[bool, int]:
    """
    Run pytest on a single file; return (ok, exit_code).

    This is mainly for per-file PASS/FAIL reporting.
    """
    # Skip files that are too large to run pytest on
    with open(test_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if len(lines) > TEST_FILE_TOO_LARGE_LINES:
            print(
                f"[pytest] Skipping pytest on large file: {test_path} "
                f"(lines={len(lines)})"
            )
            return False, -1
    print(f"[pytest] Running: {test_path}")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ok = (result.returncode == 0)
    status = "PASS" if ok else "FAIL"
    print(f"[pytest] {status}: {test_path} (exit_code={result.returncode})")
    return ok, result.returncode


def run_pytest_batch(files: list[str]) -> Tuple[int, int, int, bool]:
    """
    Run pytest once on a list of files.

    Returns:
      (exit_code, tests_run, tests_failed, session_started)
    """
    if not files:
        return 0, 0, 0, False

    cmd = [sys.executable, "-m", "pytest"] + files
    print("\n[pytest-batch] Running batch pytest command:")
    print("   ", " ".join(cmd))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    out = stdout + "\n" + stderr

    tests_run, tests_failed = parse_pytest_summary(out)
    session_started = "test session starts" in out

    print(
        f"[pytest-batch] Exit code={result.returncode}, "
        f"tests_run={tests_run}, tests_failed={tests_failed}, "
        f"session_started={session_started}"
    )

    return result.returncode, tests_run, tests_failed, session_started


# ---------------------------------------------------------------------------
# Gemini models & helpers
# ---------------------------------------------------------------------------

class GeminiPairAnalysis(BaseModel):
    id: str = Field(
        description="Pair ID passed in, e.g. 'tvm_torch:converted_tests/...'."
    )
    issues: list[str] = Field(
        description="Short bullet items describing concrete problems with the converted tests."
    )
    suggestions: list[str] = Field(
        description="Short bullet items suggesting how to fix or improve the converted tests."
    )


class GeminiBatchResponse(BaseModel):
    items: list[GeminiPairAnalysis] = Field(
        description=(
            "Only include entries for pairs where the conversion looks suspicious "
            "(missing cases, changed behavior, bugs, weaker coverage, etc.). "
            "If a pair looks fine, omit it from this list."
        )
    )


def get_file_contents(path: str, max_chars: int = 8000) -> str:
    """Read a file, truncating to max_chars so prompts don't explode."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        return f"[ERROR reading {path}: {e}]"

    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n# [Truncated for length in prompt]"
    return content


def chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def analyze_pairs_with_gemini(mode: str, results: list[TestPairResult]) -> None:
    """
    Batch Gemini calls:

    - Group multiple (original, converted) pairs into one request.
    - Gemini returns JSON matching GeminiBatchResponse.
    - Only pairs with suspected issues are returned.
    """
    pairs: list[tuple[str, TestPairResult, str, str]] = []

    for r in results:
        if not r.original_exists or not r.original_file:
            continue

        pair_id = f"{mode}:{os.path.relpath(r.converted_file)}"
        original_code = get_file_contents(r.original_file)
        converted_code = get_file_contents(r.converted_file)

        pairs.append((pair_id, r, original_code, converted_code))

    if not pairs:
        return

    client = genai.Client()
    batch_size = 3  # small-ish chunks to keep prompts reasonable

    for chunk in chunked(pairs, batch_size):
        # Build the PAIR blocks
        blocks: list[str] = []
        for pair_id, r, orig_code, conv_code in chunk:
            block = textwrap.dedent(
                f"""
                PAIR:
                ID: {pair_id}
                ORIGINAL_PATH: {os.path.relpath(r.original_file)}
                CONVERTED_PATH: {os.path.relpath(r.converted_file)}

                ORIGINAL_CODE:
                ```python
                {orig_code}
                ```

                CONVERTED_CODE:
                ```python
                {conv_code}
                ```
                """
            ).strip()
            blocks.append(block)

        pairs_text = "\n\n" + ("\n\n".join(blocks)) + "\n\n"

        base_instructions = textwrap.dedent(
            """
            You are reviewing converted unit tests between two ML compiler projects.

            You will be given several PAIR blocks. Each PAIR has:
              - "ID"
              - "ORIGINAL_CODE"
              - "CONVERTED_CODE"

            For each PAIR:

            - Decide if the converted tests faithfully cover the same behaviors
              as the original tests (scenarios, parameters, assertions).
            - ONLY report pairs that look suspicious, i.e. where:
                * important cases or assertions from the original are missing,
                * behavior or shapes change in a non-trivial way,
                * there are obvious bugs in the converted tests,
                * or coverage is clearly weaker.
            - If a pair looks like a good, faithful conversion with comparable
              coverage, DO NOT include that pair in your output at all.

            Keep the output for each reported pair VERY SHORT:
              - At most 2 "issues" bullets.
              - At most 2 "suggestions" bullets.
              - Each bullet under 120 characters.
              - Focus on the core technical differences.

            Your response must conform to the provided JSON schema.
            """
        ).strip()

        prompt = base_instructions + "\n\n" + pairs_text

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": GeminiBatchResponse.model_json_schema(),
                },
            )
            json_text = response.text
            resp_obj = GeminiBatchResponse.model_validate_json(json_text)
        except Exception as e:
            err_msg = f"[Gemini error or invalid JSON: {e}]"
            for _, r, _, _ in chunk:
                if not r.gemini_summary:
                    r.gemini_summary = err_msg
            continue

        # Map id -> combined summary
        by_id: dict[str, str] = {}

        for item in resp_obj.items:
            issues = item.issues or []
            suggestions = item.suggestions or []

            pieces: list[str] = []
            if issues:
                pieces.append("Issues:")
                for s in issues:
                    pieces.append(f"- {s}")
            if suggestions:
                if pieces:
                    pieces.append("")
                pieces.append("Suggestions:")
                for s in suggestions:
                    pieces.append(f"- {s}")

            summary = "\n".join(pieces) if pieces else ""
            if summary:
                by_id[item.id] = summary

        # Fill per-pair gemini_summary only for reported/problematic pairs
        for pair_id, r, _, _ in chunk:
            s = by_id.get(pair_id)
            if s:
                r.gemini_summary = s
            # if not in by_id: Gemini thinks conversion looks fine -> leave None


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_mode(
    mode: str,
    converted_root: str,
    original_root: str,
    use_gemini: bool,
    run_pytests: bool,
) -> Tuple[list[TestPairResult], list[str], list[str], int]:
    """
    mode: "tvm_torch" or "torch_tvm"
    converted_root: converted_tests/tvm_torch or converted_tests/torch_tvm
    original_root: repos/tvm or repos/pytorch

    Returns:
      (results, original_files_for_batch, converted_files_for_batch,
       total_original_tests_overall)
    """
    results: list[TestPairResult] = []
    original_files_batch: list[str] = []
    converted_files_batch: list[str] = []

    for root, _, files in os.walk(converted_root):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            if not fname.startswith("test_"):
                continue

            converted_path = os.path.join(root, fname)

            if mode == "tvm_torch":
                original_path = map_tvm_torch(
                    converted_root, converted_path, original_root
                )
            else:  # torch_tvm
                original_path = map_torch_tvm(
                    converted_root, converted_path, original_root
                )

            original_exists = os.path.isfile(original_path)

            num_tests_original = None
            num_tests_converted = None
            coverage_pct = None

            pytest_original_ok = None
            pytest_converted_ok = None
            pytest_original_exit_code = None
            pytest_converted_exit_code = None

            if original_exists:
                original_files_batch.append(original_path)
                converted_files_batch.append(converted_path)

                num_tests_original = find_test_functions(original_path)
                num_tests_converted = find_test_functions(converted_path)

                if num_tests_original and num_tests_original > 0:
                    coverage_pct = (
                        100.0 * num_tests_converted / num_tests_original
                    )

                if run_pytests:
                    pytest_original_ok, pytest_original_exit_code = run_pytest(
                        original_path
                    )
                    pytest_converted_ok, pytest_converted_exit_code = run_pytest(
                        converted_path
                    )
            else:
                print(
                    f"[warn] Original test file not found for converted test:\n"
                    f"  converted: {converted_path}\n"
                    f"  expected original: {original_path}"
                )

            res = TestPairResult(
                mode=mode,
                converted_file=os.path.abspath(converted_path),
                original_file=os.path.abspath(original_path)
                if original_path
                else None,
                original_exists=original_exists,
                num_tests_original=num_tests_original,
                num_tests_converted=num_tests_converted,
                coverage_pct=coverage_pct,
                pytest_original_ok=pytest_original_ok,
                pytest_converted_ok=pytest_converted_ok,
                pytest_original_exit_code=pytest_original_exit_code,
                pytest_converted_exit_code=pytest_converted_exit_code,
                gemini_summary=None,
            )
            results.append(res)

    if use_gemini:
        analyze_pairs_with_gemini(mode, results)

    # NEW: total original tests over the entire original test tree
    total_original_tests_overall = count_total_original_tests(mode, original_root)

    return results, original_files_batch, converted_files_batch, total_original_tests_overall


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def relpath_or_none(path: Optional[str]) -> str:
    if not path:
        return "(none)"
    return os.path.relpath(path)


def summarize_results(
    mode: str,
    results: list[TestPairResult],
    total_original_tests_overall: int,
    batch_orig_exit: int,
    batch_orig_tests_run: int,
    batch_orig_tests_failed: int,
    batch_orig_session_started: bool,
    batch_conv_exit: int,
    batch_conv_tests_run: int,
    batch_conv_tests_failed: int,
    batch_conv_session_started: bool,
) -> None:
    print("\n" + "=" * 80)
    print(f"SUMMARY for mode = {mode}")
    print("=" * 80)

    total_converted_tests = 0
    total_with_original = 0

    orig_files_pass = 0
    orig_files_fail = 0
    conv_files_pass = 0
    conv_files_fail = 0

    for r in results:
        if r.original_exists:
            total_with_original += 1
        if r.num_tests_converted is not None:
            total_converted_tests += r.num_tests_converted

        if r.pytest_original_ok is True:
            orig_files_pass += 1
        elif r.pytest_original_ok is False:
            orig_files_fail += 1

        if r.pytest_converted_ok is True:
            conv_files_pass += 1
        elif r.pytest_converted_ok is False:
            conv_files_fail += 1

    print(f"Number of converted test files:       {len(results)}")
    print(f"Number with matching original file:   {total_with_original}")
    print(f"Total test functions (original):      {total_original_tests_overall}")
    print(f"Total test functions (converted):     {total_converted_tests}")

    if total_original_tests_overall > 0:
        overall_coverage = (
            100.0 * total_converted_tests / total_original_tests_overall
        )
        print(f"Overall coverage (by test functions): {overall_coverage:.2f}%")
    else:
        print("Overall coverage: N/A (no test functions found in originals)")

    print("\nPer-file coverage (only where original exists):")
    for r in results:
        if not r.original_exists:
            continue
        cov_str = (
            f"{r.coverage_pct:.1f}%"
            if r.coverage_pct is not None
            else "N/A"
        )
        print(
            f"- converted: {relpath_or_none(r.converted_file)}\n"
            f"  original:  {relpath_or_none(r.original_file)}\n"
            f"  tests original={r.num_tests_original}, "
            f"converted={r.num_tests_converted}, coverage={cov_str}"
        )

    print("\nPytest file summary (from per-file runs):")
    print(
        f"  Original files:  {orig_files_pass} passed, {orig_files_fail} failed"
    )
    print(
        f"  Converted files: {conv_files_pass} passed, {conv_files_fail} failed"
    )

    print("\nPytest batch execution summary (from one pytest run per side):")

    # Originals
    if not batch_orig_session_started:
        if batch_orig_exit != 0:
            print(
                f"  Original tests: batch pytest failed during configuration "
                f"(exit code {batch_orig_exit}); test counts unavailable"
            )
        else:
            print(
                "  Original tests: no session started "
                "(no files or all deselected)"
            )
    else:
        print(
            f"  Original tests:  {batch_orig_tests_run} run, "
            f"{batch_orig_tests_failed} failed"
        )

    # Converted
    if not batch_conv_session_started:
        if batch_conv_exit != 0:
            print(
                f"  Converted tests: batch pytest failed during configuration "
                f"(exit code {batch_conv_exit}); test counts unavailable"
            )
        else:
            print(
                "  Converted tests: no session started "
                "(no files or all deselected)"
            )
    else:
        print(
            f"  Converted tests: {batch_conv_tests_run} run, "
            f"{batch_conv_tests_failed} failed"
        )

    # Pretty Gemini analysis
    any_gemini = any(r.gemini_summary for r in results)
    if any_gemini:
        print("\nGemini analysis (only pairs with suspected issues):")
        for r in results:
            if not r.gemini_summary:
                continue
            print("\n--- Problematic pair ---")
            print(f"Original:  {relpath_or_none(r.original_file)}")
            print(f"Converted: {relpath_or_none(r.converted_file)}")
            print(r.gemini_summary)

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze converted tests between TVM and PyTorch."
    )
    parser.add_argument(
        "--tvm-root",
        default="repos/tvm",
        help="Path to original TVM repo (default: repos/tvm)",
    )
    parser.add_argument(
        "--pytorch-root",
        default="repos/pytorch",
        help="Path to original PyTorch repo (default: repos/pytorch)",
    )
    parser.add_argument(
        "--converted-tvm-torch-root",
        default="converted_tests/tvm_torch",
        help="Path to converted TVM->PyTorch tests (default: converted_tests/tvm_torch)",
    )
    parser.add_argument(
        "--converted-torch-tvm-root",
        default="converted_tests/torch_tvm",
        help="Path to converted PyTorch->TVM tests (default: converted_tests/torch_tvm)",
    )
    parser.add_argument(
        "--mode",
        choices=["tvm_torch", "torch_tvm", "both"],
        default="both",
        help="Which direction(s) to analyze (default: both)",
    )
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="Call Gemini API in batches to analyze mismatches.",
    )
    parser.add_argument(
        "--no-pytest",
        action="store_true",
        help="Skip running pytest on tests (by default pytest is run).",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to write full JSON results.",
    )

    args = parser.parse_args()

    all_results: list[TestPairResult] = []

    # tvm -> torch
    if args.mode in ("tvm_torch", "both"):
        results, orig_files, conv_files, total_orig_tests = process_mode(
            mode="tvm_torch",
            converted_root=args.converted_tvm_torch_root,
            original_root=args.tvm_root,
            use_gemini=args.use_gemini,
            run_pytests=not args.no_pytest,
        )

        if not args.no_pytest:
            (
                batch_orig_exit,
                batch_orig_run,
                batch_orig_fail,
                batch_orig_started,
            ) = run_pytest_batch(orig_files)

            (
                batch_conv_exit,
                batch_conv_run,
                batch_conv_fail,
                batch_conv_started,
            ) = run_pytest_batch(conv_files)
        else:
            batch_orig_exit = batch_conv_exit = 0
            batch_orig_run = batch_orig_fail = 0
            batch_conv_run = batch_conv_fail = 0
            batch_orig_started = batch_conv_started = False

        summarize_results(
            "tvm_torch",
            results,
            total_orig_tests,
            batch_orig_exit,
            batch_orig_run,
            batch_orig_fail,
            batch_orig_started,
            batch_conv_exit,
            batch_conv_run,
            batch_conv_fail,
            batch_conv_started,
        )
        all_results.extend(results)

    # torch -> tvm
    if args.mode in ("torch_tvm", "both"):
        results, orig_files, conv_files, total_orig_tests = process_mode(
            mode="torch_tvm",
            converted_root=args.converted_torch_tvm_root,
            original_root=args.pytorch_root,
            use_gemini=args.use_gemini,
            run_pytests=not args.no_pytest,
        )

        if not args.no_pytest:
            (
                batch_orig_exit,
                batch_orig_run,
                batch_orig_fail,
                batch_orig_started,
            ) = run_pytest_batch(orig_files)

            (
                batch_conv_exit,
                batch_conv_run,
                batch_conv_fail,
                batch_conv_started,
            ) = run_pytest_batch(conv_files)
        else:
            batch_orig_exit = batch_conv_exit = 0
            batch_orig_run = batch_orig_fail = 0
            batch_conv_run = batch_conv_fail = 0
            batch_orig_started = batch_conv_started = False

        summarize_results(
            "torch_tvm",
            results,
            total_orig_tests,
            batch_orig_exit,
            batch_orig_run,
            batch_orig_fail,
            batch_orig_started,
            batch_conv_exit,
            batch_conv_run,
            batch_conv_fail,
            batch_conv_started,
        )
        all_results.extend(results)

    if args.json_output:
        serializable = [dataclasses.asdict(r) for r in all_results]
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n[info] Wrote JSON results to {args.json_output}")


if __name__ == "__main__":
    main()
