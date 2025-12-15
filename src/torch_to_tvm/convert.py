"""
convert.py

This script converts PyTorch/TorchInductor tests to TVM tests.

1. Scans PyTorch tests under ./repos/pytorch/test
2. Uses the mapping JSON from ./data/torch_tvm_mapping.json to decide which
   tests are good candidates (based on PyTorch API coverage)
3. Converts all convertible tests, sending them to Gemini in batches
   of BATCH_SIZE tests per request
4. Saves converted tests under ./converted_tests/torch_tvm/<relative_path>.py

Usage (from project root):

    python src/convert.py

You can tweak:

    --min-covered-ratio R   (default: 0.3, more relaxed)

Environment:

    Set whatever google-genai expects (e.g. GOOGLE_API_KEY), or rely on
    dotenv (.env) if configured.
"""

from dotenv import load_dotenv
load_dotenv()

import shutil
import argparse
import ast
import json
from pathlib import Path
from textwrap import dedent

from google import genai

MODEL_NAME = "gemini-2.5-flash"

# Hard-coded Gemini batch size (how many tests per request).
# Manually tune this based on context limits.
BATCH_SIZE = 1

ROOT = Path(__file__).resolve().parents[2]
TORCH_ROOT = ROOT / "repos" / "pytorch" # Changed from TVM_ROOT
MAPPING_PATH = ROOT / "data" / "torch_tvm_mapping.json" # New mapping file
OUTPUT_ROOT = ROOT / "converted_tests" / "torch_tvm" # New output path


def _get_attr_chain(node):
    # This utility function remains the same
    attrs = []
    cur = node
    while isinstance(cur, ast.Attribute):
        attrs.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        attrs.append(cur.id)
        attrs.reverse()
        return ".".join(attrs)
    return None


def collect_torch_calls_in_file(src: str) -> set[str]:
    """Collects fully-qualified PyTorch API calls (e.g., 'torch.tensor', 'torch.nn.Module')."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()

    alias_map = {}
    calls = set()

    # imports - look for 'torch' and 'torch.*'
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch":
                    # Maps 'import torch as t' to {'t': 'torch'}
                    alias_map[alias.asname or "torch"] = "torch"
        elif isinstance(node, ast.ImportFrom):
            if node.module == "torch":
                # Maps 'from torch import tensor as t' to {'t': 'torch.tensor'}
                for alias in node.names:
                    alias_map[alias.asname or alias.name] = f"torch.{alias.name}"
            elif node.module and node.module.startswith("torch."):
                # Maps 'from torch.nn import Module' to {'Module': 'torch.nn.Module'}
                for alias in node.names:
                    alias_map[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    # calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            chain = _get_attr_chain(node.func)
            if chain is None:
                continue

            parts = chain.split(".")
            root = parts[0]
            rest = parts[1:]

            # Resolve aliased imports
            if root in alias_map:
                full = ".".join([alias_map[root]] + rest)
            # Handle direct 'torch.*' calls
            elif root == "torch":
                full = chain
            else:
                continue

            # Only track API calls that start with 'torch.'
            if full.startswith("torch."):
                calls.add(full)

    return calls


def load_mappings(path: Path):
    # This function remains the same
    if not path.exists():
        # NOTE: Mapping file path changed to 'torch_tvm_mapping.json'
        raise FileNotFoundError(f"Missing mapping file: {path}. Ensure you have 'torch_tvm_mapping.json'.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Mapping JSON must be a list.")
    return data


def index_mappings(mappings):
    # This function is adapted to index 'torch_api'
    by_full = {}
    by_short = {}

    for m in mappings:
        torch_api = m.get("torch_api") # Changed from 'tvm_api'
        if not isinstance(torch_api, str):
            continue

        by_full.setdefault(torch_api, []).append(m)
        short = torch_api.split(".")[-1]
        by_short.setdefault(short, []).append(m)

    return by_full, by_short


def compute_mapping_coverage(torch_calls, by_full, by_short):
    """Computes coverage based on PyTorch calls and the indexed mapping."""
    if not torch_calls:
        return set(), set(), 0.0

    covered = set()
    uncovered = set()

    for call in torch_calls: # Iterating over torch calls
        short = call.split(".")[-1]
        candidates = by_full.get(call, []) + by_short.get(short, [])
        # Assume a mapping is 'usable' if the type is not 'no_mapping'
        usable = any(m.get("mapping_type") != "no_mapping" for m in candidates)
        if usable:
            covered.add(call)
        else:
            uncovered.add(call)

    ratio = len(covered) / len(torch_calls) if torch_calls else 0.0
    return covered, uncovered, ratio


def parse_batch_response(text: str):
    # This function remains the same
    """
    Expect output like:

        ### BEGIN FILE path/to/file.py
        ...lines...
        ### END FILE

    Returns: dict[path -> code_str]
    """
    files = {}
    lines = text.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        if line.startswith("### BEGIN FILE "):
            rel_path = line.replace("### BEGIN FILE ", "").strip()
            i += 1
            start = i
            while i < n and not lines[i].strip().startswith("### END FILE"):
                i += 1
            code = "\n".join(lines[start:i]).rstrip() + "\n"
            files[rel_path] = code
        i += 1

    return files


def call_gemini_convert_batch(selected_tests, full_mapping):
    client = genai.Client()

    mapping_json = json.dumps(full_mapping, indent=2)

    # Example block is flipped for Torch -> TVM style guidance
    example_block = dedent("""
    Example PyTorch equivalent and its TVM form (for style only):

    PyTorch:
    ===CODE===
    import torch
    x = torch.tensor([1, 2, 3])
    ===CODE===

    TVM:
    ===CODE===
    import tvm
    x = tvm.nd.array([1, 2, 3])
    ===CODE===
    """)

    tests_blob = []
    for idx, item in enumerate(selected_tests, start=1):
        tests_blob.append(
            f"""
TEST {idx}: {item['rel_path']}
TORCH_TEST_SOURCE: # Changed from TVM_TEST_SOURCE
===CODE===
{item['source']}
===CODE===
            """
        )

    tests_section = "\n".join(tests_blob)

    # Prompt is modified for PyTorch to TVM conversion
    prompt = f"""
You convert PyTorch / TorchInductor tests into equivalent TVM tests.

PRINCIPLES:
- The mapping table is a strong hint and high-precision reference, but NOT exhaustive.
- When a mapping entry exists, you should generally follow it.
- When no mapping entry exists, you MAY still rewrite the PyTorch API to a TVM
  equivalent using your own knowledge, IF you are reasonably confident.
- If you are NOT confident about a safe or correct rewrite, insert a clear TODO
  comment instead of guessing, BUT:
    * The resulting file MUST still be valid Python that can be imported and run.

RUNTIME CONSTRAINT (VERY IMPORTANT):
- Each generated test file MUST be runnable on its own without errors:
    * It must include ALL necessary imports (e.g., tvm, numpy, pytest, etc.).
    * It must NOT reference undefined variables, functions, fixtures, or modules.
    * It must NOT import or use torch or any torch.* symbols.
    * There must be no syntax errors.
    * Wherever you insert TODO comments, surrounding code must still run
      (e.g., do not leave undefined names or ellipses that would break execution).

GOALS:
- Keep the semantics of the tests: same high-level behavior, shapes, dtypes,
  assertions, and control flow.
- Replace PyTorch-specific constructs with TVM equivalents.
- Prefer idiomatic TVM code (e.g., tvm.nd.array, tvm.ir.module.IRModule, etc.).

OUTPUT FORMAT (STRICT):
- For EACH test file listed below, output:

    ### BEGIN FILE <rel_path>
    <valid, standalone Python test code>
    ### END FILE

- Use the SAME <rel_path> values that appear in the input.
- Do NOT add explanations or prose.
- Do NOT use backticks or markdown code fences.

EXAMPLE (minimal illustration of style, NOT a strict template):
{example_block}

MAPPING TABLE (JSON):
{mapping_json}

Notes about the mapping table:
- It may not contain every possible PyTorch API.
- "mapping_type" can be "direct", "composite", or "no_mapping".
- "torch_pattern" and "tvm_pattern" show example rewrite templates.
- "example_pairs" show concrete before/after code.
- Use these as guidance, but you may still convert APIs not listed
  when you know the correct TVM equivalent.

PYTORCH TESTS TO CONVERT: # Changed from TVM TESTS TO CONVERT
{tests_section}

Now, convert ALL of these tests. For each one, output exactly:

    ### BEGIN FILE <rel_path>
    <python code>
    ### END FILE

for every <rel_path> given above, and nothing else.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    text = (response.text or "").strip()
    return parse_batch_response(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-covered-ratio",
        type=float,
        default=0.3,
        help="Minimum ratio of PyTorch calls covered by mappings to consider a test."
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading mapping table...")
    # NOTE: This expects ./data/torch_tvm_mapping.json
    mappings = load_mappings(MAPPING_PATH)
    by_full, by_short = index_mappings(mappings)

    print("Scanning PyTorch tests...")
    # NOTE: Assuming PyTorch tests are under a common 'test' directory
    test_files = sorted((TORCH_ROOT / "test").rglob("test_*.py"))
    candidates = []

    for path in test_files:
        src = path.read_text(encoding="utf-8", errors="ignore")
        calls = collect_torch_calls_in_file(src) # Changed API collection function
        if not calls:
            continue

        covered, uncovered, ratio = compute_mapping_coverage(calls, by_full, by_short)
        if ratio >= args.min_covered_ratio and covered:
            # Need to find the correct relative path: assuming 'test' is the root
            try:
                rel_path = path.relative_to(TORCH_ROOT).as_posix()
            except ValueError:
                # Fallback if the file is outside TORCH_ROOT/test (shouldn't happen)
                rel_path = path.name
            candidates.append(
                {
                    "path": path,
                    "rel_path": rel_path,
                    "source": src,
                    "ratio": ratio,
                    "covered": covered,
                    "uncovered": uncovered
                }
            )

    print(f"Total PyTorch test files: {len(test_files)}")
    print(f"Tests passing coverage >= {args.min_covered_ratio}: {len(candidates)}")

    if not candidates:
        print("No convertible tests found. Try lowering --min-covered-ratio further.")
        return

    # Sort by coverage / number of covered calls (best first)
    candidates.sort(key=lambda c: (c["ratio"], len(c["covered"])), reverse=True)

    total = len(candidates)
    print(
        f"Converting all {total} candidate tests "
        f"in batches of up to {BATCH_SIZE} tests per Gemini request..."
    )
    print()

    # Process in batches
    batch_num = 0
    for start in range(0, total, BATCH_SIZE):
        batch_num += 1
        batch = candidates[start:start + BATCH_SIZE]

        print(
            f"Batch {batch_num}: converting {len(batch)} tests "
            f"(indices {start}..{start + len(batch) - 1})"
        )
        for c in batch:
            print(
                f"  - {c['rel_path']} "
                f"(coverage={c['ratio']:.2%}, covered={len(c['covered'])}, "
                f"uncovered={len(c['uncovered'])})"
            )
        print()

        rel_to_code = call_gemini_convert_batch(batch, mappings)

        print("Saving converted tests for this batch...")
        for item in batch:
            rel = item["rel_path"]
            code = rel_to_code.get(rel)
            # Output path construction is the same, just with the new root
            out_path = OUTPUT_ROOT / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if not code:
                print(f"WARNING: No output for {rel}, skipping.")
                continue

            out_path.write_text(code, encoding="utf-8")
            print(f"  Saved {out_path}")

        print()

    print("Done.")


if __name__ == "__main__":
    main()