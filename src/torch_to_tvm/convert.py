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

# --- Define Paths ---
ROOT = Path(__file__).resolve().parents[2]
TORCH_ROOT = ROOT / "repos" / "pytorch" # ASSUMING PyTorch repo is here
MAPPING_PATH = ROOT / "data" / "torch_tvm_mapping.json"
OUTPUT_ROOT = ROOT / "converted_tests" / "torch_tvm" # Separate output folder

# --- AST Utility Functions ---

def _get_attr_chain(node):
    """
    Recursively collects the full dotted name for an Attribute or Name node.
    E.g., for 'a.b.c', returns 'a.b.c'.
    """
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
    """
    Scans a Python source file for all fully qualified 'torch.*' calls.
    Handles 'import torch' and 'import torch.nn.functional as F'.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()

    alias_map = {}
    calls = set()

    # 1. Collect import aliases for 'torch' and its submodules
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch":
                    # Maps 'torch' or 't' (if 'import torch as t') -> 'torch'
                    alias_map[alias.asname or "torch"] = "torch"
        elif isinstance(node, ast.ImportFrom):
            if node.module == "torch":
                for alias in node.names:
                    # Maps 'tensor' (if 'from torch import tensor') -> 'torch.tensor'
                    alias_map[alias.asname or alias.name] = f"torch.{alias.name}"
            elif node.module and (node.module == "torch" or node.module.startswith("torch.")):
                for alias in node.names:
                    # Maps 'F' (if 'from torch.nn import functional as F') -> 'torch.nn.functional'
                    full_name = f"{node.module}.{alias.name}"
                    alias_map[alias.asname or alias.name] = full_name

    # 2. Collect 'torch.*' calls based on aliases
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            chain = _get_attr_chain(node.func)
            if chain is None:
                continue

            parts = chain.split(".")
            root = parts[0]
            rest = parts[1:]

            full = None
            if root in alias_map:
                # If root is an alias (e.g., 't.add' where 't' -> 'torch')
                full = ".".join([alias_map[root]] + rest)
            elif root == "torch":
                # Direct call (e.g., 'torch.add')
                full = chain
            
            # Check if the collected full call starts with 'torch.'
            if full and (full.startswith("torch.") or full == "torch"):
                calls.add(full)

    return calls

# --- Mapping and Coverage Functions (Reused) ---

def load_mappings(path: Path):
    """Loads and validates the JSON mapping file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing mapping file: {path}")
    # The mapping is TVM -> Torch. We need to index the Torch side for conversion.
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Mapping JSON must be a list.")
    return data


def index_mappings(mappings):
    """
    Indexes the mapping table by *Torch* API for quick lookup.
    Returns: by_full_torch, by_short_torch
    """
    by_full_torch = {}
    by_short_torch = {}

    for m in mappings:
        torch_api = m.get("torch_api")
        if not isinstance(torch_api, str):
            continue

        by_full_torch.setdefault(torch_api, []).append(m)
        short = torch_api.split(".")[-1]
        by_short_torch.setdefault(short, []).append(m)

    return by_full_torch, by_short_torch


def compute_mapping_coverage(torch_calls, by_full_torch, by_short_torch):
    """
    Computes the coverage ratio of Torch calls that have a TVM equivalent mapping.
    """
    if not torch_calls:
        return set(), set(), 0.0

    covered = set()
    uncovered = set()

    # The mapping file is TVM -> Torch, so we are checking if a torch_api key exists
    for call in torch_calls:
        # Check by full name (torch.nn.functional.relu) and short name (relu)
        short = call.split(".")[-1]
        candidates = by_full_torch.get(call, []) + by_short_torch.get(short, [])
        
        # A call is 'usable' if *any* mapping candidate is NOT 'no_mapping'
        usable = any(m.get("mapping_type") != "no_mapping" for m in candidates)
        if usable:
            covered.add(call)
        else:
            uncovered.add(call)

    ratio = len(covered) / len(torch_calls) if torch_calls else 0.0
    return covered, uncovered, ratio

# --- Gemini Interaction Functions ---

def parse_batch_response(text: str):
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
    """Generates the prompt and calls Gemini to convert the batch of tests."""
    client = genai.Client()

    # Reverse the mapping for clarity in the prompt (Torch -> TVM)
    reversed_mapping = []
    for m in full_mapping:
        if m.get("mapping_type") != "no_mapping":
            reversed_mapping.append({
                "torch_api": m.get("torch_api"),
                "tvm_api": m.get("tvm_api"),
                "mapping_type": m.get("mapping_type"),
                # Note the pattern fields are also logically reversed now
                "torch_pattern": m.get("torch_pattern"),
                "tvm_pattern": m.get("tvm_pattern"),
                "example_pairs": [
                    {"torch": pair.get("torch"), "tvm": pair.get("tvm")}
                    for pair in m.get("example_pairs", [])
                ]
            })

    mapping_json = json.dumps(reversed_mapping, indent=2)

    # No ``` anywhere — use ===CODE=== to embed examples.
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
    x = tvm.runtime.ndarray.array([1, 2, 3])
    ===CODE===
    """)

    tests_blob = []
    for idx, item in enumerate(selected_tests, start=1):
        tests_blob.append(
            f"""
TEST {idx}: {item['rel_path']}
TORCH_TEST_SOURCE:
===CODE===
{item['source']}
===CODE===
            """
        )

    tests_section = "\n".join(tests_blob)

    prompt = f"""
You convert PyTorch tests into equivalent TVM tests.

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
    * It must include ALL necessary imports (e.g., tvm, pytest, numpy, etc.).
    * It must NOT reference undefined variables, functions, fixtures, or modules.
    * It must NOT import or use torch or any torch.* symbols.
    * There must be no syntax errors.
    * Wherever you insert TODO comments, surrounding code must still run
      (e.g., do not leave undefined names or ellipses that would break execution).

GOALS:
- Keep the semantics of the tests: same high-level behavior, shapes, dtypes,
  assertions, and control flow.
- Replace PyTorch-specific constructs with TVM equivalents.
- Prefer idiomatic TVM code (e.g., tvm.runtime.ndarray.array, tvm.ir.module.IRModule, etc.).

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

MAPPING TABLE (JSON - Torch API -> TVM API):
{mapping_json}

Notes about the mapping table:
- It may not contain every possible PyTorch API.
- "mapping_type" can be "direct", "composite", or "no_mapping".
- "torch_pattern" and "tvm_pattern" show example rewrite templates.
- "example_pairs" show concrete before/after code.
- Use these as guidance, but you may still convert APIs not listed
  when you know the correct TVM equivalent.

PYTORCH TESTS TO CONVERT:
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
        "--max-tests",
        type=int,
        default=20,
        help="Maximum number of tests to convert in one run (for context size)."
    )
    parser.add_argument(
        "--min-covered-ratio",
        type=float,
        default=0.3,
        help="Minimum ratio of Torch calls covered by mappings to consider a test."
    )
    args = parser.parse_args()

    # Always delete the output folder before each run
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


    print("Loading mapping table...")
    # by_full_torch and by_short_torch are the keys for checking coverage
    # against the 'torch_api' field in the mapping.
    mappings = load_mappings(MAPPING_PATH)
    by_full_torch, by_short_torch = index_mappings(mappings)

    print("Scanning PyTorch tests...")
    # Assuming PyTorch tests live under 'repos/pytorch/test' and start with 'test_'
    torch_test_root = TORCH_ROOT / "test"
    test_files = sorted(torch_test_root.rglob("test_*.py"))
    candidates = []

    for path in test_files:
        # Skip files that are likely internal helpers
        if any(part in path.parts for part in ['__pycache__', 'cuda', 'cpp']):
            continue

        src = path.read_text(encoding="utf-8", errors="ignore")
        calls = collect_torch_calls_in_file(src)
        if not calls:
            continue

        # Use the Torch-indexed mapping for coverage check
        covered, uncovered, ratio = compute_mapping_coverage(
            calls, by_full_torch, by_short_torch
        )
        
        # Select tests that meet the minimum coverage and have at least one convertible call
        if ratio >= args.min_covered_ratio and covered:
            # The relative path should be from the PyTorch test root for better organization
            try:
                rel_path = path.relative_to(torch_test_root).as_posix()
            except ValueError:
                 # Should not happen if torch_test_root is correct, but safer to skip
                 continue 
                 
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

    print(f"Total PyTorch test files scanned: {len(test_files)}")
    print(f"Tests passing coverage >= {args.min_covered_ratio}: {len(candidates)}")

    if not candidates:
        print("No convertible tests found. Try lowering --min-covered-ratio further.")
        return

    # Sort candidates for prioritization (highest coverage first)
    candidates.sort(key=lambda c: (c["ratio"], len(c["covered"])), reverse=True)
    selected = candidates[: args.max_tests]

    print(f"Converting {len(selected)} tests in ONE Gemini request...")
    for c in selected:
        print(
            f"  - {c['rel_path']} "
            f"(coverage={c['ratio']:.2%}, covered={len(c['covered'])}, "
            f"uncovered={len(c['uncovered'])})"
        )
    print()

    # Call the conversion function
    rel_to_code = call_gemini_convert_batch(selected, mappings)

    print("Saving converted tests...")
    for item in selected:
        rel = item["rel_path"]
        code = rel_to_code.get(rel)
        out_path = OUTPUT_ROOT / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not code:
            print(f"WARNING: No output for {rel}, skipping.")
            continue

        out_path.write_text(code, encoding="utf-8")
        print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()