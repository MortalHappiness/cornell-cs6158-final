from dotenv import load_dotenv
import ast
import json
from collections import defaultdict
from pathlib import Path
from textwrap import dedent

from google import genai


load_dotenv()

MODEL_NAME = "gemini-2.5-flash"

ROOT = Path(__file__).resolve().parents[1]
TVM_ROOT = ROOT / "repos" / "tvm"
TORCH_ROOT = ROOT / "repos" / "pytorch"
OUTPUT_PATH = ROOT / "data" / "tvm_torch_mapping.json"


def extract_snippet(file_path: Path, needle: str, context_lines: int = 20) -> str:
    """
    Return a snippet of `context_lines` lines from file_path around the first
    line that contains `needle`. If not found, return the top of the file.
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return f"# ERROR: File not found: {file_path}"

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if needle in line:
            start = max(0, i - 3)
            end = min(len(lines), i + context_lines)
            return "\n".join(lines[start:end])

    return "\n".join(lines[:context_lines])


def _get_attr_chain(node):
    """
    Given an ast.Attribute chain or ast.Name, return a dotted string
    like "tvm.nd.array" or "relay.add", or None if not resolvable.
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


def collect_tvm_api_usage_from_tests(tvm_root: Path) -> set[str]:
    """
    Scan TVM's Python tests and collect fully-qualified TVM API names that
    appear in calls, e.g. "tvm.nd.array", "tvm.relay.add", etc.
    """
    tests_root = tvm_root / "tests" / "python"
    used_full_apis: set[str] = set()

    for py_file in tests_root.rglob("*.py"):
        try:
            src = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except Exception:
            continue

        # Map local import aliases -> fully qualified modules
        alias_map: dict[str, str] = {}

        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "tvm":
                        alias_map[alias.asname or "tvm"] = "tvm"
            elif isinstance(node, ast.ImportFrom):
                if node.module == "tvm":
                    # from tvm import relay, tir, ...
                    for alias in node.names:
                        alias_map[alias.asname or alias.name] = f"tvm.{alias.name}"
                elif node.module and node.module.startswith("tvm."):
                    # from tvm.relay import op as relay_op
                    for alias in node.names:
                        alias_map[alias.asname or alias.name] = f"{node.module}.{alias.name}"

        # Walk the tree to find function calls that resolve to TVM
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                chain = _get_attr_chain(node.func)
                if not chain:
                    continue

                parts = chain.split(".")
                root = parts[0]
                rest = parts[1:]

                if root in alias_map:
                    full = ".".join([alias_map[root]] + rest)
                elif root == "tvm":
                    full = chain
                else:
                    continue

                if full.startswith("tvm."):
                    used_full_apis.add(full)

    return used_full_apis


def discover_tvm_functions(tvm_root: Path):
    """
    Discover all top-level TVM functions under python/tvm.

    Returns a list of dicts:
      {
        "tvm_name": "tvm.relay.op.add",
        "file": Path(...),
        "func_name": "add"
      }
    """
    apis = []
    search_root = tvm_root / "python" / "tvm"

    for py_file in search_root.rglob("*.py"):
        parts = set(py_file.parts)
        if "tests" in parts or "contrib" in parts:
            continue

        rel_mod_path = py_file.relative_to(search_root).with_suffix("")
        module = "tvm." + ".".join(rel_mod_path.parts)

        try:
            src = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except Exception:
            continue

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                func_name = node.name
                tvm_name = f"{module}.{func_name}"
                apis.append(
                    {
                        "tvm_name": tvm_name,
                        "file": py_file,
                        "func_name": func_name,
                    }
                )

    return apis


def index_torch_functions(torch_root: Path):
    """
    Build an index: function_name -> list[Path to defining file].
    """
    name_to_files: dict[str, list[Path]] = defaultdict(list)

    for py_file in torch_root.rglob("*.py"):
        if "test" in py_file.parts:
            continue

        try:
            src = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except Exception:
            continue

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                name_to_files[node.name].append(py_file)

    return name_to_files


def build_torch_blocks(func_name: str, torch_index, max_candidates: int = 3) -> str:
    """
    For a given function name, return up to max_candidates PyTorch snippets.
    """
    files = torch_index.get(func_name, [])[:max_candidates]
    if not files:
        return "# No candidate PyTorch functions with this name were found."

    blocks = []
    for f in files:
        snippet = extract_snippet(f, f"def {func_name}(")
        blocks.append(
            f"""Name: {func_name}
File: {f}
Code:
```

{snippet}

````"""
        )
    return "\n\n".join(blocks)


def build_batch_prompt(batch_items, torch_index) -> str:
    """
    Build a prompt that asks Gemini to map ALL APIs in `batch_items`.
    The response must be a JSON array of mapping objects in the same order.
    """

    # Global instructions & schema (note: {{ }} for literal braces in f-string)
    header = dedent("""
    You are mapping TVM APIs to PyTorch/TorchInductor APIs.

    Your response will be consumed by ANOTHER LLM performing test conversion.
    Therefore, you must output PRECISE, STRUCTURED JSON.

    You are given a list of APIs. For EACH API, return ONE OBJECT
    in a JSON ARRAY, in the SAME ORDER.

    SCHEMA FOR EACH OBJECT:
    {
      "tvm_api": "str",
      "torch_api": "str or list or null",
      "mapping_type": "direct | composite | no_mapping",
      "direction": "tvm_to_torch",
      "tvm_pattern": "str or null",
      "torch_pattern": "str or null",
      "arg_mapping": {
        "tvm_arg_name": "torch_arg_or_placeholder"
      },
      "example_pairs": [
        {
          "tvm": "one-line TVM call",
          "torch": "equivalent one-line PyTorch call"
        }
      ],
      "constraints": "str",
      "notes": "str",
      "confidence": 0.0 to 1.0
    }

    IMPORTANT RULES:
    - Use ONLY the PyTorch functions shown in the provided snippets as candidates.
    - If you are unsure about a safe mapping, set "mapping_type": "no_mapping",
      "torch_api": null, and explain briefly in "notes".
    - Do NOT invent imaginary PyTorch APIs.
    - ALWAYS keep array order: the Nth object in your JSON array
      corresponds to the Nth API in the list below.
    - Be conservative. It is better to say "no_mapping" than to guess.

    EXAMPLE OF A "direct" MAPPING OBJECT:

    {
      "tvm_api": "tvm.nd.array",
      "torch_api": "torch.tensor",
      "mapping_type": "direct",
      "direction": "tvm_to_torch",
      "tvm_pattern": "tvm.nd.array({{data}}, device={{device}})",
      "torch_pattern": "torch.tensor({{data}}, device={{device}})",
      "arg_mapping": {
        "data": "data",
        "device": "device"
      },
      "example_pairs": [
        {
          "tvm": "y = tvm.nd.array({{data}}, device={{device}})",
          "torch": "y = torch.tensor({{data}}, device={{device}})"
        }
      ],
      "constraints": "Basic tensor creation only; dtype handling is separate.",
      "notes": "Wraps a host array into a tensor on a device.",
      "confidence": 0.95
    }

    If there is NO safe mapping, use this pattern:

    {
      "tvm_api": "<the TVM name>",
      "torch_api": null,
      "mapping_type": "no_mapping",
      "direction": "tvm_to_torch",
      "tvm_pattern": null,
      "torch_pattern": null,
      "arg_mapping": {},
      "example_pairs": [],
      "constraints": "Not convertible",
      "notes": "NO_MAPPING: <short reason>",
      "confidence": 0.2
    }

    Now, here is the list of TVM APIs to map:
    """)

    items_blob = []
    for idx, item in enumerate(batch_items, start=1):
        tvm_name = item["tvm_name"]
        func_name = item["func_name"]
        tvm_file = item["file"]

        tvm_snip = extract_snippet(tvm_file, f"def {func_name}(")
        torch_blocks = build_torch_blocks(func_name, torch_index)

        block = f"""
    ### API {idx}: {tvm_name}

    TVM code:
    ```
    {tvm_snip}
    ```

    PyTorch candidate functions (same name: {func_name}):
    {torch_blocks}
    """
        items_blob.append(block)

    prompt = header + "\n\n" + "\n\n".join(items_blob) + "\n\nReturn a JSON array ONLY."
    return prompt


def call_gemini(prompt: str):
    """
    Call Gemini and parse the JSON array result.
    Raises ValueError if output is empty or not a JSON array.
    """
    client = genai.Client()
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)

    # response.text is a convenience property; may be None or ""
    text = (response.text or "").strip()
    if not text:
        raise ValueError("Gemini returned empty text.")

    # Try direct parse first
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON array substring
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Gemini output not JSON array:\n" + text)
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, list):
        raise ValueError("Gemini output is not a JSON list:\n" + text)

    return parsed


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTVM root:     {TVM_ROOT}")
    print(f"PyTorch root: {TORCH_ROOT}")

    # 1. Figure out which TVM APIs are actually used in tests
    print("\n========== Collecting TVM API usage from tests ==========")
    used_full_apis = collect_tvm_api_usage_from_tests(TVM_ROOT)
    used_func_names = {api.split(".")[-1] for api in used_full_apis}
    print(f" TVM APIs used in tests: {len(used_full_apis)} call sites")
    print(f" Distinct function names used in tests: {len(used_func_names)}\n")

    # 2. Index PyTorch functions by name
    print("========== Indexing PyTorch functions ==========")
    torch_index = index_torch_functions(TORCH_ROOT)
    print(f" PyTorch unique function names indexed: {len(torch_index)}\n")

    # 3. Discover all TVM APIs, then filter:
    #    - only functions used in tests
    #    - AND having at least one PyTorch function with same name
    print("========== Discovering TVM APIs ==========")
    tvm_all = discover_tvm_functions(TVM_ROOT)
    print(f" Total TVM functions discovered: {len(tvm_all)}")

    tvm_candidates = [
        api
        for api in tvm_all
        if (api["func_name"] in used_func_names and api["func_name"] in torch_index)
    ]

    print(f" Filtered to {len(tvm_candidates)} TVM functions:")
    print("  - appear in tests")
    print("  - AND share a function name with at least one PyTorch function\n")

    if not tvm_candidates:
        print("No overlapping TVM/PyTorch function names used in tests. Nothing to map.")
        return

    # 4. Adaptive batching:
    #    - Start with one giant batch (all candidates)
    #    - On failure, halve batch size and retry
    remaining = tvm_candidates[:]  # copy
    all_mappings = []
    batch_size = len(remaining)

    batch_index = 0

    print("========== Adaptive Gemini batch mapping ==========")

    while remaining:
        batch_index += 1
        size = min(batch_size, len(remaining))
        batch = remaining[:size]
        print(f"\nAttempting batch {batch_index} with size {size} (remaining {len(remaining)})")

        prompt = build_batch_prompt(batch, torch_index)

        try:
            result = call_gemini(prompt)
        except ValueError as e:
            print(f"  Gemini failed for batch size {size}: {e}")
            if size == 1:
                # Can't shrink further; record a NO_MAPPING stub and skip
                item = batch[0]
                print(f"  Giving up on single API {item['tvm_name']} (marking as no_mapping).")
                all_mappings.append(
                    {
                        "tvm_api": item["tvm_name"],
                        "torch_api": None,
                        "mapping_type": "no_mapping",
                        "direction": "tvm_to_torch",
                        "tvm_pattern": None,
                        "torch_pattern": None,
                        "arg_mapping": {},
                        "example_pairs": [],
                        "constraints": "Not convertible (batch too large / context limit).",
                        "notes": "NO_MAPPING: Could not obtain mapping due to context/response limits.",
                        "confidence": 0.0,
                    }
                )
                remaining = remaining[1:]
                continue

            # Halve the batch size and try again
            batch_size = max(1, size // 2)
            print(f"  Reducing batch size to {batch_size} and retrying...")
            continue

        # Success
        if len(result) != len(batch):
            print(
                f"  WARNING: Gemini returned {len(result)} objects for batch of size {len(batch)}."
            )

        all_mappings.extend(result)
        remaining = remaining[size:]
        print(f"  Success mapping {len(result)} APIs. Remaining: {len(remaining)}")

        # Try to grow batch size slightly next time (optional, but we can keep it fixed too).
        # Here we just keep the same size that worked, to be conservative.

    # 5. Save all mappings
    OUTPUT_PATH.write_text(json.dumps(all_mappings, indent=2))
    print(f"\nSaved {len(all_mappings)} mappings to {OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()