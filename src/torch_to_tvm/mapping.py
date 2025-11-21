from dotenv import load_dotenv
import ast
import json
from collections import defaultdict
from pathlib import Path
from textwrap import dedent

from google import genai # Assuming this is the SDK used in the original script

# Load environment variables (e.g., for API keys)
load_dotenv()

MODEL_NAME = "gemini-2.5-flash"

# --- Define Root Paths (Adjust these paths to your system) ---
ROOT = Path(__file__).resolve().parents[2] # Adjust as necessary
TVM_ROOT = ROOT / "repos" / "tvm"         # Adjust as necessary
TORCH_ROOT = ROOT / "repos" / "pytorch"   # Adjust as necessary
OUTPUT_PATH = ROOT / "data" / "torch_tvm_mapping.json" # Inverted output name

# ==============================================================================
# === HELPER FUNCTIONS (Most can be reused or slightly modified) ===
# ==============================================================================

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
        # We need to be a bit more flexible with needle for function defs
        if needle in line:
            start = max(0, i - 3)
            end = min(len(lines), i + context_lines)
            return "\n".join(lines[start:end])

    return "\n".join(lines[:context_lines])


def _get_attr_chain(node):
    """
    Given an ast.Attribute chain or ast.Name, return a dotted string
    like "torch.ops.aten.add" or "torch.tensor", or None if not resolvable.
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


def collect_torch_api_usage_from_tests(torch_root: Path) -> set[str]:
    """
    Scan Torch's Python tests and collect fully-qualified Torch API names that
    appear in calls. (Reused as the set of Torch APIs we care about)
    """
    # This function is used to find the set of APIs that are 'in use'
    tests_root = torch_root / "test"
    used_full_apis: set[str] = set()

    for py_file in tests_root.rglob("*.py"):
        try:
            src = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except Exception:
            continue

        alias_map: dict[str, str] = {}

        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "torch":
                        alias_map[alias.asname or "torch"] = "torch"
            elif isinstance(node, ast.ImportFrom):
                if node.module == "torch":
                    for alias in node.names:
                        alias_map[alias.asname or alias.name] = f"torch.{alias.name}"
                elif node.module and node.module.startswith("torch."):
                    for alias in node.names:
                        alias_map[alias.asname or alias.name] = f"{node.module}.{alias.name}"

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
                elif root == "torch":
                    full = chain
                else:
                    continue

                if full.startswith("torch."):
                    used_full_apis.add(full)

    return used_full_apis


def discover_torch_functions(torch_root: Path):
    """
    Discover all top-level Torch functions under the main Python directories.
    (This is the **new primary API discovery function**)

    Returns a list of dicts:
      {
        "torch_name": "torch.nn.functional.relu",
        "file": Path(...),
        "func_name": "relu"
      }
    """
    apis = []
    # Search root for the main torch/ files
    search_root = torch_root / "torch"

    for py_file in search_root.rglob("*.py"):
        parts = set(py_file.parts)
        if "test" in parts or "csrc" in parts or "backends" in parts:
            continue

        # Create the module name, e.g., 'torch.nn.functional'
        rel_mod_path = py_file.relative_to(search_root.parent).with_suffix("")
        module = ".".join(rel_mod_path.parts)

        try:
            src = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except Exception:
            continue

        for node in tree.body:
            # We are looking for function definitions only
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                func_name = node.name
                torch_name = f"{module}.{func_name}"
                apis.append(
                    {
                        "torch_name": torch_name,
                        "file": py_file,
                        "func_name": func_name,
                    }
                )
    return apis


def index_tvm_functions(tvm_root: Path):
    """
    Build an index: function_name -> list[Path to defining file].
    (This is the **new indexing function**)
    """
    name_to_files: dict[str, list[Path]] = defaultdict(list)
    search_root = tvm_root / "python" / "tvm"

    for py_file in search_root.rglob("*.py"):
        parts = set(py_file.parts)
        if "tests" in parts or "contrib" in parts:
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


def build_tvm_blocks(func_name: str, tvm_index, max_candidates: int = 3) -> str:
    """
    For a given function name, return up to max_candidates TVM snippets.
    (Modified to use the TVM index)
    """
    files = tvm_index.get(func_name, [])[:max_candidates]
    if not files:
        return "# No candidate TVM functions with this name were found."

    blocks = []
    for f in files:
        # Check for function definition to get surrounding code
        snippet = extract_snippet(f, f"def {func_name}(")
        # Need to determine the full TVM module path for the snippet
        try:
            # Reconstruct the module name
            rel_mod_path = f.relative_to(TVM_ROOT / "python").with_suffix("")
            tvm_module = ".".join(rel_mod_path.parts)
        except ValueError:
            # Fallback if path reconstruction fails
            tvm_module = f.stem

        blocks.append(
            f"""Name: {func_name}
            Module: {tvm_module}
            File: {f}
            Code:
```   

{snippet}

````"""
        )
    return "\n\n".join(blocks)


def build_batch_prompt(batch_items, tvm_index) -> str:
    """
    Build a prompt that asks Gemini to map ALL APIs in `batch_items`
    from Torch to TVM.
    (Modified for Torch-to-TVM direction and schema fields)
    """

    header = dedent("""
    You are mapping PyTorch/TorchInductor APIs to TVM APIs.

    Your response will be consumed by ANOTHER LLM performing test conversion.
    Therefore, you must output PRECISE, STRUCTURED JSON.

    You are given a list of APIs. For EACH API, return ONE OBJECT
    in a JSON ARRAY, in the SAME ORDER.

    SCHEMA FOR EACH OBJECT:
    {
      "torch_api": "str",
      "tvm_api": "str or list or null",
      "mapping_type": "direct | composite | no_mapping",
      "direction": "torch_to_tvm",
      "torch_pattern": "str or null",
      "tvm_pattern": "str or null",
      "arg_mapping": {
        "torch_arg_name": "tvm_arg_or_placeholder"
      },
      "example_pairs": [
        {
          "torch": "one-line PyTorch call",
          "tvm": "equivalent one-line TVM call"
        }
      ],
      "constraints": "str",
      "notes": "str",
      "confidence": 0.0 to 1.0
    }

    IMPORTANT RULES:
    - Use ONLY the TVM functions shown in the provided snippets as candidates.
    - If you are unsure about a safe mapping, set "mapping_type": "no_mapping",
      "tvm_api": null, and explain briefly in "notes".
    - Do NOT invent imaginary TVM APIs.
    - ALWAYS keep array order: the Nth object in your JSON array
      corresponds to the Nth API in the list below.
    - Be conservative. It is better to say "no_mapping" than to guess.

    EXAMPLE OF A "direct" MAPPING OBJECT:

    {
      "torch_api": "torch.tensor",
      "tvm_api": "tvm.nd.array",
      "mapping_type": "direct",
      "direction": "torch_to_tvm",
      "torch_pattern": "torch.tensor({{data}}, device={{device}})",
      "tvm_pattern": "tvm.nd.array({{data}}, device={{device}})",
      "arg_mapping": {
        "data": "data",
        "device": "device"
      },
      "example_pairs": [
        {
          "torch": "y = torch.tensor({{data}}, device={{device}})",
          "tvm": "y = tvm.nd.array({{data}}, device={{device}})"
        }
      ],
      "constraints": "Basic tensor creation only; dtype handling is separate.",
      "notes": "Wraps a host array into a tensor on a device.",
      "confidence": 0.95
    }

    If there is NO safe mapping, use this pattern:

    {
      "torch_api": "<the Torch name>",
      "tvm_api": null,
      "mapping_type": "no_mapping",
      "direction": "torch_to_tvm",
      "torch_pattern": null,
      "tvm_pattern": null,
      "arg_mapping": {},
      "example_pairs": [],
      "constraints": "Not convertible",
      "notes": "NO_MAPPING: <short reason>",
      "confidence": 0.2
    }

    Now, here is the list of PyTorch APIs to map:
    """)

    items_blob = []
    for idx, item in enumerate(batch_items, start=1):
        torch_name = item["torch_name"]
        func_name = item["func_name"]
        torch_file = item["file"]

        # Now we look for the Torch function snippet and TVM candidate snippets
        torch_snip = extract_snippet(torch_file, f"def {func_name}(")
        tvm_blocks = build_tvm_blocks(func_name, tvm_index)

        block = f"""
    ### API {idx}: {torch_name}

    PyTorch code:
    ```
    {torch_snip}
    ```

    TVM candidate functions (same name: {func_name}):
    {tvm_blocks}
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

    text = (response.text or "").strip()
    if not text:
        raise ValueError("Gemini returned empty text.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Gemini output not JSON array:\n" + text)
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, list):
        raise ValueError("Gemini output is not a JSON list:\n" + text)

    return parsed


# ==============================================================================
# === MAIN EXECUTION LOGIC (Adapted) ===
# ==============================================================================

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nPyTorch root: {TORCH_ROOT}")
    print(f"TVM root:     {TVM_ROOT}")
    print(f"Output:       {OUTPUT_PATH}")

    # 1. Figure out which Torch APIs are actually used in tests (APIs to map)
    print("\n========== Collecting PyTorch API usage from tests ==========")
    used_full_apis = collect_torch_api_usage_from_tests(TORCH_ROOT)
    # Get only the function name, e.g., 'tensor' from 'torch.tensor'
    used_func_names = {api.split(".")[-1] for api in used_full_apis}
    print(f" PyTorch APIs used in tests: {len(used_full_apis)} call sites")
    print(f" Distinct function names used in tests: {len(used_func_names)}\n")

    # 2. Index TVM functions by name (The candidates for mapping)
    print("========== Indexing TVM functions ==========")
    tvm_index = index_tvm_functions(TVM_ROOT)
    print(f" TVM unique function names indexed: {len(tvm_index)}\n")

    # 3. Discover all Torch APIs, then filter:
    #    - only functions used in tests
    #    - AND having at least one TVM function with same name
    print("========== Discovering PyTorch APIs ==========")
    torch_all = discover_torch_functions(TORCH_ROOT)
    print(f" Total PyTorch functions discovered: {len(torch_all)}")

    torch_candidates = [
        api
        for api in torch_all
        if (api["func_name"] in used_func_names and api["func_name"] in tvm_index)
    ]

    print(f" Filtered to {len(torch_candidates)} PyTorch functions:")
    print("  - appear in tests")
    print("  - AND share a function name with at least one TVM function\n")

    if not torch_candidates:
        print("No overlapping Torch/TVM function names used in tests. Nothing to map.")
        return

    # 4. Adaptive batching:
    remaining = torch_candidates[:]  # copy
    all_mappings = []
    batch_size = len(remaining)

    batch_index = 0

    print("========== Adaptive Gemini batch mapping ==========")

    while remaining:
        batch_index += 1
        size = min(batch_size, len(remaining))
        batch = remaining[:size]
        print(f"\nAttempting batch {batch_index} with size {size} (remaining {len(remaining)})")

        prompt = build_batch_prompt(batch, tvm_index)

        try:
            result = call_gemini(prompt)
        except ValueError as e:
            print(f"  Gemini failed for batch size {size}: {e}")
            if size == 1:
                # Give up on single API
                item = batch[0]
                print(f"  Giving up on single API {item['torch_name']} (marking as no_mapping).")
                all_mappings.append(
                    {
                        "torch_api": item["torch_name"],
                        "tvm_api": None,
                        "mapping_type": "no_mapping",
                        "direction": "torch_to_tvm",
                        "torch_pattern": None,
                        "tvm_pattern": None,
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
        # Note: batch_size is not reset/grown here, which is conservative.

    # 5. Save all mappings
    OUTPUT_PATH.write_text(json.dumps(all_mappings, indent=2))
    print(f"\nSaved {len(all_mappings)} mappings to {OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()