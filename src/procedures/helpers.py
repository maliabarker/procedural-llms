import re
from typing import List, Dict, Any, Tuple, Set

def _names(items: List[Dict[str, Any]]) -> List[str]:
    return [x["name"] for x in items]

def pretty_print(procedure):
    print("\n--- Procedure: {} ---".format(procedure["NameDescription"]))
    print("Steps:")
    for step in procedure["steps"]:
        print(f"\nStep {step['id']}: {step['stepDescription']}")
        print("  **Inputs**:")
        for inp in step['inputs']:
            print(f"    - {inp['name']}: {inp['description']}")
        print("  **Outputs**:")
        for out in step['output']:
            print(f"    - {out['name']}: {out['description']}")

# Function to pull the exact numeric answer from GSM8K answer string
def extract_final_number(text: str) -> str:
    return re.search(r"####\s*([-+]?\d+(?:\.\d+)?)\s*$", text).group(1)

def _as_name_set(items: List[Dict[str, Any]]) -> Set[str]:
    return set(_names(items))

def _descriptions(items: List[Dict[str, Any]]) -> Set[str]:
    # {"name": "description"} (falls back to empty string)
    return {x["name"]: x.get("description", "") for x in items}

def _canon_details(details: Dict[str, Any]) -> Tuple:
    """
    Canonicalize details to make diagnostics dedup-able.
    Converts lists -> tuples (sorted if str/int), dicts -> tuples of items (recursively).
    """
    def canon(x):
        if isinstance(x, dict):
            return tuple(sorted((k, canon(v)) for k, v in x.items()))
        if isinstance(x, list):
            # sort simple lists for stability, else keep order but tuple-ize
            if all(isinstance(v, (str, int, float)) for v in x):
                return tuple(sorted(x))
            return tuple(canon(v) for v in x)
        if isinstance(x, set):
            return tuple(sorted(x))
        return x
    return canon(details)