from typing import List, Dict, Any, Tuple

def _names(items: List[Dict[str, Any]]) -> List[str]:
    return [x["name"] for x in items]

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