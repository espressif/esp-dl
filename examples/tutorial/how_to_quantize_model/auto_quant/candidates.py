import json
import os


def load_candidates(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_candidates(path, candidates):
    with open(path, "w") as f:
        json.dump(candidates, f, indent=4)


def _is_low_latency_candidate(record):
    s = record.get("strategy", {})
    return (not s.get("mixed_precision", False)) and (
        not s.get("horizontal_layer_split", False)
    )


def _sort_desc(records, key):
    return sorted(records, key=lambda x: (-x[key], x.get("index", 10**12)))


def _balanced_topk(candidates, k, key):
    if k <= 0:
        return []

    valid = [c for c in candidates if key in c and isinstance(c[key], (int, float))]
    if not valid:
        return []

    target = min(k, len(valid))
    low_latency_pool = _sort_desc(
        [c for c in valid if _is_low_latency_candidate(c)], key
    )
    other_candidates_pool = _sort_desc(
        [c for c in valid if not _is_low_latency_candidate(c)], key
    )

    if not other_candidates_pool:
        return low_latency_pool[:target]
    if not low_latency_pool:
        return other_candidates_pool[:target]

    n_low_latency = target // 2
    n_other_candidates = target - n_low_latency

    picked_low_latency = low_latency_pool[:n_low_latency]
    picked_other_candidates = other_candidates_pool[:n_other_candidates]
    selected = picked_low_latency + picked_other_candidates

    if len(picked_low_latency) < n_low_latency:
        need = n_low_latency - len(picked_low_latency)
        selected += other_candidates_pool[
            n_other_candidates : n_other_candidates + need
        ]
    if len(picked_other_candidates) < n_other_candidates:
        need = n_other_candidates - len(picked_other_candidates)
        selected += low_latency_pool[n_low_latency : n_low_latency + need]

    return _sort_desc(selected, key)[:target]


def update_topk_candidates(path, record, k, key):
    """Append `record`, then keep balanced Top-K by record[key]."""
    candidates = load_candidates(path)
    candidates.append(record)
    candidates = _balanced_topk(candidates, k, key)
    save_candidates(path, candidates)
    return candidates
