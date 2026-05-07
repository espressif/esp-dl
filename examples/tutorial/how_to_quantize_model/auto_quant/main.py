"""AutoQuant

Stage 1: search + quantize + accuracy eval.
Run me on a server (or any host with the calibration data + GPU/CPU).

Stage 2: latency benchmark
`test_candidates.py` takes the Top-K I pick and measures on real hardware.

Usage:
    python -m auto_quant.main                # fresh run (wipes summary/candidates)
    python -m auto_quant.main --resume       # continue interrupted run
"""

import argparse
import hashlib
import itertools
import json
import os

from auto_quant.calib_dataloader import create_calib_dataloader
from auto_quant.candidates import update_topk_candidates
from auto_quant.config import param_space, runtime_config, strategy_space
from auto_quant.eval import evaluation
from auto_quant.quantize import (
    build_quant_setting,
    get_layerwise_error_from_default_quant,
    quant_model,
)
from auto_quant.save import (
    create_run_dir,
    load_summary,
    reset_run_dir,
    save_experiment,
    update_summary,
)
from auto_quant.search import build_search_pipeline


def strategy_hash(strategy: dict, sampled_param: dict) -> str:
    """Stable hash for (strategy values, sampled_param). Used by --resume."""
    payload = {
        "strategy": {k: v["value"] for k, v in strategy.items()},
        "params": sampled_param,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AutoQuant Stage 1: search + quantize")
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip experiments already present in summary.json",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # --- Build calibration dataloader (skipped at config import time on purpose) ---
    runtime_config["calib_dataloader"] = create_calib_dataloader()

    # --- Run dir + resume bookkeeping ---
    run_dir = create_run_dir()
    candidates_path = os.path.join(run_dir, "candidates.json")

    done_hashes = set()
    next_index = 0
    if args.resume:
        prev = load_summary(run_dir)
        done_hashes = {e["hash"] for e in prev if "hash" in e}
        next_index = max((e.get("index", -1) for e in prev), default=-1) + 1
        print(
            f"[Resume] {len(done_hashes)} experiments already done; "
            f"next index={next_index}"
        )
    else:
        reset_run_dir(run_dir)

    # --- One default-setting pass to compute layerwise error ---
    layerwise_error = get_layerwise_error_from_default_quant(runtime_config)

    # --- Search loop ---
    searcher, _rules = build_search_pipeline(strategy_space, param_space)
    exp_index = next_index

    for strategy in searcher.search():
        print(f"\n=== Strategy ===")
        print("[strategy]", {k: v["value"] for k, v in strategy.items()})

        param_sets = []
        for k, v in strategy.items():
            if v["value"] and v["param_candidates"]:
                param_sets.append(v["param_candidates"])
            else:
                param_sets.append([None])

        for param_combo in itertools.product(*param_sets):
            sampled_param = dict(zip(strategy.keys(), param_combo))
            h = strategy_hash(strategy, sampled_param)
            if h in done_hashes:
                print(f"[skip] already done: {h[:8]}")
                continue

            try:
                quant_setting, num_of_bits = build_quant_setting(
                    strategy,
                    sampled_param,
                    runtime_config["target"],
                    layerwise_error,
                    runtime_config,
                )
                quant_graph = quant_model(runtime_config, quant_setting, num_of_bits)
                metrics = evaluation(quant_graph)
                assert runtime_config["primary_metric"] in metrics, (
                    f"primary_metric={runtime_config['primary_metric']!r} "
                    f"not in metrics={list(metrics)}"
                )
                print(f"[eval] {metrics}")

                record = save_experiment(
                    run_dir=run_dir,
                    index=exp_index,
                    strategy=strategy,
                    sampled_param=sampled_param,
                    export_path=runtime_config["export_path"],
                    runtime_config=runtime_config,
                )
                record.update(metrics)

                update_topk_candidates(
                    candidates_path,
                    record,
                    runtime_config["num_of_candidates"],
                    key=runtime_config["primary_metric"],
                )
                update_summary(run_dir, {"hash": h, **record})

            except Exception as exc:
                import traceback

                traceback.print_exc()
                print(
                    f"[ERROR] Experiment {exp_index} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                update_summary(
                    run_dir,
                    {
                        "hash": h,
                        "index": exp_index,
                        "strategy": {k: v["value"] for k, v in strategy.items()},
                        "params": sampled_param,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )

            exp_index += 1

    print(f"\nAll done. Total experiments tried this run: {exp_index - next_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
