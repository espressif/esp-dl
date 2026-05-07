"""AutoQuant Stage 2: hardware latency benchmark.

Run me on the local machine that has the ESP board plugged in.

Usage:
    python -m auto_quant.test_candidates                          # default run dir
    python -m auto_quant.test_candidates --port /dev/ttyUSB1
    python -m auto_quant.test_candidates --run-name my-run --target esp32s3
"""

import argparse

from auto_quant.config import runtime_config
from auto_quant.latency import test_candidates_latency
from auto_quant.save import create_run_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AutoQuant Stage 2: ESP latency")
    p.add_argument("--run-base", default="outputs", help="parent dir for runs")
    p.add_argument("--run-name", default="run", help="subdir name under --run-base")
    p.add_argument("--port", default=None, help="override runtime_config['port']")
    p.add_argument("--target", default=None, help="override runtime_config['target']")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.port is not None:
        runtime_config["port"] = args.port
    if args.target is not None:
        runtime_config["target"] = args.target

    run_dir = create_run_dir(args.run_base, args.run_name)
    test_candidates_latency(run_dir, runtime_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
