"""Run the whole pipeline, one step at a time, from download to figures.

Each step is a separate process (so memory is released between steps) and the pipeline
stops at the first failure. Downloads are skipped if their outputs already exist.
Usage:  python src/run_all.py [--from STEP_NAME]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = [
    ("atl24", ["download_atl24.py"]),
    ("prepare", ["prepare_atl24.py"]),
    ("tide_check", ["check_tide_consistency.py"]),
    ("scenes", ["select_scenes.py"]),
    ("s2", ["download_s2.py", "--k", "10"]),
    ("composite", ["build_composite.py"]),
    ("samples", ["extract_samples.py", "--image", "composite"]),
    ("samples_single", ["extract_samples.py", "--image", "single"]),
    ("gebco", ["download_gebco.py"]),
    ("eval_a", ["train_eval.py", "--schemes", "random,track", "--tag", "composite_a"]),
    ("eval_b", ["train_eval.py", "--schemes", "track_buffer,rgt_buffer", "--buffer", "500", "--tag", "composite_b"]),
    ("eval_primary", ["train_eval.py", "--schemes", "rgt_buffer", "--buffer", "1000", "--tag", "primary",
                      "--models", "stumpf,stumpf_switch,lyzenga,rf,xgb,rf_bands,rf_physics"]),
    ("eval_single", ["train_eval.py", "--image", "single", "--schemes", "rgt_buffer", "--buffer", "1000", "--tag", "single"]),
    ("eval_balanced", ["train_eval.py", "--schemes", "rgt_buffer", "--buffer", "1000", "--models", "rf,rf_balanced",
                       "--tag", "balanced"]),
    ("buffers", ["buffer_sensitivity.py"]),
    ("variogram", ["variogram.py"]),
    ("map", ["predict_map.py", "--model", "rf"]),
    ("figures", ["make_figures.py"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=STEPS[0][0], choices=[s for s, _ in STEPS])
    args = ap.parse_args()
    names = [s for s, _ in STEPS]
    for name, cmd in STEPS[names.index(args.start):]:
        if name == "buffers":
            out = SRC.parent / "results" / "buffer_sensitivity.csv"
            out.unlink(missing_ok=True)  # the script appends
        t = time.time()
        print(f"=== {name}: {' '.join(cmd)}", flush=True)
        r = subprocess.run([sys.executable, "-W", "ignore", str(SRC / cmd[0]), *cmd[1:]], cwd=SRC)
        if r.returncode:
            sys.exit(f"step '{name}' failed (exit {r.returncode}); fix it and rerun with --from {name}")
        print(f"=== {name} done in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
