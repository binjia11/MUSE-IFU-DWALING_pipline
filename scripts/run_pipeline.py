#!/usr/bin/env python3
"""Driver: run the full MUSE pipeline for all galaxies in DWALIN_Sample.

Stages executed in order for each galaxy:
  01_voronoi_binning.py
  02_ppxf_fit_bins.py
  03_subtract_continuum.py
  04_fit_emission_lines.py
  05_bpt_diagrams.py

Outputs go to  outputs/<galaxy_name>/  (line_maps, kinematics, bins, ...)
BPT PDFs go to  outputs/bpt/<galaxy_name>_bpt.pdf

Uses BPASS v2.2.1 templates + auto-measured systemic velocities.
No timeouts — runs each stage to completion.

Parallel mode:  python3 scripts/run_pipeline.py --parallel [N]
  N = number of concurrent galaxies (default: all CPU cores)
  Each galaxy gets 1 core; stage 4 uses PIPE_N_WORKERS=1.
  v_sys is measured sequentially upfront (memory-efficient).

Usage:  python3 scripts/run_pipeline.py [--parallel [N]]
"""

import os
import sys
import subprocess
import time
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)
from common import measure_systemic_velocity
SAMPLE_DIR = os.path.join(ROOT, "DWALIN_Sample")
CONDA_ENV = "uclchem_3.4"
PYTHON = os.path.expanduser(f"/opt/anaconda3/envs/{CONDA_ENV}/bin/python3")

BPT_DIR = os.path.join(ROOT, "outputs", "bpt")

SKIP_GALAXIES = {"HEN_2-10", "He2-10"}

STAGES = [
    "01_voronoi_binning.py",
    "02_ppxf_fit_bins.py",
    "03_subtract_continuum.py",
    "04_fit_emission_lines.py",
    "05_bpt_diagrams.py",
]


def build_galaxy_list():
    """Discover galaxies, measure v_sys sequentially, return list."""
    all_cubes = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.fits")))
    galaxies = []
    skipped = []

    for f in all_cubes:
        name = os.path.basename(f).replace(".fits", "")
        if any(s in name for s in SKIP_GALAXIES):
            continue
        bpt_pdf = os.path.join(BPT_DIR, f"{name}_bpt.pdf")
        if os.path.exists(bpt_pdf):
            skipped.append(name)
            continue
        v_sys = measure_systemic_velocity(f, verbose=True)
        galaxies.append((name, f, v_sys))

    return galaxies, skipped


def run_stage(galaxy, cube_path, v_sys, script, single_core=False):
    """Run one pipeline stage. Returns (ok, elapsed_seconds)."""
    script_path = os.path.join(SCRIPTS_DIR, script)
    out_dir = os.path.join(ROOT, "outputs", galaxy)

    env = os.environ.copy()
    env["PIPE_GALAXY"]       = galaxy
    env["PIPE_CUBE_PATH"]    = cube_path
    env["PIPE_OUT_DIR"]      = out_dir
    env["PIPE_V_SYS"]        = str(v_sys)
    env["PIPE_TEMPLATE_LIB"] = "bpass"

    # Single-core mode: limit numpy + stage 4 workers
    if single_core:
        env["OMP_NUM_THREADS"]     = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"]     = "1"
        env["PIPE_N_WORKERS"]      = "1"

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, script_path],
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        lines = result.stdout.strip().split("\n")
        last_info = " | ".join(l.strip()[:80] for l in lines[-3:] if l.strip())
        print(f"[{galaxy}] {script.replace('.py',''):<25s} OK {elapsed:7.1f}s  {last_info[:120]}")
        return True, elapsed
    else:
        print(f"[{galaxy}] {script} FAILED (exit {result.returncode}, {elapsed:.1f}s)")
        for line in result.stderr.strip().split("\n")[-6:]:
            print(f"    ERR: {line[:150]}")
        return False, elapsed


def process_galaxy_sequential(galaxy, cube_path, v_sys):
    """Run all 5 stages for one galaxy (used in parallel mode)."""
    t0 = time.time()
    for stage in STAGES:
        ok, _ = run_stage(galaxy, cube_path, v_sys, stage, single_core=True)
        if not ok:
            return galaxy, False, time.time() - t0
    return galaxy, True, time.time() - t0


def run_sequential(galaxies, skipped):
    """Original sequential pipeline."""
    print(f"Mode:     sequential")
    print(f"To process: {len(galaxies)}")
    if skipped:
        print(f"Skipped ({len(skipped)} already done):")
        for s in skipped:
            print(f"  - {s}")
    print()

    for i, (galaxy, cube_path, v_sys) in enumerate(galaxies):
        t0 = time.time()
        print(f"[{i+1}/{len(galaxies)}] {galaxy}  (V_SYS={v_sys:.0f} km/s)")

        for stage in STAGES:
            ok, _ = run_stage(galaxy, cube_path, v_sys, stage)
            if not ok:
                print(f"  STOPPING — {galaxy} failed at {stage}")
                break

        print(f"  [{galaxy}] total: { (time.time()-t0)/60:.1f} min\n")


def run_parallel(galaxies, skipped, n_workers):
    """Parallel pipeline: N galaxies concurrently, each on 1 core."""
    print(f"Mode:     parallel ({n_workers} concurrent galaxies)")
    print(f"To process: {len(galaxies)}")
    if skipped:
        print(f"Skipped ({len(skipped)} already done):")
        for s in skipped:
            print(f"  - {s}")
    print()

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_galaxy_sequential, g, c, v): g
            for g, c, v in galaxies
        }
        for future in as_completed(futures):
            galaxy = futures[future]
            try:
                name, ok, dt = future.result()
                results.append((name, ok, dt))
                status = "DONE" if ok else "FAILED"
                print(f"\n=== [{name}] {status} in {dt/60:.1f} min ===\n")
            except Exception as e:
                print(f"\n=== [{galaxy}] CRASHED: {e} ===\n")
                results.append((galaxy, False, 0))

    # Summary
    done = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    print(f"\nDone: {len(done)}  Failed: {len(failed)}")
    if failed:
        print("Failed galaxies:")
        for name, _, _ in failed:
            print(f"  - {name}")

    print("\n=== BPT PDFs ===")
    for f in sorted(glob.glob(os.path.join(BPT_DIR, "*_bpt.pdf"))):
        print(f"  {os.path.basename(f)}")


def main():
    parser = argparse.ArgumentParser(description="MUSE IFU pipeline driver")
    parser.add_argument("--parallel", "-p", nargs="?", const=-1, type=int,
                        default=None,
                        help="Run galaxies in parallel (default N: all cores)")
    args = parser.parse_args()

    print(f"Pipeline driver")
    print(f"Root:      {ROOT}")
    print(f"Sample:    {SAMPLE_DIR}")
    print(f"Conda:     {CONDA_ENV}")
    print(f"Template:  BPASS v2.2.1")

    galaxies, skipped = build_galaxy_list()

    if not galaxies:
        print("\nAll galaxies already processed!")
        return

    if args.parallel is not None:
        n = args.parallel if args.parallel > 0 else (os.cpu_count() or 4)
        run_parallel(galaxies, skipped, n)
    else:
        run_sequential(galaxies, skipped)

    print("\nAll done.")


if __name__ == "__main__":
    main()
