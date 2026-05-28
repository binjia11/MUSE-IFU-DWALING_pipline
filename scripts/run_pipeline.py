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

Usage:  python3 scripts/run_pipeline.py
"""

import os
import sys
import subprocess
import time
import glob

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)
SAMPLE_DIR = os.path.join(ROOT, " DWALIN_SAMPLE")
CONDA_ENV = "uclchem_3.4"
PYTHON = os.path.expanduser(f"/opt/anaconda3/envs/{CONDA_ENV}/bin/python3")

# Galaxies to process (EXCLUDE HEN_2-10 — already done)
GALAXIES = []
for f in sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.fits"))):
    name = os.path.basename(f).replace(".fits", "")
    if "HEN_2-10" in name or "He2-10" in name:
        continue
    cube_path = f
    gal_name = name
    v_sys = 0.0  # all sample galaxies have RADVEL=0 in headers
    GALAXIES.append((gal_name, cube_path, v_sys))

STAGES = [
    "01_voronoi_binning.py",
    "02_ppxf_fit_bins.py",
    "03_subtract_continuum.py",
    "04_fit_emission_lines.py",
    "05_bpt_diagrams.py",
]


def run_stage(galaxy, cube_path, v_sys, script):
    """Run one pipeline stage for one galaxy."""
    script_path = os.path.join(SCRIPTS_DIR, script)
    out_dir = os.path.join(ROOT, "outputs", galaxy)

    env = os.environ.copy()
    env["PIPE_GALAXY"]    = galaxy
    env["PIPE_CUBE_PATH"] = cube_path
    env["PIPE_OUT_DIR"]   = out_dir
    env["PIPE_V_SYS"]     = str(v_sys)

    print(f"  [{galaxy}] {script} ...", end=" ", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, script_path],
        env=env,
        capture_output=True,
        text=True,
        timeout=14400,  # 4 h per stage max
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"OK ({elapsed:.1f}s)")
        # Print last few lines
        lines = result.stdout.strip().split("\n")
        for line in lines[-4:]:
            if line.strip():
                print(f"    {line.strip()[:120]}")
    else:
        print(f"FAILED (exit {result.returncode}, {elapsed:.1f}s)")
        print("--- STDOUT (last 30 lines) ---")
        for line in result.stdout.strip().split("\n")[-30:]:
            print(f"  {line[:150]}")
        print("--- STDERR ---")
        for line in result.stderr.strip().split("\n")[-20:]:
            print(f"  {line[:150]}")
        return False
    return True


def main():
    print(f"Pipeline driver — {len(GALAXIES)} galaxies, 5 stages each")
    print(f"Root:    {ROOT}")
    print(f"Sample:  {SAMPLE_DIR}")
    print(f"Conda:   {CONDA_ENV}")
    print()

    for i, (galaxy, cube_path, v_sys) in enumerate(GALAXIES):
        print(f"[{i+1}/{len(GALAXIES)}] {galaxy}  (V_SYS={v_sys} km/s)")
        print(f"  cube: {cube_path}")

        for stage in STAGES:
            ok = run_stage(galaxy, cube_path, v_sys, stage)
            if not ok:
                print(f"  STOPPING — {galaxy} failed at {stage}")
                # Continue to next galaxy
                break

        print()

    # Summary
    print("\n=== BPT PDFs ===")
    bpt_dir = os.path.join(ROOT, "outputs", "bpt")
    for f in sorted(glob.glob(os.path.join(bpt_dir, "*_bpt.pdf"))):
        print(f"  {os.path.basename(f)}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
