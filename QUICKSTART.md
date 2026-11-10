# Quickstart

Four commands. Nothing else needed.

```bash
# 1. dependencies
pip install torch scikit-image pandas matplotlib scipy

# 2. the data: Kaggle -> mateuszbuda/lgg-mri-segmentation  (~1 GB)
#    unzip it anywhere; the scripts find it themselves

# 3. does everything work on this machine? (5 min, and it estimates the real run)
python smoke_test.py

# 4. prove the whole pipeline runs, end to end, small (~15 min)
python run_all.py --quick
```

Look at `figures/`. If those seven files exist and nothing looks insane, run the real thing:

```bash
python run_all.py
```

It is **resumable** - if it dies, run the same command again and it picks up from where it
stopped. Finished stages are skipped.

---

## If something goes wrong

| What you see | What to do |
|---|---|
| `Could not find the LGG dataset` | pass `--root path/to/kaggle_3m` (the folder whose subfolders are `TCGA_*`) |
| `No module named torch` | `pip install torch` |
| smoke test says "CPU-only build" | reinstall torch from pytorch.org with your CUDA version |
| out of memory | `python train.py --bs 8 ...`, or 4 |
| a stage fails | fix it, re-run the same `run_all.py` command - nothing already finished is lost |
| too slow | `python run_all.py --skip_restore --enhance fast` |

## Useful flags

```bash
python run_all.py --quick            # tiny version, for checking it runs
python run_all.py --skip_restore     # drop the learned enhancement arm (first thing to cut)
python run_all.py --skip_components  # drop branch B1
python run_all.py --enhance fast     # bilateral filter on the GPU instead of CPU denoising
python run_all.py --transfer         # B3: full severity transfer matrix
python run_all.py --force            # redo everything from scratch
```

## What you get

| File | Question it answers |
|---|---|
| `figures/fig0_ladder.png` | do the simulated scans look like real bad scans? **Check this first.** |
| `figures/fig1_dice_vs_severity.png` | how fast does accuracy fall, and which strategy holds up |
| `figures/fig2_decoupling.png` | **the headline** - does a better-looking image segment better? |
| `figures/fig3_label_efficiency.png` | how many local labels beat enhancement (H2) |
| `figures/fig4_components.png` | which degradation does the damage (B1) |
| `figures/fig5_confidence.png` | does the model know when it is failing (B4) |
| `figures/fig6_failure_modes.png` | who fails first, and in which direction (B5) |
| `figures/summary.csv` | every number behind every figure |
| `results/results.csv` | one row per image per arm per severity - the raw data |

Then read **Reading the results** in the handbook before deciding what any of it means.
That section was written before the data existed, on purpose.
