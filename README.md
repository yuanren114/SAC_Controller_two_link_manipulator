# Two-Link Arm SAC Tracking — Cheatsheet

## What this setup means

This project uses **direct-torque SAC** on a two-link planar arm.

Your intended experiment setup is:

* **Train** in a **mis-modeled simulation** where link-1 mass is set to **2.0 kg**.
* **Test / eval / render** in the **target / real dynamics** where link-1 mass is **1.9 kg**.
* **Train disturbance is disabled**.
* **Test disturbance is optional** and can be enabled with a fixed seed for reproducibility.

So this is a **train-test dynamics mismatch / sim-to-real-gap style setup**, not an internal model-based controller mismatch.

---

## Most useful commands

### 1) Train only

```powershell
python Core_SAC.py train `
  --trajectory-mode ellipse `
  --total-steps 45000 `
  --eval-interval 5000 `
  --eval-episodes 3 `
  --train-m1-actual 2.0 `
  --test-m1-actual 1.9
```

Use this when you want training logs, checkpoints, and periodic evaluation, but no pygame window at the end.

---

### 2) Train, then render the best checkpoint

```powershell
python Core_SAC.py train-render `
  --trajectory-mode ellipse `
  --total-steps 45000 `
  --eval-interval 5000 `
  --eval-episodes 3 `
  --train-m1-actual 2.0 `
  --test-m1-actual 1.9 `
  --test-use-disturbance `
  --disturbance-seed-test 1701
```

This will:

* train in the **2.0 kg simulation**
* periodically evaluate in the **1.9 kg test dynamics**
* save metrics and checkpoints
* open pygame rendering using the best checkpoint after training finishes

---

### 3) Evaluate an existing checkpoint

```powershell
python Core_SAC.py eval `
  --checkpoint logs/<your_run>/checkpoints/best.pt `
  --trajectory-mode ellipse `
  --eval-episodes 5 `
  --test-m1-actual 1.9 `
  --test-use-disturbance `
  --disturbance-seed-test 1701
```

Use this to evaluate a saved checkpoint under the target / real test dynamics.

---

### 4) Render an existing checkpoint

```powershell
python Core_SAC.py render `
  --checkpoint logs/<your_run>/checkpoints/best.pt `
  --trajectory-mode ellipse `
  --test-m1-actual 1.9 `
  --test-use-disturbance `
  --disturbance-seed-test 1701
```

Use this for visualization only.

---

## Smoke-test commands

### Very short training run

```powershell
python Core_SAC.py train `
  --trajectory-mode ellipse `
  --total-steps 5000 `
  --eval-interval 2500 `
  --eval-episodes 2 `
  --max-episode-steps 500 `
  --start-steps 500 `
  --update-after 500 `
  --batch-size 64 `
  --train-m1-actual 2.0 `
  --test-m1-actual 1.9
```

---

### Quick eval smoke test

```powershell
python Core_SAC.py eval `
  --checkpoint logs/<your_run>/checkpoints/best.pt `
  --trajectory-mode ellipse `
  --eval-episodes 2 `
  --test-m1-actual 1.9
```

---

## What metrics are automatically saved

### Training-side metrics

Saved in:

* `training_episodes.csv`
* `training_steps.jsonl`

Common fields include:

* `episode_reward`
* `mean_tracking_error_m`
* `rmse_tracking_error_m`
* `mean_step_compute_time_ms`

---

### Evaluation-side metrics

Saved in:

* `evaluation.csv`
* `eval/<step_label>/summary.json`
* `summary.md`

The main metrics you care about are:

* **Final RMSE tracking error**
* **Training time**
* **Mean test step time (compute only)**
* **Mean test step time (full eval loop)**

---

## Where to look after a run

Each run creates:

```text
logs/<timestamp>_<command>_<variant>_<control_mode>_seed<seed>/
```

Most useful files:

* `summary.md`

  * final high-level summary
  * includes training time, final RMSE, and test step timing
* `training_time.json`

  * total wall-clock training time
* `evaluation.csv`

  * evaluation summary across checkpoints
* `training_episodes.csv`

  * training-episode metrics
* `checkpoints/best.pt`

  * best evaluation checkpoint
* `checkpoints/final.pt`

  * last checkpoint

---

## What “step time” means

Two timing metrics are recorded during evaluation and rendering.

### 1) Compute-only step time

This includes only the main control/simulation work:

* action selection
* disturbance injection
* dynamics step
* next-state / reward computation

It **does not include pygame drawing**.

This is the better metric for comparing different controllers fairly.

### 2) Full-loop step time

This includes the whole loop, including:

* control + simulation
* drawing
* text rendering
* display update

This is closer to “actual wall-clock frame time” during visualization.

---

## Important PowerShell note

In **PowerShell**, line continuation uses the backtick:

```powershell
`
```

Do **not** use `^` there. `^` is for **cmd.exe**, not PowerShell.

So this is correct in PowerShell:

```powershell
python Core_SAC.py train `
  --trajectory-mode ellipse `
  --total-steps 45000
```

---

## Key argument meanings

* `--train-m1-actual 2.0`

  * the link-1 mass used in the **training simulation**
  * in your setup, this is the intentionally wrong simulated mass

* `--test-m1-actual 1.9`

  * the link-1 mass used in **test / eval / render**
  * in your setup, this is the target / real mass

* `--test-use-disturbance`

  * enables fixed-seed spike disturbance during test / eval / render

* `--disturbance-seed-test 1701`

  * fixes the test disturbance so repeated experiments stay comparable

---

## Recommended experiment wording for your report

You can describe the setup like this:

> The policy is trained in a mismatched simulation where the first-link mass is set to 2.0 kg, and evaluated under the target dynamics where the true first-link mass is 1.9 kg. To assess robustness, a fixed-seed spike disturbance is optionally injected during evaluation.

---

## Recommended small cleanup to the code later

The current behavior is correct for your experiment, but these argument names are easy to misunderstand:

* `train_m1_actual`
* `test_m1_actual`

A clearer naming scheme later would be:

* `train_m1_plant`
* `test_m1_plant`
* `m1_nominal`

Behavior does not need to change; this would only improve readability.
