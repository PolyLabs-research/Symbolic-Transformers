# Setting up Symbolic-Transformers on an AMD machine (instructions for a local agent)

You are setting up this repository on the user's AMD mini PC, so that it can run the project's training
experiments on the machine's GPU. The work so far has run on an NVIDIA laptop GPU (RTX 2000 Ada). Your
job is to get the same experiments running here, verify they behave the same way, and report back.
Do not run the experiments themselves until the user says so.

## Ground rules

- **Do not commit, push, or edit tracked files.** The branch below is under active development
  elsewhere. If something only works with a code change (for example an AMD-specific device issue),
  stop and report the exact error and the change you would make. Do not apply it.
- **Do not change any experiment setting:** steps, seeds, batch size, learning rate, layouts. The point
  is to compare against runs already made.
- **Keep every log.** Save each command's full output to `logs/` inside the clone (`logs/` is not
  tracked). Report outputs verbatim; do not summarise numbers you did not see.
- **Ask the user** for anything that needs their credentials, for example cloning a private repository.

## 1. Inspect the machine

Report:

- the OS and its version;
- CPU model and core count, and RAM;
- the GPU.

The machine is probably a GMKtec EVO-X2: an AMD Ryzen AI Max (Strix Halo) APU with Radeon 8060S
integrated graphics (LLVM target `gfx1151`), and memory shared between CPU and GPU. Confirm this rather
than assuming it:

- On Linux, run `lscpu`, `free -h`, and `rocminfo | grep -i gfx` if ROCm is installed.
- On Windows, use `Get-CimInstance Win32_Processor`, `Get-CimInstance Win32_VideoController` and
  `systeminfo`.

Also report which Python versions are installed. Python 3.10-3.13 works; the laptop uses 3.13.

## 2. Get the code

```bash
git clone https://github.com/PolyLabs-research/Symbolic-Transformers.git
cd Symbolic-Transformers
git checkout ccr-9f7282e8-cm8asz
git log --oneline -1
```

Report the commit hash. If the clone asks for credentials, stop and ask the user.

## 3. Python environment with GPU-enabled PyTorch

Use a plain virtual environment (`python -m venv`), not conda. On the Windows laptop, conda's OpenMP
library clashed with PyTorch's (`libiomp5md.dll` already initialised).

The code needs only `torch`, `numpy` and `pytest`. You do not need the rest of `requirements.txt`.

PyTorch on AMD GPUs uses ROCm. A ROCm build of PyTorch exposes the GPU through the usual `torch.cuda` API,
so the project's scripts pick it up with no code change. Support for this GPU generation is recent and
changes quickly, so check the current official sources rather than relying on memory: the install
selector at https://pytorch.org/get-started/locally/ and AMD's ROCm documentation for Ryzen AI Max /
Radeon 8060S. Then:

- **On Linux:**
  - Install a PyTorch build for the newest ROCm version that supports `gfx1151`, either from
    `https://download.pytorch.org/whl/rocm<version>` or from AMD's own wheels.
  - If PyTorch sees the GPU but kernels fail ("invalid device function" or similar), a common
    workaround is `HSA_OVERRIDE_GFX_VERSION=11.0.0`, which runs it as `gfx1100`. Use it only if needed,
    and report that you did.
- **On Windows:**
  - Look for AMD's current ROCm build of PyTorch for Windows that supports this GPU. Install it if it
    exists.
  - Otherwise ROCm under WSL2 may work; report what you find before going that way.
  - Do not use `torch-directml`. It needs a different device API, which the scripts do not support.
  - Set `$env:PYTHONUTF8 = "1"` in every PowerShell session. The scripts print ✓, which fails on the
    default console encoding.
  - PowerShell variables such as `$py` last only for the window they were set in.
- **If no GPU build works,** fall back to the CPU build of PyTorch, and say so clearly. The CPU is fast
  enough for some of these runs, and step 6 measures it.

After installing PyTorch, install the other two packages:

```bash
pip install numpy pytest
```

Verify the install and report the output:

```bash
python -c "import torch; print(torch.__version__, 'hip', torch.version.hip, 'gpu', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'); x = torch.randn(2048, 2048, device='cuda' if torch.cuda.is_available() else 'cpu'); print('matmul ok', float((x @ x).sum()))"
```

## 4. Run the tests

```bash
python -m pytest -q tests
```

Every test should pass (158 at the time of writing). The tests run on the CPU. Report the last line,
and any failures in full.

## 5. Regenerate the experiment datasets and check they match

The generated datasets are not in git: the JSON files are ignored, and only each dataset's
`metadata.json`, with a sha256 per split, is committed. Generate the two datasets the pending
experiments use:

```bash
python data/eval_generator.py --lookup --signature single --encoding keylast --queries 8 --train-query-sampling uniform --out datasets/eval_lookup_single_keylast_uq8
python data/eval_generator.py --lookup --signature single --encoding keylast --queries 16 --train-query-sampling uniform --out datasets/eval_lookup_single_keylast_uq16
git status --short datasets
```

`git status --short datasets` must print nothing. Output there means the regenerated data differs from
the committed metadata: stop and report it.

## 6. Smoke test and speed

This run is known to behave a particular way. It trains a tiny transformer on key-value recall, freshly
sampled at every step:

```bash
python scripts/sanity_recall.py --layout pairs --pairs 8 --queries 8 --steps 10000 --log-every 1000
```

- **Expected.** Query accuracy sits around 0.60-0.66 for the first few thousand steps, then jumps to
  1.000 and stays there.
- **What the laptop did.** It jumped between steps 4,000 and 5,000. GPU and CPU differ in random-number
  details, so the jump may come somewhat earlier or later. It should still come, within the 10,000
  steps.
- **Report.** Every logged line, and the total wall-clock time.

Then measure speed on the CPU and on the GPU, 1,000 steps each, and report both times:

```bash
python scripts/sanity_recall.py --layout pairs --pairs 8 --queries 8 --steps 1000 --log-every 1000 --device cpu
python scripts/sanity_recall.py --layout pairs --pairs 8 --queries 8 --steps 1000 --log-every 1000
```

Time them with `time` on Linux. In PowerShell, wrap each one in
`Measure-Command { <command> | Out-Default }`.

## 7. Report back

Give the user one block they can paste back into the other session, with:

1. the machine: OS, CPU, RAM, GPU;
2. the PyTorch version, the backend (ROCm/HIP version, or CPU only), and any environment variables you
   needed;
3. the commit hash;
4. the last line of the test run;
5. the output of `git status --short datasets`;
6. the smoke test's logged lines and wall time;
7. the two 1,000-step timings;
8. anything that failed or needed a workaround, with exact error messages.

## 8. Experiment queue (only when the user says to run it)

These are pre-registered in `experiments/plan_formula_evaluation.md`, in the section "Uniform training
queries, three seeds". Run them exactly as written, saving each output to `logs/`.

**Part 1: four runs, each up to 31,000 steps.**

```bash
python scripts/sanity_recall.py --layout table_keylast --pairs 21 --queries 8 --steps 31000 --log-every 3100 --fixed 50000 --schedule cosine --seed 2
python scripts/sanity_recall.py --layout table_keylast --pairs 21 --queries 16 --steps 31000 --log-every 3100 --fixed 50000 --schedule cosine --seed 0
python scripts/sanity_recall.py --layout table_keylast --pairs 21 --queries 16 --steps 31000 --log-every 3100 --fixed 50000 --schedule cosine --seed 1
python scripts/sanity_recall.py --layout table_keylast --pairs 21 --queries 16 --steps 31000 --log-every 3100 --fixed 50000 --schedule cosine --seed 2
```

**Part 2: three training runs of 40 epochs each.** Run this for seeds 1, 2 and 3:

```bash
SEED=1   # then 2, then 3
NAME=single_uq8_s$SEED
D=datasets/eval_lookup_single_keylast_uq8
python training/train.py --model-size tiny --batch-size 64 --learning-rate 3e-4 --warmup-steps 500 --num-epochs 40 --seq-format direct --seed $SEED --num-workers 0 --train-data $D/train.json --val-data $D/val.json --checkpoint-dir runs/$NAME
python scripts/assess_eval.py --data $D --splits test_iid --no-baselines --checkpoint "$NAME=runs/$NAME/best_model.pt:direct" --json $NAME.json
python -c "import json, torch; n='$NAME'; r=json.load(open(n+'.json'))[n]['test_iid']; ck=torch.load('runs/'+n+'/checkpoint_epoch_40.pt', map_location='cpu', weights_only=False); print(n, 'acc', round(100*r['accuracy'],1), 'pairs', round(100*r['pair_accuracy'],1), 'by objects', {k: round(100*v,1) for k,v in r['by_objects'].items()}); print('  val', [round(x,3) for x in ck['history']['val_loss']])"
```

In PowerShell, set the three variables with `$SEED = 1` and the like. Use them as `$SEED`, `$NAME` and
`$D`, with paths quoted, for example `"runs/$NAME"`.

Report, for every run:

- Part 1: every logged line;
- Part 2: the final two lines, `acc ... pairs ...` and `val [...]`.

Keep `--num-workers 0`, so the data order matches the laptop's runs.
