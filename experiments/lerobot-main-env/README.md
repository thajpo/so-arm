# Isolated LeRobot `main` Test Environment

This directory is for a **throwaway, isolated environment** to test whether `ThaJpo/hf_act_recordpolicy2` works with a LeRobot install from GitHub `main`, without disturbing the main `so-arm` project environment.

## Goal

Validate this hypothesis:

- the policy was trained against a newer LeRobot source version from `main`
- the main project environment is on an older published package format
- loading/inference should work in a source-installed LeRobot environment that more closely matches training

## Directory Layout

Recommended contents for this folder:

- `.venv/` — local virtual environment for this experiment
- `lerobot/` — cloned LeRobot source repo
- optional test scripts for policy loading/inference

## Setup

From the repo root:

```/dev/null/sh#L1-6
cd experiments/lerobot-main-env
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
brew install git-lfs
git lfs install
```

Then clone LeRobot:

```/dev/null/sh#L1-2
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

Install system/runtime dependencies as needed, then install LeRobot editable:

```/dev/null/sh#L1-2
pip install -e .
pip install "huggingface-hub>=1.5.0"
```

## Notes on `git-lfs`

If clone/install fails with missing LFS objects:

- confirm `git-lfs` is installed
- run `git lfs install`
- re-clone the repo after that

If the current `main` branch still fails because of a missing LFS object upstream, then use a specific commit known to work instead of current `main`.

## Suggested Validation Flow

1. Activate this local venv
2. Verify LeRobot imports from the cloned source
3. Try loading the model repo:
   - `ThaJpo/hf_act_recordpolicy2`
4. Confirm whether the checkpoint loads without the config/normalization issues seen in the main project env

## Sanity Checks

Check where LeRobot is importing from:

```/dev/null/sh#L1-1
python -c "import lerobot; print(getattr(lerobot, '__file__', 'no file'))"
```

Check version metadata if available:

```/dev/null/sh#L1-1
python -c "import lerobot; print(getattr(lerobot, '__version__', 'no __version__'))"
```

## Minimal Model Load Test

Once the environment is installed, a minimal smoke test is:

```/dev/null/test_load.py#L1-4
from lerobot.policies.act.modeling_act import ACTPolicy

policy = ACTPolicy.from_pretrained("ThaJpo/hf_act_recordpolicy2")
print(type(policy))
```

If this succeeds here but fails in the main project environment, that strongly confirms the mismatch is due to **published-package vs source-branch LeRobot version skew**.

## If This Works

If the checkpoint loads correctly here, the next step is to choose one of:

- keep policy inference isolated to this environment for now
- rebuild a fresh local runtime environment matching this source install
- identify the exact LeRobot commit used during training and pin to that commit for reproducibility

## If This Still Fails

If loading still fails here, capture:

- the exact traceback
- the LeRobot source commit
- the installed package list

That will tell you whether the issue is:
- still version mismatch
- broken current `main`
- or a different checkpoint/runtime incompatibility

## Recommendation

Treat this environment as disposable and focused:

- do not mix it with the main `so-arm` venv
- use it only to answer:  
  **“Does this policy run in a LeRobot source environment close to training?”**