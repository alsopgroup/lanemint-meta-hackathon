# Hugging Face release prep

**Status: published.** Dataset home:
https://huggingface.co/datasets/LaneMint/lanemint-meta-benchmark

## What gets uploaded

`scripts/export_hf_dataset.py` builds a frozen public task set from
NEW public seeds (default 5001, 5002), covering the easy, hard and
frontier tiers, with task CSVs, the generated tasks' own answer keys,
a `manifest.json` (seeds, counts, SHA-256 per file), and a dataset
card (`README.md` with YAML metadata: the repo's license position —
all rights reserved by LaneMint, no open-source license attached yet —
task categories, and tags including `meta-ai`, `muse-spark`,
`finance`, `benchmark`).

It refuses the held-out evaluation seeds (42, 1001, 1002) and the
planned training seeds (2001–2040). **No held-out evaluation tasks or
answer keys are included** — nothing from an eval generation is read.

## Regenerate and check

```bash
pip install -r requirements.txt
python scripts/export_hf_dataset.py            # writes ./hf_dataset/
cat hf_dataset/manifest.json                   # seeds, counts, hashes
du -sh hf_dataset                              # expect roughly 1 MB at defaults
```

## Upload (human, with your own token)

Never commit a token. Log in interactively, then:

```bash
pip install -U huggingface_hub
hf auth login            # legacy equivalent: huggingface-cli login

 # create the (empty) dataset repo, then upload the folder:
hf upload LaneMint/lanemint-meta-benchmark hf_dataset . --repo-type dataset

# Legacy CLI equivalent for the upload step:
# huggingface-cli upload LaneMint/lanemint-meta-benchmark hf_dataset . --repo-type dataset
```


## LoRA adapter weights: what exists in this repo

Checked 2026-10-31: **there are no downloadable adapter weights in
this repository** — no `*.safetensors`, `adapter_model*`, `*.bin`, or
checkpoint files anywhere. The Glimmer LoRA work exists as:

- the seed-disjoint SFT data builder (`src/finetune_prep.py`) and the
  validated, never-auto-executed Fireworks plan printer
  (`src/finetune.py`, base model `accounts/fireworks/models/muse-glimmer-30b`,
  output model names `muse-glimmer-30b-lora-answer` / `-trace`);
- the reported LoRA results in the README / `app/data/leaderboard.json`.

The tuned models themselves lived on Fireworks under
`accounts/<your-account>/models/muse-glimmer-30b-lora-<target>` (see
`docs/BENCHMARK.md`). Per the release instructions, no Fireworks or
other paid API was called to fetch them; if weights are ever released,
export them from Fireworks with your own credentials and add them as
a separate model repo — do not add them to this dataset folder.
