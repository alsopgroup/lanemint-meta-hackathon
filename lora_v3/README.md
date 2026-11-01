# Glimmer LoRA V3 (branch lora-v3, experimental)
V1/V2 failed held-out tools_frontier (0/90) because their SFT rows were single-turn, data-in-prompt, no tool calls.
V3 trains on multi-turn tool trajectories from a teacher (Muse Spark Open), filtered to strict pass and <=10 turns.

- `run.py` - teacher/eval driver (seed guard: teacher 5101-5199, Glimmer 5999 only; forbids 42/1001/1002/2001-2040/5001/5002). Shared ledger with $44 hard stop + Fireworks billing guard; per-task results appended to disk, reruns skip finished tasks; 429s back off unbilled.
- `build_sft.py` - transcripts -> SFT rows (seed 5110 = validation).
- `v3data.py` - one example per assistant message, prefix == eval generation prompt, loss on that message only (tool/user/system masked).
- `train_v3.py` - serverless Glimmer LoRA (rank 32, alpha 64, LR 5e-5, 2 epochs, batch 8); aborts unless Glimmer training/sampler is $0 and no GPU-hours accrue.
- `billing.py` - read-only billing snapshot.
Tasks are regenerated with `python -m src.generate_tasks --tier tools_frontier --seed S --count 30`.
