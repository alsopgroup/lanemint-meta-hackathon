# Glimmer LoRA V3: sanity check on seed 5999 (2026-11-01)

**What it is:** a LoRA on muse-glimmer-30b, trained with serverless SFT on strict-pass teacher (Spark Open, tools-on) trajectories from training seeds 5101–5109. The model only saw trajectories with ≤10 turns and no forced final answer. Validation used seed 5110. Seed 5999 was never used in training.
Training: 3 attempts. Attempt 1 exited 143 (timeout/kill), attempt 2 exited 2, and attempt 3 resumed and finished at 08:21 MST. Val loss: ep0 0.337 → ep1 0.161 → ep2 0.155. The eval used the ep2 sampler.

## Results: seed 5999, 30 tasks, tools-on, max_turns 12, local scorer
| | Base | V3 |
|---|---|---|
| Strict passes | **7/30** | **10/30** |
| Turn-limit hits (≥12 turns) | 21 (20 forced finals) | 9 (9 forced) |
| Empty final responses | 18 | 0 |
| Total turns | 359 | 265 |
| Errors | 0 | 0 |

| Task type (n=6 each) | Base | V3 |
|---|---|---|
| convention_risk | 4 | 3 |
| fx_calendar | 0 | 4 |
| portfolio_construction | 0 | 0 |
| reconciliation | 0 | 0 |
| split_holdings | 3 | 3 |

## Billing
`billing/train_baseline.json` (before training) vs `billing/after_train.json` (after the V3 eval, read-only GET): the Oct–Nov total stayed at **$39.3063 → $39.3063 (Δ $0.00)**. Serverless training tokens rose by 10.29M at $0/unit. RL-sampler tokens rose by 2.58M at $0.

## Interpretation
- V3 mostly fixes failures to finish: empty finals went from 18 to 0 and turn-limit hits from 21 to 9. The gain in passes is small (+3) and comes entirely from fx_calendar (0→4). convention_risk is down by 1, which is within noise.
- portfolio_construction and reconciliation are still 0/6. The model now finishes these tasks but gets them wrong, so this is a capability gap, not a format problem.
- n=30 on one seed. 7 vs 10 is not statistically significant: the Wilson 95% intervals overlap a lot (~12–41% vs ~19–51%).
