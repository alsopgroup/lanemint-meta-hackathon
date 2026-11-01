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

# Held-out eval: tools_frontier, seeds 42/1001/1002 (90 tasks, 2026-11-01)
These are the same harness and settings as the Oct 29 held-out eval: python tools, a 12-turn cap and low reasoning effort. V3 is the ep2 serverless RL sampler. All arms are re-scored with the strict local scorer (`compare_heldout_v3.py`), and pass counts match the Oct 29 report.
**Turn-limit definition:** a task counts as a hit if it was forced to give a final answer or used ≥12 turns. The forced-only counts (the Oct 29 REPORT.md definition) are in brackets.

| Arm | Pass /90 | Turn-limit hits | Empty finals | T1 | T2 | T3 | T4 | T5 |
|---|---|---|---|---|---|---|---|---|
| Muse Spark 1.3 | 63 | 50 [42] | 0 | 18 | 10 | 11 | 18 | 6 |
| Muse Spark Open | 51 | 10 [8] | 0 | 18 | 15 | 5 | 12 | 1 |
| Glimmer base | 27 | 64 [62] | 50 | 14 | 1 | 0 | 12 | 0 |
| **Glimmer V3 (ep2)** | **26** | 23 [21] | 0 | 12 | 5 | 0 | 9 | 0 |
(T1 convention_risk, T2 fx_calendar, T3 reconciliation, T4 split_holdings, T5 portfolio_construction; 18 each)

Per seed (pass/30, turn-limit hits, empty finals): base 9/9/9 (20/22/22, 17/16/17); V3 5/11/10 (7/10/6, 0/0/0).

**Infra errors:** 4 seed-1001 tasks (3003, 3007, 3008, 3009) failed in round 1 with serverless sampler HTTP 503s. All 4 were re-run successfully in round 2. The final 90 records contain 0 errors, so no fail in the table above is an infra error.
**Billing:** total $39.3063 → $39.3063 (Δ $0.00). Sampler tokens +7.29M at $0/unit. The guard (stop if the total rose by more than $0.50) checked 8 times and never tripped.

**Conclusion:** V3 ties base on held-out data (26 vs 27). It no longer leaves the final answer empty (50 → 0) and hits the turn limit far less (64 → 23). It gains on fx_calendar (+4) and loses on convention_risk (−2) and split_holdings (−3). It still scores 0 on reconciliation and portfolio_construction. The +3 on sanity seed 5999 did not hold up on held-out data. **Do not merge.**
