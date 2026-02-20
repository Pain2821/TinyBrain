# TinyBrain / BitBrain Postmortem

## Outcome
Project closed as a non-viable product implementation in its current form.

## Why It Was Stopped
- No clear user-facing usability.
- Research claims were not validated with stable, repeatable evidence.
- Architecture and metrics changed frequently, reducing trust in conclusions.
- Too much code was generated quickly without enough grounded design review.

## Technical Failure Modes
- Routing behavior remained unstable and weakly engaged.
- Continual-learning gains were inconsistent across tasks/seeds.
- Metrics were improved late, after multiple structural changes.
- Threshold-heavy logic required heavy calibration and remained brittle.

## Process Failure Modes
- Vibe-coding velocity exceeded verification discipline.
- Specification drift (theory vs implementation) happened repeatedly.
- Product definition (who uses this, for what workflow) stayed unclear.

## What Is Still Reusable
- Dataset/task loading pipeline.
- Experiment runner patterns and JSON logging structure.
- Comparative benchmarking mindset (seeded runs, forgetting analysis).

## What To Avoid Next Time
- Building architecture before defining a strict success metric.
- Accepting generated code without explicit design checkpoints.
- Mixing MVP/product goals with open-ended research goals.

## If Restarting From Scratch
1. Pick one concrete use case with a measurable user outcome.
2. Define a baseline that must be beaten (simple fine-tuning model).
3. Freeze an evaluation protocol before coding.
4. Implement minimal version first; no architectural branching.
5. Add only one new mechanism per iteration with ablation proof.

## Immediate Closeout Actions
1. Archive this repo as `TinyBrain-archive`.
2. Keep logs and dataset manifests for reference.
3. Start a new repo with strict PRD + evaluation contract.
