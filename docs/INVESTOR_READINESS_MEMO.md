# Investor presentation — safety-first LLM routing

## 1 — The problem

Sending every prompt to the largest model protects quality but wastes time and
compute when a smaller model could answer just as well. A useful router must
find those easy prompts without hiding the mistakes it creates.

The first customer is a high-volume AI platform that operates several model
tiers behind one API. The buyer is likely a head of AI platform, inference
engineering, or AI FinOps.

## 2 — The product

ModernBERT reads the prompt and estimates whether each smaller Qwen tier will
preserve the recorded quality of the trusted 7B fallback. A deterministic
policy chooses the analytically fastest eligible model. When confidence is too
low, it fails closed to the fallback.

For example, if the 1.5B and 3B tiers have safety probabilities 0.91 and 0.95
and the frozen threshold is 0.87, both are eligible; the selector sends the
prompt to the faster 1.5B tier. If they score 0.82 and 0.84, it uses 7B.

## 3 — What the completed experiment showed

Notebook 03 seed 42 is the latest completed evidence. On 1,726 sealed-test
prompts it routed 636 prompts, or 36.85%, to smaller tiers. The router produced
748 correct answers versus 740 for always using the fallback: eight more net
answers.

| Investor metric | Completed seed-42 result |
|---|---:|
| Aggregate quality-retention lower bound | 99.19% |
| Harmful routes | 32 / 1,726 = 1.85% |
| Routed-safety precision lower bound | 93.34% |
| Analytical savings at 20 ms overhead | 28.24% |
| Final predeclared gate | **Failed** |

The failure matters. One task, `bbh_object_counting`, supplied +10 net answers,
more than the experiment's overall +8. Without it, the other tasks were -2.
The macro-dataset quality gate therefore rejected a result that looked good in
aggregate. This is why the project is ready for funded validation, not a claim
of proven production savings.

## 4 — What v4 changes

Notebook 04 is a new, unrun experiment contract. Validation in notebook 03
favored the direct safety loss, so v4 trains only that objective for all five
epochs. It prints the active loss for every optimizer mini-batch and restores
the epoch with the lowest validation safety loss. The
oracle head and loss remain in the code for artifact compatibility, but their
coefficient is zero and they cannot affect training.

Numerically, if safety loss is 0.223 and the oracle diagnostic is 0.477, v4
optimizes $0.223+0\times0.477=0.223$. If epoch 3 reaches validation loss 0.181
and epoch 5 ends at 0.196, the exported router uses epoch 3. A noisy step loss
does not select the checkpoint; validation is compared after each epoch.

V4 defaults to a dataset-OOD split: complete tasks are held out from training.
That is the investor-relevant question—does routing work on a domain it has not
seen? V4 has no results yet; the notebook is intentionally output-free until a
reconstructable run artifact is produced.

## 5 — The four OOD charts investors should inspect

1. **Training and validation safety loss.** Shows whether learning is stable and
   marks the retained epoch. A widening gap warns about overfitting.
2. **Validation routing versus conservative savings frontier.** Shows the real
   tradeoff behind the chosen threshold instead of one favorable point.
3. **Held-out-domain quality retention versus routed fraction.** Each bubble is
   an unseen dataset; the dangerous quadrant is high routing with quality below
   the 98% reference line.
4. **Model allocation and harmful routes.** Shows how often 1.5B, 3B, and 7B are
   selected and where routing actually lost a correct fallback answer.

The raw domain values are exported with the dashboard so a polished chart can
always be audited against exact numbers.

## 6 — Investment decision

The investable asset is the safety and evidence system: pinned per-example
quality, calibrated fallback-relative predictions, validation-only policy
selection, sealed testing, explicit harm bounds, OOD views, and fail-closed
deployment. Funding should complete three OOD seeds and three random seeds,
measure full candidate serving cost on target hardware, and run a paid
design-partner pilot.

Candidate Qwen latency is still analytical, not measured. Therefore the honest
statement is:

> The first sealed run found promising routing and modeled economics but failed
> a cross-dataset safety gate. V4 now tests the simpler safety-only router under
> domain shift before we claim production savings or scale deployment.

The acceptable funded outcomes are a general router, a domain-specific router,
or a negative result that safely stays on the fallback.
