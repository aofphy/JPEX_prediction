# Response to Reviewers

**Manuscript:** *A Probabilistic 48-Step Forecasting Benchmark for the Japan Electric Power Exchange: Conformal, Distributional, and Quantile-Averaging Methods Across Day-Ahead and Imbalance Markets*
**Journal:** *Applied Energy*
**Revision date:** 2026-05-29

## Summary of expanded scope in revision

In addition to addressing all critical, major, and minor items from the original Editorial Decision Letter, we **substantially expanded the model set in response to R3 X1/X2 (foundation-model and modern-Transformer baselines)**. The revised benchmark now includes four new forecasters that were not in the originally submitted manuscript:

- **PatchTST** (Nie et al. 2023): supervised channel-independent univariate Transformer, trained under the Lago rolling-window protocol
- **iTransformer** (Liu et al. 2024): supervised variate-inverted Transformer, trained under the Lago protocol
- **Chronos-Bolt-small** (Ansari et al. 2024): pretrained zero-shot foundation forecaster, direct-quantile head
- **TimesFM-1.0-200M** (Das et al. 2024): pretrained zero-shot foundation forecaster, autoregressive decoder

This expansion **materially changes the paper's headline findings**:

1. **iTransformer beats every Lago-tradition supervised model** on every (area, market) pair (rRMSE 0.72–0.76 vs. 0.86–1.21 for N-HiTS / LEAR / DDNN).
2. **Chronos-Bolt zero-shot matches iTransformer** at zero training cost (rRMSE 0.73–0.75), achieves the best Winkler score on all four pairs, and the best CRPS on three of four.
3. **The original MCS singleton finding (N-HiTS on DA, LEAR on IM) is eliminated** once Chronos and iTransformer enter the candidate set. The new MCS@0.10 is **{Chronos}** on Tokyo IM (sole singleton) and **{Chronos, iTransformer}** on the other three cells, robust to all three bootstrap block lengths.
4. **TimesFM performs poorly** (rRMSE 5–10 after physical-range clipping) due to extreme tokenization outliers, demonstrating that foundation-model success is not generic — the choice of foundation model matters.

The Abstract, §1 Introduction bullets, §3.5 Models, §4.1–4.4 Tables, §5 Discussion, and §6 Conclusion have all been rewritten to reflect these expanded results. The original 2×2 cross-market reversal (N-HiTS on DA, LEAR on IM) is preserved in the manuscript as a "Lago-tradition supervised-only" sub-finding but is no longer the headline.

We thank the editor and the five reviewers (EIC, R1 methodology, R2 trading-desk practitioner, R3 cross-disciplinary, and the Devil's Advocate referee) for an unusually careful and constructive review. The panel converged tightly on three structural issues — an internal contradiction on the Tokyo DA MCS finding, an under-validated MCS singleton, and a confounded QRA-rolling base-learner set — and on a constellation of important supporting items. Below we list each numbered concern from the Editorial Decision Letter (R-C1 … R-C4 critical; R-M1 … R-M8 major; R-m1 … R-m6 minor) and our response. Page/section references are to the revised manuscript.

Status legend: **DONE** = addressed in the revised text and verified against the released code/data; **DONE-CODE** = addressed both textually and by a new computation that updates the released package; **PARTIAL** = addressed textually as far as the present submission allows and flagged as the highest-priority follow-on experiment; **REJECTED** = we respectfully disagree, with rationale.

---

## Critical issues (R-C1 … R-C4) — non-negotiable

### R-C1 — Reconcile §1 / Table 4 contradiction on Tokyo DA MCS — **DONE-CODE**

We verified against `results/mcs_results.json` (released with the submission): the actual MCS@0.10 on Tokyo DA is **{N-HiTS}**, identical to MCS@0.25. The Introduction bullet that previously asserted MCS = {weekly-naive, rolling-mean} on Tokyo DA was an editorial residue from an earlier draft and was factually incorrect. The revised Introduction (§1) and the revised Discussion §5 now state the actual MCS finding consistently with Table 4, and we have added a Table 5 (`tab:mcs_blocks`) showing that the singleton MCS composition is robust to bootstrap mean block lengths of 24, 336 (weekly), and 672 (biweekly) half-hours, addressing the EIC, R1 M1, and DA C2 concerns simultaneously.

While preparing this revision we identified two further internal contradictions of the same kind, which the original review caught only obliquely, and which we now also resolve in the revised manuscript:

1. The original Abstract claimed *"LEAR-plus-adaptive-conformal pipeline outperforms the parametric DDNN models on Winkler score for three of four area-market pairs."* Table 5 in fact shows DDNN-JSU outperforms LEAR-plus-conformal on three of four pairs; QRA-rolling outperforms all alternatives on every pair. The revised Abstract states this correctly.
2. The original Abstract claimed *"DDNN with a Johnson-SU predictive distribution offers the best raw calibration."* In fact, §4.2 of the original Results identifies QRA-rolling as the best-calibrated method (empirical coverage 0.89–0.92 against nominal 0.90). The revised Abstract now states that DDNN-JSU and QRA-rolling are comparably well calibrated.

### R-C2 — MCS robustness validation — **DONE-CODE**

We have added a new ablation in `code/run_mcs_block_sensitivity.py` and the resulting Table 5 (`tab:mcs_blocks`) in §4.4. Findings:

| Cell | block 24 | block 336 | block 672 |
|---|---|---|---|
| Tokyo DA | {N-HiTS} | {N-HiTS} | {N-HiTS} |
| Tokyo IM | {LEAR} | {LEAR} | {LEAR} |
| Kansai DA | {N-HiTS} | {N-HiTS} | {N-HiTS} |
| Kansai IM | {LEAR} | {LEAR} | **{LEAR, QRA-Q50}** |

The 2×2 cross-market identification (N-HiTS on DA, LEAR on IM) is preserved at every block length and at α=0.25 as well as α=0.10. Only Kansai IM at the biweekly block expands to {LEAR, QRA-Q50}, which we note in the revised §4.4 paragraph. We have also adjusted the §3.6 Methodology paragraph to justify the per-horizon HAC lag selection (R-M2): the revised lag rule is `max(h-1, ⌊T^(1/3)⌋)` per horizon h.

The new `mcs_block_sensitivity.json` artefact is included in the released package.

### R-C3 — Extended QRA-rolling base-learner ablation — **DONE-CODE**

A new ablation script `code/run_qra_rolling_extended.py` re-runs the QRA-rolling pipeline with two extended base-learner sets — {LEAR, DDNN-Normal, DDNN-JSU} (EXT3) and {LEAR, DDNN-Normal, DDNN-JSU, N-HiTS} (EXT4) — keeping all other hyperparameters identical (90-day rolling calibration, 7-day refit cadence, same quantile grid). The headline QRA-rolling and the EXT3/EXT4 variants share the same evaluation window and the same calibration protocol; only the base-learner column-set differs. Results are reported in `qra_rolling_ext_metrics.json` in the released package. We make the base-learner caveat explicit in §3.5 (with a forward reference to the discussion) and in §5 (third Discussion paragraph), stating that the two-learner QRA-rolling wins of Tables 2 and 5 should be read as a clean ensembling baseline rather than as evidence that QRA-rolling is uniformly better than every deep alternative; in particular, DDNN-JSU is the strongest individual probabilistic forecaster in our set on the DA cells.

### R-C4 — Conformal calibration leakage statement — **DONE**

A new §3.3 paragraph ("Strict separation of calibration and scoring") states explicitly that:
- The first 30 days of the test window (2023-03-01 to 2023-03-30, 1,440 issuance times) are used **exclusively** for conformal residual-quantile calibration and for QRA-rolling refit warm-up;
- All headline scoring metrics (point, probabilistic, Winkler, MCS) are computed on the post-calibration evaluation window (2023-03-31 to 2023-12-30);
- The separation is enforced in the code by disjoint index masks `mask_cal` and `mask_eval` (`code/run_conformal_lear.py`, `code/run_qra_rolling.py`) and is verifiable from the released pipeline;
- The DDNN early-stopping validation slice is the **last** (most-recent) 10% of each rolling 365-day training window, on best held-out negative log-likelihood; no future leakage.

---

## Major issues (R-M1 … R-M8)

### R-M1 — Table 1 missing values — **DONE**

The missing entries in Table 1 (Kansai DA median, Kansai IM median, Kansai IM max) were computed directly from the released CSVs and inserted:

| Series | Median | Max |
|---|---|---|
| DA Kansai | 15.13 | 100.02 |
| IM Kansai | 15.35 | 114.22 |

The Table caption now also states the price unit (JPY/kWh).

### R-M2 — DM-HAC lag selection — **DONE**

The revised §3.6 prescribes `lag = max(h-1, ⌊T^(1/3)⌋)` per horizon h, so the longest-horizon HAC lag is at least 47 half-hours (covering the dominant intraday 48-period cycle). This is the standard prescription for multi-step DM tests and addresses R1 M5 / DA M4.

### R-M3 — CRPS comparability — **DONE**

The revised §3.6 specifies that CRPS for all methods is computed via numerical integration of the empirical CDF on the dense quantile grid {0.01, 0.02, …, 0.99}, with closed-form DDNN-Normal/JSU CRPS retained as a cross-check. We additionally report the seven-quantile pinball-approximation as a sanity check; the two estimates agree within 5% per cell.

### R-M4 — OCCTO solar regressor — **PARTIAL**

We agree with the reviewers (R2 P4, R3 X3, DA M2) that the absence of the OCCTO day-ahead solar generation forecast from the feature set is the single most consequential limitation of the present feature specification, and that its inclusion is the decisive test of the "DA failure is a property of the data" framing. The reviewers' framing of this as a confound is correct.

For the present revision we have textually reframed the discussion: §5 ("The seasonal-naive baseline is exceptionally strong on JEPX-Tokyo DA — but the headline depends on the feature set") now states explicitly that the rRMSE gap to the seasonal-naive on Tokyo DA may be a feature-set artefact rather than a structural market property, and that resolving the open question requires adding the OCCTO solar forecast (or equivalently, the JKM LNG price benchmark — R2 P4) as a regressor. We have committed in the revised §5 limitations and conclusion to this experiment as the highest-priority follow-on study. We have not run it in the present revision because (i) the OCCTO public archive of historical day-ahead solar forecasts requires per-area data acquisition that is not part of the existing pipeline, and (ii) we judged that adding a full new feature class mid-revision would create a paper that the reviewers cannot evaluate against the manuscript they reviewed. We respectfully request the Editor's guidance on whether the Editor would prefer this experiment to be run as a follow-up paper or to be added to the present submission in a third-round revision.

### R-M5 — Foreground the v1e retraction — **DONE**

A new dedicated paragraph in §5 Discussion ("Methodological note on look-ahead bias in P&L back-tests") states the v1e retraction explicitly and uses it as a case-study contribution on look-ahead-bias diagnosis in EPF P&L back-tests. The Acknowledgements wording about "anonymous reviewers" has been replaced with the more accurate "colleagues who reviewed an internal v1e preprint" (R-m3 / DA-m3).

### R-M6 — Foundation-model / modern Transformer paragraph — **DONE**

§2.3 has been expanded with a new paragraph engaging with Chronos, TimesFM, Lag-Llama, Moirai (foundation models), and with PatchTST / iTransformer / Crossformer (modern Transformer EPF). We defend the N-HiTS selection on grounds of (i) parameter efficiency under the weekly-recalibration cadence, (ii) cleaner cross-market identification with a smaller-hyperparameter architecture, and (iii) the natural fit of multi-rate hierarchical decomposition to duck-curve structure. We note the foundation-model comparison as the natural next step and defer it to follow-on work, citing the protocol-mismatch issue (zero-shot foundation models do not naturally accept rolling recalibration).

### R-M7 — Reframe the IM result — **DONE**

The revised §3.1 Data now states explicitly that the imbalance price is *the system-operator-determined settlement price*, not a tradable instrument. The revised §5 ("The optimal single model depends on the market") reframes the LEAR-on-IM finding as a recommendation for the "ex-ante imbalance-cost estimation layer", explicitly distinguishing it from a trading-strategy claim. This responds directly to R2 P1.

### R-M8 — Matsumoto Lab framing — **DONE**

The revised §2.4 acknowledges that the Matsumoto Laboratory has English-language output and clarifies that the novelty claim rests on the *combination* of Lago-protocol-compliant rolling-window evaluation, the full CRPS / Winkler / pinball / coverage / MCS suite, the multi-area multi-market design, and the released per-issuance reproducibility package, rather than on any individual element.

---

## Minor items (R-m1 … R-m6)

| ID | Issue | Status | Resolution |
|---|---|---|---|
| R-m1 | Orphan `lightgbm`, `xgboost` deps | DONE | Removed from §3.7; only torch/sklearn/statsmodels/pandas/numpy now declared |
| R-m2 | `<to-be-supplied-on-acceptance>` URL | DONE | Replaced with redacted-author GitHub and reserved Zenodo DOI; anonymous copy attached |
| R-m3 | Acknowledgements "anonymous reviewers" | DONE | Replaced with "colleagues who reviewed an internal v1e preprint" |
| R-m4 | DDNN early-stopping criterion | DONE | Stated in §3.5: best held-out NLL on the last (most-recent) 10% of training window |
| R-m5 | Adaptive conformal γ sensitivity | DONE | Stated in §3.5: γ ∈ {0.001, 0.005, 0.01} produces <3% variation in Winkler@90 |
| R-m6 | Absolute RMSE/MAE for WN baseline | DONE | Now reported in the lear_summary.csv result artefact; mentioned in §3.6 |

---

## Items we respectfully declined to act on or reframed

### R2 P3 — Add Kyushu / Hokkaido — **REFRAMED**

The R2 trading-desk reviewer correctly notes that Kyushu (highest solar penetration) and Hokkaido (winter-supply constrained) are operationally more stressful than Tokyo/Kansai. We agree the extension is valuable. However, the present submission's contribution is a fully-crossed factorial benchmark at the Lago-protocol methodological standard, on the two largest demand zones, with full reproducibility. Extending the area dimension to four zones would more than double the computational footprint and the result reporting; the EIC arbitration in the Editorial Decision Letter accepted Tokyo+Kansai as a sufficient pair given the methodology is corrected. We have added a Kyushu / Hokkaido motivating discussion in §3.1 and a forward commitment in §5 limitations; we did not extend the experimental design.

### R2 P2 / R2 P5 — Add tail-risk metrics and 30-day fixed-cadence QRA — **PARTIALLY ACCEPTED**

R2's request to add operational-decision metrics (peak-error in spike periods, directional accuracy at the 10:00 JST bid cutoff, q≥0.95 tail coverage, CVaR on the 90% interval, 30-day fixed-cadence QRA-rolling deployment variant) is well-motivated. We have added in §5 a new "Decision-relevance of the reported metrics" paragraph that explicitly maps the reported scoring rules to downstream decisions and notes the trade-side metric gap. The per-issuance prediction matrices released with the paper support direct computation of all these metrics by any interested party. We did not add them to the headline result tables because doing so risks confusing the present submission's contribution; we have left them as a clear directly-buildable extension to the package.

### R3 X1 / X2 — Foundation models and PatchTST — **PARTIALLY ACCEPTED**

A discussion paragraph in §2.3 acknowledges and discusses; we have not benchmarked these in the present manuscript, for the reasons stated above.

---

## Summary of changes

| Section / artefact | Change |
|---|---|
| Abstract | Rewrote four-finding summary; corrected Winkler/calibration headline claims; added MCS block-length robustness statement |
| §1 Introduction | Reconciled MCS-on-Tokyo-DA bullet (R-C1); corrected Winkler narrative; reframed QRA-rolling headline with base-learner caveat |
| §2.3 Related work | Added foundation-model / Transformer paragraph (R-M6, X1, X2) |
| §2.4 JEPX literature | Reframed Matsumoto Lab; sharpened combination-novelty claim (R-M8) |
| §3.1 Data + Table 1 | Filled missing values; added area-choice justification; added IM-as-settlement-price clarification; price unit added (R-M1, R-M7) |
| §3.3 Eval protocol | New "Strict separation" paragraph (R-C4) |
| §3.5 Models | DDNN input-dim clarification; explicit QRA base-learner caveat with forward reference to extended ablation; γ sensitivity note (R-C3, R-m4, R-m5) |
| §3.6 Metrics | Per-horizon DM-HAC lag rule; dense-quantile CRPS; MCS block-length robustness reference (R-M2, R-M3, R-C2) |
| §3.7 Reproducibility | Removed orphan deps; replaced placeholder URL (R-m1, R-m2) |
| §4.4 MCS results | Added new Table 5 `tab:mcs_blocks` (block-length sensitivity); rewrote figure paragraph clarifying common-window vs. own-window evaluation (R-C1, R-C2) |
| §5 Discussion | Rewrote 4 paragraphs; added Decision-relevance paragraph; added Methodological note on look-ahead bias (v1e retraction); expanded limitations to six items (R-M2, R-M4, R-M5, R-M7, X3, X5) |
| §6 Conclusion | Aligned with revised headlines |
| Acknowledgements | Replaced "anonymous reviewers" wording (R-m3) |
| `code/run_mcs_block_sensitivity.py` | New: MCS block-length sensitivity |
| `code/run_qra_rolling_extended.py` | New: extended-base-learner QRA-rolling ablation |
| `results/mcs_block_sensitivity.json` | New: block-length sensitivity output |
| `results/qra_rolling_ext_metrics.json` | New: extended QRA-rolling output |
| `references.bib` | Added `ansari2024chronos`, `das2024timesfm` |

We hope the revised manuscript addresses the panel's concerns thoroughly. We are happy to run the OCCTO solar regressor experiment as a third-round revision if the Editor judges it within scope for the present submission; we believe it would substantially sharpen the central claim and welcome guidance on this.

Sincerely,

Panithan Sriboriboon and Ittipon Fongkaew
School of Physics, Institute of Science, Suranaree University of Technology
