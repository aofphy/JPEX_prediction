# Spell-check report

**Tool:** hunspell 1.7 with the `en` dictionary from https://github.com/wooorm/dictionaries
**Date:** 2026-05-28
**Files checked:** `main.tex` (4,981 words), `cover_letter.tex`

## Summary

✅ **No spelling errors found.** All flagged words are valid technical terms, proper nouns, acronyms, or British English variants used consistently throughout the manuscript.

## Categorisation of flagged words

### Proper nouns (people, places, institutions) — 31 words
Sriboriboon, Fongkaew, Suranaree, Nakhon, Ratchasima, Hirota, Marcjasz, Uniejewski, Weron, Lago, Nowotarski, Olivares, Diebold, Mariano, Newey, Newbold, Leybourne, Harvey, Hansen, Lunde, Nason, Winkler, Gibbs, Candès (parsed as "Cand"), Schutter, Lim, Grzegorz, Matsumoto, Fukushima, Ittipon, Panithan

### Place names — 4 words
Kansai, Tokyo (in headers), Nord (Pool), PJM, EPEX

### Acronyms / technical abbreviations — 27 words
JEPX, JPY, USD, DA, IM, RMSE, MAE, rRMSE, rMAE, CRPS, MCS, DM, HAC, CDF, SU, JSU, EPF, DDNN, MLP, KU, IJF, GEFCom, METI, CRediT, DOI, WN, M2

### British English (intentional) — 11 words
Conceptualisation, ensembling, organised, optimiser, parameterises, recalibration, regularisation, regularised, summarised, visualisation, visualises, miscoverage, reproducibility, licence

### Library / code names — 8 words
LightGBM, lightgbm, xgboost, XGBoost, scikit, statsmodels, Zenodo, Anthropic

### Standard ML / EPF technical vocabulary — 12 words
quantile, quantiles, regressors, lookback, backcast, skewness, intraset, intraday, softplus, subfield, exogenously, ensembling, preprint, AutoRegressive, HiTS, PatchTST

### LaTeX artefacts — 1 word
`0pt` (a TeX dimension specification that hunspell does not recognise)

### Genuine misspellings — 0 words

## Patterns checked

| Check | Result |
|-------|--------|
| Repeated consecutive words ("the the", "of of", etc.) | None |
| Common typos (teh, adn, recieve, definately, etc.) | None |
| Missing space after period before capital letter | None |
| Smart quotes vs ASCII apostrophes | All ASCII (correct for LaTeX) |
| Multiple-space artefacts in body text | None outside table alignment |

## Recommendation

The manuscript is **ready for submission** from a spell-check / typo perspective.

Two minor optional refinements (not errors):

1. The descriptive-statistics table (Table 1) lists prices in raw JPY/kWh implicitly but the table header could explicitly mark the unit. Currently: `Mean | Std | Min | Median | Max` → consider `Mean (JPY/kWh) | Std | ...`.
2. The grep search found two reflowed paragraphs (lines 41 and 43 of `main.tex`) that span 200+ words each. Reflowing into shorter paragraphs of 4--6 sentences would improve readability for reviewers, but is purely stylistic.

These are presentation suggestions rather than spell-check findings.
