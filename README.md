# Reproduction package — *Echo or Contain?*

This archive accompanies the manuscript "Echo or Contain? A Corpus Measure of Lexical and
Semantic Alignment Between Journalists and Spokespersons in Chinese Foreign Ministry Press
Briefings". It contains everything needed to reproduce the numbers in Tables 1–4, Figures 1–5,
the reliability statistics of Section 3.2 and the multiple-comparison check reported in
Section 3.4 of the revised manuscript.

## Contents

```
code/     analysis scripts, in pipeline order (see "How to reproduce")
coding/   the dual-coding manual, the coding scheme and the reliability report
data/     the parsed corpus and the derived feature table used by every model
dictionaries.json  all feature dictionaries, extracted from the code by extract_dictionaries.py
extract_dictionaries.py  regenerates dictionaries.json from code/
revision_checks_report.md  output of code/revision_checks.py (reliability CIs, FDR check)
```

Data files

- `data/corpus_qa.csv` — 4,005 question–answer pairs parsed from the public transcripts
  (`id, file, date, qa, agency, who, q_text, a_text, is_followup, rep`).
- `data/features_all.json`, `data/features_all.csv` — the per-pair feature table on which every
  model is fitted (18 columns: origin, topic flags, avoidance, lengths, embedding cosine, lexical
  echo, Jaccard, meeting file, spokesperson, time window).
- `data/coding_validation_samples.csv` — the 92-pair dual-coding validation set with the machine
  features and both coders' ratings.

## Environment

Python 3.10+ with `jieba`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels` and
`sentence-transformers`. Every script sets `sys.stdout.reconfigure(encoding='utf-8')` and is
run from the directory that holds its inputs.

Embedding models: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (main analysis)
and `BAAI/bge-small-zh-v1.5` (replication with a Chinese-optimised model).

## How to reproduce

| Step | Script | Produces |
|---|---|---|
| 1. Parse transcripts | `code/parse_corpus.py` | `corpus_qa.csv` |
| 2. Lexical features | `code/features.py` | `features.csv` (token counts, Jaccard, lexical echo, avoidance) |
| 3. Embeddings | `code/embed2_retry.py` | `embeddings_*.npz` (question and answer vectors) |
| 4. Assemble features | `code/build_all.py` | `features_all.json` (the table in `data/`) |
| 5. Shuffled baselines | `code/baseline_all_calc.py`, `code/baseline_strict.py` | observed-vs-baseline contrasts (Table 2) |
| 6. Models | `code/stats_all.py`, `code/reg_control.py` | mixed-effects and clustered-OLS estimates (Table 4) |
| 7. Reliability | `code/reliability.py` | Cohen's κ and the construct-validity correlations (§3.2) |
| 8. Revision checks | `code/revision_checks.py` | bootstrap confidence intervals for κ and the Benjamini–Hochberg FDR check (§3.4) |
| 9. Figures | `code/make_figs2.py` | Figures 1–5 |
| 10. Dictionaries | `extract_dictionaries.py` | `dictionaries.json` |

Random seeds are fixed and recorded in the scripts; `revision_checks.py` uses seed `20260926` for
the bootstrap. Because the transcript pages are public but are re-published under stable URLs, the
raw HTML is not redistributed; `parse_corpus.py` documents the URL pattern and the retrieval date
(September 2026) so that step 1 can be rerun from `mfa.gov.cn` and `state.gov`.

The embedding matrices (`embeddings_*.npz`, 12–19 MB each) are not included because they are
derivable from step 3; `data/features_all.*` already contains their cosine similarities, so every
model and figure in the paper can be reproduced without re-encoding the corpus.

## Diagnostics a reader may want

- The estimated between-meeting variance in the mixed models is close to zero, so the mixed models
  behave like the clustered-OLS models in `code/reg_control.py`; both return the same coefficients
  to four decimals. `revision_checks.py` prints a `ConvergenceWarning` for the same reason. The
  random intercept is retained in the reported models because pairs are nested in meetings by
  construction (1–17 pairs per meeting, median 8).
- The avoidance dictionary was validated against the coders' function ratings on the 90 pairs in
  which both coders agreed: precision 0.97, recall 0.93, F1 0.95 (`code/m3b_metrics.py`).

## Licence and citation

Code and derived data are released under CC BY 4.0 (see `LICENSE.md`). The underlying transcripts
remain the property of their publishers and are subject to their own terms; this archive
redistributes only the question and answer text needed to verify the analysis.

If you use this package, please cite the article (citation details and the repository URL, release
v1.0.0, are in `CITATION.cff`; the package is published at
<https://github.com/wks1126/echo-or-contain>) and the embedding models as credited above.
