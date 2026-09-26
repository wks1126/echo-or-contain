# Annotation addendum (round 3) — suggestion 2/6/8

Extends coding_scheme.md with two truth-value tasks and one method-design note. These need human truth labels or an external synonym resource; no automated numbers are claimed here.

## Task C — avoidance-truth labels (for dictionary validity; suggestion 8)
Purpose: report precision/recall/F1 of the automatic deflection dictionary used for the "avoidance" flag.
- For the ~150 annotation_sample.csv rows, add one column **C_avoid_true**: 1 = fully evasive (no information / refers to another authority / "no comment"), 2 = partially evasive (some substantive content but dodges the core), 3 = not evasive.
- Compute vs the automatic `avoid` flag: precision, recall, F1; and re-run the RQ3 MixedLM using C_avoid_true as an ordinal/contrast predictor to check the "semantic-only disengagement" claim under better labels.
- Cost: second rater column on the same 150 rows (merge with the existing double-coding pass).

## Task D — echo truth labels (external validity; suggestion 2)
Purpose: validate that automated semantic/lexical alignment matches analysts' "echo vs answer-around vs dodge".
- Existing B_echo (1=echo, 2=answer-around, 3=dodge) is re-used as the truth here.
- Analysis to run after coding: (i) mean cos & mirroring by B_echo; (ii) multinomial/ordinal model of B_echo on (cos, mirroring) → report AUC or rank correlation; expected: dodge lowest cos, echo highest; mirroring less discriminative (semantic-only disengagement).
- Criterion to report in the paper: agreement between automated alignment and human echo judgement (e.g., Spearman ρ or 3-class AUC).

## Method note — synonym-rewording sensitivity (suggestion 6)
Question: officials sometimes respond by paraphrasing the question with different words; cosine may undercount such "same-sense, different-word" convergence, and over-crediting the CN–WEST gap if one side paraphrases more.
- Honest status: a proper same-sense layer needs a Chinese synonym/word-embedding resource (e.g., Hownet/同义词林 or subword-alignment via a token embedder) which is not installed here.
- What we did instead: (a) the second-model (text2vec) replication already shows the gradient holds under a different embedding space; (b) net-mirroring after removing generic topic words keeps the gradient. These bound, but do not fully answer, the paraphrase concern.
- Recommended next (needs resource/annotation): add a "core-predicate paraphrase" rate (does the answer reuse the question's main verb/relation under a near-synonym) and re-test whether the gradient persists when aligning on meaning-carrying predicates rather than surface words.
- Do NOT claim this layer in the paper until implemented.
