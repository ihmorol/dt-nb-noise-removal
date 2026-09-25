# Paper outline (journal article)

## Target venues, ranked for this project + solo author

1. **Knowledge-Based Systems** (Elsevier) — strong fit: interpretable hybrids, weighting mechanisms, and this paper family (Hall 2007, Wang et al. 2006 both live here). Realistic top target.
2. **Expert Systems With Applications** (Elsevier) — the seed's venue; needs a clear applied angle + strong experiment depth. Submit here only with Phase 3 fully done.
3. **Pattern Recognition Letters** — compact method+results paper if the leakage audit carries the paper.
4. **Applied Intelligence** (Springer) — solid mid-tier fallback, method-comparison papers are common.
5. Conference warm-up (optional): ICCIT / IEEE SSCI — fast feedback; the Farid group publishes there, useful for visibility with the teacher.

## Section-by-section claims (write results-first: 4 → 5 → 6 → 2 → 3 → 1 → 7 → abstract)

1. **Introduction** — 3 paragraphs: (i) DT and NB are complementary; hybrids exist but are one-directional; (ii) the seed paper's two hybrids were never combined, its filter equates disagreement with noise, and its protocol invites CV leakage; (iii) we deliver the mutual hybrid, the audit, and confidence-aware handling with honest statistics.
2. **Related work** — three subsections mirroring the review: DT–NB architectures; instance filtering; attribute weighting for NB. MUST include Hall 2007, John 1995, Brodley & Friedl 1999, Zaidi 2013, Wong 2020.
3. **Background** — Alg. 1 + Alg. 2 restated precisely (from the PDF, not paraphrase), with the four critiques we test.
4. **Methods** — E4 mutual hybrid (one diagram: filter→tree→weights→weighted NB); E5 filter variants; nested-CV protocol definition.
5. **Setup** — 10 datasets (Table 5), 10-fold CV ×10 seeds, sklearn config, baselines (C4.5-equiv, NB, Hall-style, Wong-style sketch), statistical tests.
6. **Results** — 6.1 leakage audit (the honest hook); 6.2 mutual hybrid vs one-directional; 6.3 confidence-aware vs hard delete; 6.4 deletion diagnostics (the figure nobody else has).
7. **Discussion/limitations** — CART vs C4.5; dataset age; scope of one-filter-family claims; what the audit does NOT prove.
8. **Reproducibility statement** — code + configs released (this materially helps acceptance).

## Word budget (KBS-style ~9–11k words)

Intro 900 · Related 1,300 · Background 900 · Methods 1,800 · Setup 900 · Results 2,500 · Discussion 900 · everything else ~800.
