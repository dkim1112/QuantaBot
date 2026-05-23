# QuantaBot — Research Roadmap

Future directions for making QuantaBot's retrieval genuinely paper-worthy in the
**clinical context** domain (not necessarily multiple-choice).

This document was written when the current architecture was a standard
RAG stack: ensemble (BM25 + MMR semantic) + parent-document retrieval + cross-encoder
reranking + LLM-based multi-query expansion. Solid engineering, but no
novel research contribution.

---

## Where we are (as of 2026-05-23)

**Architecture**

- Embeddings: `all-mpnet-base-v2` (general-purpose, MTEB-mid)
- Retrieval: BM25 (k=6) + MMR semantic (k=8), ensemble weights 0.7 / 0.3
- Reranker: `BAAI/bge-reranker-v2-m3` (cross-encoder, top_n=12)
- Parent-doc retrieval: 2000-char parents → 600-char children
- Multi-query: 2 LLM-generated alt queries per question
- LLM: GPT-4 (via OpenAI API)

**MedQA baseline (n=20, USMLE Step 1 style)**

| Configuration | Letter accuracy | Notes |
|---|---|---|
| Ensemble only (no reranker) | 60.0% (12/20) | Baseline |
| Ensemble + BGE-reranker-v2-m3 | 65.0% (13/20) | +1 question — within noise on n=20 |

**Reference points (published)**

| System | MedQA |
|---|---|
| Random guess (5-choice) | 20% |
| USMLE passing threshold | ~60–65% |
| GPT-3.5 zero-shot | ~50% |
| GPT-4 zero-shot | ~80% |
| MedRAG + GPT-4 (published 2024) | ~85–90% |
| Med-PaLM 2 | ~86% |

**Honest interpretation:** Quanta is around USMLE pass level but plausibly **below
GPT-4 zero-shot**. The retrieval layer might be a net negative right now. We have
not yet confirmed Quanta's RAG actually helps over base LLM.

**The single experiment that resolves this:** Run GPT-4 alone (no retrieval) on the same
test set. If GPT-4 alone scores higher than Quanta, the retrieval pipeline needs
rethinking before adding more on top.

---

## Research directions

Below are concrete directions ordered by novelty + applicability to general clinical
context (not just multiple choice). Each notes paper-worthiness and rough effort.

### 1. Structured clinical decomposition (high priority, broadly applicable)

**Idea:** Clinical questions are multi-fact stories. Decompose the input into
structured clinical tuples — `(chief complaint, history, exam findings, labs,
imaging, prior treatment)` — using an LLM step. Retrieve for each component.
Aggregate.

This is **not** "multi-query rephrasing" — it's structured decomposition informed
by how clinicians actually parse cases. Reuses clinical reasoning ontology
(SNOMED-CT, UMLS) for term normalization.

**Why novel:** Most medical RAG papers use whole-question retrieval. Structured
clinical-tuple decomposition with ontology-grounded normalization isn't a clean
published ablation I'm aware of.

**Paper-worthiness:** High if combined with quantitative ablation on the
contribution of each tuple component.

**Effort:** Medium. ~1 week to prototype, ~2 weeks for clean ablation.

### 2. Iterative confidence-driven retrieval (high priority, broadly applicable)

**Idea:** First-pass RAG produces an answer + LLM-stated confidence + identified
knowledge gaps. If confidence below threshold, query LLM: "What specifically would
you need to know to be more confident?" Use that as the second retrieval query.

Related to FLARE / Self-RAG but with clinical-reasoning specialization (and
specifically: the gap query is in clinical terms, not general).

**Why novel:** Clinical-reasoning-aware iterative retrieval hasn't been cleanly
isolated as a contribution in medical RAG papers as far as I've seen.

**Paper-worthiness:** Moderate-to-high. Iterative retrieval is well-studied
generally; the clinical specialization is the novelty.

**Effort:** Medium. ~1 week prototype. Adds 1.5–2× latency per question
(two retrieval rounds).

### 3. Structured evidence aggregation (medium priority, broadly applicable)

**Idea:** Instead of flat `context = chunk1 + chunk2 + ...`, the LLM is given
*structured* evidence:

```
For this case:
  Supporting: <chunks>
  Conflicting: <chunks>
  Tangential / context: <chunks>
```

Achieved via a small classification step (LLM or cross-encoder) over each
retrieved chunk. The final generation prompt makes explicit reasoning over the
structure.

**Why novel:** Structured-evidence prompting for medical RAG isn't widely
ablated.

**Paper-worthiness:** Lower as a standalone contribution; high when combined
with #1 or #2.

**Effort:** Low–medium. ~3–4 days prototype.

### 4. Domain-specific embeddings (table stakes, not novel)

Swap `all-mpnet-base-v2` → MedCPT, SapBERT, or BioLink-BERT. Plus
`BAAI/bge-reranker-v2-m3` → MedCPT cross-encoder.

**Why not novel:** Domain embeddings for medical retrieval are well-established.
But mandatory for a defensible medical-RAG paper — reviewers will ask.

**Effort:** Low. Few hours plus full re-embed (~15 min).

### 5. HyDE for clinical queries (low novelty, easy test)

Generate a hypothetical answer/explanation with the LLM, embed it, retrieve.
Standard technique; worth ablating but unlikely to be the contribution.

**Effort:** Low. ~1 day to add + run.

### 6. Multi-corpus federation (effort heavy, moderate novelty)

Current Quanta uses 7 textbooks. Real medical RAG systems use PubMed +
StatPearls + Wikipedia + textbooks. Adding multi-corpus retrieval with
corpus-aware reranking would be a meaningful engineering contribution.

**Why novel:** Not a clean research novelty by itself but enables stronger
overall numbers; novel if combined with corpus-routing strategies (different
queries go to different corpora).

**Effort:** High. Probably 2+ weeks of data engineering.

### 7. Option-aware retrieval (MedQA-specific, deprioritized)

*Original idea — for MC tasks only.* Retrieve once per answer option,
evaluate per-option evidence separately. Since QuantaBot's target context isn't
limited to MC, this is deprioritized but noted for completeness if MedQA-style
benchmarks become a priority.

---

## What "paper-worthy" actually requires

Beyond a novel technique, a credible paper needs:

| Requirement | Detail |
|---|---|
| **Sample size** | Full MedQA test set (1273 questions) + ≥2 additional datasets (PubMedQA, MMLU-medical, MedMCQA, etc.) for generalization |
| **Baselines** | (a) GPT-4 zero-shot, (b) GPT-4 + naive top-k RAG, (c) at least one published medical RAG (e.g., MedRAG — code is on GitHub) |
| **Ablations** | Each proposed component evaluated individually + together |
| **Statistical significance** | Confidence intervals, paired t-test or McNemar's test for paired comparisons |
| **Compute** | ~10 hr per config at current latency. Multiple configs × multiple datasets = ~100+ hr. Consider parallelizing or running on a server. |
| **Reproducibility** | Pinned dependencies, deterministic seeds where possible, public eval script |

## Recommended sequencing

1. **First:** GPT-4 zero-shot baseline. Without this, we don't know if RAG is helping at all.
2. **Then:** #4 (domain-specific embeddings) — table-stakes, quick win.
3. **Then:** #1 (structured clinical decomposition) — highest-leverage novel direction.
4. **Then:** #2 (iterative confidence-driven retrieval) — stacks with #1.
5. **Then:** #3 (structured evidence aggregation) — refines the final prompt.
6. **Then:** Full-scale evaluation (1273 + cross-dataset) → write-up.

## Open questions to revisit

- Does the name "Quanta" need to mean something architecturally, or is it just branding? Currently no quantum-inspired component exists.
- Target venue if pursuing publication: ML/NLP (ACL, EMNLP, NeurIPS) vs medical informatics (AMIA, JAMIA, JBHI). Different reviewer expectations.
- Whether the goal is a single strong technical paper or an evolving system with applied-clinical impact.
