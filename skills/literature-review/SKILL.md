---
name: literature-review
description: Review local PDF papers with source/page evidence, compare methods, and state evidence gaps.
---

# Literature Review

Use this workflow when the question asks about papers, methods, related work, or a PDF in `data/papers/`.

1. Call `paper_search` with the most specific technical terms from the question.
2. If results are empty, retry once with shorter synonyms; never invent a paper result.
3. Preserve every useful `source` and `page` field in the notes.
4. Separate what the retrieved text states from your comparison or interpretation.
5. State when a PDF is scanned, incomplete, ambiguous, or only lexically matched by TF-IDF.

The final response should contain: evidence, method comparison, limitations, and follow-up searches when evidence is insufficient.
