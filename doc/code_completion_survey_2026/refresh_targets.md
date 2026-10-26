# Refresh targets

Survey cutoff: **2026-10-24**. This is a bounded scan, not a claim of exhaustive coverage.

Before citing the survey in a paper or choosing a final novelty claim:

1. Recheck ACL Anthology's 2026 EMNLP proceedings and the October 2026 ISSTA/ASE records; late indexing or corrected metadata can change a preprint's publication status.
2. Search ACM DL / IEEE Xplore / proceedings for repository-level FIM and code completion published after this cutoff, including industry papers and companion volumes.
3. Search arXiv monthly (cs.SE, cs.CL, cs.AI) for exact terms: “repository-level code completion,” “FIM,” “cross-file retrieval,” “dependency-aware retrieval,” “code context selection,” and cited-by chains from AIRCoder, LC², RepoShapley, ACAR, GrepRAG, and RepoFuse.
4. Verify whether CARET or other October preprints receive a venue acceptance; keep the paper's preprint version, venue, and final method differences separate.
5. Expand the graph-method search beyond code completion to typed program graphs, language-server indexes, dependency-aware retrieval, and repository-level generation—but do not transfer full-function-generation results as FIM evidence.
6. Check public artifacts and versioned evaluation scripts. In particular, validate chunking, prefix/FIM serialization, left/right context, generator version, temperature, candidate count, budget, normalization, and metric implementation before using reported EM/ES.
7. Run a novelty-focused query for “query-conditioned typed relation retrieval + single-pass FIM completion” before using “first” or “novel” in a manuscript.
