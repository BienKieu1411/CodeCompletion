# F03 — Stronger novelty, different scope: version-aware context

## Candidate contribution

Detect repository evidence that belongs to an obsolete API or commit and assemble only context compatible with the target repository state.

## Evidence and gap

A 2026 diagnostic study found that stale-only evidence can induce obsolete helper references, but it is small (17 signature-change cases in five Python repositories) and explicitly proposes no mitigation. See [S05](../sources/S05_stale_repository_context.md).

## Why it may be more novel

The candidate contribution would be a mitigation plus a commit-paired evaluation, rather than another similarity scorer or generic context filter.

## Why it is not the main choice

It requires history/stale-index scenarios and test-backed data, and is unlikely to improve a static one-snapshot CrossCodeEval/RepoEval result. Choose it only if stale indexes or multi-version repositories are part of the intended deployment.
