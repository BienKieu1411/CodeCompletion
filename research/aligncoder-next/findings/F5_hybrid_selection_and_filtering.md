# F5 — The likely practical gain is hybrid selection, not another embedding

## Claim

Once candidate recall is adequate, the main bottleneck shifts from “find one more similar chunk” to “avoid redundant or harmful context and pack the right evidence under the generator budget.”

## Evidence chain

1. **S10:** CODEFILTER measures positive, neutral, and negative chunk impact and reports that only a small subset helps; removing harmful context improves quality and reduces prompt length.
2. **S15:** Repoformer shows that some examples benefit from selective retrieval/no retrieval and that always retrieving is inefficient.
3. **S02/S12/S18:** late expansion, lexical identifiers, structure-aware deduplication, and coalition interaction provide distinct ways to improve evidence quality without changing the final generator.
4. **S13/S16:** dataflow/caller-centric structure can be more useful than raw similarity for repository context utilization.

## Proposed mechanism

Use a fixed candidate pool and train one student distribution over real candidates plus null. At serving:

1. retrieve the top `K` candidates from the student;
2. compare candidate logits to the null logit for the stop decision;
3. remove overlapping/duplicate spans deterministically;
4. optionally expand selected nodes to signature/parent/caller evidence within a fixed token budget;
5. call the frozen generator once.

No pairwise list construction is needed. A pointwise harmful-context head is an ablation, not a prerequisite for the main loss.

## Falsification test

If filtering/expansion gains disappear under exact token-budget matching, or if candidate overlap is rare, remove the extra mechanism. Do not claim context filtering as the source of a gain caused by giving the generator more tokens.
