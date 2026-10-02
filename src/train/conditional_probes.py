"""Gold-free matched probes for conditional context utility."""
import random


MAX_CONTEXTS = 24
STATE_GOALS = (0, 3, 6, 9)
ADDITIONS_PER_STATE = 5


def matched_probe_graph(renderer, seed, rank_candidates=None):
    """Build four context states and up to five feasible add probes per state.

    The returned graph contains measured contexts and directed one-item
    additions only. It does not inspect targets or create STOP labels.
    """
    rng = random.Random(seed)
    candidates = renderer.candidates
    candidate_count = len(candidates)
    model_rank_cache = {}

    def bm25_order(choices):
        return sorted(choices,
                      key=lambda i: (-float(candidates[i].get("bm25_score") or 0.0), i))

    def ordered_choices(state, choices, mode, state_id, phase):
        choices = list(choices)
        if mode == "model" and rank_candidates is not None and choices:
            key = (tuple(state), tuple(choices))
            if key not in model_rank_cache:
                model_rank_cache[key] = rank_candidates(list(state), choices)
            ranked = model_rank_cache[key]
            seen = set()
            order = []
            for candidate in ranked:
                candidate = int(candidate)
                if candidate in choices and candidate not in seen:
                    order.append(candidate)
                    seen.add(candidate)
            return order + [candidate for candidate in choices if candidate not in seen]
        if mode == "random":
            local_rng = random.Random(f"{seed}:{state_id}:{phase}:{tuple(state)}")
            local_rng.shuffle(choices)
            return choices
        return bm25_order(choices)

    def feasible_additions(state):
        choices = [i for i in range(candidate_count) if i not in state]
        if not choices:
            return []
        sets = [(*state, i) for i in choices]
        return [i for i, ok in zip(choices, renderer.feasible_many(sets)) if ok]

    def grow_state(mode, state_id, cap):
        state = ()
        while len(state) < cap:
            choices = [i for i in range(candidate_count) if i not in state]
            if not choices:
                break
            order = ordered_choices(state, choices, mode, state_id,
                                    f"state-{len(state)}")
            # Ranking depends on state and candidate only, not on other choices.
            # Check in rank order: no need to tokenize all 100 joined contexts
            # when the highest-ranked addition fits. Still exact budgets.
            candidate = next((i for i in order if renderer.feasible((*state, i))), None)
            if candidate is None:
                break
            state = renderer.canonical((*state, candidate))
        return state

    modes = (["model", "bm25", "random"] if rank_candidates is not None
             else ["bm25", "bm25", "random"])
    rng.shuffle(modes)
    mode_by_state = dict(zip((1, 2, 3), modes))

    states = {0: ()}
    used_sizes = {0}
    # Build larger targets first. If pool/budget constraints collapse goals to
    # the same size, lower goals back off to the next distinct feasible size.
    for state_id, goal in sorted(((i, g) for i, g in enumerate(STATE_GOALS) if g),
                                 key=lambda pair: -pair[1]):
        mode = mode_by_state[state_id]
        max_size = min(goal, renderer.config.max_snippets - 1,
                       candidate_count - 1)
        selected = None
        for cap in range(max_size, 0, -1):
            proposal = grow_state(mode, state_id, cap)
            if len(proposal) not in used_sizes:
                selected = proposal
                break
        if selected is None:
            selected = grow_state(mode, state_id, 0)
        states[state_id] = selected
        used_sizes.add(len(selected))
    states = {state_id: states[state_id] for state_id in range(len(STATE_GOALS))}

    eligible = {}
    probe_orders = {}
    for state_id, state in states.items():
        choices = feasible_additions(state)
        eligible[state_id] = choices
        probe_orders[state_id] = {
            mode: ordered_choices(state, choices, mode, state_id, f"probes-{mode}")
            for mode in (("model", "bm25", "random") if rank_candidates is not None
                         else ("bm25", "random"))
        }

    # Promote candidates feasible in several states so their measured gain can
    # change with the selected set. Remaining probes follow each state's source.
    states_for_candidate = {}
    for state_id, choices in eligible.items():
        for candidate in choices:
            states_for_candidate.setdefault(candidate, []).append(state_id)
    bm25_rank = {candidate: rank for rank, candidate in enumerate(bm25_order(range(candidate_count)))}
    shared = [candidate for candidate, state_ids in states_for_candidate.items()
              if len(state_ids) > 1]
    shared.sort(key=lambda candidate: (-len(states_for_candidate[candidate]),
                                       bm25_rank[candidate], candidate))
    anchors = shared[:2]

    probes = {}
    probe_sources = {}
    for state_id in states:
        choices = set(eligible[state_id])
        selected, sources = [], {}

        def add_probe(candidate, source):
            if candidate not in choices:
                return
            if candidate in sources:
                sources[candidate].add(source)
            elif len(selected) < ADDITIONS_PER_STATE:
                selected.append(candidate)
                sources[candidate] = {source}

        for candidate in anchors:
            add_probe(candidate, "shared")
        for mode in ("model", "bm25"):
            order = probe_orders[state_id].get(mode, [])
            if order:
                add_probe(order[0], mode)

        random_order = probe_orders[state_id]["random"]
        random_candidate = next((candidate for candidate in random_order
                                 if candidate not in selected),
                                random_order[0] if random_order else None)
        if random_candidate is not None:
            add_probe(random_candidate, "random")

        # Fill remaining slots by alternating sources, with random exploration
        # first so shared/BM25 anchors cannot crowd it out.
        fill_modes = (("random", "model", "bm25") if rank_candidates is not None
                      else ("random", "bm25"))
        for mode in fill_modes * ADDITIONS_PER_STATE:
            if len(selected) >= ADDITIONS_PER_STATE:
                break
            candidate = next((item for item in probe_orders[state_id].get(mode, [])
                              if item not in selected), None)
            if candidate is not None:
                add_probe(candidate, mode)
        probes[state_id] = selected[:ADDITIONS_PER_STATE]
        probe_sources[state_id] = sources

    nodes, lookup = [], {}

    def add_node(indices):
        key = renderer.canonical(indices)
        if not renderer.feasible(key):
            raise AssertionError("Probe sampler selected an infeasible rendered context")
        if key not in lookup:
            if len(nodes) >= MAX_CONTEXTS:
                raise AssertionError("Probe sampler exceeded the context cap")
            _, cost = renderer.render(key)
            lookup[key] = len(nodes)
            nodes.append({"indices": list(key), "cost": cost})
        return lookup[key]

    state_nodes = {state_id: add_node(state) for state_id, state in states.items()}
    edges, edge_keys = [], set()
    duplicate_edges = 0
    for state_id, state in states.items():
        source = state_nodes[state_id]
        for candidate in probes[state_id]:
            target_state = renderer.canonical((*state, candidate))
            target = add_node(target_state)
            key = (source, target)
            if key in edge_keys:
                duplicate_edges += 1
                continue
            edge_keys.add(key)
            edges.append({"source": source, "target": target,
                          "candidate": candidate, "state": list(state),
                          "group": "add", "state_id": state_id})

    shared_audit = []
    for candidate, state_ids in sorted(states_for_candidate.items()):
        probed = [state_id for state_id in state_ids
                  if candidate in probes[state_id]]
        distinct_states = {states[state_id] for state_id in probed}
        if len(distinct_states) > 1:
            shared_audit.append({"candidate": candidate,
                                 "state_ids": probed})

    state_audit = []
    for state_id, goal in enumerate(STATE_GOALS):
        state = states[state_id]
        state_audit.append({
            "state_id": state_id,
            "goal_size": goal,
            "actual_size": len(state),
            "indices": list(state),
            "mode": "empty" if state_id == 0 else mode_by_state[state_id],
            "probe_candidates": list(probes[state_id]),
            "probe_sources": [{"candidate": candidate,
                               "sources": sorted(probe_sources[state_id][candidate])}
                              for candidate in probes[state_id]],
            "feasible_additions": len(eligible[state_id]),
            "shortfall": max(0, goal - len(state)),
        })
    audit = {
        "max_contexts": MAX_CONTEXTS,
        "state_goals": list(STATE_GOALS),
        "states": state_audit,
        "unique_contexts": len(nodes),
        "unique_edges": len(edges),
        "duplicate_edges_suppressed": duplicate_edges,
        "distinct_state_count": len(set(states.values())),
        "bm25_score_coverage": sum(candidate.get("bm25_score") is not None
                                    for candidate in candidates),
        "shared_probe_candidates": shared_audit,
        "stop_edges": 0,
    }
    return nodes, edges, audit
