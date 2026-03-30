def compute_score(state):
    completion = state["progress"]

    vitals_score = state["vitals"]["O2"] / 100

    time_penalty = min(state["time"] / 20, 1)

    score = (
        0.5 * completion +
        0.3 * vitals_score +
        0.2 * (1 - time_penalty)
    )

    return max(0, min(1, score))