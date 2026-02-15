"""Conditional routing — Agentic behavior: loop to writer or end."""

from state import AgentState

MAX_ITERATIONS = 3


def route_after_critic(state: AgentState) -> str:
    """
    Route after critic: if IMPROVE and under iteration limit → writer
    Otherwise → end
    """
    feedback = state.get("feedback", "").upper()
    iteration = state.get("iteration", 0)

    if "IMPROVE" in feedback and iteration < MAX_ITERATIONS:
        return "writer"
    return "end"
