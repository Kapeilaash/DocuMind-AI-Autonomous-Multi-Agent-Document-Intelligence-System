"""
Agent State — The brain memory shared between all agents.
Defines the data structure that flows through the LangGraph pipeline.
"""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """State object passed between agents in the graph."""

    question: str
    plan: str
    context: str
    analysis: str
    draft_answer: str
    feedback: str
    final_answer: str
    iteration: int  # Track critic loop iterations to prevent infinite loops
