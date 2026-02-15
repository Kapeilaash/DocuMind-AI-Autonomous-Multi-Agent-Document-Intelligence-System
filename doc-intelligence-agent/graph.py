"""
LangGraph — Multi-agent orchestration with conditional routing.
Flow: Planner → Retriever → Analyst → Writer → Critic → (Writer loop | End)
"""

from langgraph.graph import StateGraph, END

from state import AgentState
from agents import planner, retriever, analyst, writer, critic, route_after_critic


def build_graph():
    """Build and compile the document intelligence graph."""
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner)
    builder.add_node("retriever", retriever)
    builder.add_node("analyst", analyst)
    builder.add_node("writer", writer)
    builder.add_node("critic", critic)

    builder.set_entry_point("planner")

    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "analyst")
    builder.add_edge("analyst", "writer")
    builder.add_edge("writer", "critic")

    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "writer": "writer",
            "end": END,
        },
    )

    return builder.compile()


# Compiled graph instance
graph = build_graph()
