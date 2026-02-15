"""Planner Agent — Creates step-by-step reasoning plan for answering the question."""

from langchain_openai import ChatOpenAI

from state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def planner(state: AgentState) -> dict:
    """Plan reasoning steps to answer the question."""
    question = state["question"]

    plan = llm.invoke(
        f"""You are a strategic planner. Create a clear, step-by-step reasoning plan to answer the following question.
Do not answer the question yet—only outline the logical steps needed.

Question: {question}

Output format (numbered steps):
1. [First step]
2. [Second step]
3. [etc.]

Plan:"""
    )

    return {"plan": plan.content}
