"""Analyst Agent — Deep analysis using plan, context, and question."""

from langchain_openai import ChatOpenAI

from state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def analyst(state: AgentState) -> dict:
    """Perform deep analysis following the plan with retrieved context."""
    question = state["question"]
    context = state.get("context", "")
    plan = state.get("plan", "")

    analysis = llm.invoke(
        f"""You are an expert analyst. Analyze the following question using the context and plan.

Question: {question}

Plan to follow:
{plan}

Context from documents:
{context}

Provide a thorough analysis. Extract key facts, identify relationships, and prepare the reasoning needed for a clear answer. Do not write the final answer yet—only the analysis."""
    )

    return {"analysis": analysis.content}
