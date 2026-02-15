"""Writer Agent — Produces structured, professional answer from analysis."""

from langchain_openai import ChatOpenAI

from state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)


def writer(state: AgentState) -> dict:
    """Write a professional, structured answer based on the analysis."""
    question = state["question"]
    analysis = state.get("analysis", "")
    context = state.get("context", "")
    feedback = state.get("feedback", "")

    base_prompt = f"""You are a professional writer. Write a clear, well-structured answer based on the analysis below.

Question: {question}

Analysis:
{analysis}

Relevant context (for reference):
{context}
"""

    if feedback and "IMPROVE" not in feedback.upper():
        # Subsequent iteration: incorporate critic feedback
        base_prompt += f"""

Critic feedback to incorporate:
{feedback}

Revise the answer according to the feedback. Maintain accuracy and structure."""
    elif feedback:
        base_prompt += f"""

Critic feedback (you must improve):
{feedback}

Rewrite the answer to address these issues. Be more precise and complete."""

    base_prompt += "\n\nWrite the final answer in a professional format. Use bullet points or sections if appropriate."

    draft = llm.invoke(base_prompt)
    return {"draft_answer": draft.content}
