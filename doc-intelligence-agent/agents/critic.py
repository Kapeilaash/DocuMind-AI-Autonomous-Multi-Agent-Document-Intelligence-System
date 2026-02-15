"""Critic Agent — Reflection loop: evaluates answer quality and triggers improvement."""

from langchain_openai import ChatOpenAI

from state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

MAX_ITERATIONS = 3  # Prevent infinite loops


def critic(state: AgentState) -> dict:
    """Critique the draft answer. Returns feedback; routing decides if we improve or finish."""
    draft = state.get("draft_answer", "")
    question = state.get("question", "")
    iteration = state.get("iteration", 0) + 1

    feedback_prompt = f"""You are a rigorous critic. Evaluate this answer for the question below.

Question: {question}

Draft answer:
{draft}

Check for:
- Accuracy (is it supported by the context?)
- Completeness (does it fully address the question?)
- Clarity (is it well-structured and easy to understand?)
- Hallucination (does it make claims not in the context?)

If the answer needs significant improvement, reply with exactly: IMPROVE
Then on the next line, explain what to fix.

If the answer is good enough (accurate, complete, clear), reply with exactly: FINAL
Then optionally add brief improvement suggestions.

Your verdict:"""

    feedback = llm.invoke(feedback_prompt)
    content = feedback.content.strip().upper()

    return {
        "feedback": feedback.content,
        "iteration": iteration,
    }
