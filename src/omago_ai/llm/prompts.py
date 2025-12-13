from __future__ import annotations


def reading_level_rules(reading_level: str) -> str:
    if reading_level == "simple":
        return (
            "Use very simple English. Short sentences. Avoid jargon. "
            "Use bullet points when listing steps. Ask 2-4 short confirmation questions."
        )
    return (
        "Use clear professional English. Keep it concise. "
        "Use bullet points for steps and questions."
    )


def system_prompt_base(reading_level: str) -> str:
    return (
        "You are OmagoAI, an accessible trading companion for education-only demos. "
        "You must not provide financial advice. "
        + reading_level_rules(reading_level)
    )


def jargon_prompt(user_question: str, *, reading_level: str, retrieved_snippets: list[str]) -> tuple[str, str]:
    system = system_prompt_base(reading_level)
    joined = "\n\n---\n\n".join(retrieved_snippets[:3])
    user = (
        "Explain the user question using ONLY the provided glossary snippets. "
        "If the snippets do not contain the answer, say you do not know.\n\n"
        f"User question: {user_question}\n\n"
        "Glossary snippets:\n"
        f"{joined}\n\n"
        "Return a short explanation and one short follow-up question."
    )
    return system, user


def synthesis_prompt(
    *,
    reading_level: str,
    trade_summary: str,
    risk_level: str,
    risk_score: int,
    reasons: list[str],
    safer_actions: list[str],
) -> tuple[str, str]:
    system = system_prompt_base(reading_level)

    user = (
        "Turn the trade + risk analysis into a short, accessible checklist.\n"
        "Return STRICT JSON with keys: steps (array of strings), confirmation_questions (array of strings).\n\n"
        f"Trade: {trade_summary}\n"
        f"Risk level: {risk_level}\n"
        f"Risk score: {risk_score}/100\n"
        f"Reasons: {reasons}\n"
        f"Safer actions: {safer_actions}\n"
    )

    return system, user


def general_prompt(user_input: str, *, reading_level: str) -> tuple[str, str]:
    system = system_prompt_base(reading_level)
    user = (
        "Help the user. If they want to trade, ask for missing details (ticker, buy/sell, amount). "
        "If they ask for jargon, explain it simply. Keep it short.\n\n"
        f"User: {user_input}"
    )
    return system, user
