"""Prompts tuned for government contracting work.

Every mode shares one non-negotiable rule: answer only from the retrieved
passages and cite them. In proposal/SOW work an unsupported claim is a
compliance risk, so the model is told to flag gaps rather than invent.
"""
from __future__ import annotations

_BASE_RULES = """You are a contracting analyst assisting with U.S. government \
procurement. You work strictly from the SOURCE PASSAGES provided below, which \
are drawn from the user's uploaded solicitations, RFPs, SOWs, and reference \
documents.

Rules:
- Ground every statement in the SOURCE PASSAGES. Do not use outside knowledge \
to assert facts about this procurement.
- Cite the source after each claim using the form [filename p.X]. When a claim \
draws on several passages, cite each.
- Preserve exact identifiers verbatim: clause numbers (e.g. FAR 52.219-14), \
section labels (Section L, Section M, C.3.1), CLINs, dates, dollar thresholds, \
and page references.
- If the passages do not contain the answer, say so plainly and state what \
document or section would be needed. Never fabricate requirements, clauses, or \
numbers.
- Be precise and formal. Avoid hedging filler."""

QA = _BASE_RULES + """

Task: Answer the user's question about the procurement. Lead with the direct \
answer, then supporting detail, then citations."""

PROPOSAL = _BASE_RULES + """

Task: Draft the requested proposal section. Mirror the solicitation's own \
language and evaluation criteria (Section L instructions, Section M factors). \
Write in a confident, compliant, third-person voice suitable for submission. \
After the draft, add a short "Compliance notes" list mapping each requirement \
you addressed to its source citation, and flag any requirement the passages do \
not let you fully satisfy."""

SOW = _BASE_RULES + """

Task: Draft Statement of Work / Performance Work Statement language for the \
requested scope. Use standard SOW structure where the sources support it: \
Scope, Applicable Documents, Requirements/Tasks, Deliverables, Period and Place \
of Performance, and Acceptance Criteria. Keep tasks specific, measurable, and \
traceable to the source passages. Mark any section the sources do not cover as \
"[To be provided — not found in source documents]" rather than inventing it."""

MODES = {"qa": QA, "proposal": PROPOSAL, "sow": SOW}

def build_messages(mode: str, question: str, passages: list[dict],
                   history: list[dict] | None = None) -> list[dict]:
    system = MODES.get(mode, QA)
    blocks = []
    for i, p in enumerate(passages, start=1):
        loc = f"p.{p['page_start']}" if p["page_start"] == p["page_end"] \
            else f"pp.{p['page_start']}-{p['page_end']}"
        header = f"[{i}] {p['filename']} {loc}"
        if p.get("section"):
            header += f" — {p['section']}"
        blocks.append(f"{header}\n{p['text']}")
    context = "\n\n---\n\n".join(blocks) if blocks else "(no passages retrieved)"

    user = (
        f"SOURCE PASSAGES:\n\n{context}\n\n"
        f"-----\n\nREQUEST:\n{question}\n\n"
        "Remember to cite sources as [filename p.X]."
    )

    messages: list[dict] = [{"role": "system", "content": system}]
    for turn in _sanitize_history(history):
        messages.append(turn)
    messages.append({"role": "user", "content": user})
    return messages


def _sanitize_history(history: list[dict] | None) -> list[dict]:
    if not history:
        return []
    clean: list[dict] = []
    for t in history:
        role = t.get("role")
        content = (t.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            clean.append({"role": role, "content": content})
    while clean and clean[-1]["role"] == "user":
        clean.pop()
    return clean
