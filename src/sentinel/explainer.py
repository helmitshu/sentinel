"""Alert explanations in plain English.

The default TemplateExplainer is deterministic and fully grounded: every
finding it reports comes from measured feature deviations (see diagnostics),
nothing is invented. LLMExplainer is an optional upgrade that calls an
Anthropic model with a strictly grounded prompt; it falls back to the template
when no API key is configured, so the demo never depends on a key.
"""
from __future__ import annotations

import os

from .diagnostics import Finding


class TemplateExplainer:
    """Deterministic, grounded explanations. No model, no hallucination."""

    def explain(
        self,
        machine_id: str,
        findings: list[Finding],
        stable: list[str],
        health: int,
        trend: str,
        fault_hint: str,
        confidence: str,
        window_days: int,
        action: str,
    ) -> str:
        movement = "; ".join(f.text for f in findings) or "no single dominant change"
        stable_txt = (
            f"{', '.join(s.replace('_', ' ') for s in stable)} remained stable, "
            f"which points away from process-wide causes"
            if stable
            else "no stable reference channels"
        )
        return (
            f"{machine_id} shows signs of {fault_hint}. "
            f"{movement} across {window_days} {'day' if window_days == 1 else 'days'}. "
            f"{stable_txt}. "
            f"Health score {health}, trend {trend}. "
            f"Confidence: {confidence}. "
            f"Recommended action: {action}."
        )


SYSTEM_PROMPT = (
    "You are a reliability engineer writing an alert for a field technician. "
    "Use ONLY the numbers provided. Never invent readings, dates, or causes. "
    "If the evidence is ambiguous, say so and state what would confirm it. "
    "Keep it under 120 words, plain language, no jargon."
)


class LLMExplainer:
    """Anthropic-backed explanations with template fallback."""

    def __init__(self, model: str = "claude-sonnet-4-5"):
        self.model = model
        self._template = TemplateExplainer()

    def explain(self, machine_id: str, context: dict, **kwargs) -> str:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return self._template.explain(machine_id=machine_id, **kwargs)
        try:
            import anthropic  # lazy: optional dependency
        except ImportError:
            return self._template.explain(machine_id=machine_id, **kwargs)

        client = anthropic.Anthropic(api_key=api_key)
        numbers = "\n".join(f"- {k}: {v}" for k, v in context.items())
        message = client.messages.create(
            model=self.model,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Machine {machine_id} telemetry summary:\n{numbers}\n\nWrite the alert.",
                }
            ],
        )
        return message.content[0].text.strip()
