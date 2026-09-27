from __future__ import annotations

import json
import re
from typing import Any, Callable

from .config import SETTINGS
from .providers import call_gemini, call_muse, call_nemotron


_AGREEMENTS = {"CONSENSUS", "PARTIAL_AGREEMENT", "DISAGREEMENT"}


def build_grounded_context(state: dict[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False, indent=2, default=str)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract a JSON object even when a model wraps it in prose or markdown."""
    if not text:
        return None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            candidate, _ = decoder.raw_decode(text[match.start() :])
            if isinstance(candidate, dict):
                return candidate
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _normalize_agreement(value: Any) -> str:
    normalized = str(value or "").strip().upper().replace(" ", "_")
    if normalized in {"PARTIAL", "PARTIAL_AGREEMENT"}:
        return "PARTIAL_AGREEMENT"
    if normalized in _AGREEMENTS:
        return normalized
    return "UNASSESSED"


def _call_and_record(
    name: str,
    fn: Callable[[str], tuple[str, float]],
    prompt: str,
    responses: dict[str, str],
    latencies: dict[str, float],
    errors: dict[str, str],
) -> None:
    try:
        text, latency = fn(prompt)
        if not text.strip():
            errors[name] = "Model returned an empty visible response. Raw reasoning is intentionally not used as fallback."
            return
        responses[name] = text.strip()
        latencies[name] = latency
    except Exception as exc:
        errors[name] = str(exc)




def _analysis_fallback(responses: dict[str, str]) -> str | None:
    primary = responses.get("Nemotron Primary")
    if primary:
        return primary
    gemini = responses.get("Gemini Second Opinion")
    if gemini:
        parsed = _extract_json_object(gemini)
        if parsed and isinstance(parsed.get("analysis"), str) and parsed["analysis"].strip():
            return parsed["analysis"].strip()
        return gemini
    muse = responses.get("Muse Glimmer")
    return muse or None

def _primary_prompt(question: str, context: str) -> str:
    return (
        "You are the PRIMARY ANALYSIS sub-agent inside FloodGuard AI. Analyze the VERIFIED FloodGuard evidence "
        "and work out the substance of the operator's question for downstream cross-checking. Use ONLY the "
        "verified evidence. Do not invent coordinates, roads, distances, shelters, weather, route choices, or "
        "scenario results that are not present. The deterministic FloodGuard simulation and routing data are the "
        "source of truth. Treat the flood engine as a prototype hazard heuristic, not a certified hydrodynamic "
        "forecast or emergency navigation system. Explain the evidence, route impact, and uncertainty in plain "
        "prose. Do not output <think> tags. Do not make a hidden reasoning trace your answer.\n\n"
        f"VERIFIED FLOODGUARD CONTEXT:\n{context}\n\n"
        f"OPERATOR QUESTION:\n{question}"
    )


def _gemini_prompt(question: str, context: str, primary: str | None) -> str:
    return (
        "You are the INDEPENDENT SECOND-OPINION agent for FloodGuard AI. Review the verified FloodGuard evidence "
        "and the primary analysis below. Independently check whether the primary analysis is consistent with the "
        "deterministic evidence. You may disagree. Do not invent facts. Do not calculate or invent geographic data "
        "that is not explicitly present. Return JSON with agreement and a concise analysis for a final synthesis "
        "agent. The FloodGuard hazard model is a prototype heuristic, not a certified flood forecast.\n\n"
        f"VERIFIED FLOODGUARD CONTEXT:\n{context}\n\n"
        f"PRIMARY ANALYSIS:\n{primary or '[Primary analysis unavailable]'}\n\n"
        f"OPERATOR QUESTION:\n{question}"
    )


def _muse_prompt(question: str, context: str, primary: str | None) -> str:
    return (
        "You are the OPTIONAL MODEL REVIEWER (Muse Glimmer) inside FloodGuard AI. Provide an independent review "
        "of the verified evidence and primary analysis. Focus on missed contradictions, unsupported claims, and "
        "whether the route and flood-impact interpretation is grounded in the supplied evidence. Do not invent "
        "coordinates, roads, distances, weather, shelters, or route choices. Do not output <think> tags.\n\n"
        f"VERIFIED FLOODGUARD CONTEXT:\n{context}\n\n"
        f"PRIMARY ANALYSIS:\n{primary or '[Primary analysis unavailable]'}\n\n"
        f"OPERATOR QUESTION:\n{question}"
    )


def _synthesis_prompt(question: str, context: str, analyses: dict[str, str]) -> str:
    joined = json.dumps(analyses, ensure_ascii=False, indent=2)
    return (
        "You are NEMOTRON SUPER, the CROSS-CHECK + FINAL SYNTHESIS agent for FloodGuard AI. "
        "This is the last model in the pipeline and your output becomes the operator-facing answer. "
        "First cross-check the independent analyses against the VERIFIED FLOODGUARD CONTEXT. "
        "The deterministic simulation and real-road routing data are authoritative for coordinates, road names, "
        "distances, route candidates, exposure scores, rainfall values, and weather observations. "
        "The models are not allowed to invent geographic truth. If an analysis conflicts with verified evidence, "
        "discard that claim. If the evidence is insufficient, say so explicitly. "
        "Remember this is a decision-support prototype using a simulated hazard index, not a certified emergency "
        "navigation or hydrodynamic forecasting system.\n\n"
        "Return ONLY one JSON object in this exact shape:\n"
        '{"agreement":"CONSENSUS|PARTIAL_AGREEMENT|DISAGREEMENT","answer":"concise grounded operator answer"}\n\n'
        "The answer should directly address the operator question, normally in <= 180 words, and should identify "
        "the relevant route/evidence when present. Do not output markdown fences, <think> tags, or extra prose.\n\n"
        f"VERIFIED FLOODGUARD CONTEXT:\n{context}\n\n"
        f"INDEPENDENT MODEL REVIEWS:\n{joined}\n\n"
        f"OPERATOR QUESTION:\n{question}"
    )


def copilot(question: str, state: dict[str, Any]) -> dict[str, Any]:
    """Run the presentation architecture: analyze → cross-check → synthesize."""
    context = build_grounded_context(state)
    responses: dict[str, str] = {}
    latencies: dict[str, float] = {}
    errors: dict[str, str] = {}

    _call_and_record(
        "Nemotron Primary",
        lambda prompt: call_nemotron(prompt, role="primary"),
        _primary_prompt(question, context),
        responses,
        latencies,
        errors,
    )

    primary = responses.get("Nemotron Primary")

    if SETTINGS.enable_gemini and SETTINGS.gemini_api_key:
        gemini_name = "Gemini Second Opinion"
        _call_and_record(
            gemini_name,
            call_gemini,
            _gemini_prompt(question, context, primary),
            responses,
            latencies,
            errors,
        )
    else:
        errors["Gemini Second Opinion"] = "Gemini is disabled or not configured; final synthesis will proceed without it."

    if SETTINGS.enable_muse and SETTINGS.nvidia_api_key:
        _call_and_record(
            "Muse Glimmer",
            call_muse,
            _muse_prompt(question, context, primary),
            responses,
            latencies,
            errors,
        )

    # A final synthesizer is still useful even when one or more reviewers fail.
    synthesis_inputs = dict(responses)
    final_prompt = _synthesis_prompt(question, context, synthesis_inputs)

    final_answer = "No final synthesis model responded. Check the AI Configuration panel."
    agreement = "NO MODEL"

    try:
        synthesis, latency = call_nemotron(final_prompt, role="synthesis")
        latencies["Nemotron Super"] = latency
        if not synthesis:
            errors["Nemotron Super"] = (
                "Final synthesis model returned an empty visible response. "
                "Its raw reasoning is intentionally not used as an answer."
            )
        else:
            responses["Nemotron Super"] = synthesis
            parsed = _extract_json_object(synthesis)
            if parsed and isinstance(parsed.get("answer"), str) and parsed["answer"].strip():
                final_answer = parsed["answer"].strip()
                agreement = _normalize_agreement(parsed.get("agreement"))
            else:
                errors["Nemotron Super"] = "Final model did not satisfy the JSON contract; deterministic fallback used."
                # Do not silently display malformed synthesis as the final answer.
                final_answer = _analysis_fallback(responses) or "No grounded model analysis is available."
                agreement = "UNASSESSED"
    except Exception as exc:
        errors["Nemotron Super"] = str(exc)
        final_answer = _analysis_fallback(responses) or final_answer
        agreement = "UNASSESSED"

    if not responses:
        final_answer = "No enabled AI provider responded. Check the AI Configuration panel."
        agreement = "NO MODEL"

    return {
        "answer": final_answer,
        "responses": responses,
        "latencies": latencies,
        "errors": errors,
        "agreement": agreement,
        "pipeline": [
            "Nemotron Primary",
            "Gemini Second Opinion",
            "Muse Glimmer (optional)",
            "Nemotron Super",
        ],
    }
