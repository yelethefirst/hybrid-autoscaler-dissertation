"""Retrieval-bounded LLM narrator (§3.8, §3.12).

Architecture
------------
1. An evidence bundle row (Decision JSON) + SHAP attribution + faithfulness
   metrics are formatted into a STRUCTURED CONTEXT block — every value the
   LLM is permitted to cite is materialised here.

2. A SYSTEM PROMPT instructs the LLM to:
   a) Use ONLY values from the STRUCTURED CONTEXT.
   b) Describe the scaling decision in 2–3 sentences.
   c) Explain WHY the system chose this action (driven by top SHAP feature).
   d) Mention uncertainty level (σ̂) and any active state.
   e) Not invent numbers, model names, or domain facts not in the context.

3. The generated narrative is stored in NarrativeResult along with prompt
   tokens and the full prompt (for FActScore evaluation).

Retrieval-bounded guarantee
---------------------------
The template function `_format_context()` returns a string that contains
every number/name the LLM is expected to cite. FActScore then verifies that
each atomic claim in the narrative is grounded in that context string.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Optional

from explain.attribution import Attribution
from explain.faithfulness import FaithfulnessMetrics


SYSTEM_PROMPT = textwrap.dedent("""\
    You are an autoscaling explainer. Your job is to narrate one Kubernetes
    scaling decision in 2–3 clear sentences for a technical SRE audience.

    Rules (strictly enforced):
    1. Use ONLY values from the STRUCTURED CONTEXT provided.
       Do not invent numbers, names, or facts not present in that context.
    2. Your narrative must cover:
       a. What the system decided (scale up, scale down, or hold).
       b. Why (the top SHAP feature that drove the forecast).
       c. How confident the forecast was (σ̂ value and state).
    3. If the autoscaler was in a FALLBACK state, explain that the
       predictive model was unavailable and HPA-equivalent logic was used.
    4. Write at the level of a senior SRE — use technical vocabulary but
       keep it concise.
    5. Output only the narrative — no headers, no bullet points, no JSON.
""")

CONTEXT_TEMPLATE = textwrap.dedent("""\
    STRUCTURED CONTEXT (narrator must use ONLY these values):

    Service: {service}
    Namespace: {namespace}
    Timestamp: {timestamp}
    State: {state}
    Autoscaler action: replicas changed from {current_replicas} to {new_replicas}
      (recommended: {recommended_replicas}; rate-limited: {rate_limited})
    Fallback active: {fallback_engaged}
    Fallback reason: {fallback_reason}

    Forecast (horizon {horizon_seconds}s):
      Point estimate (f̂): {forecast_point}
      Uncertainty (σ̂):   {forecast_sigma}
      Forecaster:         {forecaster_name}

    Instantaneous utilisation (u_t): {observed_metric}

    SHAP attribution (method: {shap_method}):
      Top feature 1: {shap_f1_name} = {shap_f1_val}
      Top feature 2: {shap_f2_name} = {shap_f2_val}
      Top feature 3: {shap_f3_name} = {shap_f3_val}
      Expected value (φ₀): {shap_expected_value}

    Faithfulness (insertion AUC: {insertion_auc}; deletion AUC: {deletion_auc})
""")


def _fmt(val: Any, decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _format_context(
    decision: Dict[str, Any],
    attribution: Attribution,
    fm: Optional[FaithfulnessMetrics] = None,
) -> str:
    top = attribution.top_features
    f1 = top[0] if len(top) > 0 else ("N/A", None)
    f2 = top[1] if len(top) > 1 else ("N/A", None)
    f3 = top[2] if len(top) > 2 else ("N/A", None)

    return CONTEXT_TEMPLATE.format(
        service=decision.get("service", "N/A"),
        namespace=decision.get("namespace", "N/A"),
        timestamp=decision.get("timestamp", "N/A"),
        state=decision.get("state", "N/A"),
        current_replicas=decision.get("current_replicas", "N/A"),
        new_replicas=decision.get("new_replicas", "N/A"),
        recommended_replicas=decision.get("recommended_replicas", "N/A"),
        rate_limited=decision.get("rate_limited", False),
        fallback_engaged=decision.get("fallback_engaged", False),
        fallback_reason=decision.get("forecaster_fault_reason") or "none",
        horizon_seconds=decision.get("horizon_seconds", "N/A"),
        forecast_point=_fmt(decision.get("forecast_point")),
        forecast_sigma=_fmt(decision.get("forecast_sigma")),
        forecaster_name=decision.get("forecaster_name") or "N/A",
        observed_metric=_fmt(decision.get("observed_metric")),
        shap_method=attribution.method,
        shap_f1_name=f1[0],
        shap_f1_val=_fmt(f1[1]),
        shap_f2_name=f2[0],
        shap_f2_val=_fmt(f2[1]),
        shap_f3_name=f3[0],
        shap_f3_val=_fmt(f3[1]),
        shap_expected_value=_fmt(attribution.expected_value),
        insertion_auc=_fmt(fm.insertion_auc if fm else None, 3),
        deletion_auc=_fmt(fm.deletion_auc if fm else None, 3),
    )


@dataclass
class NarrativeResult:
    """Output of one narrator invocation."""

    narrative: str
    context: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "narrative": self.narrative,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "error": self.error,
        }


class Narrator:
    """Generates retrieval-bounded LLM narratives for scaling decisions.

    Parameters
    ----------
    client:
        An ``openai.OpenAI`` client (or compatible API client).
    model:
        OpenAI model name. Default: "gpt-4o-mini" (§3.12).
    max_tokens:
        Maximum tokens for the narrative completion.
    temperature:
        LLM temperature. Low (0.2) reduces hallucination risk.
    """

    def __init__(
        self,
        client: Any,
        model: str = "gpt-4o-mini",
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> None:
        self._client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def narrate(
        self,
        decision: Dict[str, Any],
        attribution: Attribution,
        faithfulness: Optional[FaithfulnessMetrics] = None,
    ) -> NarrativeResult:
        """Generate a narrative for one scaling decision.

        Parameters
        ----------
        decision:
            A dict parsed from one evidence-bundle JSONL row.
        attribution:
            SHAP attribution for this decision.
        faithfulness:
            Optional faithfulness metrics.

        Returns
        -------
        NarrativeResult with the narrative and token counts.
        """
        context = _format_context(decision, attribution, faithfulness)
        user_msg = f"Please narrate the following scaling decision.\n\n{context}"

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            narrative = response.choices[0].message.content.strip()
            usage = response.usage
            return NarrativeResult(
                narrative=narrative,
                context=context,
                model=self.model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )
        except Exception as exc:
            return NarrativeResult(
                narrative="",
                context=context,
                model=self.model,
                error=f"{type(exc).__name__}: {exc}",
            )

    def narrate_offline(
        self,
        decision: Dict[str, Any],
        attribution: Attribution,
        faithfulness: Optional[FaithfulnessMetrics] = None,
    ) -> NarrativeResult:
        """Generate a narrative without calling the LLM (returns the context block).

        Useful for testing the prompt template without an API key, or for
        preregistration review where the researcher checks the prompt before
        running the full evaluation.
        """
        context = _format_context(decision, attribution, faithfulness)
        return NarrativeResult(
            narrative=f"[offline mode — context only]\n\n{context}",
            context=context,
            model="offline",
        )
