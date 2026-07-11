"""FActScore faithfulness evaluation (§3.12).

FActScore (Min et al. 2023) measures factual precision: for each atomic claim
in the generated narrative, a judge model checks whether the claim is entailed
by the source document (the evidence bundle context).

Implementation
--------------
We use a simplified two-step version:
    1. Decompose the narrative into atomic claims via the LLM.
    2. For each claim, ask the LLM whether it is supported by the context.

This differs from the original FActScore (which uses Wikipedia as the source)
but is equivalent in principle: the source is the structured evidence bundle
context that the narrator was given.

Full evaluation pipeline (§3.12)
---------------------------------
The dissertation evaluation runs FActScore on n=30 narrative samples per
(autoscaler, workload) cell. The `FActScoreEvaluator.batch_evaluate()` method
handles this, writing per-claim verdicts to a JSONL log for manual review.

Human Likert rating (§3.12)
----------------------------
The `HumanRatingBatch` class generates a human annotation workbook (CSV)
for the 5-point Likert items:
    1. Accuracy  — does the narrative accurately describe the decision?
    2. Clarity   — is the narrative clear to an SRE?
    3. Utility   — would this narrative be useful in a post-incident review?

References
----------
    Min et al. (2023). FActScore: Fine-grained Atomic Evaluation of Factual
    Precision in Long Form Text Generation. EMNLP 2023.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


DECOMPOSE_PROMPT = textwrap.dedent("""\
    Break the following narrative into a list of simple, self-contained atomic
    claims. Each claim should be a single sentence asserting one fact.
    Return ONLY a JSON array of strings — no explanation, no markdown.

    Narrative:
    {narrative}
""")

VERIFY_PROMPT = textwrap.dedent("""\
    You are a fact-checker. Given a SOURCE document and a CLAIM, answer
    "supported" if the claim is entailed by the source, or "not_supported"
    if it is not. Reply with ONLY "supported" or "not_supported".

    SOURCE:
    {source}

    CLAIM:
    {claim}
""")


@dataclass
class ClaimVerdict:
    claim: str
    verdict: str          # "supported" | "not_supported" | "error"
    raw_response: str = ""


@dataclass
class FActScoreResult:
    """FActScore result for one narrative."""

    narrative: str
    claims: List[ClaimVerdict] = field(default_factory=list)
    factscore: Optional[float] = None    # fraction of claims supported
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factscore": self.factscore,
            "n_claims": len(self.claims),
            "n_supported": sum(1 for c in self.claims if c.verdict == "supported"),
            "claims": [{"claim": c.claim, "verdict": c.verdict} for c in self.claims],
            "error": self.error,
        }


class FActScoreEvaluator:
    """LLM-based FActScore evaluator for autoscaling narratives.

    Parameters
    ----------
    client:
        An ``openai.OpenAI`` client.
    judge_model:
        Model used for claim decomposition and verification. Default: "gpt-4o-mini".
    max_claims:
        Maximum number of atomic claims to extract (prevents runaway token usage).
    """

    def __init__(
        self,
        client: Any,
        judge_model: str = "gpt-4o-mini",
        max_claims: int = 10,
    ) -> None:
        self._client = client
        self.judge_model = judge_model
        self.max_claims = max_claims

    def _decompose(self, narrative: str) -> List[str]:
        prompt = DECOMPOSE_PROMPT.format(narrative=narrative)
        response = self._client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        try:
            claims = json.loads(raw)
            if isinstance(claims, list):
                return [str(c) for c in claims[: self.max_claims]]
        except json.JSONDecodeError:
            pass
        # Fallback: split by newlines
        return [line.strip(" -•") for line in raw.splitlines() if line.strip()][: self.max_claims]

    def _verify(self, claim: str, source: str) -> ClaimVerdict:
        prompt = VERIFY_PROMPT.format(source=source, claim=claim)
        try:
            response = self._client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip().lower()
            # Order matters: "not_supported"/"not supported" CONTAINS the
            # substring "supported", so the negative form must be checked
            # first or every negative verdict flips positive and FActScore
            # inflates towards 1.0 (2026-07-05 code review).
            if "not_supported" in raw or "not supported" in raw or raw.startswith("not"):
                verdict = "not_supported"
            elif "supported" in raw:
                verdict = "supported"
            else:
                verdict = "not_supported"  # unparseable judge output counts against
            return ClaimVerdict(claim=claim, verdict=verdict, raw_response=raw)
        except Exception as exc:
            return ClaimVerdict(claim=claim, verdict="error", raw_response=str(exc))

    def score(self, narrative: str, source: str) -> FActScoreResult:
        """Compute FActScore for one narrative against a source string.

        Parameters
        ----------
        narrative:
            The generated natural-language narrative.
        source:
            The evidence bundle context (from Narrator.narrate().context).

        Returns
        -------
        FActScoreResult with factscore ∈ [0, 1].
        """
        if not narrative.strip():
            return FActScoreResult(narrative=narrative, error="empty narrative")
        try:
            claims_text = self._decompose(narrative)
            verdicts = [self._verify(c, source) for c in claims_text]
            n_sup = sum(1 for v in verdicts if v.verdict == "supported")
            fs = n_sup / len(verdicts) if verdicts else 0.0
            return FActScoreResult(narrative=narrative, claims=verdicts, factscore=fs)
        except Exception as exc:
            return FActScoreResult(narrative=narrative, error=f"{type(exc).__name__}: {exc}")

    def batch_evaluate(
        self,
        samples: List[Dict[str, Any]],
        result_path: Path,
    ) -> List[FActScoreResult]:
        """Evaluate a batch of narrative samples and write results to JSONL.

        Parameters
        ----------
        samples:
            List of dicts with keys "narrative" and "context".
        result_path:
            Path to write JSONL results (one line per sample).

        Returns
        -------
        List of FActScoreResult objects.
        """
        results = []
        with open(result_path, "w") as f:
            for i, s in enumerate(samples):
                result = self.score(s["narrative"], s["context"])
                f.write(json.dumps(result.to_dict()) + "\n")
                f.flush()
                results.append(result)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Human Likert rating workbook
# ─────────────────────────────────────────────────────────────────────────────

def generate_likert_workbook(
    narratives: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Write a CSV workbook for human Likert rating of narratives.

    Each row contains the narrative text (blinded — no autoscaler label) and
    five 1–5 Likert scale columns for raters to fill in:
        accuracy, clarity, utility, completeness, trust.

    Parameters
    ----------
    narratives:
        List of dicts with keys "trial_id", "narrative", and optionally
        "autoscaler" and "workload" (blinded in the CSV).
    output_path:
        Where to write the CSV.
    """
    import csv

    fields = ["row_id", "narrative", "accuracy", "clarity", "utility", "completeness", "trust"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, n in enumerate(narratives, 1):
            writer.writerow({
                "row_id": i,
                "narrative": n.get("narrative", ""),
                "accuracy": "",
                "clarity": "",
                "utility": "",
                "completeness": "",
                "trust": "",
            })
