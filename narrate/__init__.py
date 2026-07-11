"""Phase 7 LLM narrative generation (§3.8, §3.12).

Generates a retrieval-bounded natural-language explanation for each scaling
decision from the evidence bundle, SHAP attribution, and faithfulness metrics.

"Retrieval-bounded" means the narrator is structurally prevented from making
claims beyond the values in the evidence bundle — the prompt template
interpolates concrete numbers directly, and the LLM is instructed to use
only those values.

Public interface
----------------
    from narrate import Narrator, NarrativeResult
    from narrate.factscore import FActScoreEvaluator

    narrator = Narrator(client=openai_client, model="gpt-4o-mini")
    result = narrator.narrate(evidence_row, attribution, faithfulness_metrics)
    # result.narrative: str
    # result.prompt_tokens: int

    evaluator = FActScoreEvaluator(client=openai_client)
    score = evaluator.score(result.narrative, evidence_row)
    # score.factscore: float in [0, 1]
"""

from .narrator import Narrator, NarrativeResult
from .factscore import FActScoreEvaluator, FActScoreResult

__all__ = ["Narrator", "NarrativeResult", "FActScoreEvaluator", "FActScoreResult"]
