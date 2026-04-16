from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fraud_system.agents import FraudAgentOrchestrator
from fraud_system.config import Settings, load_settings
from fraud_system.data import load_dataset
from fraud_system.features import CandidateTransaction, baseline_decision, score_transactions


@dataclass(slots=True)
class DetectionResult:
    fraud_transaction_ids: list[str]
    reviewed_candidates: list[dict]


class FraudDetectionPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dataset = load_dataset(settings.dataset_dir)
        self.orchestrator = FraudAgentOrchestrator(settings)

    def run(
        self,
        threshold: float = 0.5,
        max_fraud_ratio: float = 0.75,
        top_k: int = 0,
    ) -> DetectionResult:
        candidates = score_transactions(self.dataset)
        reviewed: list[dict] = []
        scored_items: list[dict] = []

        for candidate in candidates:
            baseline_fraud, baseline_reason = baseline_decision(candidate)
            final_label = "fraud" if baseline_fraud else "legit"
            final_score = candidate.combined_score
            final_reasons = [baseline_reason]

            if self.orchestrator.enabled:
                agent_decision = self.orchestrator.evaluate(candidate)
                final_label = agent_decision.label
                final_score = max(final_score, agent_decision.score)
                final_reasons = agent_decision.reasons or final_reasons

            item = _serialize_candidate(candidate, final_label, final_score, final_reasons)
            reviewed.append(item)
            scored_items.append(item)

        max_fraud_count = max(1, int(len(scored_items) * max_fraud_ratio))
        fraud_candidates = [
            item
            for item in scored_items
            if item["final_label"] == "fraud" or item["final_score"] >= threshold
        ]
        fraud_candidates = sorted(
            fraud_candidates,
            key=lambda item: (item["final_score"], item["combined_score"], item["economic_risk_score"]),
            reverse=True,
        )
        if top_k > 0:
            fraud_candidates = fraud_candidates[: min(top_k, max_fraud_count)]
        else:
            fraud_candidates = fraud_candidates[:max_fraud_count]
        fraud_ids = [item["transaction_id"] for item in fraud_candidates]

        return DetectionResult(
            fraud_transaction_ids=fraud_ids,
            reviewed_candidates=reviewed,
        )


def _serialize_candidate(
    candidate: CandidateTransaction,
    final_label: str,
    final_score: float,
    final_reasons: list[str],
) -> dict:
    return {
        "transaction_id": candidate.transaction.transaction_id,
        "timestamp": candidate.transaction.timestamp.isoformat(),
        "sender_id": candidate.transaction.sender_id,
        "recipient_id": candidate.transaction.recipient_id,
        "amount": candidate.transaction.amount,
        "transaction_type": candidate.transaction.transaction_type,
        "heuristic_score": candidate.heuristic_score,
        "economic_risk_score": candidate.economic_risk_score,
        "combined_score": candidate.combined_score,
        "final_label": final_label,
        "final_score": round(final_score, 3),
        "reasons": final_reasons,
        "candidate_reasons": candidate.reasons,
        "feature_map": candidate.feature_map,
        "recent_phishing_examples": candidate.communication_summary.recent_phishing_examples,
    }


def build_pipeline(dataset_dir: str | Path | None = None) -> FraudDetectionPipeline:
    settings = load_settings(dataset_dir)
    return FraudDetectionPipeline(settings)
