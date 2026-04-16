from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fraud_system.agents import FraudAgentOrchestrator
from fraud_system.config import Settings, load_settings
from fraud_system.data import load_dataset
from fraud_system.features import CandidateTransaction, baseline_decision, score_transactions
from fraud_system.progress import ProgressReporter


@dataclass(slots=True)
class DetectionResult:
    fraud_transaction_ids: list[str]
    reviewed_candidates: list[dict]


class FraudDetectionPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dataset = load_dataset(settings.dataset_dir)
        self.orchestrator = FraudAgentOrchestrator(settings)
        self.progress = ProgressReporter(enabled=True)

    def run(
        self,
        threshold: float = 0.5,
        max_fraud_ratio: float = 0.75,
        top_k: int = 0,
    ) -> DetectionResult:
        self.progress.stage(f"Loaded dataset from {self.settings.dataset_dir.name}")
        candidates = score_transactions(self.dataset)
        self.progress.stage(f"Scored {len(candidates)} user transactions")
        reviewed: list[dict] = []
        scored_items: list[dict] = []
        candidate_by_id = {candidate.transaction.transaction_id: candidate for candidate in candidates}

        for candidate in candidates:
            baseline_fraud, baseline_reason = baseline_decision(candidate)
            final_label = "fraud" if baseline_fraud else "legit"
            final_score = candidate.combined_score
            final_reasons = [baseline_reason]

            item = _serialize_candidate(candidate, final_label, final_score, final_reasons)
            reviewed.append(item)
            scored_items.append(item)

        if self.orchestrator.enabled:
            llm_candidates = self._select_llm_candidates(candidate_by_id, scored_items)
            batches = self._build_llm_batches(llm_candidates, batch_size=18)
            self.progress.stage(
                f"Prepared {len(llm_candidates)} candidates for LLM rerank in {len(batches)} batches"
            )
            for idx, batch in enumerate(batches, start=1):
                decisions = self.orchestrator.evaluate_batch(batch)
                for candidate in batch:
                    decision = decisions.get(candidate.transaction.transaction_id)
                    if not decision:
                        continue
                    for item in scored_items:
                        if item["transaction_id"] != candidate.transaction.transaction_id:
                            continue
                        item["final_label"] = decision.label
                        item["final_score"] = round(max(item["final_score"], decision.score), 3)
                        if decision.reasons:
                            item["reasons"] = decision.reasons
                        break
                self.progress.progress("LLM rerank", idx, len(batches))

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

    def _select_llm_candidates(
        self,
        candidate_by_id: dict[str, CandidateTransaction],
        scored_items: list[dict],
    ) -> list[CandidateTransaction]:
        frame = pd.DataFrame(scored_items)
        frame = frame.sort_values(
            by=["combined_score", "economic_risk_score", "heuristic_score"],
            ascending=False,
        )
        shortlist = frame[
            (frame["combined_score"] >= 0.75)
            | (frame["economic_risk_score"] >= 1.0)
            | (frame["heuristic_score"] >= 1.0)
        ]
        top_per_sender = (
            shortlist.sort_values(
                by=["combined_score", "economic_risk_score", "heuristic_score"],
                ascending=False,
            )
            .groupby("sender_id", sort=False)
            .head(12)
        )
        ids = top_per_sender["transaction_id"].tolist()
        return [candidate_by_id[item_id] for item_id in ids if item_id in candidate_by_id]

    def _build_llm_batches(
        self,
        candidates: list[CandidateTransaction],
        batch_size: int,
    ) -> list[list[CandidateTransaction]]:
        grouped: dict[str, list[CandidateTransaction]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.transaction.sender_id, []).append(candidate)
        batches: list[list[CandidateTransaction]] = []
        for sender_id in sorted(grouped):
            sender_candidates = sorted(
                grouped[sender_id],
                key=lambda item: item.combined_score,
                reverse=True,
            )
            for start in range(0, len(sender_candidates), batch_size):
                batches.append(sender_candidates[start : start + batch_size])
        return batches


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
