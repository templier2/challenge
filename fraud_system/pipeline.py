from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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
        self.dataset = load_dataset(settings.dataset_dir, include_audio=settings.include_audio)
        self.orchestrator = FraudAgentOrchestrator(settings)
        self.progress = ProgressReporter(enabled=True)

    def run(
        self,
        threshold: float = 0.5,
        max_fraud_ratio: float = 0.75,
        top_k: int = 0,
    ) -> DetectionResult:
        self.progress.stage(f"Loaded dataset from {self.settings.dataset_dir.name}")
        if self.orchestrator.enabled:
            self.progress.stage(f"Langfuse session id: {self.orchestrator.session_id}")
        candidates = score_transactions(self.dataset, self.settings)
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
            item_by_id = {item["transaction_id"]: item for item in scored_items}
            max_workers = min(self.settings.llm_concurrency, max(1, len(batches)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(self.orchestrator.evaluate_batch, batch): batch for batch in batches
                }
                for idx, future in enumerate(as_completed(future_map), start=1):
                    batch = future_map[future]
                    decisions = future.result()
                    for candidate in batch:
                        decision = decisions.get(candidate.transaction.transaction_id)
                        if not decision:
                            continue
                        item = item_by_id.get(candidate.transaction.transaction_id)
                        if not item:
                            continue
                        item["final_label"] = decision.label
                        item["final_score"] = round(max(item["final_score"], decision.score), 3)
                        if decision.reasons:
                            item["reasons"] = decision.reasons
                    self.progress.progress("LLM rerank", idx, len(batches))

        max_fraud_count = max(1, int(len(scored_items) * max_fraud_ratio))
        fraud_candidates = [
            item
            for item in scored_items
            if item["final_label"] == "fraud"
            or item["final_score"] >= threshold
            or self._passes_borderline_rule(item, threshold)
        ]
        fraud_candidates = sorted(
            fraud_candidates,
            key=lambda item: (
                item["final_score"],
                item["combined_score"],
                item["campaign_score"],
                item["sequence_score"],
                item["economic_risk_score"],
            ),
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

    def _passes_borderline_rule(self, item: dict, threshold: float) -> bool:
        score = float(item["final_score"])
        if score < 3.0 or score >= threshold:
            return False

        feature_map = item.get("feature_map", {})
        transaction_type = str(item.get("transaction_type", ""))
        amount = float(item.get("amount", 0.0))
        description = str(feature_map.get("description", "")).lower()
        recent_phishing_count = int(feature_map.get("recent_phishing_count", 0))
        payment_issue_count = int(feature_map.get("payment_issue_count", 0))
        recipient_global_count = int(feature_map.get("recipient_global_count", 0))
        new_recipient = bool(feature_map.get("new_recipient", False))
        archetype = str(item.get("fraud_archetype", ""))

        if (
            transaction_type in {"direct debit", "e-commerce"}
            and new_recipient
            and recipient_global_count == 1
        ):
            return True
        if "payment_issue_to_remote_charge" in archetype:
            return True
        if (
            transaction_type == "transfer"
            and recent_phishing_count >= 2
            and any(marker in description for marker in ("bill", "utility bill", "phone bill"))
        ):
            return True
        if (
            transaction_type == "e-commerce"
            and amount >= 700
            and recent_phishing_count >= 1
        ):
            return True
        if (
            transaction_type == "direct debit"
            and payment_issue_count >= 1
            and recent_phishing_count >= 1
        ):
            return True
        return False

    def _select_llm_candidates(
        self,
        candidate_by_id: dict[str, CandidateTransaction],
        scored_items: list[dict],
    ) -> list[CandidateTransaction]:
        frame = pd.DataFrame(scored_items)
        frame = frame.sort_values(
            by=["combined_score", "campaign_score", "sequence_score", "economic_risk_score", "heuristic_score"],
            ascending=False,
        )
        strong = frame[
            (frame["combined_score"] >= 3.0)
            | (frame["campaign_score"] >= 1.0)
            | (frame["sequence_score"] >= 1.5)
            | (frame["economic_risk_score"] >= 2.0)
            | ((frame["final_label"] == "fraud") & (frame["combined_score"] >= 2.0))
        ]
        reserve = frame[
            (frame["combined_score"] >= 1.25)
            | (frame["campaign_score"] >= 0.7)
            | (frame["sequence_score"] >= 1.0)
            | (frame["economic_risk_score"] >= 1.4)
            | (frame["heuristic_score"] >= 1.4)
        ]
        reserve = reserve[~reserve["transaction_id"].isin(set(strong["transaction_id"]))]

        strong_sender_cap = 14
        reserve_sender_cap = 4
        reserve_global_cap = max(24, min(len(frame), int(len(frame) * 0.08)))

        strong_per_sender = strong.groupby("sender_id", sort=False).head(strong_sender_cap)
        reserve_global = reserve.head(reserve_global_cap)
        reserve_per_sender = reserve.groupby("sender_id", sort=False).head(reserve_sender_cap)

        merged = pd.concat(
            [strong_per_sender, reserve_global, reserve_per_sender],
            ignore_index=True,
        ).drop_duplicates(
            subset=["transaction_id"]
        )
        ids = merged["transaction_id"].tolist()
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
        "sequence_score": candidate.sequence_score,
        "campaign_score": candidate.campaign_score,
        "combined_score": candidate.combined_score,
        "final_label": final_label,
        "final_score": round(final_score, 3),
        "reasons": final_reasons,
        "candidate_reasons": candidate.reasons,
        "feature_map": candidate.feature_map,
        "fraud_archetype": candidate.network_summary["fraud_archetype"],
        "campaign_tag": candidate.network_summary["campaign_tag"],
        "recent_phishing_examples": candidate.communication_summary.recent_phishing_examples,
    }


def build_pipeline(
    dataset_dir: str | Path | None = None,
    include_audio: bool = False,
    enable_campaign_features: bool = False,
    enable_domain_features: bool = False,
    enable_credential_link_features: bool = False,
    enable_message_transaction_fit: bool = False,
    enable_bill_pattern_features: bool = False,
    llm_concurrency: int = 2,
) -> FraudDetectionPipeline:
    settings = load_settings(
        dataset_dir,
        include_audio=include_audio,
        enable_campaign_features=enable_campaign_features,
        enable_domain_features=enable_domain_features,
        enable_credential_link_features=enable_credential_link_features,
        enable_message_transaction_fit=enable_message_transaction_fit,
        enable_bill_pattern_features=enable_bill_pattern_features,
        llm_concurrency=llm_concurrency,
    )
    return FraudDetectionPipeline(settings)
