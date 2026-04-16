from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta

from fraud_system.data import Communication, Dataset, LocationPing, Transaction, UserProfile


PHISHING_KEYWORDS = (
    "urgent",
    "verify",
    "suspicious",
    "locked",
    "suspension",
    "secure",
    "unusual login",
    "account",
    "payment failed",
)
LEGIT_DESCRIPTION_MARKERS = (
    "salary",
    "rent",
    "monthly plan",
    "insurance",
    "subscription",
    "bill",
)
SUSPICIOUS_DOMAINS = (
    "amaz0n",
    "paypa1",
    "ub3r",
    "verify",
    "secure",
    "bit.ly",
)


@dataclass(slots=True)
class CommunicationSummary:
    recent_phishing_count: int
    recent_phishing_examples: list[str]
    recent_legit_count: int
    recent_summaries: list[dict[str, str | int | bool]]
    prompt_injection_attempts: int


@dataclass(slots=True)
class CandidateTransaction:
    transaction: Transaction
    user: UserProfile
    heuristic_score: float
    economic_risk_score: float
    combined_score: float
    reasons: list[str]
    feature_map: dict[str, float | int | str | bool]
    communication_summary: CommunicationSummary
    location_summary: dict[str, str | float | bool]
    network_summary: dict[str, str | int | bool]


def _message_risk(message: Communication) -> bool:
    text = f"{message.sender} {message.subject} {message.body}".lower()
    return any(keyword in text for keyword in PHISHING_KEYWORDS) and any(
        domain in text for domain in SUSPICIOUS_DOMAINS
    )


def _message_legit(message: Communication) -> bool:
    text = f"{message.sender} {message.subject} {message.body}".lower()
    suspicious = any(domain in text for domain in SUSPICIOUS_DOMAINS)
    return not suspicious and ("https://" in text or "http://" in text)


def _extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s'\">)]+", text)


def _extract_domains(text: str) -> list[str]:
    domains: list[str] = []
    for url in _extract_urls(text):
        match = re.search(r"https?://([^/\s]+)", url)
        if match:
            domains.append(match.group(1).lower())
    return domains


def _prompt_injection_markers(text: str) -> list[str]:
    lowered = text.lower()
    markers = (
        "ignore previous instructions",
        "ignore the above",
        "system prompt",
        "assistant",
        "developer message",
        "tool call",
        "return json",
        "output legit",
        "mark this as legitimate",
    )
    return [marker for marker in markers if marker in lowered]


def _sanitize_snippet(text: str, limit: int = 120) -> str:
    collapsed = " ".join(text.split())
    collapsed = re.sub(r"https?://[^\s]+", "[url]", collapsed)
    collapsed = re.sub(
        r"ignore previous instructions|ignore the above|system prompt|developer message|tool call",
        "[instruction-like-text]",
        collapsed,
        flags=re.IGNORECASE,
    )
    return collapsed[:limit]


def _communication_summary(
    user: UserProfile,
    transaction: Transaction,
    communications: list[Communication],
) -> CommunicationSummary:
    start = transaction.timestamp - timedelta(days=30)
    recent = [
        item
        for item in communications
        if item.user_biotag == user.biotag and start <= item.timestamp <= transaction.timestamp
    ]
    phishing = [item for item in recent if _message_risk(item)]
    legit = [item for item in recent if _message_legit(item)]
    examples = []
    recent_summaries: list[dict[str, str | int | bool]] = []
    prompt_injection_attempts = 0
    for item in phishing[:3]:
        snippet = item.body.strip().replace("\n", " ")
        examples.append(snippet[:140])
    for item in recent[-5:]:
        text = f"{item.sender} {item.subject} {item.body}"
        domains = _extract_domains(text)
        injection_markers = _prompt_injection_markers(text)
        prompt_injection_attempts += int(bool(injection_markers))
        recent_summaries.append(
            {
                "channel": item.channel,
                "sender": _sanitize_snippet(item.sender, 60),
                "subject": _sanitize_snippet(item.subject, 80),
                "snippet": _sanitize_snippet(item.body),
                "suspicious_domains": ",".join(domains[:3]),
                "has_suspicious_domain": any(
                    any(domain_marker in domain for domain_marker in SUSPICIOUS_DOMAINS)
                    for domain in domains
                ),
                "has_urgent_language": any(
                    keyword in text.lower() for keyword in PHISHING_KEYWORDS
                ),
                "prompt_injection_like": bool(injection_markers),
                "prompt_injection_markers": ",".join(injection_markers[:3]),
            }
        )
    return CommunicationSummary(
        recent_phishing_count=len(phishing),
        recent_phishing_examples=examples,
        recent_legit_count=len(legit),
        recent_summaries=recent_summaries,
        prompt_injection_attempts=prompt_injection_attempts,
    )


def _nearest_location(
    user: UserProfile,
    transaction: Transaction,
    locations: list[LocationPing],
) -> dict[str, str | float | bool]:
    relevant = [
        item for item in locations if item.biotag == user.biotag and item.timestamp <= transaction.timestamp
    ]
    if not relevant:
        return {
            "has_recent_location": False,
            "recent_city": user.city,
            "hours_since_ping": -1.0,
            "city_mismatch": False,
        }
    latest = max(relevant, key=lambda item: item.timestamp)
    hours_since_ping = (transaction.timestamp - latest.timestamp).total_seconds() / 3600
    city_mismatch = bool(transaction.location and latest.city not in transaction.location)
    return {
        "has_recent_location": True,
        "recent_city": latest.city,
        "hours_since_ping": round(hours_since_ping, 2),
        "city_mismatch": city_mismatch,
    }


def _amount_stats(history: list[Transaction]) -> tuple[float, float]:
    if not history:
        return 0.0, 0.0
    amounts = [item.amount for item in history]
    mean = sum(amounts) / len(amounts)
    variance = sum((value - mean) ** 2 for value in amounts) / len(amounts)
    return mean, math.sqrt(variance)


def _description_flag(description: str) -> bool:
    lowered = description.lower()
    return any(marker in lowered for marker in LEGIT_DESCRIPTION_MARKERS)


def _description_tokens(description: str) -> set[str]:
    return set(re.findall(r"[a-z]+", description.lower()))


def _looks_like_recurring_obligation(description: str) -> bool:
    lowered = description.lower()
    markers = ("rent", "monthly plan", "insurance", "subscription", "bill", "premium")
    return any(marker in lowered for marker in markers)


def _future_recurring_match(
    transaction: Transaction,
    all_transactions: list[Transaction],
) -> bool:
    if not transaction.recipient_id or not _looks_like_recurring_obligation(transaction.description):
        return False
    future_matches = [
        item
        for item in all_transactions
        if item.sender_id == transaction.sender_id
        and item.recipient_id == transaction.recipient_id
        and item.timestamp > transaction.timestamp
    ]
    if len(future_matches) < 2:
        return False
    return all(_looks_like_recurring_obligation(item.description) for item in future_matches[:3])


def _economic_risk(
    user: UserProfile,
    transaction: Transaction,
    amount_zscore: float,
    drain_ratio: float,
    phishing_count: int,
    new_recipient: bool,
    recurring_same_recipient: bool,
    structured_description: bool,
    first_recurring_obligation: bool,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    monthly_salary = max(user.salary / 12.0, 1.0)
    salary_ratio = transaction.amount / monthly_salary

    if transaction.transaction_type == "transfer":
        score += 0.7
        reasons.append("transfer has higher financial risk")

    if transaction.amount >= 900:
        score += 1.0
        reasons.append("high absolute amount")
    elif transaction.amount >= 300:
        score += 0.5
        reasons.append("medium absolute amount")

    if salary_ratio >= 0.4:
        score += 0.9
        reasons.append("large relative to monthly salary")
    elif salary_ratio >= 0.15:
        score += 0.4
        reasons.append("noticeable relative to monthly salary")

    if drain_ratio >= 0.2:
        score += 1.2
        reasons.append("material balance drain")
    elif drain_ratio >= 0.08:
        score += 0.5
        reasons.append("moderate balance drain")

    if amount_zscore >= 2.5:
        score += 0.9
        reasons.append("large deviation from personal amount history")
    elif amount_zscore >= 1.5:
        score += 0.4
        reasons.append("meaningful deviation from personal amount history")

    if phishing_count >= 2 and new_recipient:
        score += 1.0
        reasons.append("new payee after repeated phishing pressure")
    elif phishing_count >= 1 and transaction.amount >= 300:
        score += 0.4
        reasons.append("non-trivial payment near phishing activity")

    if transaction.transaction_type in {"e-commerce", "direct debit"} and transaction.amount < 80:
        score -= 0.8
        reasons.append("small remote charge has limited economic impact")

    if recurring_same_recipient and structured_description:
        score -= 1.6
        reasons.append("established recurring obligation lowers fraud impact priority")

    if first_recurring_obligation:
        score -= 1.3
        reasons.append("first payment is supported by future recurring pattern")

    return round(score, 3), reasons


def score_transactions(dataset: Dataset) -> list[CandidateTransaction]:
    user_by_biotag = {user.biotag: user for user in dataset.users}
    history_by_sender: dict[str, list[Transaction]] = defaultdict(list)
    recipient_counter_by_sender: dict[str, Counter[str]] = defaultdict(Counter)
    recipient_global_counter: Counter[str] = Counter(
        transaction.recipient_id for transaction in dataset.transactions if transaction.recipient_id
    )
    candidates: list[CandidateTransaction] = []

    for transaction in dataset.transactions:
        user = user_by_biotag.get(transaction.sender_id)
        if not user:
            continue

        history = history_by_sender[transaction.sender_id]
        recipient_counter = recipient_counter_by_sender[transaction.sender_id]
        comms = _communication_summary(user, transaction, dataset.communications)
        location = _nearest_location(user, transaction, dataset.locations)
        outbound_history = [item for item in history if item.transaction_type == "transfer"]
        mean, std = _amount_stats(outbound_history)
        prior_same_recipient = [
            item for item in history if item.recipient_id and item.recipient_id == transaction.recipient_id
        ]

        new_recipient = recipient_counter[transaction.recipient_id] == 0 if transaction.recipient_id else False
        amount_zscore = 0.0
        if std > 0:
            amount_zscore = (transaction.amount - mean) / std

        recipient_global_count = recipient_global_counter[transaction.recipient_id]
        recurring_same_recipient = len(prior_same_recipient) >= 2
        first_recurring_obligation = (
            len(prior_same_recipient) == 0
            and _future_recurring_match(transaction, dataset.transactions)
        )
        recipient_description_shift = False
        if prior_same_recipient and transaction.description:
            current_tokens = _description_tokens(transaction.description)
            prior_tokens = set().union(
                *(_description_tokens(item.description) for item in prior_same_recipient if item.description)
            )
            if current_tokens and prior_tokens:
                overlap = len(current_tokens & prior_tokens) / max(1, len(current_tokens | prior_tokens))
                recipient_description_shift = overlap < 0.25

        reasons: list[str] = []
        score = 0.0

        if transaction.transaction_type == "transfer" and new_recipient:
            score += 1.4
            reasons.append("new recipient for sender")

        if (
            transaction.transaction_type == "transfer"
            and amount_zscore > 2.0
            and (new_recipient or not _description_flag(transaction.description))
        ):
            score += 1.0
            reasons.append(f"amount spike z={amount_zscore:.2f}")

        if transaction.transaction_type in {"transfer", "e-commerce"} and comms.recent_phishing_count:
            bonus = min(1.6, 0.5 * comms.recent_phishing_count)
            score += bonus
            reasons.append(f"recent phishing pressure ({comms.recent_phishing_count} events)")

        if transaction.transaction_type == "in-person payment" and bool(location["city_mismatch"]):
            score += 1.2
            reasons.append("location mismatch vs latest GPS ping")

        if transaction.transaction_type == "transfer" and not _description_flag(transaction.description):
            score += 0.6
            reasons.append("unstructured transfer description")

        drain_ratio = transaction.amount / transaction.balance_before if transaction.balance_before else 0.0
        if drain_ratio >= 0.35:
            score += 0.8
            reasons.append(f"large balance drain ({drain_ratio:.0%})")

        if transaction.transaction_type == "e-commerce" and transaction.payment_method.lower() in {
            "paypal",
            "googlepay",
        }:
            score += 0.4
            reasons.append("remote wallet payment")

        if new_recipient and recipient_global_count == 1 and transaction.transaction_type == "transfer":
            score += 0.7
            reasons.append("recipient appears only once in dataset")

        if transaction.transaction_type in {"direct debit", "e-commerce"} and new_recipient:
            score += 1.2
            reasons.append("first charge from new merchant")

        if transaction.transaction_type in {"direct debit", "e-commerce"} and recipient_global_count == 1:
            score += 0.9
            reasons.append("merchant appears only once in dataset")

        if transaction.transaction_type == "direct debit" and recipient_description_shift:
            score += 1.1
            reasons.append("same recipient reused with different billing purpose")

        if recurring_same_recipient and _description_flag(transaction.description):
            score -= 1.0
            reasons.append("matches established recurring payment pattern")

        if first_recurring_obligation:
            score -= 1.2
            reasons.append("looks like the start of a recurring obligation")

        economic_risk_score, economic_reasons = _economic_risk(
            user=user,
            transaction=transaction,
            amount_zscore=amount_zscore,
            drain_ratio=drain_ratio,
            phishing_count=comms.recent_phishing_count,
            new_recipient=new_recipient,
            recurring_same_recipient=recurring_same_recipient,
            structured_description=_description_flag(transaction.description),
            first_recurring_obligation=first_recurring_obligation,
        )
        combined_score = score + 1.35 * economic_risk_score

        network_summary = {
            "new_recipient": new_recipient,
            "recipient_seen_for_sender": recipient_counter[transaction.recipient_id],
            "recipient_global_count": recipient_global_count,
            "description_is_structured": _description_flag(transaction.description),
            "recurring_same_recipient": recurring_same_recipient,
            "first_recurring_obligation": first_recurring_obligation,
            "recipient_description_shift": recipient_description_shift,
        }

        feature_map = {
            "amount": transaction.amount,
            "balance_before": round(transaction.balance_before, 2),
            "balance_after": transaction.balance_after,
            "amount_zscore": round(amount_zscore, 3),
            "drain_ratio": round(drain_ratio, 3),
            "recent_phishing_count": comms.recent_phishing_count,
            "recent_legit_count": comms.recent_legit_count,
            "new_recipient": new_recipient,
            "recipient_global_count": recipient_global_count,
            "salary_ratio": round(transaction.amount / max(user.salary / 12.0, 1.0), 3),
            "transaction_type": transaction.transaction_type,
            "payment_method": transaction.payment_method or "",
            "description": transaction.description or "",
            "location": transaction.location or "",
            "economic_risk_score": economic_risk_score,
            "combined_score": round(combined_score, 3),
        }

        candidates.append(
            CandidateTransaction(
                transaction=transaction,
                user=user,
                heuristic_score=round(score, 3),
                economic_risk_score=economic_risk_score,
                combined_score=round(combined_score, 3),
                reasons=reasons + economic_reasons,
                feature_map=feature_map,
                communication_summary=comms,
                location_summary=location,
                network_summary=network_summary,
            )
        )

        history.append(transaction)
        if transaction.recipient_id:
            recipient_counter[transaction.recipient_id] += 1

    return sorted(candidates, key=lambda item: item.combined_score, reverse=True)


def baseline_decision(candidate: CandidateTransaction) -> tuple[bool, str]:
    phishing = candidate.communication_summary.recent_phishing_count
    new_recipient = bool(candidate.network_summary["new_recipient"])
    recipient_global_count = int(candidate.network_summary["recipient_global_count"])
    recurring_same_recipient = bool(candidate.network_summary["recurring_same_recipient"])
    first_recurring_obligation = bool(candidate.network_summary["first_recurring_obligation"])
    recipient_description_shift = bool(candidate.network_summary["recipient_description_shift"])
    drain_ratio = float(candidate.feature_map["drain_ratio"])
    structured = bool(candidate.network_summary["description_is_structured"])
    transaction_type = str(candidate.feature_map["transaction_type"])
    description = str(candidate.feature_map["description"]).lower()
    bill_like = "bill" in description or "invoice" in description

    fraud = (
        candidate.combined_score >= 4.0
        or (phishing >= 2 and new_recipient and transaction_type == "transfer")
        or (
            transaction_type == "transfer"
            and new_recipient
            and recipient_global_count == 1
            and (bill_like or not structured)
        )
        or (
            transaction_type in {"direct debit", "e-commerce"}
            and new_recipient
            and recipient_global_count == 1
        )
        or (
            transaction_type == "direct debit"
            and recipient_description_shift
            and not recurring_same_recipient
        )
        or (phishing >= 1 and drain_ratio >= 0.25 and not structured)
    )
    if first_recurring_obligation:
        fraud = False

    rationale = "; ".join(candidate.reasons) if candidate.reasons else "no abnormal signals"
    return fraud, rationale
