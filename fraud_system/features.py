from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta

from fraud_system.config import Settings
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
CAMPAIGN_DESCRIPTION_MARKERS = (
    "donation",
    "supplier payment",
    "consultant fee",
    "invoice settlement",
    "emergency fund transfer",
    "phone bill",
    "savings transfer",
)
BRAND_DOMAIN_HINTS = {
    "paypal": ("paypal.com", "paypal."),
    "amazon": ("amazon.",),
    "uber": ("uber.com", "uber."),
    "dhl": ("dhl.com", "dhl.", "dhlparcel."),
    "barclays": ("barclays.co.uk", "barclays."),
    "britishgas": ("britishgas.co.uk", "britishgas."),
    "commerzbank": ("commerzbank.de", "commerzbank."),
    "chase": ("chase.com", "chase."),
    "unicredit": ("unicredit.it", "unicredit."),
}


@dataclass(slots=True)
class CommunicationSummary:
    recent_phishing_count: int
    recent_phishing_examples: list[str]
    recent_legit_count: int
    recent_summaries: list[dict[str, str | int | bool]]
    prompt_injection_attempts: int
    login_alert_count: int
    verify_link_count: int
    payment_issue_count: int
    delivery_fee_count: int
    brand_mismatch_count: int
    credential_link_count: int
    recent_event_types: list[str]


@dataclass(slots=True)
class CandidateTransaction:
    transaction: Transaction
    user: UserProfile
    heuristic_score: float
    economic_risk_score: float
    sequence_score: float
    campaign_score: float
    combined_score: float
    reasons: list[str]
    feature_map: dict[str, float | int | str | bool]
    communication_summary: CommunicationSummary
    location_summary: dict[str, str | float | bool]
    network_summary: dict[str, str | int | float | bool]


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


def _message_event_types(message: Communication) -> set[str]:
    text = f"{message.sender} {message.subject} {message.body}".lower()
    event_types: set[str] = set()
    if any(term in text for term in ("suspicious login", "unusual login", "sign-in", "logged in")):
        event_types.add("login_alert")
    if any(term in text for term in ("verify", "confirm", "restore access", "secure your account")):
        event_types.add("verify_link")
    if any(term in text for term in ("payment failed", "renewal failed", "update payment", "billing")):
        event_types.add("payment_issue")
    if any(term in text for term in ("parcel", "customs", "delivery fee", "release fee", "dhl")):
        event_types.add("delivery_fee")
    if _extract_urls(text):
        event_types.add("has_link")
    return event_types


def _sender_domain(sender: str) -> str:
    match = re.search(r"<[^@<>]+@([^<>]+)>", sender)
    if match:
        return match.group(1).lower()
    match = re.search(r"@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", sender)
    if match:
        return match.group(1).lower()
    return ""


def _brand_domain_mismatch(message: Communication) -> tuple[bool, str]:
    text = f"{message.sender} {message.subject} {message.body}".lower()
    domain = _sender_domain(message.sender)
    if not domain:
        return False, ""
    for brand, expected_hints in BRAND_DOMAIN_HINTS.items():
        if brand in text:
            if not any(hint in domain for hint in expected_hints):
                return True, brand
            suspicious_brand_variant = brand.replace("a", "4").replace("o", "0").replace("i", "1")
            if suspicious_brand_variant in domain and suspicious_brand_variant != brand:
                return True, brand
    if any(marker in domain for marker in SUSPICIOUS_DOMAINS):
        for brand, expected_hints in BRAND_DOMAIN_HINTS.items():
            if brand in text and not any(hint in domain for hint in expected_hints):
                return True, brand
    return False, ""


def _credential_harvest_link(message: Communication) -> bool:
    text = f"{message.sender} {message.subject} {message.body}".lower()
    event_types = _message_event_types(message)
    has_link = "has_link" in event_types
    suspicious_domain = any(
        any(marker in domain for marker in SUSPICIOUS_DOMAINS)
        for domain in _extract_domains(text)
    )
    brand_mismatch, _ = _brand_domain_mismatch(message)
    lure_language = any(
        term in text
        for term in (
            "verify",
            "confirm",
            "restore access",
            "update payment",
            "suspicious login",
            "unusual login",
        )
    )
    return has_link and lure_language and (suspicious_domain or brand_mismatch)


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
    event_counter: Counter[str] = Counter()
    brand_mismatch_count = 0
    credential_link_count = 0
    for item in phishing[:3]:
        snippet = item.body.strip().replace("\n", " ")
        examples.append(snippet[:140])
    for item in recent[-5:]:
        text = f"{item.sender} {item.subject} {item.body}"
        domains = _extract_domains(text)
        injection_markers = _prompt_injection_markers(text)
        event_types = _message_event_types(item)
        brand_mismatch, mismatch_brand = _brand_domain_mismatch(item)
        credential_link = _credential_harvest_link(item)
        prompt_injection_attempts += int(bool(injection_markers))
        event_counter.update(event_types)
        brand_mismatch_count += int(brand_mismatch)
        credential_link_count += int(credential_link)
        recent_summaries.append(
            {
                "channel": item.channel,
                "sender": _sanitize_snippet(item.sender, 60),
                "subject": _sanitize_snippet(item.subject, 80),
                "snippet": _sanitize_snippet(item.body),
                "event_types": ",".join(sorted(event_types)),
                "brand_domain_mismatch": brand_mismatch,
                "brand_mismatch_brand": mismatch_brand,
                "credential_link": credential_link,
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
        login_alert_count=event_counter["login_alert"],
        verify_link_count=event_counter["verify_link"],
        payment_issue_count=event_counter["payment_issue"],
        delivery_fee_count=event_counter["delivery_fee"],
        brand_mismatch_count=brand_mismatch_count,
        credential_link_count=credential_link_count,
        recent_event_types=sorted(event_counter),
    )


def _latest_phishing_delta_hours(
    user: UserProfile,
    transaction: Transaction,
    communications: list[Communication],
) -> float | None:
    phishing_messages = [
        item
        for item in communications
        if item.user_biotag == user.biotag
        and item.timestamp <= transaction.timestamp
        and _message_risk(item)
    ]
    if not phishing_messages:
        return None
    latest = max(phishing_messages, key=lambda item: item.timestamp)
    return (transaction.timestamp - latest.timestamp).total_seconds() / 3600


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


def _campaign_template(description: str) -> str:
    lowered = description.lower().strip()
    if not lowered:
        return ""
    if lowered.startswith("rent payment") or _looks_like_recurring_obligation(description):
        return ""
    for marker in CAMPAIGN_DESCRIPTION_MARKERS:
        if marker in lowered:
            return marker
    return ""


def _bill_like_one_off_label(description: str) -> bool:
    lowered = description.lower()
    markers = (
        "bill",
        "invoice",
        "supplier payment",
        "consultant fee",
        "service payment",
        "phone bill",
        "utility bill",
        "donation",
    )
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


def _unusual_hour(transaction: Transaction, history: list[Transaction]) -> bool:
    if len(history) < 6:
        return False
    prior_hours = [item.timestamp.hour for item in history]
    current_bucket = transaction.timestamp.hour // 6
    bucket_counts = Counter(hour // 6 for hour in prior_hours)
    return bucket_counts[current_bucket] <= 1


def _new_payment_channel(transaction: Transaction, history: list[Transaction]) -> bool:
    payment_method = transaction.payment_method.strip().lower()
    if not payment_method:
        return False
    prior_methods = {
        item.payment_method.strip().lower()
        for item in history
        if item.payment_method.strip()
    }
    return bool(prior_methods) and payment_method not in prior_methods


def _recurring_break(
    transaction: Transaction,
    prior_same_recipient: list[Transaction],
    amount_zscore: float,
    description_shift: bool,
) -> bool:
    if len(prior_same_recipient) < 2:
        return False
    if description_shift and amount_zscore >= 1.0:
        return True
    latest_gap_days = (
        transaction.timestamp - prior_same_recipient[-1].timestamp
    ).total_seconds() / 86400
    prior_gap_days = [
        (current.timestamp - previous.timestamp).total_seconds() / 86400
        for previous, current in zip(prior_same_recipient, prior_same_recipient[1:])
    ]
    if not prior_gap_days:
        return False
    typical_gap = sum(prior_gap_days) / len(prior_gap_days)
    return latest_gap_days < typical_gap * 0.4 or latest_gap_days > typical_gap * 1.8


def _sequence_risk(
    transaction: Transaction,
    history: list[Transaction],
    comms: CommunicationSummary,
    latest_phishing_hours: float | None,
    new_recipient: bool,
    amount_zscore: float,
    drain_ratio: float,
    unusual_hour: bool,
    new_payment_channel: bool,
    recurring_break: bool,
    recurring_same_recipient: bool,
    recipient_global_count: int,
    first_recurring_obligation: bool,
    enable_domain_features: bool,
    enable_credential_link_features: bool,
    enable_message_transaction_fit: bool,
    enable_bill_pattern_features: bool,
) -> tuple[float, list[str], str]:
    score = 0.0
    reasons: list[str] = []
    archetypes: list[str] = []

    if latest_phishing_hours is not None and latest_phishing_hours <= 72:
        if new_recipient and transaction.transaction_type == "transfer":
            score += 1.7
            reasons.append("transfer follows phishing communication within 72h")
            archetypes.append("phishing_to_new_payee")
        elif transaction.transaction_type in {"e-commerce", "direct debit"}:
            score += 1.1
            reasons.append("remote charge follows phishing communication within 72h")
            archetypes.append("phishing_to_remote_charge")
        elif drain_ratio >= 0.18:
            score += 0.8
            reasons.append("meaningful balance drain shortly after phishing")
            archetypes.append("phishing_to_balance_drain")

    if (
        comms.login_alert_count >= 1
        and comms.verify_link_count >= 1
        and new_recipient
        and transaction.transaction_type == "transfer"
    ):
        score += 1.0
        reasons.append("new payee transfer follows login alert and verification lure")
        archetypes.append("login_alert_to_new_payee")

    if (
        comms.payment_issue_count >= 1
        and transaction.transaction_type in {"e-commerce", "direct debit"}
        and new_recipient
    ):
        score += 0.8
        reasons.append("first remote charge follows payment issue message")
        archetypes.append("payment_issue_to_remote_charge")

    if (
        comms.delivery_fee_count >= 1
        and transaction.transaction_type == "transfer"
        and new_recipient
        and transaction.amount <= 250
    ):
        score += 0.9
        reasons.append("small new-payee transfer matches delivery-fee scam pattern")
        archetypes.append("delivery_fee_to_small_transfer")

    if enable_domain_features and comms.brand_mismatch_count >= 1:
        if new_recipient and transaction.transaction_type == "transfer":
            score += 1.0
            reasons.append("new payee transfer follows brand/domain mismatch communication")
            archetypes.append("brand_mismatch_to_new_payee")
        elif transaction.transaction_type in {"e-commerce", "direct debit"}:
            score += 0.8
            reasons.append("remote charge follows brand/domain mismatch communication")
            archetypes.append("brand_mismatch_to_remote_charge")
        else:
            score += 0.4
            reasons.append("transaction follows communication with impersonated brand domain")
            archetypes.append("brand_mismatch_contact")

    if enable_credential_link_features and comms.credential_link_count >= 1:
        if new_recipient and transaction.transaction_type == "transfer":
            score += 1.1
            reasons.append("new payee transfer follows credential-harvest link")
            archetypes.append("credential_link_to_new_payee")
        elif transaction.transaction_type in {"e-commerce", "direct debit"}:
            score += 0.9
            reasons.append("remote charge follows credential-harvest link")
            archetypes.append("credential_link_to_remote_charge")
        else:
            score += 0.4
            reasons.append("transaction follows message with likely credential-harvest link")
            archetypes.append("credential_link_contact")

    if enable_message_transaction_fit:
        if (
            comms.delivery_fee_count >= 1
            and transaction.transaction_type == "transfer"
            and transaction.amount <= 250
        ):
            score += 0.8
            reasons.append("transaction amount/type fits a delivery-fee scam narrative")
            archetypes.append("message_transaction_fit_delivery")
        if (
            comms.payment_issue_count >= 1
            and transaction.transaction_type in {"e-commerce", "direct debit"}
        ):
            score += 0.7
            reasons.append("remote payment fits a payment-update scam narrative")
            archetypes.append("message_transaction_fit_payment")
        if (
            comms.login_alert_count >= 1
            and comms.verify_link_count >= 1
            and transaction.transaction_type == "transfer"
            and new_recipient
        ):
            score += 0.8
            reasons.append("new-payee transfer fits an account-takeover narrative")
            archetypes.append("message_transaction_fit_takeover")
        if (
            comms.delivery_fee_count >= 1
            and transaction.transaction_type == "transfer"
            and transaction.amount >= 500
        ):
            score -= 0.6
            reasons.append("large transfer is a poor fit for a delivery-fee message narrative")
        if (
            comms.payment_issue_count >= 1
            and transaction.transaction_type == "transfer"
            and transaction.amount >= 500
        ):
            score -= 0.5
            reasons.append("large bank transfer is a weak fit for a payment-update narrative")

    if enable_bill_pattern_features:
        if (
            transaction.transaction_type == "transfer"
            and new_recipient
            and _bill_like_one_off_label(transaction.description)
            and not recurring_same_recipient
            and not first_recurring_obligation
        ):
            score += 0.9
            reasons.append("one-off bill-like transfer to a new payee is structurally suspicious")
            archetypes.append("one_off_fake_bill")
        if (
            transaction.transaction_type == "transfer"
            and recurring_same_recipient
            and _bill_like_one_off_label(transaction.description)
            and not recurring_break
        ):
            score -= 0.7
            reasons.append("bill-like transfer stays consistent with an established payment pattern")

    if unusual_hour and transaction.amount >= 200:
        score += 0.7
        reasons.append("activity lands in a rare time window for sender")
        archetypes.append("temporal_shift")

    if new_payment_channel and transaction.transaction_type in {"e-commerce", "in-person payment"}:
        score += 0.5
        reasons.append("uses a previously unseen payment channel")
        archetypes.append("channel_shift")

    if recurring_break and not first_recurring_obligation:
        score += 1.2
        reasons.append("known recipient pattern breaks in timing or purpose")
        archetypes.append("recurring_pattern_break")

    if (
        recipient_global_count == 1
        and new_recipient
        and transaction.amount >= 250
        and transaction.transaction_type in {"transfer", "direct debit", "e-commerce"}
    ):
        score += 0.9
        reasons.append("single-use counterparty paired with material amount")
        archetypes.append("single_use_counterparty")

    if (
        len(history) >= 4
        and new_recipient
        and comms.recent_phishing_count >= 1
        and drain_ratio >= 0.12
    ):
        score += 0.8
        reasons.append("sender shifts from normal behavior into a risky sequence")
        archetypes.append("behavioral_shift_after_contact")

    if recurring_same_recipient and not recurring_break:
        score -= 0.8
        reasons.append("recipient pattern remains stable across the sender timeline")

    if first_recurring_obligation:
        score -= 1.0

    archetype = ",".join(dict.fromkeys(archetypes)) if archetypes else "none"
    return round(score, 3), reasons, archetype


def _campaign_risk(
    transaction: Transaction,
    template_sender_count: int,
    template_occurrence_count: int,
    prefix_sender_count: int,
    prefix_occurrence_count: int,
    new_recipient: bool,
) -> tuple[float, list[str], str]:
    score = 0.0
    reasons: list[str] = []
    tags: list[str] = []

    template = _campaign_template(transaction.description)
    if template and template_sender_count >= 2 and template_occurrence_count >= 3:
        score += 1.1
        reasons.append("description matches a repeated cross-user transfer template")
        tags.append("cross_user_template")
        if new_recipient:
            score += 0.5
            reasons.append("template appears on a new recipient for this sender")
            tags.append("template_to_new_payee")

    prefix = (transaction.recipient_id or "")[:5]
    if (
        transaction.transaction_type == "transfer"
        and prefix
        and prefix_sender_count >= 2
        and prefix_occurrence_count >= 3
        and template
    ):
        score += 0.7
        reasons.append("recipient prefix recurs across users inside the same suspicious template family")
        tags.append("cross_user_recipient_family")

    campaign_tag = ",".join(dict.fromkeys(tags)) if tags else "none"
    return round(score, 3), reasons, campaign_tag


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


def score_transactions(dataset: Dataset, settings: Settings | None = None) -> list[CandidateTransaction]:
    enable_campaign_features = bool(settings and settings.enable_campaign_features)
    enable_domain_features = bool(settings and settings.enable_domain_features)
    enable_credential_link_features = bool(settings and settings.enable_credential_link_features)
    enable_message_transaction_fit = bool(settings and settings.enable_message_transaction_fit)
    enable_bill_pattern_features = bool(settings and settings.enable_bill_pattern_features)
    user_by_biotag = {user.biotag: user for user in dataset.users}
    history_by_sender: dict[str, list[Transaction]] = defaultdict(list)
    recipient_counter_by_sender: dict[str, Counter[str]] = defaultdict(Counter)
    recipient_global_counter: Counter[str] = Counter(
        transaction.recipient_id for transaction in dataset.transactions if transaction.recipient_id
    )
    template_occurrence_counter: Counter[str] = Counter()
    template_sender_sets: dict[str, set[str]] = defaultdict(set)
    prefix_occurrence_counter: Counter[str] = Counter()
    prefix_sender_sets: dict[str, set[str]] = defaultdict(set)
    for item in dataset.transactions:
        template = _campaign_template(item.description)
        if template:
            template_occurrence_counter[template] += 1
            template_sender_sets[template].add(item.sender_id)
        prefix = (item.recipient_id or "")[:5]
        if prefix:
            prefix_occurrence_counter[prefix] += 1
            prefix_sender_sets[prefix].add(item.sender_id)
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
        unusual_hour = _unusual_hour(transaction, history)
        new_payment_channel = _new_payment_channel(transaction, history)
        recipient_description_shift = False
        if prior_same_recipient and transaction.description:
            current_tokens = _description_tokens(transaction.description)
            prior_tokens = set().union(
                *(_description_tokens(item.description) for item in prior_same_recipient if item.description)
            )
            if current_tokens and prior_tokens:
                overlap = len(current_tokens & prior_tokens) / max(1, len(current_tokens | prior_tokens))
                recipient_description_shift = overlap < 0.25
        recurring_break = _recurring_break(
            transaction=transaction,
            prior_same_recipient=prior_same_recipient,
            amount_zscore=amount_zscore,
            description_shift=recipient_description_shift,
        )
        latest_phishing_hours = _latest_phishing_delta_hours(user, transaction, dataset.communications)
        campaign_template = _campaign_template(transaction.description)
        recipient_prefix = (transaction.recipient_id or "")[:5]

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
        sequence_score, sequence_reasons, fraud_archetype = _sequence_risk(
            transaction=transaction,
            history=history,
            comms=comms,
            latest_phishing_hours=latest_phishing_hours,
            new_recipient=new_recipient,
            amount_zscore=amount_zscore,
            drain_ratio=drain_ratio,
            unusual_hour=unusual_hour,
            new_payment_channel=new_payment_channel,
            recurring_break=recurring_break,
            recurring_same_recipient=recurring_same_recipient,
            recipient_global_count=recipient_global_count,
            first_recurring_obligation=first_recurring_obligation,
            enable_domain_features=enable_domain_features,
            enable_credential_link_features=enable_credential_link_features,
            enable_message_transaction_fit=enable_message_transaction_fit,
            enable_bill_pattern_features=enable_bill_pattern_features,
        )
        if enable_campaign_features:
            campaign_score, campaign_reasons, campaign_tag = _campaign_risk(
                transaction=transaction,
                template_sender_count=len(template_sender_sets[campaign_template]) if campaign_template else 0,
                template_occurrence_count=template_occurrence_counter[campaign_template]
                if campaign_template
                else 0,
                prefix_sender_count=len(prefix_sender_sets[recipient_prefix]) if recipient_prefix else 0,
                prefix_occurrence_count=prefix_occurrence_counter[recipient_prefix] if recipient_prefix else 0,
                new_recipient=new_recipient,
            )
        else:
            campaign_score, campaign_reasons, campaign_tag = 0.0, [], "disabled"
        combined_score = score + 1.2 * economic_risk_score + 1.1 * sequence_score + campaign_score

        network_summary = {
            "new_recipient": new_recipient,
            "recipient_seen_for_sender": recipient_counter[transaction.recipient_id],
            "recipient_global_count": recipient_global_count,
            "description_is_structured": _description_flag(transaction.description),
            "recurring_same_recipient": recurring_same_recipient,
            "first_recurring_obligation": first_recurring_obligation,
            "recipient_description_shift": recipient_description_shift,
            "recurring_break": recurring_break,
            "unusual_hour": unusual_hour,
            "new_payment_channel": new_payment_channel,
            "latest_phishing_hours": round(latest_phishing_hours, 2)
            if latest_phishing_hours is not None
            else -1.0,
            "fraud_archetype": fraud_archetype,
            "campaign_tag": campaign_tag,
        }

        feature_map = {
            "amount": transaction.amount,
            "balance_before": round(transaction.balance_before, 2),
            "balance_after": transaction.balance_after,
            "amount_zscore": round(amount_zscore, 3),
            "drain_ratio": round(drain_ratio, 3),
            "recent_phishing_count": comms.recent_phishing_count,
            "recent_legit_count": comms.recent_legit_count,
            "login_alert_count": comms.login_alert_count,
            "verify_link_count": comms.verify_link_count,
            "payment_issue_count": comms.payment_issue_count,
            "delivery_fee_count": comms.delivery_fee_count,
            "brand_mismatch_count": comms.brand_mismatch_count,
            "credential_link_count": comms.credential_link_count,
            "campaign_score": campaign_score,
            "campaign_template": campaign_template,
            "new_recipient": new_recipient,
            "recipient_global_count": recipient_global_count,
            "salary_ratio": round(transaction.amount / max(user.salary / 12.0, 1.0), 3),
            "transaction_type": transaction.transaction_type,
            "payment_method": transaction.payment_method or "",
            "description": transaction.description or "",
            "location": transaction.location or "",
            "economic_risk_score": economic_risk_score,
            "sequence_score": sequence_score,
            "combined_score": round(combined_score, 3),
        }

        candidates.append(
            CandidateTransaction(
                transaction=transaction,
                user=user,
                heuristic_score=round(score, 3),
                economic_risk_score=economic_risk_score,
                sequence_score=sequence_score,
                campaign_score=campaign_score,
                combined_score=round(combined_score, 3),
                reasons=reasons + economic_reasons + sequence_reasons + campaign_reasons,
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
    recurring_break = bool(candidate.network_summary["recurring_break"])
    drain_ratio = float(candidate.feature_map["drain_ratio"])
    sequence_score = float(candidate.feature_map["sequence_score"])
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
        or (recurring_break and sequence_score >= 1.0 and candidate.combined_score >= 3.2)
        or (phishing >= 1 and drain_ratio >= 0.25 and not structured)
    )
    if first_recurring_obligation:
        fraud = False

    rationale = "; ".join(candidate.reasons) if candidate.reasons else "no abnormal signals"
    return fraud, rationale
