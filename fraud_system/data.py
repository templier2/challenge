from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def parse_datetime(value: str, fmt: str | None = None) -> datetime:
    if fmt:
        return datetime.strptime(value, fmt)
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.strptime(value, "%a, %d %b %Y %H:%M:%S %z").replace(tzinfo=None)


@dataclass(slots=True)
class UserProfile:
    biotag: str
    first_name: str
    last_name: str
    salary: float
    job: str
    iban: str
    city: str
    residence_lat: float
    residence_lng: float
    description: str

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


@dataclass(slots=True)
class Transaction:
    transaction_id: str
    sender_id: str
    recipient_id: str
    transaction_type: str
    amount: float
    location: str
    payment_method: str
    sender_iban: str
    recipient_iban: str
    balance_after: float
    description: str
    timestamp: datetime

    @property
    def balance_before(self) -> float:
        return self.balance_after + self.amount


@dataclass(slots=True)
class LocationPing:
    biotag: str
    timestamp: datetime
    lat: float
    lng: float
    city: str


@dataclass(slots=True)
class Communication:
    channel: str
    user_biotag: str
    sender: str
    timestamp: datetime
    subject: str
    body: str


@dataclass(slots=True)
class Dataset:
    users: list[UserProfile]
    transactions: list[Transaction]
    locations: list[LocationPing]
    communications: list[Communication]


def _derive_biotag(first_name: str, last_name: str, city: str, index: int) -> str:
    return (
        f"{last_name[:4].upper():<4}".replace(" ", "X")
        + "-"
        + f"{first_name[:4].upper():<4}".replace(" ", "X")
        + f"-AUTO-{city[:3].upper()}-{index}"
    )


def load_users(dataset_dir: Path) -> list[UserProfile]:
    payload = json.loads((dataset_dir / "users.json").read_text())
    biotag_by_iban: dict[str, str] = {}
    for row in csv.DictReader((dataset_dir / "transactions.csv").open()):
        sender_id = row["sender_id"]
        sender_iban = row["sender_iban"]
        if "-" in sender_id and sender_iban:
            biotag_by_iban[sender_iban] = sender_id
    users: list[UserProfile] = []
    for idx, row in enumerate(payload):
        biotag = biotag_by_iban.get(row["iban"]) or _derive_biotag(
            row["first_name"],
            row["last_name"],
            row["residence"]["city"],
            idx,
        )
        users.append(
            UserProfile(
                biotag=biotag,
                first_name=row["first_name"],
                last_name=row["last_name"],
                salary=float(row["salary"]),
                job=row["job"],
                iban=row["iban"],
                city=row["residence"]["city"],
                residence_lat=float(row["residence"]["lat"]),
                residence_lng=float(row["residence"]["lng"]),
                description=row["description"],
            )
        )
    return users


def load_transactions(dataset_dir: Path) -> list[Transaction]:
    with (dataset_dir / "transactions.csv").open() as handle:
        rows = csv.DictReader(handle)
        transactions = [
            Transaction(
                transaction_id=row["transaction_id"],
                sender_id=row["sender_id"],
                recipient_id=row["recipient_id"],
                transaction_type=row["transaction_type"],
                amount=float(row["amount"]),
                location=row["location"],
                payment_method=row["payment_method"],
                sender_iban=row["sender_iban"],
                recipient_iban=row["recipient_iban"],
                balance_after=float(row["balance_after"]),
                description=row["description"],
                timestamp=parse_datetime(row["timestamp"]),
            )
            for row in rows
        ]
    return sorted(transactions, key=lambda item: item.timestamp)


def load_locations(dataset_dir: Path) -> list[LocationPing]:
    payload = json.loads((dataset_dir / "locations.json").read_text())
    return [
        LocationPing(
            biotag=row["biotag"],
            timestamp=parse_datetime(row["timestamp"]),
            lat=float(row["lat"]),
            lng=float(row["lng"]),
            city=row["city"],
        )
        for row in payload
    ]


def _extract_line(block: str, prefix: str) -> str:
    match = re.search(rf"^{re.escape(prefix)}\s*(.+)$", block, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _assign_user(text: str, users: list[UserProfile]) -> str | None:
    lowered = text.lower()
    for user in users:
        if user.first_name.lower() in lowered or user.last_name.lower() in lowered:
            return user.biotag
    return None


def load_sms(dataset_dir: Path, users: list[UserProfile]) -> list[Communication]:
    payload = json.loads((dataset_dir / "sms.json").read_text())
    communications: list[Communication] = []
    for row in payload:
        sms = row["sms"]
        user_biotag = _assign_user(sms, users)
        if not user_biotag:
            continue
        communications.append(
            Communication(
                channel="sms",
                user_biotag=user_biotag,
                sender=_extract_line(sms, "From:"),
                timestamp=parse_datetime(_extract_line(sms, "Date:"), "%Y-%m-%d %H:%M:%S"),
                subject="",
                body=_extract_line(sms, "Message:"),
            )
        )
    return communications


def load_mails(dataset_dir: Path, users: list[UserProfile]) -> list[Communication]:
    payload = json.loads((dataset_dir / "mails.json").read_text())
    communications: list[Communication] = []
    for row in payload:
        mail = row["mail"]
        user_biotag = _assign_user(mail, users)
        if not user_biotag:
            continue
        communications.append(
            Communication(
                channel="mail",
                user_biotag=user_biotag,
                sender=_extract_line(mail, "From:"),
                timestamp=parse_datetime(_extract_line(mail, "Date:")),
                subject=_extract_line(mail, "Subject:"),
                body=mail,
            )
        )
    return communications


def load_dataset(dataset_dir: Path) -> Dataset:
    users = load_users(dataset_dir)
    communications = load_sms(dataset_dir, users) + load_mails(dataset_dir, users)
    return Dataset(
        users=users,
        transactions=load_transactions(dataset_dir),
        locations=load_locations(dataset_dir),
        communications=sorted(communications, key=lambda item: item.timestamp),
    )
