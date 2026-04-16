from __future__ import annotations

import argparse
import json

from fraud_system.pipeline import build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect fraudulent transactions in The Truman Show dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        default="The Truman Show - train",
        help="Directory containing transactions.csv, users.json, locations.json, sms.json, mails.json",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print the full candidate report instead of only fraud transaction IDs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fraud score threshold used when building the final suspicious set.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="If > 0, return the top-K scored suspicious transactions instead of using only thresholding.",
    )
    parser.add_argument(
        "--max-fraud-ratio",
        type=float,
        default=0.75,
        help="Upper bound for reported suspicious transactions as a fraction of all scored user transactions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = build_pipeline(args.dataset_dir)
    result = pipeline.run(
        threshold=args.threshold,
        max_fraud_ratio=args.max_fraud_ratio,
        top_k=args.top_k,
    )
    if args.report:
        payload = {
            "fraud_transaction_ids": result.fraud_transaction_ids,
            "reviewed_candidates": result.reviewed_candidates,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    for transaction_id in result.fraud_transaction_ids:
        print(transaction_id)


if __name__ == "__main__":
    main()
