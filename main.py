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
        "--output-file",
        help="Path to the ASCII output file where suspicious transaction IDs will be written, one per line.",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Enable optional audio transcription input if the dataset contains an audio directory.",
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
    parser.add_argument(
        "--print-langfuse-session",
        action="store_true",
        help="Print Langfuse session id to stdout before transaction IDs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.report and not args.output_file:
        raise SystemExit("--output-file is required unless --report is used.")

    pipeline = build_pipeline(args.dataset_dir, include_audio=args.audio)
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

    if args.print_langfuse_session:
        print(f"langfuse_session_id={pipeline.orchestrator.session_id}")
        print(f"llm_enabled={str(pipeline.orchestrator.enabled).lower()}")
        print(f"langfuse_enabled={str(pipeline.settings.langfuse_enabled).lower()}")

    output_path = args.output_file
    if output_path is not None:
        with open(output_path, "w", encoding="ascii", newline="\n") as handle:
            for transaction_id in result.fraud_transaction_ids:
                handle.write(f"{transaction_id}\n")


if __name__ == "__main__":
    main()
