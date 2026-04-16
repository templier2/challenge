from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv() -> bool:
        return False


@dataclass(slots=True)
class Settings:
    dataset_dir: Path
    include_audio: bool
    enable_campaign_features: bool
    enable_domain_features: bool
    enable_credential_link_features: bool
    enable_message_transaction_fit: bool
    enable_bill_pattern_features: bool
    llm_concurrency: int
    openrouter_api_key: str | None
    openrouter_model: str
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_host: str
    team_name: str

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


def load_settings(
    dataset_dir: str | Path | None = None,
    include_audio: bool = False,
    enable_campaign_features: bool = False,
    enable_domain_features: bool = False,
    enable_credential_link_features: bool = False,
    enable_message_transaction_fit: bool = False,
    enable_bill_pattern_features: bool = False,
    llm_concurrency: int = 2,
) -> Settings:
    load_dotenv()
    resolved_dataset_dir = Path(dataset_dir or "The Truman Show - train").resolve()
    return Settings(
        dataset_dir=resolved_dataset_dir,
        include_audio=include_audio,
        enable_campaign_features=enable_campaign_features,
        enable_domain_features=enable_domain_features,
        enable_credential_link_features=enable_credential_link_features,
        enable_message_transaction_fit=enable_message_transaction_fit,
        enable_bill_pattern_features=enable_bill_pattern_features,
        llm_concurrency=max(1, llm_concurrency),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        langfuse_host=os.getenv(
            "LANGFUSE_HOST",
            "https://challenges.reply.com/langfuse",
        ),
        team_name=os.getenv("TEAM_NAME", "truman-show"),
    )
