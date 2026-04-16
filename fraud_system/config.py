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


def load_settings(dataset_dir: str | Path | None = None) -> Settings:
    load_dotenv()
    resolved_dataset_dir = Path(dataset_dir or "The Truman Show - train").resolve()
    return Settings(
        dataset_dir=resolved_dataset_dir,
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
