"""Settings, loaded once from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Raised at startup with an actionable message, never mid-conversation."""


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    nvidia_api_key: str
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def load(cls) -> "Settings":
        missing = [k for k in ("GROQ_API_KEY", "NVIDIA_API_KEY") if not os.getenv(k)]
        if missing:
            raise ConfigError(
                f"Missing {', '.join(missing)}. Copy .env.example to .env and fill it in — "
                "both keys are free. See the README for where to get them."
            )
        return cls(
            groq_api_key=os.environ["GROQ_API_KEY"],
            nvidia_api_key=os.environ["NVIDIA_API_KEY"],
            host=os.getenv("COACH_HOST", "127.0.0.1"),
            port=int(os.getenv("COACH_PORT", "8000")),
        )
