import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _parse_cors_origins(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ("http://localhost:3000", "http://127.0.0.1:3000")
    return tuple(origin.strip() for origin in raw.split(",") if origin.strip())


@dataclass(frozen=True)
class Settings:
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    cors_origins: tuple[str, ...] = _parse_cors_origins(os.getenv("CORS_ORIGINS"))

    archia_token: str | None = os.getenv("ARCHIA_TOKEN")
    archia_api_key: str | None = os.getenv("ARCHIA_API_KEY")  # optional fallback
    archia_sdk_base_url: str = os.getenv("ARCHIA_SDK_BASE_URL", "https://registry.archia.app/v1")
    archia_transcription_model: str = os.getenv("ARCHIA_TRANSCRIPTION_MODEL", "whisper-1")



settings = Settings()
