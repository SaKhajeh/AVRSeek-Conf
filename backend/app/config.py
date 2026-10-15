from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_base_url: str = "https://api.avalai.ir/v1"

    model_text: str = "gpt-4o"
    model_vlm: str = "gpt-4o"
    model_standardize: str = "gpt-4o"

    # OCR / vision behavior
    enable_ocr: bool = True
    enable_vlm_page_read: bool = True
    ocsr_local_model: str = ""  # "rxnscribe" | "molscribe" | empty
    # Tesseract langs now: eng+rus+chi_sim+chi_tra. Later: fas+spa+heb
    ocr_languages: str = "eng+rus+chi_sim+chi_tra"

    # When false, re-uploading the same PDF always re-runs the full pipeline
    # (useful while developing). Set ENABLE_RESULT_CACHE=true to skip duplicates.
    enable_result_cache: bool = False

    # Native-text docs at or below this size use one full-document LLM call
    # instead of per-chunk extraction (avoids duplicate steps from chunk merge).
    single_shot_max_chars: int = 20000

    # Render + VLM on figure-heavy pages even when native text exists (vector schemes).
    enable_vlm_figure_pages: bool = True
    figure_page_min_drawings: int = 15
    vlm_figure_max_pages: int = 8

    avrseek_host: str = "127.0.0.1"
    avrseek_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    upload_dir: Path = BACKEND_ROOT / "uploads"
    output_dir: Path = BACKEND_ROOT / "outputs"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return settings
