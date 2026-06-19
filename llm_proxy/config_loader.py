"""配置加载/保存/重载"""

import json
import os
import logging
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
_CONFIG_DEFAULT = BASE_DIR / "config.json"
CONFIG_PATH = Path(os.environ.get("LLM_PROXY_CONFIG_PATH", str(_CONFIG_DEFAULT)))

logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    if "sidecar" in config:
        warnings.warn(
            "config.json 'sidecar' section is deprecated: sidecar has been replaced "
            "by native Python Anthropic↔OpenAI conversion in protocol/anthropic_openai/. "
            "The 'sidecar' key is ignored and can be safely removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning(
            "config.json 'sidecar' section is deprecated and ignored. "
            "Remove it from your config to silence this warning."
        )
    return config


def save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
