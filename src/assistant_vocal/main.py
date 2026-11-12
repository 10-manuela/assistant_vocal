"""Point d'entrée : vérifie que l'environnement est prêt."""
import json
import sys
import urllib.request

from assistant_vocal.config import load_config
from assistant_vocal.logger import get_logger, setup_logging


def check_ollama(host: str, model: str) -> tuple[bool, str]:
    """Vérifie qu'Ollama répond et que le modèle est installé."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as resp:
            models = [m["name"] for m in json.load(resp).get("models", [])]
    except Exception as exc:
        return False, f"Ollama injoignable ({exc})"
    if not any(m.startswith(model) for m in models):
        return False, f"Modèle '{model}' absent. Lance : ollama pull {model}"
    return True, f"Ollama OK, modèle '{model}' disponible"


def main() -> int:
    cfg = load_config()
    setup_logging(cfg.log_level)
    log = get_logger("main")

    log.info("Python %s", sys.version.split()[0])
    log.info("Langue : %s | Audio : %s Hz", cfg.language, cfg.audio.sample_rate)

    ok, msg = check_ollama(cfg.llm.host, cfg.llm.model)
    (log.info if ok else log.warning)(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())