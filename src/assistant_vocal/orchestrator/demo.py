"""Démo : conversation vocale avec voix centrale. Casque recommandé. Ctrl+C pour arrêter."""
from assistant_vocal.audio_io import MicCapture, SpeechDetector
from assistant_vocal.config import load_config
from assistant_vocal.llm import LLMClient, LLMError
from assistant_vocal.orchestrator import Orchestrator
from assistant_vocal.storage import open_store
from assistant_vocal.stt import Transcriber
from assistant_vocal.tts import TTSError, build_tts
from assistant_vocal.voiceprint import OWNER, build_gate


def show(kind, data):
    if kind == "state":
        print(f"   [{data}]")
    elif kind == "user":
        print(f"Toi > {data}")
    elif kind == "assistant":
        print(f"IA  > {data}")
    elif kind == "latency":
        print("   " + " | ".join(f"{k} {v:.2f} s" for k, v in data.items()))
    elif kind == "interrupted":
        print("   [interrompu]")
    elif kind == "ignored":
        score = data.get("score")
        shown = f"{score:.2f}" if score is not None else "n/a"
        print(f"   [voix ignorée : {data.get('reason')}, score {shown}]")
    elif kind == "enrolling":
        print(f"   [enregistrement de la voix « {data} » : parle]")
    elif kind == "enrolled":
        print(f"   [voix « {data} » enregistrée]")
    elif kind == "intent":
        print(f"   [commande vocale : {data}]")
    elif kind == "error":
        print(f"   Erreur : {data}")


def main() -> None:
    cfg = load_config()
    a = cfg.audio

    print("Chargement des modèles (STT, voix, LLM)...")
    try:
        stt = Transcriber.from_config(cfg)
        stt.warmup()
        tts = build_tts(cfg)
        tts.warmup()
        llm = LLMClient.from_config(cfg)
        llm.warmup()
    except (TTSError, LLMError) as exc:
        print(f"Erreur : {exc}")
        return

    gate = None
    if cfg.voiceprint.enabled:
        print("Chargement du modèle de reconnaissance de voix...")
        try:
            gate = build_gate(cfg)
        except Exception as exc:  # téléchargement impossible, paquet absent...
            print(f"Voix centrale désactivée : {exc}")

    store = open_store(cfg)
    vad = SpeechDetector(a.sample_rate, a.vad_threshold, a.min_silence_ms, a.block_ms)
    orch = Orchestrator.from_config(cfg, stt, llm, tts, store, vad, gate=gate, on_event=show)

    if gate is not None:
        orch.say("Bonjour. Pour commencer, présente-toi en quelques phrases, "
                 "pour que je reconnaisse ta voix.")
        orch.begin_enrollment(OWNER)
    else:
        orch.say("Bonjour, je suis prête. Je t'écoute.")

    print("\nParle ! Coupe-moi la parole quand tu veux. Ctrl+C pour arrêter.\n")
    try:
        with MicCapture(a.sample_rate, a.block_ms, a.input_device) as mic:
            orch.run(mic)
    except KeyboardInterrupt:
        print("\nArrêt.")
    finally:
        orch.close()
        store.close()


if __name__ == "__main__":
    main()