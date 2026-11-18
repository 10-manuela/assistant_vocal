"""Démo : tape un texte pour l'entendre, ou '/ia question' pour passer par le LLM."""
import time

from assistant_vocal.audio_io import play
from assistant_vocal.config import load_config
from assistant_vocal.llm import LLMClient, LLMError, make_message
from assistant_vocal.tts import TTSError, build_tts

SYSTEM = (
    "Tu es un assistant vocal pour un entretien professionnel. "
    "Tu parles à l'oral : réponds en français, en deux ou trois phrases courtes maximum, "
    "sans liste, sans numérotation, sans markdown."
)


def main() -> None:
    cfg = load_config()
    tts = build_tts(cfg)
    try:
        print("Chargement de la voix...")
        tts.warmup()
    except TTSError as exc:
        print(f"Erreur : {exc}")
        return

    llm = LLMClient.from_config(cfg)
    print("Tape un texte pour l'entendre, ou '/ia ta question'. Ligne vide pour quitter.")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break

        if line.startswith("/ia "):
            history = [make_message("system", SYSTEM), make_message("user", line[4:])]
            start = time.perf_counter()
            first_audio = None
            try:
                for speech in tts.stream(llm.stream(history)):
                    if first_audio is None:
                        first_audio = time.perf_counter() - start
                        print(f"   1er son après {first_audio:.2f} s")
                    print(f"IA > {speech.text}  (RTF {speech.rtf:.2f})")
                    play(speech.audio, speech.sample_rate, cfg.audio.output_device)
            except LLMError as exc:
                print(f"Erreur LLM : {exc}")
        else:
            speech = tts.synthesize(line)
            print(f"   {speech.duration:.1f} s d'audio | calcul {speech.elapsed:.2f} s | RTF {speech.rtf:.2f}")
            play(speech.audio, speech.sample_rate, cfg.audio.output_device)
    print("Au revoir.")


if __name__ == "__main__":
    main()