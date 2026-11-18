"""Démo : tu parles, le texte et la latence s'affichent dans le terminal."""
from collections import deque

import numpy as np

from assistant_vocal.audio_io import MicCapture, SpeechDetector
from assistant_vocal.config import load_config
from assistant_vocal.stt import Transcriber


def main() -> None:
    cfg = load_config()
    a = cfg.audio

    print("Chargement du modèle (le premier lancement télécharge ~500 Mo)...")
    stt = Transcriber.from_config(cfg)
    stt.warmup()

    vad = SpeechDetector(a.sample_rate, a.vad_threshold, a.min_silence_ms, a.block_ms)
    preroll = deque(maxlen=10)  # ~320 ms avant le début détecté
    recording: list[np.ndarray] = []
    capturing = False

    print("\nParle (Ctrl+C pour arrêter).\n")
    with MicCapture(a.sample_rate, a.block_ms, a.input_device) as mic:
        try:
            while True:
                block = mic.read()
                event = vad.process(block)
                if event == "start":
                    capturing, recording = True, list(preroll)
                if capturing:
                    recording.append(block)
                else:
                    preroll.append(block)
                if event == "end":
                    capturing = False
                    audio = np.concatenate(recording)
                    r = stt.transcribe(audio)
                    print(f"« {r.text or '(rien compris)'} »")
                    print(f"   {r.audio_seconds:.1f} s d'audio | calcul {r.elapsed:.2f} s | RTF {r.rtf:.2f}\n")
                    vad.reset()
                    preroll.clear()
        except KeyboardInterrupt:
            print("\nArrêt.")


if __name__ == "__main__":
    main()