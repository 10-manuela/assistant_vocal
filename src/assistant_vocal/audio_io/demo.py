"""Démo : liste les périphériques, affiche parole/silence en direct, puis rejoue ta voix."""
import time

import numpy as np

from assistant_vocal.audio_io import MicCapture, SpeechDetector, beep, list_devices, play
from assistant_vocal.config import load_config


def main() -> None:
    cfg = load_config().audio

    print("=== Périphériques ===")
    for d in list_devices():
        print(f"[{d['index']}] {d['name']}  (in={d['inputs']}, out={d['outputs']})")

    print("\nTest haut-parleurs : un bip va retentir...")
    beep(device=cfg.output_device)

    vad = SpeechDetector(cfg.sample_rate, cfg.vad_threshold, cfg.min_silence_ms, cfg.block_ms)
    recording: list[np.ndarray] = []
    capturing = False

    print("\nParle (Ctrl+C pour arrêter). Chaque phrase sera rejouée.\n")
    with MicCapture(cfg.sample_rate, cfg.block_ms, cfg.input_device) as mic:
        try:
            while True:
                block = mic.read()
                event = vad.process(block)
                if event == "start":
                    print("🟢 parole détectée")
                    capturing, recording = True, []
                if capturing:
                    recording.append(block)
                if event == "end":
                    print("⚪ silence")
                    capturing = False
                    audio = np.concatenate(recording)
                    print(f"   rejoue {len(audio) / cfg.sample_rate:.1f} s...")
                    play(audio, cfg.sample_rate, cfg.output_device)
                    vad.reset()
                    time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nArrêt.")


if __name__ == "__main__":
    main()