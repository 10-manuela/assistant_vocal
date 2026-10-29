import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
import ollama
import pyttsx3
import os

# --- CONFIGURATION ---
FREQ = 16000  
DUREE_MAX = 5  # Enregistrement de 5 secondes pour le test
FICHIER_AUDIO = "temp.wav"
MODELE_OLLAMA = "llama3.2:3b"

print("⏳ Initialisation des moteurs (STT, LLM, TTS)...")

# 1. Initialisation de la Transcription (STT)
model_stt = WhisperModel("tiny", device="cpu", compute_type="int8")

# 2. Initialisation de la Voix Windows (TTS)
engine_tts = pyttsx3.init()
# Optionnel : Forcer une voix française si disponible sur votre Windows
voices = engine_tts.getProperty('voices')
for voice in voices:
    if "FR" in voice.id or "French" in voice.name:
        engine_tts.setProperty('voice', voice.id)
        break
engine_tts.setProperty('rate', 170)  # Vitesse de parole (170 est un bon rythme)

def enregistrer_vocal():
    print("\n🎤 ÉCOUTE en cours... Parlez maintenant (5 sec) :")
    audio = sd.rec(int(DUREE_MAX * FREQ), samplerate=FREQ, channels=1, dtype='int16')
    sd.wait()
    wav.write(FICHIER_AUDIO, FREQ, audio)

def transcrire_audio():
    segments, _ = model_stt.transcribe(FICHIER_AUDIO, beam_size=5, language="fr")
    return "".join([segment.text for segment in segments]).strip()

def interroger_ollama(texte_utilisateur):
    print(f"🧠 Réflexion de l'IA...")
    response = ollama.chat(model=MODELE_OLLAMA, messages=[
        {'role': 'system', 'content': 'Tu es un assistant vocal rapide et concis. Réponds en français en une ou deux phrases maximum.'},
        {'role': 'user', 'content': texte_utilisateur}
    ])
    return response['message']['content']

def faire_parler_ia(texte_reponse):
    print(f"🤖 L'IA dit : {texte_reponse}")
    engine_tts.say(texte_reponse)
    engine_tts.runAndWait()

# --- BOUCLE PRINCIPALE ---
if __name__ == "__main__":
    print("\n✅ Système prêt ! Lancez la conversation.")
    
    try:
        while True:
            # 1. Écouter l'utilisateur
            enregistrer_vocal()
            texte_prompt = transcrire_audio()
            
            if not texte_prompt:
                print("💤 Je n'ai rien entendu...")
                continue
                
            print(f"📝 Vous avez dit : \"{texte_prompt}\"")
            
            # 2. Obtenir la réponse d'Ollama
            reponse_ia = interroger_ollama(texte_prompt)
            
            # 3. Faire parler l'IA
            faire_parler_ia(reponse_ia)
            
            # Nettoyage
            if os.path.exists(FICHIER_AUDIO):
                os.remove(FICHIER_AUDIO)
                
    except KeyboardInterrupt:
        print("\n👋 Au revoir !")
