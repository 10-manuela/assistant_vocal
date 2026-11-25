"""Orchestrateur : relie micro, VAD, STT, LLM, TTS et haut-parleurs."""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from assistant_vocal.config import DEFAULT_SYSTEM_PROMPT, AppConfig
from assistant_vocal.dialogue.steps import Compute, Generate, Quit, RecordUser, Say, Session
from assistant_vocal.llm import make_message
from assistant_vocal.orchestrator.speaker import Speaker
from assistant_vocal.voiceprint import ADD_GUEST, OWNER, REMOVE_GUEST, SpeakerGate, detect_intent

log = logging.getLogger("orchestrator")

MSG_OWNER_OK = "Merci, j'ai reconnu ta voix. On peut commencer."
MSG_GUEST_OK = "Merci, je vous entends aussi maintenant."
MSG_MORE = "Encore quelques secondes, s'il te plaît."
MSG_ADD_GUEST = "D'accord. Que la personne parle quelques secondes, pour que je reconnaisse sa voix."
MSG_GUEST_REMOVED = "C'est noté, je n'écoute plus que toi."
MSG_NO_GUEST = "Il n'y a aucune autre voix enregistrée."


class State(str, Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass(eq=False)
class Turn:
    """Un tour de parole : l'utilisateur parle, puis l'assistant répond."""
    cancel: threading.Event = field(default_factory=threading.Event)
    played: list = field(default_factory=list)  # phrases réellement jouées en entier
    thread: threading.Thread | None = None
    t_end: float | None = field(default_factory=time.perf_counter)
    first_audio: float | None = None


class Orchestrator:
    def __init__(
        self,
        stt,
        llm,
        tts,
        store,
        vad,
        speaker: Speaker,
        *,
        gate: SpeakerGate | None = None,
        dialogue=None,
        block_ms: int = 32,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        barge_in_ms: int = 300,
        verified_barge_in_ms: int = 1000,
        min_utterance_ms: int = 300,
        preroll_ms: int = 320,
        history_limit: int = 12,
        guest_timeout_s: float = 60.0,
        conversation_id: int | None = None,
        on_event=None,
    ):
        self.stt, self.llm, self.tts, self.store, self.vad = stt, llm, tts, store, vad
        self.speaker = speaker
        self.speaker.on_start = self._on_speaker_start
        self.gate = gate
        self.dialogue = dialogue  # chef de dialogue (Mervi) ; None = assistant simple
        self.system_prompt = system_prompt
        self.history_limit = history_limit
        self.guest_timeout_s = guest_timeout_s
        self.on_event = on_event
        self.finished = False  # passe à True quand l'utilisateur quitte la discussion

        self.barge_in_blocks = max(1, barge_in_ms // block_ms)
        self.verified_barge_in_blocks = max(1, verified_barge_in_ms // block_ms)
        self.min_utterance_blocks = max(1, min_utterance_ms // block_ms)
        self._preroll: deque = deque(maxlen=max(1, preroll_ms // block_ms))

        self.state = State.LISTENING
        self._lock = threading.Lock()
        self._turn: Turn | None = None
        self._capturing = False
        self._recording: list = []
        self._voiced_run = 0
        self._voiced_total = 0
        self._barge_checked = False

        # Enrôlement d'une voix en cours (le prochain énoncé sert à l'enregistrer)
        self._enroll_lock = threading.Lock()
        self._pending: str | None = None
        self._pending_audio: list = []
        self._pending_deadline: float | None = None

        if conversation_id is None:
            conversation_id = store.create_conversation().id
            self.history: list[dict] = []
        else:  # reprise d'une conversation existante
            old = store.get_messages(conversation_id, limit=history_limit)
            self.history = [make_message(m.role, m.content) for m in old]
        self.conversation_id = conversation_id

    @classmethod
    def from_config(cls, cfg: AppConfig, stt, llm, tts, store, vad, gate=None, dialogue=None,
                    conversation_id=None, on_event=None) -> "Orchestrator":
        o = cfg.orchestrator
        return cls(
            stt, llm, tts, store, vad, Speaker(device=cfg.audio.output_device),
            gate=gate,
            dialogue=dialogue,
            block_ms=cfg.audio.block_ms,
            system_prompt=o.system_prompt,
            barge_in_ms=o.barge_in_ms,
            verified_barge_in_ms=o.verified_barge_in_ms,
            min_utterance_ms=o.min_utterance_ms,
            preroll_ms=o.preroll_ms,
            history_limit=o.history_limit,
            guest_timeout_s=cfg.voiceprint.guest_timeout_s,
            conversation_id=conversation_id,
            on_event=on_event,
        )

    # --- événements et états ---------------------------------------------

    def _emit(self, kind: str, data=None) -> None:
        if self.on_event:
            try:
                self.on_event(kind, data)
            except Exception:
                log.exception("Erreur dans on_event")

    def _set_state(self, state: State, turn: Turn | None = None) -> None:
        with self._lock:
            # Un tour annulé ou dépassé ne doit plus modifier l'état
            if turn is not None and (turn is not self._turn or turn.cancel.is_set()):
                return
            if state == self.state:
                return
            self.state = state
        self._emit("state", state.value)

    def _on_speaker_start(self, turn: Turn, speech) -> None:
        if turn.first_audio is None and turn.t_end is not None:
            turn.first_audio = time.perf_counter() - turn.t_end
            self._emit("latency", {"first_audio_s": turn.first_audio})
        self._set_state(State.SPEAKING, turn)

    # --- flux audio entrant ----------------------------------------------

    def handle_block(self, block: np.ndarray) -> None:
        """Traite un bloc du micro (appelé toutes les 32 ms)."""
        event = self.vad.process(block)
        voiced = self.vad.last_probability >= self.vad.threshold

        if event == "start":
            self._capturing = True
            self._recording = list(self._preroll) + [block]
            self._voiced_run = self._voiced_total = 0
            self._barge_checked = False
        elif self._capturing:
            self._recording.append(block)
        else:
            self._preroll.append(block)

        if self._capturing:
            if voiced:
                self._voiced_run += 1
                self._voiced_total += 1
            else:
                self._voiced_run = 0
            if self.state != State.LISTENING:
                self._maybe_barge_in()

        if event == "end":
            self._capturing = False
            if self.state == State.LISTENING and self._voiced_total >= self.min_utterance_blocks:
                self._start_turn(np.concatenate(self._recording))
            self._recording = []
            self._preroll.clear()

    def _maybe_barge_in(self) -> None:
        """Interruption : voix pendant que l'assistant réfléchit ou parle."""
        gated = self.gate is not None and self.gate.active
        if not gated:
            if self._voiced_run >= self.barge_in_blocks:
                self.interrupt()
            return
        # Filtre actif : on attend assez de voix pour vérifier QUI parle, une seule fois
        if self._barge_checked or self._voiced_total < self.verified_barge_in_blocks:
            return
        self._barge_checked = True
        verdict = self.gate.check(np.concatenate(self._recording), remember=False)
        if verdict.accepted:
            self.interrupt()
        else:
            self._emit("ignored", {"score": verdict.score, "reason": "barge_in"})

    def run(self, mic, stop_event: threading.Event | None = None) -> None:
        """Boucle principale : lit le micro jusqu'à l'arrêt ou jusqu'à « au revoir »."""
        stop_event = stop_event or threading.Event()
        while not stop_event.is_set() and not self.finished:
            try:
                block = mic.read(timeout=0.5)
            except queue.Empty:
                continue
            self.handle_block(block)

    # --- démarrage -------------------------------------------------------

    def start(self) -> None:
        """Accueil de Mervi, puis enregistrement de la voix si un filtre est actif. Bloquant."""
        if self.dialogue is None:
            raise RuntimeError("start() demande un chef de dialogue")
        ask_voice = self.gate is not None
        turn = Turn(t_end=None)
        self._turn = turn
        if ask_voice:
            self.begin_enrollment(OWNER)  # avant de parler : l'utilisateur peut couper la parole
        steps = self.dialogue.greeting_steps(ask_voice=ask_voice)
        if not ask_voice:
            steps += self.dialogue.menu_steps()
        self._execute(turn, steps, None)
        self._set_state(State.LISTENING, turn)

    # --- enrôlement des voix ---------------------------------------------

    def begin_enrollment(self, name: str = OWNER, timeout_s: float | None = None) -> None:
        """Le(s) prochain(s) énoncé(s) serviront à enregistrer la voix `name`."""
        if self.gate is None:
            raise RuntimeError("Aucun filtre de locuteur configuré")
        with self._enroll_lock:
            self._pending = name
            self._pending_audio = []
            self._pending_deadline = None if timeout_s is None else time.monotonic() + timeout_s
        self._emit("enrolling", name)

    def _handle_enrollment(self, turn: Turn, audio: np.ndarray) -> bool:
        """Retourne True si l'énoncé a servi à l'enrôlement (pas de LLM)."""
        if self.gate is None:
            return False
        with self._enroll_lock:
            name = self._pending
            if name is None:
                return False
            if self._pending_deadline is not None and time.monotonic() > self._pending_deadline:
                self._pending, self._pending_audio = None, []
                return False
            self._pending_audio.append(audio)
            total = np.concatenate(self._pending_audio)
            done = len(total) >= self.gate.min_enroll_samples
            if done:
                self.gate.enroll(name, total)
                self._pending, self._pending_audio = None, []
        if not done:
            self._say_fixed(turn, MSG_MORE)
            return True

        self._emit("enrolled", name)
        if name == OWNER and self.dialogue is not None:
            # La présentation sert aussi à retenir le prénom, puis Mervi présente ses modes
            try:
                intro = (self.stt.transcribe(total).text or "").strip()
            except Exception:
                log.exception("Transcription de la présentation impossible")
                intro = ""
            self.dialogue.set_user_intro(intro)
            self._execute(turn, self.dialogue.on_enrolled(), None)
        else:
            self._say_fixed(turn, MSG_OWNER_OK if name == OWNER else MSG_GUEST_OK)
        return True

    def _say_fixed(self, turn: Turn, text: str) -> None:
        """Phrase de service : jouée, mais absente de l'historique de la conversation."""
        if turn.cancel.is_set():
            return
        if self.speaker.say(turn, self.tts.synthesize(text)):
            self.speaker.wait()
        turn.played.clear()

    def _handle_intent(self, turn: Turn, intent: str) -> None:
        self._emit("intent", intent)
        if intent == ADD_GUEST:
            self.begin_enrollment(self.gate.new_guest_name(), self.guest_timeout_s)
            self._say_fixed(turn, MSG_ADD_GUEST)
        elif intent == REMOVE_GUEST:
            removed = self.gate.clear_guests()
            self._say_fixed(turn, MSG_GUEST_REMOVED if removed else MSG_NO_GUEST)

    # --- tours de parole -------------------------------------------------

    def interrupt(self) -> None:
        turn = self._turn
        if turn is not None:
            turn.cancel.set()  # d'abord annuler, ensuite couper le son
        self.speaker.stop()
        self._set_state(State.LISTENING)
        self._emit("interrupted")

    def _start_turn(self, audio: np.ndarray) -> None:
        prev = self._turn
        turn = Turn()
        self._turn = turn
        self._set_state(State.THINKING, turn)
        turn.thread = threading.Thread(
            target=self._run_turn, args=(turn, prev, audio), daemon=True, name="turn"
        )
        turn.thread.start()

    def _run_turn(self, turn: Turn, prev: Turn | None, audio: np.ndarray) -> None:
        try:
            if self._handle_enrollment(turn, audio):
                return
            if self.gate is not None and self.gate.active:
                t0 = time.perf_counter()
                verdict = self.gate.check(audio)
                self._emit("latency", {"gate_s": time.perf_counter() - t0})
                if not verdict.accepted:
                    self._emit("ignored", {"score": verdict.score, "reason": verdict.reason})
                    return
            result = self.stt.transcribe(audio)
            text = (result.text or "").strip()
            self._emit("latency", {"stt_s": getattr(result, "elapsed", 0.0)})
            # Le tour précédent doit être terminé avant de toucher à l'historique
            if prev is not None and prev.thread is not None:
                prev.thread.join()
            if turn.cancel.is_set() or len(text) < 2:
                return
            self._emit("user", text)
            intent = detect_intent(text) if self.gate is not None and self.gate.active else None
            if intent is not None:
                self._handle_intent(turn, intent)
                return
            if self.dialogue is not None:
                self._execute(turn, self.dialogue.handle(text), text)
                return
            self._record("user", text)
            self._respond(turn)
        except Exception as exc:
            log.exception("Erreur pendant le tour de parole")
            self._emit("error", str(exc))
        finally:
            self._finish(turn)

    # --- exécution des étapes du dialogue --------------------------------

    def _execute(self, turn: Turn, steps: list, user_text: str | None) -> None:
        pending = list(steps)
        while pending and not turn.cancel.is_set():
            step = pending.pop(0)
            if isinstance(step, Say):
                self._speak(turn, step.text, step.record)
            elif isinstance(step, RecordUser):
                if user_text:
                    self._record("user", user_text)
            elif isinstance(step, Session):
                self._new_session(step.title)
            elif isinstance(step, Generate):
                self._respond(turn, step.system, step.instruction)
            elif isinstance(step, Compute):
                pending[0:0] = step.fn()  # étapes calculées maintenant (ex. : bilan de l'entretien)
            elif isinstance(step, Quit):
                self.finished = True
                self._emit("quit")

    def _speak(self, turn: Turn, text: str, record: bool) -> None:
        """Joue un texte préparé, phrase par phrase (interruptible)."""
        before = len(turn.played)
        for speech in self.tts.stream([text]):
            if turn.cancel.is_set() or not self.speaker.say(turn, speech):
                break
        if not turn.cancel.is_set():
            self.speaker.wait()
        said = " ".join(turn.played[before:]).strip()
        del turn.played[before:]  # seules les réponses du LLM restent pour _finish
        if record and said:
            self._record("assistant", said)
            self._emit("assistant", said)

    def _record(self, role: str, text: str) -> None:
        self.history.append(make_message(role, text))
        self.store.add_message(self.conversation_id, role, text)

    def _new_session(self, title: str) -> None:
        """Nouvelle conversation enregistrée (la précédente est supprimée si elle est vide)."""
        if not self.store.get_messages(self.conversation_id):
            self.store.delete_conversation(self.conversation_id)
        self.conversation_id = self.store.create_conversation(title).id
        self.history.clear()
        self._emit("session", title)

    def _respond(self, turn: Turn, system: str | None = None, instruction: str | None = None) -> None:
        messages = [make_message("system", system or self.system_prompt)] + self.history[-self.history_limit:]
        if instruction:  # consigne cachée : envoyée au LLM mais pas enregistrée
            messages.append(make_message("user", instruction))
        stream = self.llm.stream(messages)
        t0 = time.perf_counter()

        def tokens():
            first = True
            for piece in stream:
                if turn.cancel.is_set():
                    return
                if first:
                    self._emit("latency", {"llm_first_token_s": time.perf_counter() - t0})
                    first = False
                yield piece

        try:
            for speech in self.tts.stream(tokens()):
                if turn.cancel.is_set() or not self.speaker.say(turn, speech):
                    break
            if not turn.cancel.is_set():
                self.speaker.wait()
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()  # coupe la requête vers Ollama

    def _finish(self, turn: Turn) -> None:
        reply = " ".join(turn.played).strip()  # uniquement ce qui a été réellement dit
        if reply:
            self._record("assistant", reply)
            self._emit("assistant", reply)
        self._set_state(State.LISTENING, turn)

    # --- utilitaires -----------------------------------------------------

    def say(self, text: str) -> None:
        """Fait parler l'assistant sans passer par le LLM (accueil...). Bloquant."""
        turn = Turn(t_end=None)
        self._turn = turn
        self.speaker.say(turn, self.tts.synthesize(text))
        self.speaker.wait()
        self._set_state(State.LISTENING, turn)

    def wait_idle(self, timeout: float = 10.0) -> bool:
        turn = self._turn
        if turn is not None and turn.thread is not None:
            turn.thread.join(timeout)
        return self.state == State.LISTENING

    def close(self) -> None:
        if self._turn is not None:
            self._turn.cancel.set()
        self.speaker.close()
