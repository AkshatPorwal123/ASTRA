"""
Offline voice assistant. Uses pyttsx3 (fully offline, OS-native TTS engines)
when available; falls back to a no-op that just logs the phrase, so the
rest of the system works even on machines without an audio backend
(e.g. a headless CI box or this dev sandbox).

Status: IMPLEMENTED for text generation/triggering logic;
        PARTIALLY IMPLEMENTED for actual audio output (depends on pyttsx3
        + a local speech engine being available on the host OS).
"""
from __future__ import annotations
import logging

logger = logging.getLogger("astra.voice")

try:
    import pyttsx3
    _ENGINE = pyttsx3.init()
    _AVAILABLE = True
except Exception:
    _ENGINE = None
    _AVAILABLE = False


class VoiceAssistant:
    def __init__(self):
        self.available = _AVAILABLE

    def say(self, text: str):
        logger.info(f"[VOICE] {text}")
        if self.available and _ENGINE is not None:
            try:
                _ENGINE.say(text)
                _ENGINE.runAndWait()
            except Exception as e:
                logger.warning(f"TTS playback failed, logged only: {e}")

    # Convenience wrappers matching Section 19's required event types
    def step_completed(self, step_name: str):
        self.say(f"Step completed: {step_name}.")

    def next_step(self, instruction: str):
        self.say(instruction)

    def warning(self, message: str):
        self.say(f"Warning. {message}")

    def error(self, message: str):
        self.say(f"Error. {message}")

    def recovery(self, message: str):
        self.say(message)

    def experiment_start(self):
        self.say("Experiment started.")

    def experiment_complete(self):
        self.say("Experiment complete.")
