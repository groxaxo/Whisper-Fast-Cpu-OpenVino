#!/usr/bin/env python3
"""
Global Dictation Client for Whisper-Fast-Cpu-OpenVINO
With Visual Feedback Overlay
"""

import sys
import time
import threading
import queue
import signal
import io
import wave
import logging
import tkinter as tk
from tkinter import font as tkfont

import sounddevice as sd
import numpy as np
import requests
from pynput import keyboard, mouse
from scipy.io.wavfile import write as write_wav

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
API_URL = "http://localhost:8000/v1/audio/transcriptions"
HOTKEY = {keyboard.Key.ctrl, keyboard.Key.alt, keyboard.Key.space}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dictation_client")

class StatusOverlay:
    def __init__(self, root):
        self.root = root
        self.root.title("Dictation Overlay")
        self.root.geometry("200x60")
        
        # Make frameless and floating
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.85)
        
        # Center the window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - 100
        y = (screen_height // 2) - 30
        self.root.geometry(f"+{x}+{y}")
        
        # Styling
        self.bg_color = "#333333"
        self.text_color = "#FFFFFF"
        self.root.configure(bg=self.bg_color)
        
        self.label = tk.Label(
            self.root, 
            text="Ready", 
            font=("Arial", 16, "bold"),
            bg=self.bg_color,
            fg=self.text_color
        )
        self.label.pack(expand=True, fill='both')
        
        # Hide initially
        self.root.withdraw()
        
    def show(self, text, color="#333333"):
        self.label.config(text=text, bg=color)
        self.root.configure(bg=color)
        self.root.deiconify()
        self.root.update()

    def hide(self):
        self.root.withdraw()

class DictationClient:
    def __init__(self, root):
        self.recording = False
        self.audio_queue = queue.Queue()
        self.audio_data = []
        self.current_keys = set()
        self.keyboard_controller = keyboard.Controller()
        
        self.root = root
        self.overlay = StatusOverlay(root)
        
    def start_listeners(self):
        # Start keyboard listener in non-blocking mode
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        
        logger.info("Dictation client started. Press Ctrl+Alt+Space to record.")

    def on_press(self, key):
        if key in HOTKEY:
            self.current_keys.add(key)
            if self.current_keys == HOTKEY:
                # Schedule toggle in main thread
                self.root.after_idle(self.toggle_recording)
        
    def on_release(self, key):
        try:
            self.current_keys.remove(key)
        except KeyError:
            pass

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            # We need to run the stop/transcribe logic in a thread to keep UI responsive
            # but updates back to UI must happen in main thread
             threading.Thread(target=self.stop_recording_and_transcribe).start()

    def start_recording(self):
        self.recording = True
        self.audio_data = []
        
        logger.info("🔴 Recording started...")
        self.overlay.show("🔴 Listening...", color="#CC0000") # Red
        
        self.record_thread = threading.Thread(target=self._record_loop)
        self.record_thread.start()

    def _record_loop(self):
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=self._audio_callback):
            while self.recording:
                time.sleep(0.1)

    def _audio_callback(self, indata, frames, time, status):
        self.audio_data.append(indata.copy())

    def stop_recording_and_transcribe(self):
        logger.info("🛑 Stopping...")
        self.root.after_idle(lambda: self.overlay.show("⏳ Capturing...", color="#CC8800")) # Orange
        
        # Stop recording FIRST, then add a small delay to ensure buffer is flushed
        self.recording = False
        time.sleep(0.3)  # Allow audio callback to flush
        self.record_thread.join()
        
        # Processing UI
        self.root.after_idle(lambda: self.overlay.show("⚡ Transcribing...", color="#0066CC")) # Blue
        
        if not self.audio_data:
            logger.warning("No audio data captured!")
            self.root.after_idle(lambda: self.overlay.hide())
            return

        audio_array = np.concatenate(self.audio_data, axis=0)
        
        # Log audio stats for debugging
        duration_sec = len(audio_array) / SAMPLE_RATE
        max_level = np.max(np.abs(audio_array))
        logger.info(f"📊 Audio: {duration_sec:.2f}s, max level: {max_level:.4f}")
        
        # Send to API
        try:
            wav_buffer = io.BytesIO()
            write_wav(wav_buffer, SAMPLE_RATE, audio_array)
            wav_buffer.seek(0)
            
            response = requests.post(API_URL, files={'file': ('audio.wav', wav_buffer, 'audio/wav')})
            
            if response.status_code == 200:
                text = response.json().get('text', '').strip()
                if text:
                    logger.info(f"Transcription: {text}")
                    self.root.after_idle(lambda: self.type_text(text))
                else:
                    self.root.after_idle(lambda: self.overlay.show("❌ No speech"))
                    self.root.after(1000, lambda: self.overlay.hide())
            else:
                logger.error(f"Error: {response.status_code}")
                self.root.after_idle(lambda: self.overlay.show("❌ Error"))
                self.root.after(1000, lambda: self.overlay.hide())
                
        except Exception as e:
            logger.error(f"Failed: {e}")
            self.root.after_idle(lambda: self.overlay.show("❌ Failed"))
            self.root.after(1000, lambda: self.overlay.hide())

    def type_text(self, text):
        self.overlay.hide()
        for char in text:
            self.keyboard_controller.type(char)
            time.sleep(0.005)

def main():
    root = tk.Tk()
    client = DictationClient(root)
    client.start_listeners()
    root.mainloop()

if __name__ == "__main__":
    main()
