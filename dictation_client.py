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
import evdev
from pynput import mouse
from pynput.keyboard import Controller, Key

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
API_URL = "http://100.85.200.52:8887/v1/audio/transcriptions"
OPUS_BITRATE = 24000  # 24 kbps - good quality for speech, ~10x compression
# evdev Key Codes
EV_CTRL = {evdev.ecodes.KEY_LEFTCTRL, evdev.ecodes.KEY_RIGHTCTRL}
EV_ALT = {evdev.ecodes.KEY_LEFTALT, evdev.ecodes.KEY_RIGHTALT}
EV_SPACE = {evdev.ecodes.KEY_SPACE}
HOTKEY_CODES = EV_CTRL | EV_ALT | EV_SPACE

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
        self.keyboard_controller = Controller()
        self.last_hotkey_time = 0
        self.hotkey_debounce = 0.3  # 300ms debounce
        self.toggle_lock = threading.Lock()
        
        self.root = root
        self.overlay = StatusOverlay(root)
        
    def start_listeners(self):
        self.listener_thread = threading.Thread(target=self._evdev_listener_loop, daemon=True)
        self.listener_thread.start()
        logger.info("Dictation client started. Press Ctrl+Alt+Space to record.")

    def _evdev_listener_loop(self):
        import select
        try:
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            keyboards = []
            for device in devices:
                try:
                    caps = device.capabilities()
                    if evdev.ecodes.EV_KEY in caps:
                        if evdev.ecodes.KEY_SPACE in caps[evdev.ecodes.EV_KEY]:
                            keyboards.append(device)
                except Exception:
                    continue
            
            if not keyboards:
                logger.error("No keyboards found or permission denied!")
                logger.error("Run: sudo usermod -aG input $USER && logout")
                return

            logger.info(f"Monitoring keyboards: {[k.name for k in keyboards]}")
            
            held_ctrl = False
            held_alt = False
            held_space = False

            while True:
                # Use select to wait for events from any keyboard
                r, w, x = select.select(keyboards, [], [])
                for dev in r:
                    for event in dev.read():
                        if event.type == evdev.ecodes.EV_KEY:
                            key_event = evdev.categorize(event)
                            code = key_event.scancode
                            is_down = (key_event.keystate != evdev.events.KeyEvent.key_up)
                            
                            if code in EV_CTRL: held_ctrl = is_down
                            elif code in EV_ALT: held_alt = is_down
                            elif code in EV_SPACE: held_space = is_down
                            
                            if held_ctrl and held_alt and held_space:
                                if key_event.keystate == evdev.events.KeyEvent.key_down:
                                    # Debounce: prevent multiple triggers
                                    current_time = time.time()
                                    if current_time - self.last_hotkey_time > self.hotkey_debounce:
                                        self.last_hotkey_time = current_time
                                        logger.info("Hotkey detected!")
                                        self.root.after_idle(self.toggle_recording)
        except Exception as e:
            logger.error(f"Listener error: {e}")
    
    def toggle_recording(self):
        # Use lock to prevent race conditions
        with self.toggle_lock:
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
    
    def encode_to_ogg(self, audio_array):
        """Encode audio to OGG format for efficient network transfer."""
        import soundfile as sf
        import tempfile
        import os
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Write as OGG (uses Opus codec internally)
            sf.write(tmp_path, audio_array, SAMPLE_RATE, format='OGG')
            
            # Read back the OGG file
            with open(tmp_path, 'rb') as f:
                ogg_data = f.read()
            
            return io.BytesIO(ogg_data)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

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
            # Encode to OGG for efficient network transfer
            ogg_buffer = self.encode_to_ogg(audio_array)
            ogg_buffer.seek(0)
            
            ogg_size_kb = len(ogg_buffer.getvalue()) / 1024
            logger.info(f"📦 OGG encoded: {ogg_size_kb:.1f} KB")
            
            response = requests.post(API_URL, files={'file': ('audio.ogg', ogg_buffer, 'audio/ogg')})
            
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
