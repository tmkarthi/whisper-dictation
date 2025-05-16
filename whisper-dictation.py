import argparse
import time
import threading
import pyaudio
import numpy as np
from pynput import keyboard
from faster_whisper import WhisperModel
import platform
import math
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global PyAudio instance and lock
_pyaudio_instance = None
_pyaudio_lock = threading.Lock()

def get_pyaudio_instance():
    """Get or create a global PyAudio instance with thread safety."""
    global _pyaudio_instance
    with _pyaudio_lock:
        if _pyaudio_instance is None:
            _pyaudio_instance = pyaudio.PyAudio()
        return _pyaudio_instance

def play_tone(frequency, duration=0.067, volume=0.3):
    """Play a tone with the given frequency, duration, and volume."""
    try:
        sample_rate = 44100  # samples per second
        
        # Generate samples
        samples = (np.sin(2 * np.pi * np.arange(sample_rate * duration) * frequency / sample_rate)).astype(np.float32)
        samples = samples * volume
        
        # Apply envelope to avoid clicks
        envelope = np.ones_like(samples)
        ramp_samples = int(0.015 * sample_rate)  # 15ms ramp
        if ramp_samples * 2 < len(samples):
            envelope[:ramp_samples] = np.linspace(0, 1, ramp_samples)
            envelope[-ramp_samples:] = np.linspace(1, 0, ramp_samples)
        else:
            mid_point = len(samples) // 2
            envelope[:mid_point] = np.linspace(0, 1, mid_point)
            envelope[mid_point:] = np.linspace(1, 0, len(samples) - mid_point)
        
        samples = samples * envelope
        
        # Complete waveform cycle to avoid clicks
        sample_length = len(samples)
        cycles = frequency * duration
        if not math.isclose(cycles, round(cycles), abs_tol=0.1):
            last_sample_idx = int(round(cycles) * sample_rate / frequency)
            if last_sample_idx < sample_length:
                fade_len = sample_length - last_sample_idx
                fade_envelope = np.linspace(1, 0, fade_len)
                samples[last_sample_idx:] = samples[last_sample_idx:] * fade_envelope
        
        # Convert to int16
        samples = (samples * 32767).astype(np.int16)
            
        p = get_pyaudio_instance()
        with _pyaudio_lock:
            # Open and play stream
            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=sample_rate,
                            output=True)
            stream.write(samples.tobytes())
            stream.stop_stream()
            stream.close()
            # Don't terminate PyAudio here as we're reusing the instance
    except Exception as e:
        logger.error(f"Error playing tone: {e}")

class SpeechTranscriber:
    def __init__(self, model):
        self.model = model
        self.pykeyboard = keyboard.Controller()

    def transcribe(self, audio_data, language=None):
        # faster-whisper returns segments instead of a dictionary with text
        segments, _ = self.model.transcribe(audio_data, language=language, beam_size=5)
        text = ""
        for segment in segments:
            text += segment.text
        logger.info(f"Transcribed text: {text}")
        
        is_first = True
        for element in text:
            if is_first and element == " ":
                is_first = False
                continue
            try:
                self.pykeyboard.type(element)
                time.sleep(0.0025)
            except Exception as e:
                logger.error(f"Error typing character: {e}")
        logger.info("Typing complete.")

class Recorder:
    def __init__(self, transcriber):
        self.recording = False
        self.transcriber = transcriber

    def start(self, language=None):
        thread = threading.Thread(target=self._record_impl, args=(language,))
        thread.start()

    def stop(self):
        self.recording = False

    def _record_impl(self, language):
        self.recording = True
        frames_per_buffer = 1024
        p = get_pyaudio_instance()
        with _pyaudio_lock:
            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=16000,
                            frames_per_buffer=frames_per_buffer,
                            input=True)
            frames = []

            try:
                i = 0
                logger.info("Listening...")
                while self.recording:
                    data = stream.read(frames_per_buffer)
                    frames.append(data)
                    i += 1
                    if i % 10 == 0:
                        print(".", end="", flush=True)
                print()
            finally:
                stream.stop_stream()
                stream.close()
        # Don't terminate PyAudio here as we're reusing the instance
        logger.info("Done.")
        logger.info("Transcribing...")
        # For faster-whisper, we can pass the audio as a numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data_fp32 = audio_data.astype(np.float32) / 32768.0
        self.transcriber.transcribe(audio_data_fp32, language)

class RecordingManager:
    def __init__(self, recorder, language, max_time):
        self.recorder = recorder
        self.language = language
        self.max_time = max_time
        self.recording = False
        self.timer = None

    def start(self):
        if not self.recording:
            self.recording = True
            self.recorder.start(self.language)
            if self.max_time is not None:
                self.timer = threading.Timer(self.max_time, self.stop)
                self.timer.start()

    def stop(self):
        if self.recording:
            if self.timer is not None:
                self.timer.cancel()
            self.recording = False
            self.recorder.stop()

    def toggle(self):
        if self.recording:
            self.stop()
        else:
            self.start()

class GlobalKeyListener:
    def __init__(self, recording_manager, key_combination):
        self.recording_manager = recording_manager
        self.key1, self.key2 = self.parse_key_combination(key_combination)
        self.key1_pressed = False
        self.key2_pressed = False

    def parse_key_combination(self, key_combination):
        key1_name, key2_name = key_combination.split('+')
        key1 = getattr(keyboard.Key, key1_name, keyboard.KeyCode(char=key1_name))
        key2 = getattr(keyboard.Key, key2_name, keyboard.KeyCode(char=key2_name))
        return key1, key2

    def on_key_press(self, key):
        if key == self.key1:
            self.key1_pressed = True
        elif key == self.key2:
            self.key2_pressed = True
        if self.key1_pressed and self.key2_pressed:
            self.recording_manager.toggle()

    def on_key_release(self, key):
        if key == self.key1:
            self.key1_pressed = False
        elif key == self.key2:
            self.key2_pressed = False

class DoubleCommandKeyListener:
    def __init__(self, recording_manager):
        self.recording_manager = recording_manager
        self.key = keyboard.Key.cmd_r
        self.last_press_time = 0

    def on_key_press(self, key):
        if key == self.key:
            current_time = time.time()
            if not self.recording_manager.recording and current_time - self.last_press_time < 0.5:
                self.recording_manager.start()
            elif self.recording_manager.recording:
                self.recording_manager.stop()
            self.last_press_time = current_time

    def on_key_release(self, key):
        pass

class PushToTalkListener:
    def __init__(self, recording_manager):
        self.recording_manager = recording_manager
        self.key = keyboard.Key.cmd_r
        self.active = False
        self.last_press_time = 0

    def on_key_press(self, key):
        if key == self.key:
            current_time = time.time()
            if not self.active and current_time - self.last_press_time < 0.5:
                self.active = True
                play_tone(300)
                self.recording_manager.start()
            self.last_press_time = current_time

    def on_key_release(self, key):
        if key == self.key and self.active:
            self.active = False
            self.recording_manager.stop()
            play_tone(600)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Dictation app using the faster-whisper ASR model. By default the keyboard shortcut cmd+option '
                    'starts and stops dictation')
    parser.add_argument('-m', '--model_name', type=str,
                        choices=['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2', 'large-v3'],
                        default='base',
                        help='Specify the whisper ASR model to use.')
    parser.add_argument('-k', '--key_combination', type=str, default='cmd_l+alt' if platform.system() == 'Darwin' else 'ctrl+alt',
                        help='Key combination to toggle recording.')
    parser.add_argument('--k_double_cmd', action='store_true',
                        help='Use double Right Command key press to start recording, single press to stop.')
    parser.add_argument('--ptt', action='store_true',
                        help='Use double tap of Right Command key to activate push-to-talk mode.')
    parser.add_argument('-l', '--language', type=str, default=None,
                        help='Specify the two-letter language code (e.g., "en" for English).')
    parser.add_argument('-t', '--max_time', type=float, default=30,
                        help='Maximum recording time in seconds.')
    args = parser.parse_args()

    if args.language is not None:
        args.language = args.language.split(',')
    if args.model_name.endswith('.en') and args.language is not None and any(lang != 'en' for lang in args.language):
        raise ValueError('If using a .en model, language must be English.')
    return args

if __name__ == "__main__":
    # Play startup tone
    threading.Thread(target=play_tone, args=(800, 0.3, 0.5)).start()

    args = parse_args()

    logger.info("Loading model...")
    # Initialize faster-whisper model
    # Use CPU by default, but you can change to 'cuda' for GPU acceleration if available
    model = WhisperModel(args.model_name, device="cpu", compute_type="int8", download_root=None, local_files_only=False)
    logger.info(f"{args.model_name} model loaded")
    threading.Thread(target=play_tone, args=(500, 0.2, 0.4)).start()

    transcriber = SpeechTranscriber(model)
    recorder = Recorder(transcriber)
    language = args.language[0] if args.language else None
    recording_manager = RecordingManager(recorder, language, args.max_time)

    if args.ptt:
        key_listener = PushToTalkListener(recording_manager)
    elif args.k_double_cmd:
        key_listener = DoubleCommandKeyListener(recording_manager)
    else:
        key_listener = GlobalKeyListener(recording_manager, args.key_combination)

    listener = keyboard.Listener(on_press=key_listener.on_key_press, on_release=key_listener.on_key_release)
    listener.start()

    logger.info("Running... Press Ctrl+C to exit.")
    try:
        listener.join()
    except KeyboardInterrupt:
        logger.info("Exiting...")
        listener.stop()
        if recording_manager.recording:
            recording_manager.stop()
    finally:
        # Clean up the global PyAudio instance
        with _pyaudio_lock:
            if _pyaudio_instance is not None:
                _pyaudio_instance.terminate()
                _pyaudio_instance = None
