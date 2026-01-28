import os
import time
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
import soundfile as sf
from pydub import AudioSegment, silence
import io
import magic
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allow device override via environment variable
device_env = os.getenv("QWEN_DEVICE", "auto")
if device_env == "auto":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
else:
    device = device_env

# Select appropriate dtype based on device
# CPU performs better with float32, GPU can use bfloat16
if device == "cpu":
    dtype = torch.float32
    logging.info("Using CPU with float32 dtype")
else:
    dtype = torch.bfloat16
    logging.info(f"Using {device} with bfloat16 dtype")

# Initialize Qwen3-TTS model
from qwen_tts import Qwen3TTSModel

logging.info(f"Loading Qwen3-TTS model on {device}...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map=device,
    dtype=dtype,
)
logging.info("Qwen3-TTS model loaded successfully")

# Initialize Whisper for transcription (Qwen3-TTS doesn't have built-in transcription)
import whisper

logging.info("Loading Whisper model for transcription...")
whisper_model = whisper.load_model("base", device=device)
logging.info("Whisper model loaded successfully")

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

resources_dir = 'resources'
os.makedirs(resources_dir, exist_ok=True)

# Default reference audio and text for base_tts
default_ref_audio = None
default_ref_text = "Some call me nature, others call me mother nature."
default_voice_prompt = None

# Cache for voice data: {voice_name: {"processed_audio": path, "ref_text": str, "prompt": object}}
voice_cache = {}


def convert_to_wav(input_path: str, output_path: str):
    """Convert any audio format to WAV using pydub."""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(24000)  # Set sample rate
    audio.export(output_path, format='wav')


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using Whisper."""
    result = whisper_model.transcribe(audio_path)
    return result["text"].strip()


def detect_leading_silence(audio, silence_threshold=-42, chunk_size=10):
    """Detect silence at the beginning of the audio."""
    trim_ms = 0
    while audio[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < len(audio):
        trim_ms += chunk_size
    return trim_ms


def remove_silence_edges(audio, silence_threshold=-42):
    """Remove silence from the beginning and end of the audio."""
    start_trim = detect_leading_silence(audio, silence_threshold)
    end_trim = detect_leading_silence(audio.reverse(), silence_threshold)
    duration = len(audio)
    return audio[start_trim:duration - end_trim]


def process_reference_audio(reference_file: str) -> tuple[str, str]:
    """
    Process reference audio: clip to max 15s and transcribe.
    Returns (processed_audio_path, transcription).
    """
    temp_short_ref = f'{output_dir}/temp_short_ref.wav'
    aseg = AudioSegment.from_file(reference_file)

    # 1. try to find long silence for clipping
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
            logging.info("Audio is over 15s, clipping short. (1)")
            break
        non_silent_wave += non_silent_seg

    # 2. try to find short silence for clipping if 1. failed
    if len(non_silent_wave) > 15000:
        non_silent_segs = silence.split_on_silence(
            aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                logging.info("Audio is over 15s, clipping short. (2)")
                break
            non_silent_wave += non_silent_seg

    aseg = non_silent_wave

    # 3. if no proper silence found for clipping
    if len(aseg) > 15000:
        aseg = aseg[:15000]
        logging.info("Audio is over 15s, clipping short. (3)")

    aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
    aseg.export(temp_short_ref, format='wav')

    # Transcribe the short clip
    ref_text = transcribe_audio(temp_short_ref)
    logging.info(f'Reference text transcribed from first 15s: {ref_text}')

    return temp_short_ref, ref_text


def apply_speed(audio_data: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Apply speed adjustment to audio using time stretching."""
    if speed == 1.0:
        return audio_data
    
    try:
        import librosa
        # Time stretch: speed > 1 = faster, speed < 1 = slower
        return librosa.effects.time_stretch(audio_data, rate=speed)
    except ImportError:
        logging.warning("librosa not installed, speed adjustment not available")
        return audio_data


def generate_speech_with_prompt(text: str, voice_prompt, speed: float = 1.0) -> tuple[np.ndarray, int]:
    """Generate speech using cached voice clone prompt (optimized)."""
    import time
    start_time = time.time()
    
    # Set fixed seed for reproducible output
    torch.manual_seed(42)
    
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="Auto",
        voice_clone_prompt=voice_prompt,
    )
    
    audio_data = wavs[0]
    
    # Apply speed adjustment if needed
    if speed != 1.0:
        audio_data = apply_speed(audio_data, sr, speed)
    
    generation_time = time.time() - start_time
    audio_duration = len(audio_data) / sr
    logging.info(f"Generation completed in {generation_time:.2f}s (audio duration: {audio_duration:.2f}s, RTF: {generation_time/audio_duration:.2f}x)")
    
    return audio_data, sr


def generate_speech(text: str, ref_audio_path: str, ref_text: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
    """Generate speech using Qwen3-TTS voice cloning (non-cached fallback)."""
    import time
    start_time = time.time()
    
    # Set fixed seed for reproducible output
    torch.manual_seed(42)
    
    # Load reference audio as numpy array
    ref_audio_data, ref_sr = sf.read(ref_audio_path)
    
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="Auto",
        ref_audio=(ref_audio_data, ref_sr),
        ref_text=ref_text,
    )
    
    audio_data = wavs[0]
    
    # Apply speed adjustment if needed
    if speed != 1.0:
        audio_data = apply_speed(audio_data, sr, speed)
    
    generation_time = time.time() - start_time
    audio_duration = len(audio_data) / sr
    logging.info(f"Generation completed in {generation_time:.2f}s (audio duration: {audio_duration:.2f}s, RTF: {generation_time/audio_duration:.2f}x)")
    
    return audio_data, sr


def audio_to_wav_bytes(audio_data: np.ndarray, sr: int) -> io.BytesIO:
    """Convert numpy audio array to WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sr, format='WAV')
    buffer.seek(0)
    return buffer


def get_or_create_voice_cache(voice: str, reference_file: str) -> dict:
    """
    Get cached voice data or create new cache entry.
    Caches: processed audio path, transcription, and voice clone prompt.
    This avoids repeated Whisper transcription on every request.
    """
    global voice_cache
    
    if voice in voice_cache:
        logging.info(f"Using cached voice data for: {voice}")
        return voice_cache[voice]
    
    logging.info(f"Creating voice cache for: {voice}")
    
    # Process reference audio (clip to 15s, remove silence)
    processed_ref, ref_text = process_reference_audio(reference_file)
    
    # Create reusable voice clone prompt
    ref_audio_data, ref_sr = sf.read(processed_ref)
    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=(ref_audio_data, ref_sr),
        ref_text=ref_text,
    )
    
    # Store in cache
    voice_cache[voice] = {
        "processed_audio": processed_ref,
        "ref_text": ref_text,
        "prompt": voice_prompt,
        "audio_data": ref_audio_data,
        "sample_rate": ref_sr,
    }
    
    logging.info(f"Voice cache created for: {voice} (transcription: '{ref_text[:50]}...')")
    return voice_cache[voice]


@app.on_event("startup")
async def startup_event():
    """Warmup the model on startup."""
    global default_ref_audio, default_voice_prompt
    
    # Check if we have a default voice file
    default_files = [f for f in os.listdir(resources_dir) if f.startswith("default_en")]
    if default_files:
        default_ref_audio = f"{resources_dir}/{default_files[0]}"
        if not default_ref_audio.endswith('.wav'):
            wav_path = f"{resources_dir}/default_en.wav"
            convert_to_wav(default_ref_audio, wav_path)
            default_ref_audio = wav_path
    
    # Warmup with demo_speaker0 if available
    demo_files = [f for f in os.listdir(resources_dir) if f.startswith("demo_speaker0")]
    if demo_files:
        logging.info("Warming up model with demo_speaker0...")
        test_text = "This is a test sentence generated by the Qwen3-TTS API."
        try:
            await synthesize_speech(test_text, "demo_speaker0")
            logging.info("Warmup complete")
        except Exception as e:
            logging.warning(f"Warmup failed: {e}")


@app.get("/base_tts/")
async def base_tts(text: str, speed: Optional[float] = 1.0):
    """
    Perform text-to-speech conversion using only the base speaker.
    """
    try:
        return await synthesize_speech(text=text, voice="default_en", speed=speed)
    except Exception as e:
        logging.error(f"Error in base_tts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/change_voice/")
async def change_voice(reference_speaker: str = Form(...), file: UploadFile = File(...)):
    """
    Change the voice of an existing audio file.
    """
    try:
        logging.info(f'Changing voice to {reference_speaker}...')

        contents = await file.read()
        
        # Save the input audio temporarily
        input_path = f'{output_dir}/input_audio.wav'
        with open(input_path, 'wb') as f:
            f.write(contents)

        # Find the reference audio file
        matching_files = [f for f in os.listdir(resources_dir) if f.startswith(str(reference_speaker))]
        if not matching_files:
            raise HTTPException(status_code=400, detail="No matching reference speaker found.")
        
        reference_file = f'{resources_dir}/{matching_files[0]}'
        
        # Convert reference file to WAV if it's not already
        if not reference_file.lower().endswith('.wav'):
            ref_wav_path = f'{output_dir}/ref_converted.wav'
            convert_to_wav(reference_file, ref_wav_path)
            reference_file = ref_wav_path
        
        # Transcribe the input audio
        text = transcribe_audio(input_path)
        logging.info(f'Transcribed input audio: {text}')
        
        # Get or create cached voice data for the reference speaker
        cache_data = get_or_create_voice_cache(reference_speaker, reference_file)
        
        # Generate with the new voice using cached prompt
        audio_data, sr = generate_speech_with_prompt(text, cache_data["prompt"])
        
        # Save output
        save_path = f'{output_dir}/output_converted.wav'
        sf.write(save_path, audio_data, sr)

        return StreamingResponse(open(save_path, 'rb'), media_type="audio/wav")
    except Exception as e:
        logging.error(f"Error in change_voice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_audio/")
async def upload_audio(audio_file_label: str = Form(...), file: UploadFile = File(...)):
    """
    Upload an audio file for later use as the reference audio.
    """
    try:
        contents = await file.read()

        allowed_extensions = {'wav', 'mp3', 'flac', 'ogg'}
        max_file_size = 5 * 1024 * 1024  # 5MB

        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in allowed_extensions:
            return {"error": "Invalid file type. Allowed types are: wav, mp3, flac, ogg"}

        if len(contents) > max_file_size:
            return {"error": "File size is over limit. Max size is 5MB."}

        temp_file = io.BytesIO(contents)
        file_format = magic.from_buffer(temp_file.read(), mime=True)

        if 'audio' not in file_format:
            return {"error": "Invalid file content."}

        stored_file_name = f"{audio_file_label}.{file_ext}"

        with open(f"{resources_dir}/{stored_file_name}", "wb") as f:
            f.write(contents)

        # Also create a WAV version
        wav_path = f"{resources_dir}/{audio_file_label}.wav"
        convert_to_wav(f"{resources_dir}/{stored_file_name}", wav_path)
        
        # Clear cached voice data if it exists (will be regenerated on next use)
        if audio_file_label in voice_cache:
            del voice_cache[audio_file_label]
            logging.info(f"Cleared voice cache for: {audio_file_label}")

        return {"message": f"File {file.filename} uploaded successfully with label {audio_file_label}."}
    except Exception as e:
        logging.error(f"Error in upload_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/synthesize_speech/")
async def synthesize_speech(
        text: str,
        voice: str,
        speed: Optional[float] = 1.0,
):
    """
    Synthesize speech from text using a specified voice and style.
    """
    start_time = time.time()
    try:
        logging.info(f'Generating speech for voice: {voice}')

        # First try to find a WAV version
        matching_files = [f for f in os.listdir(resources_dir) if f.startswith(voice) and f.lower().endswith('.wav')]
        
        # If no WAV found, try other formats and convert
        if not matching_files:
            matching_files = [f for f in os.listdir(resources_dir) if f.startswith(voice)]
            if not matching_files:
                raise HTTPException(status_code=400, detail="No matching voice found.")
            
            # Convert to WAV
            input_file = f'{resources_dir}/{matching_files[0]}'
            wav_path = f'{output_dir}/ref_converted.wav'
            convert_to_wav(input_file, wav_path)
            reference_file = wav_path
        else:
            reference_file = f'{resources_dir}/{matching_files[0]}'

        # Get or create cached voice data (includes transcription and voice prompt)
        if voice == "default_en" and default_ref_audio:
            # Use default voice with known transcription
            cache_data = get_or_create_voice_cache(voice, default_ref_audio)
        else:
            cache_data = get_or_create_voice_cache(voice, reference_file)
        
        # Generate speech using cached voice prompt (no re-transcription needed)
        audio_data, sr = generate_speech_with_prompt(text, cache_data["prompt"], speed)
        
        # Save output
        save_path = f'{output_dir}/output_synthesized.wav'
        sf.write(save_path, audio_data, sr)

        result = StreamingResponse(open(save_path, 'rb'), media_type="audio/wav")

        end_time = time.time()
        elapsed_time = end_time - start_time

        result.headers["X-Elapsed-Time"] = str(elapsed_time)
        result.headers["X-Device-Used"] = device

        # Add CORS headers
        result.headers["Access-Control-Allow-Origin"] = "*"
        result.headers["Access-Control-Allow-Credentials"] = "true"
        result.headers["Access-Control-Allow-Headers"] = "Origin, Content-Type, X-Amz-Date, Authorization, X-Api-Key, X-Amz-Security-Token, locale"
        result.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"

        return result
    except Exception as e:
        logging.error(f"Error in synthesize_speech: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
