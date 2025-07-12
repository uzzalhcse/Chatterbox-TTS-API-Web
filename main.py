import random
import numpy as np
import torch
from chatterbox.src.chatterbox.tts import ChatterboxTTS
import gradio as gr
import spaces
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional
import io
import wave
import tempfile
import os
import uvicorn
from contextlib import asynccontextmanager

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# --- Global Model Initialization ---
MODEL = None


def get_or_load_model():
    """Loads the ChatterboxTTS model if it hasn't been loaded already,
    and ensures it's on the correct device."""
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        try:
            MODEL = ChatterboxTTS.from_pretrained(DEVICE)
            if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
                MODEL.to(DEVICE)
            print(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return MODEL


# Attempt to load the model at startup.
try:
    get_or_load_model()
except Exception as e:
    print(f"CRITICAL: Failed to load model on startup. Application may not function. Error: {e}")


def set_seed(seed: int):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


@spaces.GPU
def generate_tts_audio(
        text_input: str,
        audio_prompt_path_input: str = None,
        exaggeration_input: float = 0.5,
        temperature_input: float = 0.8,
        seed_num_input: int = 0,
        cfgw_input: float = 0.5
) -> tuple[int, np.ndarray]:
    """
    Generate high-quality speech audio from text using ChatterboxTTS model with optional reference audio styling.

    This tool synthesizes natural-sounding speech from input text. When a reference audio file 
    is provided, it captures the speaker's voice characteristics and speaking style. The generated audio 
    maintains the prosody, tone, and vocal qualities of the reference speaker, or uses default voice if no reference is provided.

    Args:
        text_input (str): The text to synthesize into speech (maximum 300 characters)
        audio_prompt_path_input (str, optional): File path or URL to the reference audio file that defines the target voice style. Defaults to None.
        exaggeration_input (float, optional): Controls speech expressiveness (0.25-2.0, neutral=0.5, extreme values may be unstable). Defaults to 0.5.
        temperature_input (float, optional): Controls randomness in generation (0.05-5.0, higher=more varied). Defaults to 0.8.
        seed_num_input (int, optional): Random seed for reproducible results (0 for random generation). Defaults to 0.
        cfgw_input (float, optional): CFG/Pace weight controlling generation guidance (0.2-1.0). Defaults to 0.5.

    Returns:
        tuple[int, np.ndarray]: A tuple containing the sample rate (int) and the generated audio waveform (numpy.ndarray)
    """
    current_model = get_or_load_model()

    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    print(f"Generating audio for text: '{text_input[:50]}...'")

    # Handle optional audio prompt
    generate_kwargs = {
        "exaggeration": exaggeration_input,
        "temperature": temperature_input,
        "cfg_weight": cfgw_input,
    }

    if audio_prompt_path_input:
        generate_kwargs["audio_prompt_path"] = audio_prompt_path_input

    wav = current_model.generate(
        text_input[:300],  # Truncate text to max chars
        **generate_kwargs
    )
    print("Audio generation complete.")
    return (current_model.sr, wav.squeeze(0).numpy())


# --- REST API Models ---
class TTSRequest(BaseModel):
    text: str = Field(..., max_length=300, description="Text to synthesize into speech")
    exaggeration: float = Field(0.5, ge=0.25, le=2.0, description="Speech expressiveness")
    temperature: float = Field(0.8, ge=0.05, le=5.0, description="Generation randomness")
    seed: int = Field(0, ge=0, description="Random seed (0 for random)")
    cfg_weight: float = Field(0.5, ge=0.2, le=1.0, description="CFG/Pace weight")


class TTSResponse(BaseModel):
    message: str
    sample_rate: int
    audio_length_seconds: float


# --- Gradio Interface ---
with gr.Blocks(title="ChatterboxTTS") as demo:
    gr.Markdown(
        """
        # Chatterbox TTS Demo
        Generate high-quality speech from text with reference audio styling.

        **API Endpoints:**
        - `GET /` - API info
        - `GET /health` - Health check
        - `POST /generate` - Generate TTS from text only
        - `POST /generate-with-reference` - Generate TTS with reference audio
        - `GET /web` - This Gradio interface
        """
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize (max chars 300)",
                max_lines=5
            )
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File (Optional)",
                value="https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_shadowheart4.flac"
            )
            exaggeration = gr.Slider(
                0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5
            )
            cfg_weight = gr.Slider(
                0.2, 1, step=.05, label="CFG/Pace", value=0.5
            )

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    run_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
        ],
        outputs=[audio_output],
    )


# --- FastAPI App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting FastAPI server...")
    yield
    # Shutdown
    print("🛑 Shutting down FastAPI server...")


app = FastAPI(
    title="ChatterboxTTS API",
    description="High-quality text-to-speech API with voice cloning capabilities",
    version="1.0.0",
    lifespan=lifespan
)


def numpy_to_wav_bytes(audio_data: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy array to WAV bytes."""
    # Ensure audio is in the right format
    if audio_data.dtype != np.int16:
        # Convert float to int16
        audio_data = (audio_data * 32767).astype(np.int16)

    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    return wav_buffer.getvalue()


@app.get("/")
async def root():
    return {
        "message": "ChatterboxTTS API is running!",
        "endpoints": ["/generate", "/generate-with-reference", "/health", "/web", "/docs"]
    }


@app.get("/health")
async def health_check():
    try:
        model = get_or_load_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "device": DEVICE
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "device": DEVICE
        }


@app.post("/generate")
async def generate_audio(request: TTSRequest):
    """Generate TTS audio from text."""
    try:
        sample_rate, audio_data = generate_tts_audio(
            text_input=request.text,
            audio_prompt_path_input=None,
            exaggeration_input=request.exaggeration,
            temperature_input=request.temperature,
            seed_num_input=request.seed,
            cfgw_input=request.cfg_weight
        )

        # Convert to WAV bytes
        wav_bytes = numpy_to_wav_bytes(audio_data, sample_rate)

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=generated_audio.wav",
                "X-Sample-Rate": str(sample_rate),
                "X-Audio-Length": str(len(audio_data) / sample_rate)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")


@app.post("/generate-with-reference")
async def generate_audio_with_reference(
        text: str = Form(..., max_length=300),
        reference_audio: UploadFile = File(...),
        exaggeration: float = Form(0.5),
        temperature: float = Form(0.8),
        seed: int = Form(0),
        cfg_weight: float = Form(0.5)
):
    """Generate TTS audio with reference audio for voice cloning."""
    try:
        # Validate parameters
        if not (0.25 <= exaggeration <= 2.0):
            raise HTTPException(status_code=400, detail="Exaggeration must be between 0.25 and 2.0")
        if not (0.05 <= temperature <= 5.0):
            raise HTTPException(status_code=400, detail="Temperature must be between 0.05 and 5.0")
        if not (0.2 <= cfg_weight <= 1.0):
            raise HTTPException(status_code=400, detail="CFG weight must be between 0.2 and 1.0")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await reference_audio.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            sample_rate, audio_data = generate_tts_audio(
                text_input=text,
                audio_prompt_path_input=tmp_file_path,
                exaggeration_input=exaggeration,
                temperature_input=temperature,
                seed_num_input=seed,
                cfgw_input=cfg_weight
            )

            # Convert to WAV bytes
            wav_bytes = numpy_to_wav_bytes(audio_data, sample_rate)

            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=generated_audio_with_reference.wav",
                    "X-Sample-Rate": str(sample_rate),
                    "X-Audio-Length": str(len(audio_data) / sample_rate)
                }
            )
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")


# Mount Gradio app at /web
app = gr.mount_gradio_app(app, demo, path="/web")

if __name__ == "__main__":
    import threading
    import time
    import argparse

    parser = argparse.ArgumentParser(description='Run ChatterboxTTS with REST API and Web Interface')
    parser.add_argument('--use-ngrok', action='store_true', help='Use ngrok for public access')
    parser.add_argument('--ngrok-token', type=str, help='ngrok authtoken (required for static domains)')
    parser.add_argument('--static-domain', type=str, help='ngrok static domain (requires Pro/Business plan)')
    parser.add_argument('--port', type=int, default=8000, help='Port for server')
    args = parser.parse_args()

    # Setup ngrok if requested
    if args.use_ngrok:
        try:
            from pyngrok import ngrok
            import atexit

            # Set auth token if provided
            if args.ngrok_token:
                ngrok.set_auth_token(args.ngrok_token)
            elif args.static_domain:
                print("❌ Static domain requires ngrok auth token. Use --ngrok-token")
                exit(1)

            # Create ngrok tunnel with static domain if specified
            if args.static_domain:
                public_url = ngrok.connect(args.port, domain=args.static_domain)
                print(f"🚀 Static Domain: {public_url}")
                print(f"🔗 Your services are permanently available at:")
            else:
                public_url = ngrok.connect(args.port)
                print(f"🚀 Public URL: {public_url}")
                print("⚠️  Note: This is a temporary URL. Use --static-domain for permanent access.")
                print(f"🔗 Your services are available at:")

            print(f"   🎯 Web Interface: {public_url}/web")
            print(f"   📚 API Documentation: {public_url}/docs")
            print(f"   🏥 Health Check: {public_url}/health")
            print(f"   🔧 API Endpoints: {public_url}/generate, {public_url}/generate-with-reference")

            # Cleanup on exit
            atexit.register(ngrok.disconnect, public_url)

        except ImportError:
            print("❌ pyngrok not installed. Install with: pip install pyngrok")
            print("🔄 Continuing without ngrok...")
        except Exception as e:
            print(f"❌ ngrok setup failed: {e}")
            if "domain not found" in str(e).lower():
                print("💡 Make sure your static domain is created in ngrok dashboard")
            elif "authentication failed" in str(e).lower():
                print("💡 Check your ngrok auth token")
            print("🔄 Continuing without ngrok...")

    # Start the unified server
    print(f"🌐 Starting unified server on http://0.0.0.0:{args.port}")
    print(f"   🎯 Web Interface: http://0.0.0.0:{args.port}/web")
    print(f"   📚 API Documentation: http://0.0.0.0:{args.port}/docs")

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")