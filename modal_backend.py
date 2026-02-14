"""
Tail a Tale — Chatterbox TTS Voice Cloning API on Modal
========================================================
Deploy: modal deploy modal_backend.py
Test:   modal run modal_backend.py

Prerequisites:
  1. pip install modal
  2. modal setup              (creates your API token)
  3. modal secret create hf-token HF_TOKEN=hf_YOUR_TOKEN
     (get token from https://huggingface.co/settings/tokens)

After deploying, copy the endpoint URL and paste it into the
Tail a Tale app settings (gear icon → TTS API URL).
"""

import modal

# --------------- Container Image ---------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "chatterbox-tts>=0.1.1",
        "fastapi[standard]>=0.115.0",
        "peft>=0.14.0",
        "torchaudio>=2.0.0",
        "python-multipart>=0.0.9",
    )
)

app = modal.App("tailtale-tts", image=image)

# --------------- TTS Service ---------------
with image.imports():
    import io
    import base64
    import tempfile
    import os

    import torch
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS
    from fastapi import Request
    from fastapi.responses import Response
    from fastapi.middleware.cors import CORSMiddleware


@app.cls(
    gpu="a10g",
    scaledown_window=60 * 5,  # keep warm 5 min after last request
    secrets=[modal.Secret.from_name("hf-token")],
)
@modal.concurrent(max_inputs=10)
class TTS:
    @modal.enter()
    def load(self):
        """Load Chatterbox model on container start."""
        self.model = ChatterboxTTS.from_pretrained(device="cuda")
        print("[TTS] ✅ Chatterbox model loaded")

    @modal.fastapi_endpoint(docs=True, method="POST")
    async def generate(self, request: Request):
        """
        Generate speech from text using voice cloning.

        Expects JSON body:
          - text: str           — text to synthesize
          - voice_b64: str      — base64-encoded WAV/WebM of reference voice
          - language: str       — language code (en, he, es, fr, ar, ru, pt, zh)
          - exaggeration: float — emotion intensity 0.0-1.0 (default 0.5)
        
        Returns: audio/wav
        """
        body = await request.json()
        text = body.get("text", "")
        voice_b64 = body.get("voice_b64", "")
        language = body.get("language", "en")
        exaggeration = float(body.get("exaggeration", 0.5))

        if not text:
            return Response(content='{"error":"no text"}', status_code=400,
                            media_type="application/json")
        if not voice_b64:
            return Response(content='{"error":"no voice audio"}', status_code=400,
                            media_type="application/json")

        # Decode voice audio from base64
        voice_bytes = base64.b64decode(voice_b64)

        # Save to temp file (Chatterbox needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            # If it's WebM (from MediaRecorder), convert to WAV via torchaudio
            if voice_bytes[:4] != b'RIFF':
                # Write raw bytes and convert
                raw_path = tmp_path + ".webm"
                with open(raw_path, "wb") as rf:
                    rf.write(voice_bytes)
                try:
                    waveform, sr = ta.load(raw_path)
                    # Resample to 24kHz mono
                    if sr != 24000:
                        waveform = ta.functional.resample(waveform, sr, 24000)
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    ta.save(tmp_path, waveform, 24000)
                finally:
                    os.unlink(raw_path)
            else:
                f.write(voice_bytes)

        try:
            # Generate speech with voice cloning
            wav = self.model.generate(
                text,
                audio_prompt_path=tmp_path,
                exaggeration=exaggeration,
            )

            # Encode to WAV bytes
            buffer = io.BytesIO()
            ta.save(buffer, wav, self.model.sr, format="wav")
            buffer.seek(0)
            audio_bytes = buffer.read()

            return Response(
                content=audio_bytes,
                media_type="audio/wav",
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                },
            )
        finally:
            os.unlink(tmp_path)

    @modal.fastapi_endpoint(docs=True, method="GET")
    async def health(self):
        """Health check — also useful to pre-warm the container."""
        return {"status": "ok", "model": "chatterbox"}

    @modal.fastapi_endpoint(docs=True, method="OPTIONS")
    async def cors_preflight(self, request: Request):
        """Handle CORS preflight."""
        return Response(
            content="",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS, GET",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Max-Age": "86400",
            },
        )


# --------------- Local test entrypoint ---------------
@app.local_entrypoint()
def test(
    text: str = "Once upon a time, in a land far away, a little bear yawned and snuggled into bed.",
    output_path: str = "/tmp/tailtale-test.wav",
):
    """Test the TTS endpoint locally."""
    tts = TTS()
    import pathlib

    # Simple test without voice cloning (would need a voice file)
    print(f"[Test] Generating: {text[:60]}...")
    print(f"[Test] Note: Full voice cloning requires a reference audio file.")
    print(f"[Test] Deploy with: modal deploy modal_backend.py")
    print(f"[Test] Then use the endpoint URL in Tail a Tale settings.")
