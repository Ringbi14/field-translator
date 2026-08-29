import os
import json
import base64
import io
import ssl
import uvicorn
import trustme
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
from gtts import gTTS

app = FastAPI()

api_key = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=api_key)

def generate_fast_audio(text: str, lang: str) -> str:
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return base64.b64encode(fp.read()).decode("utf-8")
    except Exception:
        return ""

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Tamil ↔ English Field Translator</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: #f1f5f9; color: #0f172a; padding: 1.25rem; display: flex; justify-content: center; }
    .card { background: #ffffff; max-width: 480px; width: 100%; border-radius: 16px; padding: 1.5rem; box-shadow: 0 4px 14px rgba(0,0,0,0.08); display: flex; flex-direction: column; gap: 1.25rem; }
    h1 { font-size: 1.25rem; font-weight: 700; text-align: center; }
    .mode-switch { display: flex; background: #e2e8f0; border-radius: 12px; padding: 4px; gap: 4px; }
    .mode-btn { flex: 1; padding: 12px; border: none; background: transparent; font-weight: 600; font-size: 0.95rem; border-radius: 8px; cursor: pointer; color: #475569; }
    .mode-btn.active { background: #2563eb; color: #ffffff; }
    .record-container { display: flex; flex-direction: column; align-items: center; gap: 0.75rem; margin: 0.75rem 0; }
    .record-btn { width: 85px; height: 85px; border-radius: 50%; border: none; background-color: #ef4444; color: white; font-size: 1.75rem; cursor: pointer; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4); }
    .record-btn.recording { animation: pulse 1.2s infinite; background-color: #dc2626; }
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.1); } 100% { transform: scale(1); } }
    .status-text { font-size: 0.95rem; color: #64748b; font-weight: 500; }
    .box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem; }
    .box-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem; }
    .box-title { font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; }
    .box-content { font-size: 1.15rem; line-height: 1.45; color: #0f172a; min-height: 28px; word-break: break-word; }
    .pronunciation { font-size: 0.95rem; color: #475569; font-style: italic; margin-top: 0.35rem; }
    .speak-btn { background: #e2e8f0; border: none; border-radius: 50%; width: 34px; height: 34px; cursor: pointer; font-size: 1rem; display: flex; align-items: center; justify-content: center; }
    .speak-btn:hover { background: #cbd5e1; }
    .toggle-row { display: flex; align-items: center; justify-content: flex-end; gap: 8px; font-size: 0.85rem; color: #64748b; font-weight: 600; }
    .toggle-row input { cursor: pointer; width: 16px; height: 16px; accent-color: #2563eb; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Field Translator</h1>
    
    <div class="mode-switch">
      <button id="btn-ta-en" class="mode-btn active" onclick="setDirection('ta_to_en')">Tamil ➔ English</button>
      <button id="btn-en-ta" class="mode-btn" onclick="setDirection('en_to_ta')">English ➔ Tamil</button>
    </div>

    <div class="toggle-row">
      <label for="auto-speak" style="cursor: pointer;">Auto-speak translation</label>
      <input type="checkbox" id="auto-speak" checked>
    </div>

    <div class="record-container">
      <button id="record-btn" class="record-btn" onclick="toggleRecord()">🎤</button>
      <div id="status" class="status-text">Tap mic to speak</div>
    </div>

    <div class="box">
      <div class="box-header">
        <div class="box-title">Spoken Text</div>
      </div>
      <div id="spoken-output" class="box-content">—</div>
    </div>

    <div class="box">
      <div class="box-header">
        <div class="box-title">Translation</div>
        <button id="btn-play-trans" class="speak-btn" onclick="playCurrentAudio()" title="Speak translation">🔊</button>
      </div>
      <div id="trans-output" class="box-content" style="color: #2563eb; font-weight: 600;">—</div>
      <div id="pronunciation-output" class="pronunciation"></div>
    </div>
  </div>

  <audio id="audio-player"></audio>

  <script>
    let currentDirection = 'ta_to_en';
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let currentAudioBase64 = '';

    function setDirection(dir) {
      currentDirection = dir;
      document.getElementById('btn-ta-en').classList.toggle('active', dir === 'ta_to_en');
      document.getElementById('btn-en-ta').classList.toggle('active', dir === 'en_to_ta');
      document.getElementById('pronunciation-output').innerText = '';
    }

    function playCurrentAudio() {
      if (!currentAudioBase64) return;
      const player = document.getElementById('audio-player');
      player.src = 'data:audio/mp3;base64,' + currentAudioBase64;
      player.play();
    }

    async function toggleRecord() {
      if (!isRecording) {
        startRecording();
      } else {
        stopRecording();
      }
    }

    async function startRecording() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
          if (event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = sendAudioToServer;
        mediaRecorder.start();

        isRecording = true;
        document.getElementById('record-btn').classList.add('recording');
        document.getElementById('status').innerText = 'Listening... Tap to finish';
      } catch (err) {
        alert('Microphone access denied or not supported.');
      }
    }

    function stopRecording() {
      if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        isRecording = false;
        document.getElementById('record-btn').classList.remove('recording');
        document.getElementById('status').innerText = 'Translating...';
      }
    }

    async function sendAudioToServer() {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      const formData = new FormData();
      formData.append('audio', audioBlob, 'speech.webm');
      formData.append('direction', currentDirection);

      try {
        const response = await fetch('/translate-audio', {
          method: 'POST',
          body: formData
        });

        const data = await response.json();
        document.getElementById('status').innerText = 'Tap mic to speak';
        
        if (data.error) {
          document.getElementById('trans-output').innerText = data.error;
          document.getElementById('spoken-output').innerText = '—';
          document.getElementById('pronunciation-output').innerText = '';
          currentAudioBase64 = '';
        } else {
          document.getElementById('spoken-output').innerText = data.spoken || '—';
          document.getElementById('trans-output').innerText = data.translation || '—';
          document.getElementById('pronunciation-output').innerText = data.pronunciation ? `Pronounce: "${data.pronunciation}"` : '';
          currentAudioBase64 = data.audio_base64 || '';

          if (document.getElementById('auto-speak').checked && currentAudioBase64) {
            playCurrentAudio();
          }
        }
      } catch (err) {
        document.getElementById('status').innerText = 'Tap mic to speak';
        document.getElementById('trans-output').innerText = 'Server error. Please try again.';
        document.getElementById('pronunciation-output').innerText = '';
        currentAudioBase64 = '';
      }
    }
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_CONTENT

@app.post("/translate-audio")
async def translate_audio(audio: UploadFile = File(...), direction: str = Form(...)):
    audio_bytes = await audio.read()
    
    if direction == "ta_to_en":
        prompt = (
            "You are an assistant for field researchers in Tamil Nadu. "
            "Listen to this Tamil speech. Extract spoken Tamil, translate into clean, natural English. "
            "Respond ONLY with a JSON object containing keys: spoken, translation, pronunciation (leave pronunciation empty string)."
        )
        target_lang = "en"
    else:
        prompt = (
            "You are an assistant for field researchers in Tamil Nadu. "
            "Listen to this English speech. Extract spoken English, translate into natural colloquial spoken Tamil script. "
            "Provide phonetic Tanglish pronunciation in English letters. "
            "Respond ONLY with a JSON object containing keys: spoken, translation, pronunciation."
        )
        target_lang = "ta"

    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=audio.content_type or 'audio/webm'),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        data = json.loads(response.text)
        
        translation_text = data.get("translation", "")
        data["audio_base64"] = generate_fast_audio(translation_text, target_lang) if translation_text else ""
        
        return data
    except Exception as e:
        return {"error": f"Error processing audio: {str(e)}"}

if __name__ == "__main__":
    # Generate local SSL certs on the fly
    ca = trustme.CA()
    cert = ca.issue_cert("localhost", "127.0.0.1", "0.0.0.0", "10.186.72.151")
    cert.private_key_pem.write_to_path("key.pem")
    cert.cert_chain_pems[0].write_to_path("cert.pem")

    print("\n\n>>> SERVER STARTED WITH HTTPS <<<")
    print("Open on your phone: https://10.186.72.151:8080\n\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem"
    )