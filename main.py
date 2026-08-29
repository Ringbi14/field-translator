import os
import json
import base64
import io
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
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

@app.get("/manifest.json")
async def manifest():
    return JSONResponse({
        "name": "Field Translator",
        "short_name": "Translator",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0f172a",
        "theme_color": "#2563eb",
        "icons": [
            {
                "src": "https://cdn-icons-png.flaticon.com/512/2875/2875338.png",
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    })

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Field Translator</title>
  
  <link rel="manifest" href="/manifest.json">
  <meta name="theme-color" content="#2563eb">
  <meta name="mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <meta name="apple-mobile-web-app-title" content="Translator">
  <link rel="apple-touch-icon" href="https://cdn-icons-png.flaticon.com/512/2875/2875338.png">

  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; -webkit-tap-highlight-color: transparent; }
    body { background-color: #0f172a; color: #f8fafc; display: flex; justify-content: center; height: 100vh; overflow: hidden; }
    .app-shell { width: 100%; max-width: 480px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; padding: 1.25rem; }
    .top-bar { display: flex; align-items: center; justify-content: space-between; padding-bottom: 0.5rem; }
    .app-title { font-size: 1.1rem; font-weight: 700; letter-spacing: 0.5px; color: #e2e8f0; }
    .mode-switch { display: flex; background: #1e293b; border-radius: 14px; padding: 5px; gap: 6px; border: 1px solid #334155; }
    .mode-btn { flex: 1; padding: 14px 10px; border: none; background: transparent; font-weight: 600; font-size: 0.95rem; border-radius: 10px; cursor: pointer; color: #94a3b8; transition: all 0.2s ease; }
    .mode-btn.active { background: #2563eb; color: #ffffff; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4); }
    .feed-area { display: flex; flex-direction: column; gap: 1rem; flex: 1; margin: 1rem 0; overflow-y: auto; }
    .card-box { background: #1e293b; border: 1px solid #334155; border-radius: 16px; padding: 1rem 1.15rem; }
    .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; }
    .card-title { font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
    .card-content { font-size: 1.15rem; line-height: 1.45; color: #f8fafc; min-height: 28px; word-break: break-word; }
    .pronunciation { font-size: 0.95rem; color: #38bdf8; font-style: italic; margin-top: 0.4rem; }
    .speak-btn { background: #334155; border: none; border-radius: 50%; width: 34px; height: 34px; cursor: pointer; font-size: 1rem; color: #ffffff; display: flex; align-items: center; justify-content: center; }
    .controls { display: flex; flex-direction: column; align-items: center; gap: 0.75rem; padding-bottom: 0.5rem; }
    .record-btn { width: 90px; height: 90px; border-radius: 50%; border: none; background: linear-gradient(135deg, #ef4444, #dc2626); color: white; font-size: 2rem; cursor: pointer; display: flex; align-items: center; justify-content: center; box-shadow: 0 6px 20px rgba(239, 68, 68, 0.45); }
    .record-btn.recording { animation: pulse 1.1s infinite; background: linear-gradient(135deg, #f87171, #ef4444); }
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.12); } 100% { transform: scale(1); } }
    .status-text { font-size: 0.9rem; color: #94a3b8; font-weight: 500; }
    .toggle-row { display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: #94a3b8; font-weight: 500; }
    .toggle-row input { cursor: pointer; width: 16px; height: 16px; accent-color: #2563eb; }
  </style>
</head>
<body>
  <div class="app-shell">
    <div class="top-bar">
      <div class="app-title">⚡ Field Translator</div>
      <div class="toggle-row">
        <label for="auto-speak" style="cursor: pointer;">Auto-Speak</label>
        <input type="checkbox" id="auto-speak" checked>
      </div>
    </div>

    <div class="mode-switch">
      <button id="btn-ta-en" class="mode-btn active" onclick="setDirection('ta_to_en')">Tamil ➔ English</button>
      <button id="btn-en-ta" class="mode-btn" onclick="setDirection('en_to_ta')">English ➔ Tamil</button>
    </div>

    <div class="feed-area">
      <div class="card-box">
        <div class="card-header">
          <div class="card-title">Original Speech</div>
        </div>
        <div id="spoken-output" class="card-content">—</div>
      </div>

      <div class="card-box" style="border-color: #3b82f6;">
        <div class="card-header">
          <div class="card-title" style="color: #60a5fa;">Translation</div>
          <button id="btn-play-trans" class="speak-btn" onclick="playCurrentAudio()">🔊</button>
        </div>
        <div id="trans-output" class="card-content" style="color: #60a5fa; font-weight: 600;">—</div>
        <div id="pronunciation-output" class="pronunciation"></div>
      </div>
    </div>

    <div class="controls">
      <button id="record-btn" class="record-btn" onclick="toggleRecord()">🎤</button>
      <div id="status" class="status-text">Tap mic to speak</div>
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
          document.getElementById('pronunciation-output').innerText = data.pronunciation ? `Tanglish: "${data.pronunciation}"` : '';
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
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
