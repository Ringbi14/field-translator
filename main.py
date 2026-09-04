import os
import json
import base64
import io
import asyncio
from datetime import datetime, date, timedelta
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from google import genai
from google.genai import types
from gtts import gTTS

app = FastAPI(title="Field Translation Assistant")

api_key = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(api_version="v1alpha")
)

class TranslationResponse(BaseModel):
    spoken: str
    translation: str
    pronunciation: str

def generate_tts_audio(text: str, lang: str, gender: str = "male") -> str:
    if not text or text.strip() == "" or text.startswith("["):
        return ""
    try:
        if lang == "en":
            tld_accent = "co.in" if gender == "male" else "us"
        else:
            tld_accent = "co.in" if gender == "male" else "com"

        tts = gTTS(text=text, lang=lang, tld=tld_accent, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return base64.b64encode(fp.read()).decode("utf-8")
    except Exception:
        return ""

@app.get("/manifest.json")
async def manifest():
    return JSONResponse({
        "name": "Field Translation Assistant",
        "short_name": "FieldAssistant",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#06090e",
        "theme_color": "#0284c7",
        "icons": [
            {
                "src": "https://cdn-icons-png.flaticon.com/512/2875/2875338.png",
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    })

# ====================================================================
# LIVE INTERPRETER WEBSOCKET
# ====================================================================
@app.websocket("/ws/live-interpreter")
async def live_interpreter_proxy(websocket: WebSocket):
    await websocket.accept()

    if not api_key:
        await websocket.send_json({"error": "GEMINI_API_KEY not configured on server."})
        await websocket.close()
        return

    sys_prompt = (
        "You are a real-time live interpreter for social fieldwork in Tamil Nadu. "
        "Perform immediate bidirectional spoken interpretation between conversational Tamil and English. "
        "If you hear Tamil, immediately output the spoken translation in conversational English. "
        "If you hear English, immediately output the spoken translation in polite colloquial Tamil (Pechu Tamizh). "
        "Do not add conversational filler, intros, or explanations. Translate directly."
    )

    live_config = types.LiveConnectConfig(
        response_modalities=[types.LiveModality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
            )
        ),
        system_instruction=types.Content(parts=[types.Part.from_text(text=sys_prompt)])
    )

    try:
        async with client.aio.live.connect(model="gemini-2.0-flash-exp", config=live_config) as session:
            await websocket.send_json({"type": "ready"})

            async def client_inbound():
                try:
                    while True:
                        msg = await websocket.receive_text()
                        data = json.loads(msg)
                        if "audio_pcm" in data:
                            pcm_data = base64.b64decode(data["audio_pcm"])
                            chunk = types.Blob(data=pcm_data, mime_type="audio/pcm;rate=16000")
                            realtime_input = types.LiveClientRealtimeInput(media_chunks=[chunk])
                            await session.send(input=realtime_input)
                except (WebSocketDisconnect, asyncio.CancelledError):
                    pass
                except Exception as e:
                    print(f"Error reading client socket: {e}")

            async def gemini_outbound():
                try:
                    async for response in session.receive():
                        server_content = response.server_content
                        if server_content is not None:
                            model_turn = server_content.model_turn
                            if model_turn is not None:
                                for part in model_turn.parts:
                                    if part.inline_data:
                                        b64_str = base64.b64encode(part.inline_data.data).decode("utf-8")
                                        await websocket.send_json({"type": "audio", "data": b64_str})
                                    if part.text:
                                        await websocket.send_json({"type": "text", "data": part.text})

                            if server_content.turn_complete:
                                await websocket.send_json({"type": "turn_complete"})
                            if server_content.interrupted:
                                await websocket.send_json({"type": "interrupted"})
                except (WebSocketDisconnect, asyncio.CancelledError):
                    pass
                except Exception as e:
                    print(f"Error streaming Gemini output: {e}")

            in_task = asyncio.create_task(client_inbound())
            out_task = asyncio.create_task(gemini_outbound())
            await asyncio.wait([in_task, out_task], return_when=asyncio.FIRST_COMPLETED)
            in_task.cancel()
            out_task.cancel()

    except Exception as e:
        print(f"Session error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

# ====================================================================
# FAST TWO-WAY REAL-TIME FALLBACK ENDPOINT
# ====================================================================
@app.post("/api/live-turn")
async def live_turn(
    audio: UploadFile = File(...),
    gender: str = Form("male")
):
    audio_bytes = await audio.read()
    mime = audio.content_type or 'audio/webm'

    prompt = (
        "You are a real-time live interpreter for social fieldwork in Tamil Nadu. "
        "Detect the spoken language automatically. "
        "1. If spoken in Tamil, translate into clear conversational English. "
        "2. If spoken in English, translate into polite colloquial spoken Tamil (Pechu Tamizh). "
        "Return strictly JSON with keys: 'detected_lang', 'spoken', 'translation', 'pronunciation'."
    )

    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        res_data = json.loads(response.text)
        det_lang = res_data.get("detected_lang", "ta").lower()
        target_lang = "en" if "tam" in det_lang or det_lang == "ta" else "ta"
        
        trans_text = res_data.get("translation", "")
        res_data["audio_base64"] = generate_tts_audio(trans_text, target_lang, gender)
        return res_data
    except Exception as e:
        return {"error": f"Live error: {str(e)}"}

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Field Translation Assistant</title>
  
  <link rel="manifest" href="/manifest.json">
  <meta name="theme-color" content="#0284c7">
  <meta name="mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <link rel="apple-touch-icon" href="https://cdn-icons-png.flaticon.com/512/2875/2875338.png">

  <style>
    :root {
      --bg-dark: #070b12;
      --panel-bg: #0f172a;
      --card-bg: #131d31;
      --border-color: #1e293b;
      --accent-cyan: #38bdf8;
      --accent-blue: #0284c7;
      --accent-emerald: #10b981;
      --accent-rose: #f43f5e;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; -webkit-tap-highlight-color: transparent; }
    body { background-color: var(--bg-dark); color: var(--text-main); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

    header { background: #0c1322; padding: 0.75rem 1rem; border-bottom: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; z-index: 10; }
    .brand { display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 1rem; color: var(--accent-cyan); letter-spacing: 0.5px; }
    .header-badges { display: flex; align-items: center; gap: 8px; }
    .status-pill { font-size: 0.7rem; padding: 3px 8px; border-radius: 20px; background: #064e3b; color: #34d399; font-weight: 700; text-transform: uppercase; }
    .status-pill.offline { background: #7f1d1d; color: #fca5a5; }

    .voice-select { background: #131d31; border: 1px solid var(--border-color); color: var(--accent-cyan); font-size: 0.75rem; font-weight: 600; padding: 3px 6px; border-radius: 8px; outline: none; }

    nav.app-tabs { display: flex; background: #0c1322; border-bottom: 1px solid var(--border-color); overflow-x: auto; }
    .tab-btn { flex: 1; min-width: 78px; padding: 12px 6px; border: none; background: transparent; color: var(--text-muted); font-weight: 600; font-size: 0.75rem; cursor: pointer; border-bottom: 2px solid transparent; display: flex; align-items: center; justify-content: center; gap: 4px; white-space: nowrap; }
    .tab-btn.active { color: var(--accent-cyan); border-bottom-color: var(--accent-cyan); background: rgba(56, 189, 248, 0.05); }
    .tab-btn.live-tab { color: #f43f5e; }
    .tab-btn.live-tab.active { border-bottom-color: #f43f5e; color: #f43f5e; background: rgba(244, 63, 94, 0.08); }

    .privacy-banner { background: #09101d; padding: 5px 12px; font-size: 0.7rem; color: #64748b; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #141e33; }

    main { flex: 1; overflow-y: auto; padding: 1rem; display: flex; flex-direction: column; }
    .view-section { display: none; flex-direction: column; height: 100%; gap: 1rem; }
    .view-section.active { display: flex; }

    .direction-container { display: flex; background: #0b1325; border: 1px solid #1e2d4a; border-radius: 14px; padding: 4px; gap: 6px; align-items: center; }
    .dir-toggle-btn { flex: 1; padding: 12px; border: none; background: transparent; font-weight: 700; font-size: 0.9rem; border-radius: 10px; cursor: pointer; color: var(--text-muted); transition: all 0.2s; }
    .dir-toggle-btn.active { background: var(--accent-blue); color: #ffffff; box-shadow: 0 2px 10px rgba(2, 132, 199, 0.4); }
    .swap-btn { background: #1a2744; border: 1px solid #2a3c63; width: 36px; height: 36px; border-radius: 50%; color: var(--accent-cyan); cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 1rem; }

    .feed-container { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 0.85rem; padding-bottom: 0.5rem; }
    .record-card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 16px; padding: 1rem; display: flex; flex-direction: column; gap: 0.6rem; position: relative; }
    .record-card.ta-source { border-left: 4px solid var(--accent-emerald); }
    .record-card.en-source { border-left: 4px solid var(--accent-blue); }

    .card-meta-row { display: flex; justify-content: space-between; align-items: center; font-size: 0.7rem; color: var(--text-muted); font-weight: 700; text-transform: uppercase; }
    .expiry-tag { color: #f59e0b; font-weight: 600; font-size: 0.7rem; }
    
    .section-title { font-size: 0.68rem; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 2px; }
    .transcript-text { font-size: 1.05rem; color: #e2e8f0; line-height: 1.45; word-break: break-word; }
    .translation-text { font-size: 1.15rem; font-weight: 600; color: #38bdf8; line-height: 1.45; word-break: break-word; }
    .record-card.ta-source .translation-text { color: #34d399; }
    .phonetic-guide { font-size: 0.85rem; color: #7dd3fc; font-style: italic; }

    .dual-audio-dock { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 6px; padding-top: 8px; border-top: 1px solid #1c2b48; }
    .audio-play-btn { background: #17233d; border: 1px solid #273a63; border-radius: 10px; padding: 8px 10px; color: #e2e8f0; font-size: 0.75rem; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 6px; }
    .audio-play-btn.accent { background: #0c3358; border-color: #0284c7; color: #38bdf8; }

    /* Live Interpreter Console */
    .live-console-box { background: #0c1527; border: 1px solid #1e335a; border-radius: 16px; padding: 1.25rem; display: flex; flex-direction: column; align-items: center; gap: 1rem; text-align: center; }
    .live-badge-status { font-size: 0.85rem; font-weight: 700; padding: 6px 14px; border-radius: 20px; background: #1f293d; color: #94a3b8; display: flex; align-items: center; gap: 8px; }
    .live-badge-status.listening { background: #064e3b; color: #34d399; }
    .live-badge-status.speaking { background: #0369a1; color: #38bdf8; }
    .live-badge-status.error { background: #7f1d1d; color: #fca5a5; }

    .live-stream-feed { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 0.75rem; margin-top: 0.5rem; }
    .live-bubble { background: var(--card-bg); border-radius: 14px; padding: 0.85rem 1rem; border: 1px solid var(--border-color); display: flex; flex-direction: column; gap: 4px; }
    .live-bubble .bubble-lang { font-size: 0.7rem; font-weight: 700; color: var(--accent-cyan); text-transform: uppercase; }
    .live-bubble .bubble-text { font-size: 1.15rem; line-height: 1.4; color: #ffffff; }

    .live-trigger-btn { width: 100%; max-width: 260px; padding: 16px; border-radius: 30px; font-size: 1.05rem; font-weight: 800; border: none; cursor: pointer; letter-spacing: 0.5px; transition: all 0.25s ease; display: flex; align-items: center; justify-content: center; gap: 10px; }
    .live-trigger-btn.start { background: linear-gradient(135deg, #10b981, #059669); color: #fff; box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4); }
    .live-trigger-btn.end { background: linear-gradient(135deg, #ef4444, #dc2626); color: #fff; box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4); }

    .controls-cockpit { background: #0c1322; border-top: 1px solid var(--border-color); padding: 0.85rem 1rem; display: flex; flex-direction: column; align-items: center; gap: 0.6rem; z-index: 10; }
    .visualizer-canvas { width: 100%; height: 26px; display: none; }
    .visualizer-canvas.active { display: block; }
    
    .cockpit-row { display: flex; align-items: center; justify-content: center; width: 100%; gap: 1.8rem; position: relative; }
    .sub-action-btn { width: 46px; height: 46px; border-radius: 50%; background: #131d31; border: 1px solid var(--border-color); color: #e2e8f0; font-size: 1.2rem; display: flex; align-items: center; justify-content: center; cursor: pointer; }
    
    .mic-rig { position: relative; width: 88px; height: 88px; display: flex; align-items: center; justify-content: center; }
    .mic-sonar-wave { position: absolute; width: 100%; height: 100%; border-radius: 50%; border: 2px solid var(--accent-rose); opacity: 0; pointer-events: none; }
    .is-listening .mic-sonar-wave { animation: sonar 1.4s cubic-bezier(0, 0.2, 0.8, 1) infinite; }
    @keyframes sonar { 0% { transform: scale(1); opacity: 0.8; } 100% { transform: scale(1.65); opacity: 0; } }

    .main-mic-trigger { width: 84px; height: 84px; border-radius: 50%; border: none; background: linear-gradient(135deg, #ef4444, #b91c1c); color: white; font-size: 2rem; cursor: pointer; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 20px rgba(239, 68, 68, 0.45); z-index: 2; transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1); }
    .main-mic-trigger.is-recording { background: linear-gradient(135deg, #f43f5e, #be123c); transform: scale(1.08); box-shadow: 0 0 28px rgba(244, 63, 94, 0.8); }
    .main-mic-trigger.is-translating { background: linear-gradient(135deg, #0284c7, #0369a1); pointer-events: none; animation: radar 1.5s linear infinite; }
    @keyframes radar { 0% { filter: hue-rotate(0deg); } 100% { filter: hue-rotate(90deg); } }

    .cockpit-status { font-size: 0.85rem; font-weight: 600; color: var(--text-muted); display: flex; align-items: center; gap: 6px; }

    .form-box { width: 100%; min-height: 110px; background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 14px; padding: 0.85rem; color: #fff; font-size: 1rem; resize: none; outline: none; }
    .form-box:focus { border-color: var(--accent-cyan); }
    .submit-btn { background: var(--accent-blue); color: #fff; border: none; padding: 12px 18px; border-radius: 12px; font-weight: 700; font-size: 0.95rem; cursor: pointer; width: 100%; display: flex; align-items: center; justify-content: center; gap: 8px; }
    
    .upload-dropzone { border: 2px dashed #243556; background: #0d1527; border-radius: 16px; padding: 2rem 1rem; text-align: center; display: flex; flex-direction: column; align-items: center; gap: 0.75rem; cursor: pointer; }
    .library-stream { display: flex; flex-direction: column; gap: 0.85rem; overflow-y: auto; }
    .library-item { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 14px; padding: 1rem; display: flex; flex-direction: column; gap: 0.5rem; }
  </style>
</head>
<body>
  <header>
    <div class="brand">⚡ Field Assistant</div>
    <div class="header-badges">
      <select id="voice-gender" class="voice-select" onchange="updateVoiceGender(this.value)">
        <option value="male" selected>🎙️ Male Voice</option>
        <option value="female">🎙️ Female Voice</option>
      </select>
      <span id="network-pill" class="status-pill">ONLINE</span>
      <button class="audio-play-btn" style="padding:4px 8px;" onclick="exportRecordsTxt()">📤 Export</button>
    </div>
  </header>

  <div class="privacy-banner">
    <span>🔒 Privacy: All recordings auto-delete after 4 days</span>
    <span id="storage-counter">0 Saved</span>
  </div>

  <nav class="app-tabs">
    <button class="tab-btn live-tab" onclick="switchView('live')">⚡ Live Interpreter</button>
    <button class="tab-btn active" onclick="switchView('translate')">🎙️ Translate</button>
    <button class="tab-btn" onclick="switchView('convo')">💬 Conversation</button>
    <button class="tab-btn" onclick="switchView('text')">📝 Text</button>
    <button class="tab-btn" onclick="switchView('upload')">📁 Upload</button>
    <button class="tab-btn" onclick="switchView('library')">📚 Library (<span id="tab-lib-count">0</span>)</button>
  </nav>

  <main>
    <!-- 0. LIVE INTERPRETER VIEW -->
    <div id="view-live" class="view-section">
      <div class="live-console-box">
        <div style="font-size:1.1rem; font-weight:800; color:#38bdf8;">LIVE INTERPRETER</div>
        <div style="font-size:0.95rem; color:#e2e8f0; font-weight:600;">🇮🇳 Tamil ↔ English 🇬🇧</div>
        <div id="live-connection-badge" class="live-badge-status">🔴 Standby</div>
        
        <button id="live-toggle-btn" class="live-trigger-btn start" onclick="toggleLiveInterpreterSession()">
          🎙️ START LIVE
        </button>
        <div style="font-size:0.75rem; color:#64748b;">Continuous hands-free bidirectional interpretation.</div>
      </div>

      <div id="live-stream-feed" class="live-stream-feed">
        <div class="live-bubble">
          <div class="bubble-lang">System Ready</div>
          <div class="bubble-text" style="font-size:0.95rem; color:#94a3b8;">
            Tap [ START LIVE ] to begin continuous interpretation. Speak freely in either Tamil or English.
          </div>
        </div>
      </div>
    </div>

    <!-- 1. LIVE TRANSLATE VIEW (WALKIE-TALKIE) -->
    <div id="view-translate" class="view-section active">
      <div class="direction-container">
        <button id="main-dir-ta" class="dir-toggle-btn active" onclick="setDirection('ta_to_en')">Tamil ➔ English</button>
        <button class="swap-btn" onclick="swapCurrentDirection()">⇄</button>
        <button id="main-dir-en" class="dir-toggle-btn" onclick="setDirection('en_to_ta')">English ➔ Tamil</button>
      </div>

      <div id="translate-feed" class="feed-container">
        <div class="record-card ta-source">
          <div class="card-meta-row">
            <span>System Console</span>
            <span class="expiry-tag" id="current-voice-badge">Voice: Male</span>
          </div>
          <div class="transcript-text">Ready for fieldwork. Tap the microphone to record speech. Original audio is preserved separately from the translated voice.</div>
        </div>
      </div>
    </div>

    <!-- 2. CONVERSATION VIEW -->
    <div id="view-convo" class="view-section">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:0.8rem; color:var(--text-muted); font-weight:700;">CONVERSATION MODE</span>
        <button class="audio-play-btn" style="background:#7f1d1d; border-color:#991b1b;" onclick="clearConversationFeed()">Clear Chat</button>
      </div>
      <div id="convo-feed" class="feed-container"></div>
    </div>

    <!-- 3. TEXT TRANSLATE VIEW -->
    <div id="view-text" class="view-section">
      <div class="direction-container">
        <button id="text-dir-ta" class="dir-toggle-btn active" onclick="setTextDirection('ta_to_en')">Tamil ➔ English</button>
        <button class="swap-btn" onclick="swapTextDirection()">⇄</button>
        <button id="text-dir-en" class="dir-toggle-btn" onclick="setTextDirection('en_to_ta')">English ➔ Tamil</button>
      </div>
      <textarea id="text-input" class="form-box" placeholder="Type Tamil or English text here..."></textarea>
      <button class="submit-btn" onclick="submitTextTranslation()">⚡ Translate Text</button>
      
      <div id="text-result-box" class="record-card" style="display:none;">
        <div class="card-meta-row">
          <span>Translation Output</span>
          <span class="expiry-tag" id="text-voice-badge">Voice: Male</span>
        </div>
        <div id="text-result-trans" class="translation-text">—</div>
        <div id="text-result-phonetic" class="phonetic-guide"></div>
        <div class="dual-audio-dock">
          <button class="audio-play-btn accent" onclick="playTextAudio()">🔊 Play Translated Voice</button>
          <button class="audio-play-btn" onclick="saveTextTranslationRecord()">💾 Save to Library</button>
        </div>
      </div>
    </div>

    <!-- 4. UPLOAD AUDIO FILE VIEW -->
    <div id="view-upload" class="view-section">
      <div class="direction-container">
        <button id="upload-dir-ta" class="dir-toggle-btn active" onclick="setUploadDirection('ta_to_en')">Tamil Audio ➔ English</button>
        <button class="swap-btn" onclick="swapUploadDirection()">⇄</button>
        <button id="upload-dir-en" class="dir-toggle-btn" onclick="setUploadDirection('en_to_ta')">English Audio ➔ Tamil</button>
      </div>

      <label class="upload-dropzone">
        <span style="font-size:2.2rem;">📁</span>
        <div style="font-weight:700; color:#e2e8f0;">Select Audio File</div>
        <div style="font-size:0.75rem; color:#64748b;">Supports MP3, WAV, M4A, WEBM, OGG</div>
        <input type="file" id="file-uploader" accept="audio/*" style="display:none;" onchange="handleDirectFileUpload(event)">
      </label>

      <div id="upload-status" class="cockpit-status" style="justify-content:center;"></div>
    </div>

    <!-- 5. 4-DAY SAVED RECORDINGS LIBRARY -->
    <div id="view-library" class="view-section">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:0.8rem; color:#94a3b8;">Recordings Auto-Delete After 4 Days</span>
        <button class="audio-play-btn" style="background:#7f1d1d; border-color:#991b1b;" onclick="clearEntireLibrary()">Delete All</button>
      </div>
      <div id="library-list" class="library-stream"></div>
    </div>
  </main>

  <div class="controls-cockpit" id="bottom-cockpit">
    <canvas id="waveform-scope" class="visualizer-canvas"></canvas>

    <div class="cockpit-row">
      <button class="sub-action-btn" title="Record without translating immediately (Poor connection)" onclick="recordOfflineDraft()">
        💾
      </button>

      <div class="mic-rig" id="mic-rig">
        <div class="mic-sonar-wave"></div>
        <button id="main-mic-btn" class="main-mic-trigger" onclick="toggleMainRecording()">🎤</button>
      </div>

      <button class="sub-action-btn" id="auto-speak-toggle" title="Toggle Auto-Speak" onclick="toggleAutoSpeak()">
        🔊
      </button>
    </div>
    
    <div id="status-readout" class="cockpit-status">Tap mic to speak</div>
  </div>

  <audio id="primary-audio-player" playsinline preload="auto"></audio>

  <script>
    let currentDirection = 'ta_to_en';
    let textDirection = 'ta_to_en';
    let uploadDirection = 'ta_to_en';
    let selectedGender = localStorage.getItem('voice_gender') || 'male';
    let autoSpeak = true;
    let isRecording = false;
    let isDraftMode = false;
    let mediaRecorder = null;
    let audioChunks = [];
    let audioContext = null;
    let analyserNode = null;
    let canvasCtx = null;
    let animationId = null;
    let recordTimer = null;
    let elapsedSeconds = 0;
    let lastTextRecord = null;

    const player = document.getElementById('primary-audio-player');

    // =======================================================
    // ROBUST LIVE INTERPRETER CONTINUOUS ENGINE
    // =======================================================
    let liveActive = false;
    let liveMediaStream = null;
    let liveRecorder = null;
    let liveInterval = null;

    async function toggleLiveInterpreterSession() {
      unlockAudio();
      if (!liveActive) {
        startLiveContinuous();
      } else {
        stopLiveContinuous();
      }
    }

    async function startLiveContinuous() {
      const btn = document.getElementById('live-toggle-btn');
      const badge = document.getElementById('live-connection-badge');

      badge.className = 'live-badge-status listening';
      badge.innerText = '🎙️ Listening Continuously...';
      btn.className = 'live-trigger-btn end';
      btn.innerText = '⏹️ END LIVE';
      liveActive = true;

      try {
        liveMediaStream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true, noiseSuppression: true }
        });

        // Run continuous turn-based interpretation slices
        startNextLiveSlice();
      } catch (err) {
        badge.className = 'live-badge-status error';
        badge.innerText = '🔴 Mic Access Required';
        alert('Microphone permission required for Live Interpreter.');
        stopLiveContinuous();
      }
    }

    function startNextLiveSlice() {
      if (!liveActive || !liveMediaStream) return;

      let sliceChunks = [];
      let options = {};
      if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
        options = { mimeType: 'audio/webm;codecs=opus' };
      } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
        options = { mimeType: 'audio/mp4' };
      }

      liveRecorder = new MediaRecorder(liveMediaStream, options);
      liveRecorder.ondataavailable = e => {
        if (e.data.size > 0) sliceChunks.push(e.data);
      };

      liveRecorder.onstop = async () => {
        if (sliceChunks.length > 0) {
          const mime = liveRecorder.mimeType || 'audio/webm';
          const blob = new Blob(sliceChunks, { type: mime });
          if (blob.size > 3000) {
            sendLiveAudioTurn(blob);
          }
        }
        if (liveActive) {
          startNextLiveSlice();
        }
      };

      liveRecorder.start();
      setTimeout(() => {
        if (liveRecorder && liveRecorder.state === 'recording') {
          liveRecorder.stop();
        }
      }, 4500);
    }

    async function sendLiveAudioTurn(blob) {
      const badge = document.getElementById('live-connection-badge');
      badge.className = 'live-badge-status speaking';
      badge.innerText = '🧠 Interpreting...';

      const formData = new FormData();
      formData.append('audio', blob, 'speech.webm');
      formData.append('gender', selectedGender);

      try {
        const res = await fetch('/api/live-turn', { method: 'POST', body: formData });
        const data = await res.json();
        
        if (data.translation && !data.translation.includes("Could not understand")) {
          appendLiveResult(data.spoken, data.translation, data.detected_lang);
          if (data.audio_base64) {
            playBase64Audio(data.audio_base64);
          }
        }
      } catch (e) {
        console.error(e);
      } finally {
        if (liveActive) {
          badge.className = 'live-badge-status listening';
          badge.innerText = '🎙️ Listening Continuously...';
        }
      }
    }

    function appendLiveResult(spoken, translation, lang) {
      const feed = document.getElementById('live-stream-feed');
      const bubble = document.createElement('div');
      bubble.className = 'live-bubble';
      const label = (lang && lang.toLowerCase().includes('en')) ? 'English ➔ Tamil' : 'Tamil ➔ English';
      bubble.innerHTML = `
        <div class="bubble-lang">${label}</div>
        <div style="font-size:0.95rem; color:#94a3b8;">"${spoken}"</div>
        <div class="bubble-text" style="margin-top:4px;">${translation}</div>
      `;
      feed.appendChild(bubble);
      feed.scrollTop = feed.scrollHeight;
    }

    function stopLiveContinuous() {
      liveActive = false;
      if (liveRecorder && liveRecorder.state === 'recording') {
        liveRecorder.stop();
      }
      if (liveMediaStream) {
        liveMediaStream.getTracks().forEach(t => t.stop());
        liveMediaStream = null;
      }
      const btn = document.getElementById('live-toggle-btn');
      const badge = document.getElementById('live-connection-badge');
      if (btn) {
        btn.className = 'live-trigger-btn start';
        btn.innerText = '🎙️ START LIVE';
      }
      if (badge) {
        badge.className = 'live-badge-status';
        badge.innerText = '🔴 Standby';
      }
    }

    // =======================================================
    // EXISTING NORMAL TRANSLATOR, INDEXEDDB & RECORDING LOGIC
    // =======================================================
    document.addEventListener('DOMContentLoaded', () => {
      const select = document.getElementById('voice-gender');
      if (select) select.value = selectedGender;
      updateVoiceBadgeText();
    });

    function updateVoiceGender(gender) {
      selectedGender = gender;
      localStorage.setItem('voice_gender', gender);
      updateVoiceBadgeText();
    }

    function updateVoiceBadgeText() {
      const badge = document.getElementById('current-voice-badge');
      const textBadge = document.getElementById('text-voice-badge');
      const label = selectedGender === 'male' ? 'Voice: Male' : 'Voice: Female';
      if (badge) badge.innerText = label;
      if (textBadge) textBadge.innerText = label;
    }

    function unlockAudio() {
      if (player.paused && !player.src) {
        player.src = 'data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA';
        player.play().catch(() => {});
      }
    }

    function playBase64Audio(b64) {
      if (!b64) return;
      player.src = 'data:audio/mp3;base64,' + b64;
      player.play().catch(e => console.log('Audio playback error:', e));
    }

    function playBlobAudio(blob) {
      if (!blob) return;
      player.src = URL.createObjectURL(blob);
      player.play().catch(e => console.log('Blob playback error:', e));
    }

    function playTextAudio() {
      if (lastTextRecord && lastTextRecord.audioBase64) {
        playBase64Audio(lastTextRecord.audioBase64);
      }
    }

    let db;
    const dbReq = indexedDB.open('FieldAssistantDB', 2);
    dbReq.onupgradeneeded = e => {
      db = e.target.result;
      if (!db.objectStoreNames.contains('records')) {
        db.createObjectStore('records', { keyPath: 'id', autoIncrement: true });
      }
    };
    dbReq.onsuccess = e => {
      db = e.target.result;
      purgeExpiredRecords();
      renderLibrary();
    };

    function calculateExpiry(createdDateStr) {
      const created = new Date(createdDateStr);
      const expiry = new Date(created);
      expiry.setDate(created.getDate() + 4);
      
      const now = new Date();
      const diffMs = expiry - now;
      const daysLeft = Math.ceil(diffMs / (1000 * 60 * 60 * 24));
      
      return {
        expiryDateStr: expiry.toLocaleDateString([], { month: 'short', day: 'numeric' }),
        daysLeft: Math.max(0, daysLeft),
        isExpired: diffMs <= 0
      };
    }

    function purgeExpiredRecords() {
      if (!db) return;
      const tx = db.transaction('records', 'readwrite');
      const store = tx.objectStore('records');
      store.openCursor().onsuccess = e => {
        const cursor = e.target.result;
        if (cursor) {
          const item = cursor.value;
          const exp = calculateExpiry(item.isoDate);
          if (exp.isExpired) cursor.delete();
          cursor.continue();
        }
      };
      tx.oncomplete = () => updateRecordCounts();
    }

    function updateRecordCounts() {
      if (!db) return;
      const tx = db.transaction('records', 'readonly');
      const req = tx.objectStore('records').count();
      req.onsuccess = () => {
        document.getElementById('storage-counter').innerText = `${req.result} Saved`;
        document.getElementById('tab-lib-count').innerText = req.result;
      };
    }

    function switchView(viewName) {
      if (viewName !== 'live' && liveActive) {
        stopLiveContinuous();
      }

      document.querySelectorAll('.tab-btn').forEach((btn, idx) => {
        const match = (viewName === 'live' && idx === 0) ||
                      (viewName === 'translate' && idx === 1) ||
                      (viewName === 'convo' && idx === 2) ||
                      (viewName === 'text' && idx === 3) ||
                      (viewName === 'upload' && idx === 4) ||
                      (viewName === 'library' && idx === 5);
        btn.classList.toggle('active', match);
      });

      document.getElementById('view-live').classList.toggle('active', viewName === 'live');
      document.getElementById('view-translate').classList.toggle('active', viewName === 'translate');
      document.getElementById('view-convo').classList.toggle('active', viewName === 'convo');
      document.getElementById('view-text').classList.toggle('active', viewName === 'text');
      document.getElementById('view-upload').classList.toggle('active', viewName === 'upload');
      document.getElementById('view-library').classList.toggle('active', viewName === 'library');

      const cockpit = document.getElementById('bottom-cockpit');
      cockpit.style.display = (viewName === 'translate' || viewName === 'convo') ? 'flex' : 'none';

      if (viewName === 'library') renderLibrary();
    }

    function setDirection(dir) {
      currentDirection = dir;
      document.getElementById('main-dir-ta').classList.toggle('active', dir === 'ta_to_en');
      document.getElementById('main-dir-en').classList.toggle('active', dir === 'en_to_ta');
    }

    function swapCurrentDirection() {
      setDirection(currentDirection === 'ta_to_en' ? 'en_to_ta' : 'ta_to_en');
    }

    function setTextDirection(dir) {
      textDirection = dir;
      document.getElementById('text-dir-ta').classList.toggle('active', dir === 'ta_to_en');
      document.getElementById('text-dir-en').classList.toggle('active', dir === 'en_to_ta');
    }

    function swapTextDirection() {
      setTextDirection(textDirection === 'ta_to_en' ? 'en_to_ta' : 'ta_to_en');
    }

    function setUploadDirection(dir) {
      uploadDirection = dir;
      document.getElementById('upload-dir-ta').classList.toggle('active', dir === 'ta_to_en');
      document.getElementById('upload-dir-en').classList.toggle('active', dir === 'en_to_ta');
    }

    function swapUploadDirection() {
      setUploadDirection(uploadDirection === 'ta_to_en' ? 'en_to_ta' : 'ta_to_en');
    }

    function toggleAutoSpeak() {
      autoSpeak = !autoSpeak;
      document.getElementById('auto-speak-toggle').style.opacity = autoSpeak ? '1' : '0.4';
    }

    async function toggleMainRecording() {
      unlockAudio();
      if (!isRecording) {
        startRecording(false);
      } else {
        stopRecording();
      }
    }

    function recordOfflineDraft() {
      unlockAudio();
      if (!isRecording) {
        startRecording(true);
      } else {
        stopRecording();
      }
    }

    async function startRecording(draftMode = false) {
      isDraftMode = draftMode;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true, noiseSuppression: true }
        });

        try {
          audioContext = new (window.AudioContext || window.webkitAudioContext)();
          const source = audioContext.createMediaStreamSource(stream);
          analyserNode = audioContext.createAnalyser();
          analyserNode.fftSize = 64;
          source.connect(analyserNode);
          startVisualizerAnimation();
        } catch (e) {}

        let options = {};
        if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
          options = { mimeType: 'audio/webm;codecs=opus' };
        } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
          options = { mimeType: 'audio/mp4' };
        }

        mediaRecorder = new MediaRecorder(stream, options);
        audioChunks = [];
        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
        mediaRecorder.onstop = handleRecordingComplete;
        mediaRecorder.start();
        isRecording = true;

        const mic = document.getElementById('main-mic-btn');
        const rig = document.getElementById('mic-rig');
        mic.classList.add('is-recording');
        mic.innerText = '⏹️';
        rig.classList.add('is-listening');

        elapsedSeconds = 0;
        document.getElementById('status-readout').innerText = isDraftMode ? 'Saving Audio... (00:00)' : 'Listening... (00:00)';
        recordTimer = setInterval(() => {
          elapsedSeconds++;
          const mins = String(Math.floor(elapsedSeconds / 60)).padStart(2, '0');
          const secs = String(elapsedSeconds % 60).padStart(2, '0');
          document.getElementById('status-readout').innerText = `${isDraftMode ? 'Saving Audio' : 'Listening'} (${mins}:${secs})`;
        }, 1000);

      } catch (err) {
        alert('Microphone permission required.');
      }
    }

    function stopRecording() {
      if (!mediaRecorder || !isRecording) return;
      clearInterval(recordTimer);
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(t => t.stop());
      isRecording = false;

      stopVisualizerAnimation();

      const mic = document.getElementById('main-mic-btn');
      const rig = document.getElementById('mic-rig');
      mic.classList.remove('is-recording');
      rig.classList.remove('is-listening');

      if (!isDraftMode) {
        mic.classList.add('is-translating');
        mic.innerText = '⏳';
        document.getElementById('status-readout').innerText = `AI Translating (${selectedGender} voice)...`;
      } else {
        mic.innerText = '🎤';
        document.getElementById('status-readout').innerText = 'Saved to Library';
      }
    }

    function startVisualizerAnimation() {
      const canvas = document.getElementById('waveform-scope');
      canvas.classList.add('active');
      canvasCtx = canvas.getContext('2d');
      const bufferLength = analyserNode.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      function draw() {
        animationId = requestAnimationFrame(draw);
        analyserNode.getByteFrequencyData(dataArray);
        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
        
        const barWidth = (canvas.width / bufferLength) * 2;
        let x = 0;
        for (let i = 0; i < bufferLength; i++) {
          const barHeight = (dataArray[i] / 255) * canvas.height;
          canvasCtx.fillStyle = '#38bdf8';
          canvasCtx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
          x += barWidth;
        }
      }
      draw();
    }

    function stopVisualizerAnimation() {
      if (animationId) cancelAnimationFrame(animationId);
      const canvas = document.getElementById('waveform-scope');
      canvas.classList.remove('active');
    }

    async function handleRecordingComplete() {
      const mime = mediaRecorder.mimeType || 'audio/webm';
      const audioBlob = new Blob(audioChunks, { type: mime });
      
      if (audioBlob.size < 1200 && !isDraftMode) {
        alert("No speech detected. Please speak clearly and try again.");
        resetMicUi();
        return;
      }

      await processAudioRecord(audioBlob, currentDirection, isDraftMode);
    }

    function handleDirectFileUpload(e) {
      unlockAudio();
      const file = e.target.files[0];
      if (!file) return;

      document.getElementById('upload-status').innerText = 'Processing file translation...';
      processAudioRecord(file, uploadDirection, false).then(() => {
        document.getElementById('upload-status').innerText = 'Translation ready! Saved to Library.';
        switchView('translate');
      });
    }

    async function processAudioRecord(blobOrFile, direction, isOfflineDraft) {
      const now = new Date();
      const recordItem = {
        isoDate: now.toISOString(),
        dateStr: now.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' }),
        timeStr: now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        direction: direction,
        gender: selectedGender,
        originalAudioBlob: blobOrFile,
        spoken: '',
        translation: '',
        pronunciation: '',
        audioBase64: '',
        status: isOfflineDraft ? 'draft_saved' : (navigator.onLine ? 'processing' : 'draft_saved')
      };

      if (!isOfflineDraft && navigator.onLine) {
        const formData = new FormData();
        formData.append('audio', blobOrFile, blobOrFile.name || 'speech.webm');
        formData.append('direction', direction);
        formData.append('gender', selectedGender);

        try {
          const res = await fetch('/translate-audio', { method: 'POST', body: formData });
          const data = await res.json();
          if (!data.error) {
            recordItem.spoken = data.spoken || '';
            recordItem.translation = data.translation || '';
            recordItem.pronunciation = data.pronunciation || '';
            recordItem.audioBase64 = data.audio_base64 || '';
            recordItem.status = 'completed';

            if (autoSpeak && recordItem.audioBase64) {
              playBase64Audio(recordItem.audioBase64);
            }
          } else {
            recordItem.spoken = '[Indistinct speech]';
            recordItem.translation = data.error;
            recordItem.status = 'error';
          }
        } catch (err) {
          recordItem.status = 'draft_saved';
          recordItem.translation = 'Saved offline (Connection dropped). Translate later.';
        }
      }

      saveRecordToIndexedDB(recordItem);
      appendLiveRecordCard(recordItem);
      resetMicUi();
    }

    function resetMicUi() {
      const mic = document.getElementById('main-mic-btn');
      mic.classList.remove('is-translating');
      mic.classList.remove('is-recording');
      mic.innerText = '🎤';
      document.getElementById('status-readout').innerText = 'Tap mic to speak';
    }

    async function submitTextTranslation() {
      unlockAudio();
      const text = document.getElementById('text-input').value.trim();
      if (!text) return;

      document.getElementById('text-result-box').style.display = 'none';
      const formData = new FormData();
      formData.append('text', text);
      formData.append('direction', textDirection);
      formData.append('gender', selectedGender);

      try {
        const res = await fetch('/translate-text', { method: 'POST', body: formData });
        const data = await res.json();
        
        lastTextRecord = {
          isoDate: new Date().toISOString(),
          dateStr: new Date().toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' }),
          timeStr: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          direction: textDirection,
          gender: selectedGender,
          originalAudioBlob: null,
          spoken: text,
          translation: data.translation || '',
          pronunciation: data.pronunciation || '',
          audioBase64: data.audio_base64 || '',
          status: 'completed'
        };

        document.getElementById('text-result-box').style.display = 'flex';
        document.getElementById('text-result-trans').innerText = data.translation;
        document.getElementById('text-result-phonetic').innerText = data.pronunciation ? `Tanglish: "${data.pronunciation}"` : '';

        if (autoSpeak && lastTextRecord.audioBase64) {
          playBase64Audio(lastTextRecord.audioBase64);
        }
      } catch (err) {
        alert('Translation failed. Check network connection.');
      }
    }

    function saveTextTranslationRecord() {
      if (!lastTextRecord) return;
      saveRecordToIndexedDB(lastTextRecord);
      alert('Saved to Library (4-day auto retention).');
    }

    function appendLiveRecordCard(item) {
      const exp = calculateExpiry(item.isoDate);
      const isTa = item.direction === 'ta_to_en';
      const voiceLabel = (item.gender || selectedGender) === 'male' ? 'Male Voice' : 'Female Voice';
      
      const cardHtml = `
        <div class="record-card ${isTa ? 'ta-source' : 'en-source'}">
          <div class="card-meta-row">
            <span>${isTa ? 'Tamil ➔ English' : 'English ➔ Tamil'}</span>
            <span class="expiry-tag">Expires in ${exp.daysLeft}d</span>
          </div>
          
          <div class="section-title">🎙️ Original Speech</div>
          <div class="transcript-text">${item.spoken || (item.status === 'draft_saved' ? 'Audio recorded (Saved offline)' : 'Processing...')}</div>
          
          <div class="section-title" style="margin-top:6px;">🌐 Translation (${voiceLabel})</div>
          <div class="translation-text">${item.translation || ''}</div>
          ${item.pronunciation ? `<div class="phonetic-guide">Tanglish: "${item.pronunciation}"</div>` : ''}
          
          <div class="dual-audio-dock">
            <button class="audio-play-btn" onclick="playRecordOriginal(${item.id})">🎙️ Original Voice</button>
            ${item.audioBase64 ? `<button class="audio-play-btn accent" onclick="playBase64Audio('${item.audioBase64}')">🔊 ${voiceLabel}</button>` : ''}
            ${item.status === 'draft_saved' ? `<button class="audio-play-btn accent" onclick="translateDraftRecord(${item.id})">⚡ Translate</button>` : ''}
          </div>
        </div>
      `;

      const transFeed = document.getElementById('translate-feed');
      transFeed.insertAdjacentHTML('afterbegin', cardHtml);

      const convoFeed = document.getElementById('convo-feed');
      convoFeed.insertAdjacentHTML('beforeend', cardHtml);
      convoFeed.scrollTop = convoFeed.scrollHeight;
    }

    function saveRecordToIndexedDB(item) {
      if (!db) return;
      const tx = db.transaction('records', 'readwrite');
      const store = tx.objectStore('records');
      const req = store.add(item);
      req.onsuccess = e => {
        item.id = e.target.result;
        updateRecordCounts();
      };
    }

    function renderLibrary() {
      if (!db) return;
      const list = document.getElementById('library-list');
      list.innerHTML = '';

      const tx = db.transaction('records', 'readonly');
      const store = tx.objectStore('records');
      store.openCursor(null, 'prev').onsuccess = e => {
        const cursor = e.target.result;
        if (cursor) {
          const row = cursor.value;
          const id = cursor.key;
          const exp = calculateExpiry(row.isoDate);
          const voiceLabel = (row.gender || 'male') === 'male' ? 'Male Voice' : 'Female Voice';

          const div = document.createElement('div');
          div.className = 'library-item';
          div.innerHTML = `
            <div class="card-meta-row">
              <span>${row.direction === 'ta_to_en' ? 'Tamil ➔ English' : 'English ➔ Tamil'} • ${row.timeStr} (${row.dateStr})</span>
              <span class="expiry-tag">Expires: ${exp.expiryDateStr} (${exp.daysLeft}d)</span>
            </div>
            <div class="transcript-text" style="font-size:0.95rem;">${row.spoken || 'Audio Recording (Offline Draft)'}</div>
            <div class="translation-text" style="font-size:1rem;">${row.translation || 'Pending Translation'}</div>
            ${row.pronunciation ? `<div class="phonetic-guide">${row.pronunciation}</div>` : ''}
            
            <div class="dual-audio-dock">
              ${row.originalAudioBlob ? `<button class="audio-play-btn" onclick="playRecordOriginal(${id})">🎙️ Original</button>` : ''}
              ${row.audioBase64 ? `<button class="audio-play-btn accent" onclick="playBase64Audio('${row.audioBase64}')">🔊 ${voiceLabel}</button>` : ''}
              ${row.status === 'draft_saved' ? `<button class="audio-play-btn accent" onclick="translateDraftRecord(${id})">⚡ Translate</button>` : ''}
              <button class="audio-play-btn" style="background:#7f1d1d; border-color:#991b1b;" onclick="deleteSingleRecord(${id})">🗑 Delete</button>
            </div>
          `;
          list.appendChild(div);
          cursor.continue();
        }
      };
      updateRecordCounts();
    }

    function playRecordOriginal(id) {
      if (!db) return;
      const tx = db.transaction('records', 'readonly');
      tx.objectStore('records').get(id).onsuccess = e => {
        const item = e.target.result;
        if (item && item.originalAudioBlob) playBlobAudio(item.originalAudioBlob);
      };
    }

    async function translateDraftRecord(id) {
      if (!navigator.onLine) {
        alert('Internet connection required to translate saved audio.');
        return;
      }
      const tx = db.transaction('records', 'readwrite');
      const store = tx.objectStore('records');
      store.get(id).onsuccess = async e => {
        const item = e.target.result;
        const formData = new FormData();
        formData.append('audio', item.originalAudioBlob, 'speech.webm');
        formData.append('direction', item.direction);
        formData.append('gender', selectedGender);

        try {
          const res = await fetch('/translate-audio', { method: 'POST', body: formData });
          const data = await res.json();
          item.spoken = data.spoken || '';
          item.translation = data.translation || '';
          item.pronunciation = data.pronunciation || '';
          item.audioBase64 = data.audio_base64 || '';
          item.gender = selectedGender;
          item.status = 'completed';

          const updateTx = db.transaction('records', 'readwrite');
          updateTx.objectStore('records').put(item);
          updateTx.oncomplete = () => {
            renderLibrary();
            alert('Audio translated and updated in Library.');
          };
        } catch (err) {
          alert('Failed to connect to translation server.');
        }
      };
    }

    function deleteSingleRecord(id) {
      if (!db) return;
      const tx = db.transaction('records', 'readwrite');
      tx.objectStore('records').delete(id);
      tx.oncomplete = () => renderLibrary();
    }

    function clearEntireLibrary() {
      if (!confirm('Permanently delete all saved recordings from this device?')) return;
      const tx = db.transaction('records', 'readwrite');
      tx.objectStore('records').clear();
      tx.oncomplete = () => renderLibrary();
    }

    function clearConversationFeed() {
      document.getElementById('convo-feed').innerHTML = '';
    }

    function exportRecordsTxt() {
      if (!db) return;
      const tx = db.transaction('records', 'readonly');
      let text = `FIELDWORK TRANSLATION DOSSIER\\nGenerated: ${new Date().toLocaleString()}\\nAuto-Deletion Policy: 4-Day Retention\\n==========================================\\n\\n`;
      tx.objectStore('records').openCursor().onsuccess = e => {
        const cursor = e.target.result;
        if (cursor) {
          const item = cursor.value;
          const exp = calculateExpiry(item.isoDate);
          text += `[${item.dateStr} ${item.timeStr}] Direction: ${item.direction === 'ta_to_en' ? 'Tamil -> English' : 'English -> Tamil'}\\n`;
          text += `Retention: Expires on ${exp.expiryDateStr}\\n`;
          text += `Original Transcript: ${item.spoken || 'N/A'}\\n`;
          text += `Meaning Translation: ${item.translation || 'N/A'}\\n`;
          if (item.pronunciation) text += `Tanglish: ${item.pronunciation}\\n`;
          text += `------------------------------------------\\n\\n`;
          cursor.continue();
        } else {
          const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
          const a = document.createElement('a');
          a.href = URL.createObjectURL(blob);
          a.download = `fieldwork_records_${Date.now()}.txt`;
          a.click();
        }
      };
    }

    function updateOnlineBadge() {
      const pill = document.getElementById('network-pill');
      if (navigator.onLine) {
        pill.innerText = 'ONLINE';
        pill.className = 'status-pill';
      } else {
        pill.innerText = 'OFFLINE';
        pill.className = 'status-pill offline';
      }
    }
    window.addEventListener('online', updateOnlineBadge);
    window.addEventListener('offline', updateOnlineBadge);
    updateOnlineBadge();
  </script>
</body>
</html>
"""

@app.post("/translate-audio")
async def translate_audio(
    audio: UploadFile = File(...),
    direction: str = Form(...),
    gender: str = Form("male")
):
    audio_bytes = await audio.read()
    mime = audio.content_type or 'audio/webm'

    if direction == "ta_to_en":
        prompt = (
            "You are an expert field research translator in Tamil Nadu assisting an interviewer. "
            "1. Transcribe conversational, colloquial spoken Tamil with high tolerance for ambient field noise. "
            "2. Translate into clear, practical, meaning-focused English (never stiff or overly literal). "
            "3. If the audio is completely indistinct or contains no spoken words, set spoken to '[Indistinct speech]' and translation to 'Could not understand clearly. Please speak again.' "
            "Leave pronunciation empty."
        )
        target_voice_lang = "en"
    else:
        prompt = (
            "You are an expert field research translator in Tamil Nadu assisting an interviewer communicating with local residents. "
            "1. Accurately capture the spoken English intent. "
            "2. Translate into natural, polite colloquial spoken Tamil (Pechu Tamizh / பேச்சுத் தமிழ்) that local community members easily understand. "
            "3. Provide a clear, natural phonetic Tanglish pronunciation guide in English letters. "
            "4. If audio is unclear, set spoken to '[Indistinct speech]' and translation to 'Could not understand clearly. Please try again.'"
        )
        target_voice_lang = "ta"

    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TranslationResponse,
                temperature=0.1
            )
        )
        data = json.loads(response.text)
        trans_text = data.get("translation", "")
        data["audio_base64"] = generate_tts_audio(trans_text, target_voice_lang, gender)
        return data
    except Exception as e:
        return {"error": f"Audio processing error: {str(e)}"}

@app.post("/translate-text")
async def translate_text(
    text: str = Form(...),
    direction: str = Form(...),
    gender: str = Form("male")
):
    if direction == "ta_to_en":
        prompt = (
            f"Translate this spoken Tamil to simple, natural conversational English: '{text}'. "
            "Leave pronunciation empty."
        )
        target_voice_lang = "en"
    else:
        prompt = (
            f"Translate this English text into polite colloquial spoken Tamil (Pechu Tamizh) and provide phonetic Tanglish pronunciation: '{text}'."
        )
        target_voice_lang = "ta"

    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TranslationResponse,
                temperature=0.1
            )
        )
        data = json.loads(response.text)
        trans_text = data.get("translation", "")
        data["audio_base64"] = generate_tts_audio(trans_text, target_voice_lang, gender)
        return data
    except Exception as e:
        return {"error": f"Text processing error: {str(e)}"}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_CONTENT

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
