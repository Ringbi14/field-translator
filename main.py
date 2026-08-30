import os
import json
import base64
import io
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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
        "name": "Field Translator Assistant",
        "short_name": "FieldAssistant",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#090d16",
        "theme_color": "#1d4ed8",
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
  <title>Field Translation Assistant</title>
  
  <link rel="manifest" href="/manifest.json">
  <meta name="theme-color" content="#1d4ed8">
  <meta name="mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <link rel="apple-touch-icon" href="https://cdn-icons-png.flaticon.com/512/2875/2875338.png">

  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; -webkit-tap-highlight-color: transparent; }
    body { background-color: #0b0f19; color: #f1f5f9; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
    
    /* Top Header */
    header { background: #111827; padding: 0.75rem 1rem; border-bottom: 1px solid #1f2937; display: flex; justify-content: space-between; align-items: center; }
    .logo-badge { font-weight: 700; font-size: 1rem; color: #60a5fa; display: flex; align-items: center; gap: 6px; }
    .header-actions { display: flex; align-items: center; gap: 10px; }
    .status-pill { font-size: 0.75rem; padding: 3px 8px; border-radius: 12px; background: #064e3b; color: #34d399; font-weight: 600; }
    .status-pill.offline { background: #7f1d1d; color: #f87171; }

    /* Navigation Tabs */
    .nav-tabs { display: flex; background: #111827; border-bottom: 1px solid #1f2937; }
    .nav-tab { flex: 1; padding: 10px; border: none; background: transparent; color: #94a3b8; font-weight: 600; font-size: 0.85rem; cursor: pointer; border-bottom: 2px solid transparent; }
    .nav-tab.active { color: #38bdf8; border-bottom-color: #38bdf8; background: #1e293b; }

    /* Main Container */
    main { flex: 1; overflow-y: auto; padding: 1rem; display: flex; flex-direction: column; gap: 1rem; }
    .view-panel { display: none; flex-direction: column; height: 100%; gap: 1rem; }
    .view-panel.active { display: flex; }

    /* Direction Switcher */
    .mode-switch { display: flex; background: #1e293b; border-radius: 12px; padding: 4px; gap: 4px; border: 1px solid #334155; }
    .mode-btn { flex: 1; padding: 10px; border: none; background: transparent; font-weight: 600; font-size: 0.85rem; border-radius: 8px; cursor: pointer; color: #94a3b8; }
    .mode-btn.active { background: #2563eb; color: #ffffff; }

    /* Conversation Stream */
    .convo-feed { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 0.75rem; padding-bottom: 0.5rem; }
    .msg-card { background: #1e293b; border: 1px solid #334155; border-radius: 14px; padding: 0.85rem; display: flex; flex-direction: column; gap: 0.4rem; }
    .msg-card.ta { border-left: 4px solid #10b981; }
    .msg-card.en { border-left: 4px solid #3b82f6; }
    .msg-meta { display: flex; justify-content: space-between; font-size: 0.7rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; }
    .msg-original { font-size: 1rem; color: #e2e8f0; line-height: 1.4; }
    .msg-translated { font-size: 1.05rem; font-weight: 600; color: #60a5fa; line-height: 1.4; }
    .msg-card.ta .msg-translated { color: #34d399; }
    .msg-phonetic { font-size: 0.85rem; color: #38bdf8; font-style: italic; }
    .msg-tools { display: flex; gap: 8px; margin-top: 4px; }
    .tool-btn { background: #334155; border: none; border-radius: 6px; padding: 4px 8px; color: #cbd5e1; font-size: 0.75rem; cursor: pointer; display: flex; align-items: center; gap: 4px; }

    /* Visualizer & Mic Area */
    .controls-panel { background: #111827; border-top: 1px solid #1f2937; padding: 0.75rem 1rem; display: flex; flex-direction: column; align-items: center; gap: 0.5rem; }
    .mic-row { display: flex; align-items: center; justify-content: center; width: 100%; gap: 1.5rem; position: relative; }
    .action-btn-circle { width: 44px; height: 44px; border-radius: 50%; background: #1e293b; border: 1px solid #334155; color: #e2e8f0; font-size: 1.1rem; display: flex; align-items: center; justify-content: center; cursor: pointer; }
    
    /* Dedicated Mic Wrapper for animations */
    .mic-wrapper { position: relative; width: 84px; height: 84px; display: flex; align-items: center; justify-content: center; }
    
    /* Ripple Rings for recording */
    .ripple-ring { position: absolute; width: 100%; height: 100%; border-radius: 50%; border: 2px solid #ef4444; opacity: 0; pointer-events: none; }
    .recording .ripple-ring { animation: ring-pulse 1.6s cubic-bezier(0, 0.2, 0.8, 1) infinite; }
    @keyframes ring-pulse { 0% { transform: scale(1); opacity: 0.8; } 100% { transform: scale(1.6); opacity: 0; } }

    /* Recording / Translating Button Base */
    .record-btn { width: 80px; height: 80px; border-radius: 50%; border: none; background: linear-gradient(135deg, #ef4444, #dc2626); color: white; font-size: 1.8rem; cursor: pointer; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4); z-index: 2; transition: all 0.25s ease; }
    
    /* RECORDING ACTIVE STATE */
    .record-btn.is-recording { background: linear-gradient(135deg, #f43f5e, #e11d48); transform: scale(1.06); box-shadow: 0 0 25px rgba(244, 63, 94, 0.7); animation: breathe 1.2s ease-in-out infinite alternate; }
    @keyframes breathe { 0% { transform: scale(1.02); } 100% { transform: scale(1.1); } }

    /* TRANSLATING ACTIVE STATE */
    .record-btn.is-translating { background: linear-gradient(135deg, #2563eb, #1d4ed8); pointer-events: none; box-shadow: 0 0 25px rgba(37, 99, 235, 0.8); animation: radar-spin 2s linear infinite; }
    @keyframes radar-spin { 0% { filter: hue-rotate(0deg); } 100% { filter: hue-rotate(60deg); } }

    /* Animated Sound Waveform Bars */
    .waveform-bar-container { display: none; align-items: center; justify-content: center; gap: 4px; height: 20px; }
    .waveform-bar-container.active { display: flex; }
    .wave-bar { width: 3px; height: 6px; background: #f43f5e; border-radius: 2px; animation: wave-jump 0.8s ease-in-out infinite alternate; }
    .wave-bar:nth-child(1) { animation-delay: 0.1s; }
    .wave-bar:nth-child(2) { animation-delay: 0.3s; }
    .wave-bar:nth-child(3) { animation-delay: 0.5s; }
    .wave-bar:nth-child(4) { animation-delay: 0.2s; }
    .wave-bar:nth-child(5) { animation-delay: 0.4s; }
    @keyframes wave-jump { 0% { height: 4px; } 100% { height: 18px; } }

    .status-text { font-size: 0.85rem; color: #94a3b8; font-weight: 600; letter-spacing: 0.3px; display: flex; align-items: center; gap: 6px; }

    /* Form & Text Mode */
    .text-input-box { width: 100%; min-height: 90px; background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 0.75rem; color: #fff; font-size: 1rem; resize: none; }
    .primary-btn { background: #2563eb; color: #fff; border: none; padding: 10px 16px; border-radius: 10px; font-weight: 600; font-size: 0.95rem; cursor: pointer; }
    .secondary-btn { background: #334155; color: #fff; border: none; padding: 8px 12px; border-radius: 8px; font-size: 0.85rem; cursor: pointer; }

    /* Library List */
    .library-item { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 0.85rem; display: flex; flex-direction: column; gap: 0.5rem; }
    .library-header { display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8; }
    .library-title { font-weight: 600; font-size: 0.95rem; color: #f1f5f9; }
  </style>
</head>
<body>
  <header>
    <div class="logo-badge">⚡ Field Assistant</div>
    <div class="header-actions">
      <span id="network-pill" class="status-pill">ONLINE</span>
      <button class="secondary-btn" style="padding: 4px 8px; font-size: 0.75rem;" onclick="exportRecordsTxt()">📤 Export</button>
    </div>
  </header>

  <nav class="nav-tabs">
    <button class="nav-tab active" onclick="switchView('convo')">💬 Conversation</button>
    <button class="nav-tab" onclick="switchView('text')">📝 Text Translate</button>
    <button class="nav-tab" onclick="switchView('library')">📚 Library (<span id="lib-count">0</span>)</button>
  </nav>

  <main>
    <!-- 1. LIVE CONVERSATION VIEW -->
    <div id="view-convo" class="view-panel active">
      <div class="mode-switch">
        <button id="btn-ta-en" class="mode-btn active" onclick="setDirection('ta_to_en')">Tamil ➔ English</button>
        <button id="btn-en-ta" class="mode-btn" onclick="setDirection('en_to_ta')">English ➔ Tamil</button>
      </div>

      <div id="convo-feed" class="convo-feed">
        <div class="msg-card ta">
          <div class="msg-meta"><span>System</span><span>Ready</span></div>
          <div class="msg-original">Tap the microphone to speak or upload audio. Offline recordings will be saved locally.</div>
        </div>
      </div>
    </div>

    <!-- 2. DIRECT TEXT VIEW -->
    <div id="view-text" class="view-panel">
      <div class="mode-switch">
        <button id="text-ta-en" class="mode-btn active" onclick="setTextDirection('ta_to_en')">Tamil ➔ English</button>
        <button id="text-en-ta" class="mode-btn" onclick="setTextDirection('en_to_ta')">English ➔ Tamil</button>
      </div>
      <textarea id="text-input" class="text-input-box" placeholder="Type words or phrase here..."></textarea>
      <button class="primary-btn" onclick="submitTextTranslation()">Translate Text</button>

      <div id="text-result" class="msg-card en" style="display: none;">
        <div class="msg-meta"><span>Result</span></div>
        <div id="text-result-body" class="msg-translated">—</div>
        <div id="text-result-phonetic" class="msg-phonetic"></div>
        <div class="msg-tools">
          <button class="tool-btn" onclick="playTextResultAudio()">🔊 Listen</button>
        </div>
      </div>
    </div>

    <!-- 3. SAVED RECORDINGS LIBRARY -->
    <div id="view-library" class="view-panel">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 0.85rem; color: #94a3b8;">Local Device Storage</span>
        <button class="secondary-btn" onclick="clearAllLibrary()">Clear All</button>
      </div>
      <div id="library-list" style="display: flex; flex-direction: column; gap: 0.75rem; overflow-y: auto;">
        <!-- Saved items populate here -->
      </div>
    </div>
  </main>

  <!-- BOTTOM CONTROLS (Active during conversation) -->
  <div class="controls-panel" id="bottom-controls">
    <!-- Live Waveform Visualizer -->
    <div id="waveform-container" class="waveform-bar-container">
      <div class="wave-bar"></div>
      <div class="wave-bar"></div>
      <div class="wave-bar"></div>
      <div class="wave-bar"></div>
      <div class="wave-bar"></div>
    </div>

    <div class="mic-row">
      <label class="action-btn-circle" title="Upload Audio File">
        📁
        <input type="file" id="audio-upload-input" accept="audio/*" style="display: none;" onchange="handleFileUpload(event)">
      </label>

      <!-- Animated Mic Wrapper -->
      <div class="mic-wrapper" id="mic-wrapper">
        <div class="ripple-ring"></div>
        <button id="record-btn" class="record-btn" onclick="toggleRecord()">🎤</button>
      </div>

      <button class="action-btn-circle" id="auto-speak-toggle" title="Auto-speak on/off" onclick="toggleAutoSpeak()">
        🔊
      </button>
    </div>
    
    <div id="status" class="status-text">Tap mic to speak</div>
  </div>

  <audio id="audio-player"></audio>

  <script>
    let currentDirection = 'ta_to_en';
    let textDirection = 'ta_to_en';
    let autoSpeak = true;
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let textResultAudioBase64 = '';
    let recordTimerInterval = null;
    let recordSeconds = 0;

    // Database Initialization (IndexedDB for offline persistence)
    let db;
    const dbReq = indexedDB.open('FieldTranslatorDB', 1);
    dbReq.onupgradeneeded = e => {
      db = e.target.result;
      if (!db.objectStoreNames.contains('recordings')) {
        db.createObjectStore('recordings', { keyPath: 'id', autoIncrement: true });
      }
    };
    dbReq.onsuccess = e => {
      db = e.target.result;
      renderLibrary();
    };

    // Network Status Watcher
    function updateNetworkStatus() {
      const pill = document.getElementById('network-pill');
      if (navigator.onLine) {
        pill.innerText = 'ONLINE';
        pill.className = 'status-pill';
      } else {
        pill.innerText = 'OFFLINE';
        pill.className = 'status-pill offline';
      }
    }
    window.addEventListener('online', updateNetworkStatus);
    window.addEventListener('offline', updateNetworkStatus);
    updateNetworkStatus();

    // UI Tab Navigation
    function switchView(viewName) {
      document.querySelectorAll('.nav-tab').forEach((t, i) => {
        t.classList.toggle('active', (viewName === 'convo' && i === 0) || (viewName === 'text' && i === 1) || (viewName === 'library' && i === 2));
      });
      document.getElementById('view-convo').classList.toggle('active', viewName === 'convo');
      document.getElementById('view-text').classList.toggle('active', viewName === 'text');
      document.getElementById('view-library').classList.toggle('active', viewName === 'library');
      document.getElementById('bottom-controls').style.display = (viewName === 'convo') ? 'flex' : 'none';
      if (viewName === 'library') renderLibrary();
    }

    function setDirection(dir) {
      currentDirection = dir;
      document.getElementById('btn-ta-en').classList.toggle('active', dir === 'ta_to_en');
      document.getElementById('btn-en-ta').classList.toggle('active', dir === 'en_to_ta');
    }

    function setTextDirection(dir) {
      textDirection = dir;
      document.getElementById('text-ta-en').classList.toggle('active', dir === 'ta_to_en');
      document.getElementById('text-en-ta').classList.toggle('active', dir === 'en_to_ta');
    }

    function toggleAutoSpeak() {
      autoSpeak = !autoSpeak;
      document.getElementById('auto-speak-toggle').style.opacity = autoSpeak ? '1' : '0.4';
    }

    function playBase64Audio(b64) {
      if (!b64) return;
      const player = document.getElementById('audio-player');
      player.src = 'data:audio/mp3;base64,' + b64;
      player.play();
    }

    function playBlobAudio(blob) {
      if (!blob) return;
      const player = document.getElementById('audio-player');
      player.src = URL.createObjectURL(blob);
      player.play();
    }

    // Recording Logic with dynamic UI states
    async function toggleRecord() {
      if (!isRecording) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          mediaRecorder = new MediaRecorder(stream);
          audioChunks = [];
          mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
          mediaRecorder.onstop = handleRecordingComplete;
          mediaRecorder.start();
          isRecording = true;

          // State 1: RECORDING VISUALS
          const btn = document.getElementById('record-btn');
          const wrapper = document.getElementById('mic-wrapper');
          const wave = document.getElementById('waveform-container');
          
          btn.classList.add('is-recording');
          btn.innerText = '⏹️';
          wrapper.classList.add('recording');
          wave.classList.add('active');

          recordSeconds = 0;
          document.getElementById('status').innerText = 'Listening... (00:00)';
          recordTimerInterval = setInterval(() => {
            recordSeconds++;
            const mins = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
            const secs = String(recordSeconds % 60).padStart(2, '0');
            document.getElementById('status').innerText = `Listening... (${mins}:${secs})`;
          }, 1000);

        } catch (err) {
          alert('Microphone permission required.');
        }
      } else {
        clearInterval(recordTimerInterval);
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(t => t.stop());
        isRecording = false;

        // State 2: TRANSLATING VISUALS
        const btn = document.getElementById('record-btn');
        const wrapper = document.getElementById('mic-wrapper');
        const wave = document.getElementById('waveform-container');

        btn.classList.remove('is-recording');
        wrapper.classList.remove('recording');
        wave.classList.remove('active');

        btn.classList.add('is-translating');
        btn.innerText = '⏳';
        document.getElementById('status').innerText = 'AI Translating...';
      }
    }

    async function handleRecordingComplete() {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      processAndSaveAudio(audioBlob, currentDirection);
    }

    function handleFileUpload(e) {
      const file = e.target.files[0];
      if (!file) return;

      const btn = document.getElementById('record-btn');
      btn.classList.add('is-translating');
      btn.innerText = '⏳';
      document.getElementById('status').innerText = 'Processing file...';

      processAndSaveAudio(file, currentDirection);
    }

    async function processAndSaveAudio(blobOrFile, direction) {
      const recordItem = {
        timestamp: new Date().toLocaleString(),
        direction: direction,
        audioBlob: blobOrFile,
        spoken: '',
        translation: '',
        pronunciation: '',
        audioBase64: '',
        status: navigator.onLine ? 'translating' : 'saved_offline'
      };

      // If offline, save directly to IndexedDB
      if (!navigator.onLine) {
        saveToDB(recordItem);
        appendConvoCard(recordItem, true);
        resetMicButton();
        document.getElementById('status').innerText = 'Saved offline in Library';
        return;
      }

      // Online: send to server
      const formData = new FormData();
      formData.append('audio', blobOrFile, 'speech.webm');
      formData.append('direction', direction);

      try {
        const response = await fetch('/translate-audio', { method: 'POST', body: formData });
        const data = await response.json();
        
        if (data.error) {
          recordItem.status = 'error';
          recordItem.translation = data.error;
        } else {
          recordItem.spoken = data.spoken || '';
          recordItem.translation = data.translation || '';
          recordItem.pronunciation = data.pronunciation || '';
          recordItem.audioBase64 = data.audio_base64 || '';
          recordItem.status = 'completed';

          if (autoSpeak && recordItem.audioBase64) {
            playBase64Audio(recordItem.audioBase64);
          }
        }
      } catch (err) {
        recordItem.status = 'saved_offline';
        recordItem.translation = 'Network dropped. Saved locally in Library.';
      }

      saveToDB(recordItem);
      appendConvoCard(recordItem);
      resetMicButton();
      document.getElementById('status').innerText = 'Tap mic to speak';
    }

    function resetMicButton() {
      const btn = document.getElementById('record-btn');
      btn.classList.remove('is-translating');
      btn.classList.remove('is-recording');
      btn.innerText = '🎤';
    }

    // Direct Text Translation
    async function submitTextTranslation() {
      const text = document.getElementById('text-input').value.trim();
      if (!text) return;

      document.getElementById('text-result').style.display = 'none';
      const formData = new FormData();
      formData.append('text', text);
      formData.append('direction', textDirection);

      try {
        const res = await fetch('/translate-text', { method: 'POST', body: formData });
        const data = await res.json();
        document.getElementById('text-result').style.display = 'flex';
        document.getElementById('text-result-body').innerText = data.translation || 'Error translating';
        document.getElementById('text-result-phonetic').innerText = data.pronunciation ? `Tanglish: "${data.pronunciation}"` : '';
        textResultAudioBase64 = data.audio_base64 || '';
        if (autoSpeak && textResultAudioBase64) playBase64Audio(textResultAudioBase64);
      } catch (err) {
        alert('Translation failed. Check connection.');
      }
    }

    function playTextResultAudio() {
      if (textResultAudioBase64) playBase64Audio(textResultAudioBase64);
    }

    // UI DOM Helpers
    function appendConvoCard(item, isOffline = false) {
      const feed = document.getElementById('convo-feed');
      const card = document.createElement('div');
      const isTa = item.direction === 'ta_to_en';
      card.className = `msg-card ${isTa ? 'ta' : 'en'}`;
      
      card.innerHTML = `
        <div class="msg-meta">
          <span>${isTa ? 'Tamil ➔ English' : 'English ➔ Tamil'}</span>
          <span>${item.timestamp || 'Just now'}</span>
        </div>
        <div class="msg-original">${item.spoken || (isOffline ? 'Audio recorded (Saved offline)' : 'Processing...')}</div>
        <div class="msg-translated">${item.translation || ''}</div>
        ${item.pronunciation ? `<div class="msg-phonetic">Tanglish: "${item.pronunciation}"</div>` : ''}
        <div class="msg-tools">
          <button class="tool-btn" onclick="playOriginalItemAudio(${item.id})">▶ Original</button>
          ${item.audioBase64 ? `<button class="tool-btn" onclick="playBase64Audio('${item.audioBase64}')">🔊 Speak</button>` : ''}
        </div>
      `;
      feed.appendChild(card);
      feed.scrollTop = feed.scrollHeight;
    }

    // IndexedDB Operations
    function saveToDB(item) {
      if (!db) return;
      const tx = db.transaction('recordings', 'readwrite');
      tx.objectStore('recordings').add(item);
      tx.oncomplete = () => {
        updateLibraryBadge();
      };
    }

    function updateLibraryBadge() {
      if (!db) return;
      const tx = db.transaction('recordings', 'readonly');
      const countReq = tx.objectStore('recordings').count();
      countReq.onsuccess = () => {
        document.getElementById('lib-count').innerText = countReq.result;
      };
    }

    function renderLibrary() {
      if (!db) return;
      const list = document.getElementById('library-list');
      list.innerHTML = '';
      const tx = db.transaction('recordings', 'readonly');
      const store = tx.objectStore('recordings');
      store.openCursor(null, 'prev').onsuccess = e => {
        const cursor = e.target.result;
        if (cursor) {
          const row = cursor.value;
          const id = cursor.key;
          const div = document.createElement('div');
          div.className = 'library-item';
          div.innerHTML = `
            <div class="library-header">
              <span>${row.direction === 'ta_to_en' ? 'Tamil ➔ English' : 'English ➔ Tamil'}</span>
              <span>${row.timestamp}</span>
            </div>
            <div class="library-title">${row.spoken || 'Offline Audio Recording'}</div>
            <div style="font-size: 0.9rem; color: #60a5fa;">${row.translation || 'Pending Translation'}</div>
            <div class="msg-tools">
              <button class="tool-btn" onclick="playOriginalItemAudio(${id})">▶ Play Audio</button>
              ${row.status === 'saved_offline' ? `<button class="tool-btn" style="background:#2563eb;" onclick="translateOfflineRecord(${id})">⚡ Translate Now</button>` : ''}
              <button class="tool-btn" style="background:#7f1d1d;" onclick="deleteRecord(${id})">🗑 Delete</button>
            </div>
          `;
          list.appendChild(div);
          cursor.continue();
        }
      };
      updateLibraryBadge();
    }

    function playOriginalItemAudio(id) {
      if (!db) return;
      const tx = db.transaction('recordings', 'readonly');
      tx.objectStore('recordings').get(id).onsuccess = e => {
        const item = e.target.result;
        if (item && item.audioBlob) playBlobAudio(item.audioBlob);
      };
    }

    async function translateOfflineRecord(id) {
      if (!navigator.onLine) {
        alert('Still offline. Connect to mobile data or Wi-Fi to translate.');
        return;
      }
      const tx = db.transaction('recordings', 'readwrite');
      const store = tx.objectStore('recordings');
      store.get(id).onsuccess = async e => {
        const item = e.target.result;
        const formData = new FormData();
        formData.append('audio', item.audioBlob, 'speech.webm');
        formData.append('direction', item.direction);

        try {
          const response = await fetch('/translate-audio', { method: 'POST', body: formData });
          const data = await response.json();
          item.spoken = data.spoken || '';
          item.translation = data.translation || '';
          item.pronunciation = data.pronunciation || '';
          item.audioBase64 = data.audio_base64 || '';
          item.status = 'completed';

          const updateTx = db.transaction('recordings', 'readwrite');
          updateTx.objectStore('recordings').put(item);
          updateTx.oncomplete = () => renderLibrary();
        } catch (err) {
          alert('Failed to connect to translation server.');
        }
      };
    }

    function deleteRecord(id) {
      if (!db) return;
      const tx = db.transaction('recordings', 'readwrite');
      tx.objectStore('recordings').delete(id);
      tx.oncomplete = () => renderLibrary();
    }

    function clearAllLibrary() {
      if (!confirm('Permanently delete all saved local recordings?')) return;
      const tx = db.transaction('recordings', 'readwrite');
      tx.objectStore('recordings').clear();
      tx.oncomplete = () => renderLibrary();
    }

    function exportRecordsTxt() {
      if (!db) return;
      const tx = db.transaction('recordings', 'readonly');
      let textContent = `FIELDWORK TRANSLATION EXPORT\\nGenerated: ${new Date().toLocaleString()}\\n====================================\\n\\n`;
      tx.objectStore('recordings').openCursor().onsuccess = e => {
        const cursor = e.target.result;
        if (cursor) {
          const item = cursor.value;
          textContent += `[${item.timestamp}] (${item.direction === 'ta_to_en' ? 'Tamil -> English' : 'English -> Tamil'})\\n`;
          textContent += `Spoken: ${item.spoken || 'N/A'}\\n`;
          textContent += `Translation: ${item.translation || 'N/A'}\\n`;
          if (item.pronunciation) textContent += `Tanglish: ${item.pronunciation}\\n`;
          textContent += `------------------------------------\\n\\n`;
          cursor.continue();
        } else {
          const blob = new Blob([textContent], { type: 'text/plain' });
          const a = document.createElement('a');
          a.href = URL.createObjectURL(blob);
          a.download = `fieldwork_translations_${Date.now()}.txt`;
          a.click();
        }
      };
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
            "You are an expert field assistant translating spoken Tamil to English for community researchers in Tamil Nadu. "
            "1. Listen carefully to the Tamil audio. "
            "2. Accurately capture colloquial and conversational Tamil expressions. "
            "3. Translate into clear, practical, meaning-focused English (avoid rigid word-for-word translation). "
            "4. If the speech is indistinct or background noise only, state '[Indistinct speech]' without making up facts. "
            "Respond strictly with a JSON object containing keys: 'spoken', 'translation', 'pronunciation' (leave pronunciation as empty string)."
        )
        target_lang = "en"
    else:
        prompt = (
            "You are an expert field assistant translating English into spoken Tamil for community researchers in Tamil Nadu. "
            "1. Translate English into natural, polite, everyday spoken Tamil (colloquial conversational register). "
            "2. Provide an easy-to-read Tanglish pronunciation guide in English Latin letters. "
            "3. If speech is indistinct, state '[Indistinct speech]'. "
            "Respond strictly with a JSON object containing keys: 'spoken', 'translation', 'pronunciation'."
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
        return {"error": f"Audio processing error: {str(e)}"}

@app.post("/translate-text")
async def translate_text(text: str = Form(...), direction: str = Form(...)):
    if direction == "ta_to_en":
        prompt = (
            "Translate this Tamil text into clear, simple conversational English focusing on the core meaning: "
            f"'{text}'. Respond strictly in JSON with keys: 'spoken', 'translation', 'pronunciation'."
        )
        target_lang = "en"
    else:
        prompt = (
            "Translate this English text into natural, polite spoken Tamil script and provide a Tanglish phonetic pronunciation: "
            f"'{text}'. Respond strictly in JSON with keys: 'spoken', 'translation', 'pronunciation'."
        )
        target_lang = "ta"

    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        data = json.loads(response.text)
        translation_text = data.get("translation", "")
        data["audio_base64"] = generate_fast_audio(translation_text, target_lang) if translation_text else ""
        return data
    except Exception as e:
        return {"error": f"Text processing error: {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
