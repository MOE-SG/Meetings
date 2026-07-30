// Dedicated worker for real speaker diarization (segmentation + speaker
// embedding + clustering), using k2-fsa/sherpa-onnx compiled to WebAssembly.
//
// Models baked into sherpa-onnx-wasm-main-speaker-diarization.data at build
// time (this is how sherpa-onnx's WASM build works — models are preloaded
// into the virtual filesystem when the .wasm/.data pair is compiled, not
// fetched dynamically at runtime like the Whisper model is):
//   segmentation: pyannote/segmentation-3.0
//   embedding:    3D-Speaker ERes2Net-base (speaker embedding / voiceprint)
//
// This is a classic (non-module) worker on purpose — the glue code below
// uses `importScripts` and a global `Module` object, Emscripten's standard
// browser output pattern, not ES modules.

var Module = {
  // Force every secondary file (.wasm, .data) to resolve relative to this
  // worker script's own location, regardless of how the glue code would
  // otherwise guess it.
  locateFile: function (path) {
    return './' + path;
  },
};

var diarizationReady = false;
var sd = null;
var reconstructedDataUrl = null;
var reconstructedDataPromise = null;
var cacheKey = 'speaker-diarization-data-v1';

var remoteDataParts = [
  'https://github.com/user-attachments/files/30528688/sherpa-onnx-wasm-main-speaker-diarization.data.part1.zip',
  'https://github.com/user-attachments/files/30528691/sherpa-onnx-wasm-main-speaker-diarization.data.part2.zip',
  'https://github.com/user-attachments/files/30528695/sherpa-onnx-wasm-main-speaker-diarization.data.part3.zip'
];

Module.onRuntimeInitialized = function () {
  diarizationReady = true;
  self.postMessage({ type: 'ready', payload: { sampleRate: getSampleRateSafely() } });
};

function getSampleRateSafely(){
  try {
    // Instantiating early just to read the expected sample rate; cheap
    // relative to the model already being fully loaded at this point.
    if (!sd) sd = createOfflineSpeakerDiarization(Module);
    return sd.sampleRate;
  } catch (err) {
    return 16000; // both underlying models are trained at 16kHz
  }
}

async function ensureReconstructedDataUrl() {
  if (reconstructedDataUrl) return reconstructedDataUrl;
  if (!reconstructedDataPromise) {
    reconstructedDataPromise = (async function () {
      try {
        const db = await openModelCacheDb();
        const cached = await getCachedModelData(db);
        if (cached) {
          reconstructedDataUrl = URL.createObjectURL(new Blob([cached], { type: 'application/octet-stream' }));
          return reconstructedDataUrl;
        }

        const buffers = [];
        for (let i = 0; i < remoteDataParts.length; i++) {
          const url = remoteDataParts[i];
          self.postMessage({ type: 'progress', payload: { stage: 'downloading', part: i + 1, total: remoteDataParts.length, percent: Math.round(((i + 1) / remoteDataParts.length) * 100) } });
          const response = await fetch(url, { cache: 'reload' });
          if (!response.ok) {
            throw new Error('Failed to download speaker-model asset ' + response.status);
          }
          buffers.push(new Uint8Array(await response.arrayBuffer()));
        }

        const totalLength = buffers.reduce((sum, chunk) => sum + chunk.byteLength, 0);
        const merged = new Uint8Array(totalLength);
        let offset = 0;
        for (const chunk of buffers) {
          merged.set(chunk, offset);
          offset += chunk.byteLength;
        }

        self.postMessage({ type: 'progress', payload: { stage: 'caching', part: remoteDataParts.length, total: remoteDataParts.length, percent: 100 } });
        await storeCachedModelData(db, merged);

        const blob = new Blob([merged], { type: 'application/octet-stream' });
        reconstructedDataUrl = URL.createObjectURL(blob);
        return reconstructedDataUrl;
      } catch (err) {
        throw err;
      }
    })();
  }
  return reconstructedDataPromise;
}

function openModelCacheDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('meeting-scope-model-cache', 1);
    request.onupgradeneeded = function () {
      const db = request.result;
      if (!db.objectStoreNames.contains('models')) {
        db.createObjectStore('models');
      }
    };
    request.onsuccess = function () { resolve(request.result); };
    request.onerror = function () { reject(request.error || new Error('Unable to open model cache database')); };
  });
}

function getCachedModelData(db) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('models', 'readonly');
    const store = tx.objectStore('models');
    const req = store.get(cacheKey);
    req.onsuccess = function () { resolve(req.result || null); };
    req.onerror = function () { reject(req.error || new Error('Unable to read cached model data')); };
  });
}

function storeCachedModelData(db, data) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('models', 'readwrite');
    const store = tx.objectStore('models');
    const req = store.put(data, cacheKey);
    req.onsuccess = function () { resolve(); };
    req.onerror = function () { reject(req.error || new Error('Unable to store cached model data')); };
  });
}

async function bootstrapDiarizationModule() {
  try {
    const dataUrl = await ensureReconstructedDataUrl();
    Module.locateFile = function (path) {
      if (path && String(path).endsWith('.data')) return dataUrl;
      return './' + path;
    };

    importScripts('./sherpa-onnx-speaker-diarization.js');
    importScripts('./sherpa-onnx-wasm-main-speaker-diarization.js');
  } catch (err) {
    self.postMessage({ type: 'error', payload: String((err && err.message) || err) });
  }
}

self.onmessage = function (e) {
  var msg = e.data || {};
  if (msg.type !== 'diarize') return;

  if (!diarizationReady) {
    self.postMessage({ type: 'error', payload: 'Speaker model is still loading — try again in a moment.' });
    return;
  }

  try {
    if (!sd) sd = createOfflineSpeakerDiarization(Module);

    var numClusters = (msg.payload && msg.payload.numClusters) || -1;
    var threshold = (msg.payload && msg.payload.threshold) || 0.5;
    sd.setConfig({ clustering: { numClusters: numClusters, threshold: threshold } });

    var segments = sd.process(msg.payload.audio);
    self.postMessage({ type: 'result', payload: { segments: segments } });
  } catch (err) {
    self.postMessage({ type: 'error', payload: String((err && err.message) || err) });
  }
};

bootstrapDiarizationModule();
