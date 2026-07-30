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

importScripts('./sherpa-onnx-speaker-diarization.js');
importScripts('./sherpa-onnx-wasm-main-speaker-diarization.js');

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
