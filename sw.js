// Optional companion service worker for SOE Meeting.
// Only needed if you deploy index.html to a real host (GitHub Pages, etc).
// It has nothing to do with the Whisper model weights — those are already
// cached separately by the browser's Cache Storage via transformers.js.
// This just lets the app shell (index.html itself) load instantly and
// work offline after the first visit.

const CACHE_NAME = 'meeting-scope-shell-v3';
const SHELL_FILES = ['./', './index.html', './manifest.json', './icon-192.png', './icon-512.png'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_FILES))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  // Only handle same-origin GET requests for the shell itself.
  // Everything else (CDN libraries, model weights) passes straight through
  // to the network / the browser's own Cache Storage handling.
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || url.origin !== self.location.origin) return;

  // Network-first: while this app is under active development, a stale
  // cached shell is worse than a slightly slower repeat load. Falls back
  // to cache only if the network is actually unavailable (offline).
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        if (response.ok) {
          const copy = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy));
        }
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
