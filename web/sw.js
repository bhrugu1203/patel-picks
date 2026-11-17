// Patel Picks offline support. The board (slate.json, stats.json) and the page always come from
// the network first, so the app is never staler than the website; the saved copy is only used
// when the phone is offline, so the last board still opens.
const VERSION = "pp-v1";
const SHELL = ["./", "pwa.js", "pwa.css", "manifest.webmanifest", "icons/icon-192.png", "icons/icon-512.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

async function networkFirst(req, key) {
  const cache = await caches.open(VERSION);
  try {
    const res = await fetch(req);
    if (res.ok) cache.put(key || req, res.clone());
    return res;
  } catch (err) {
    const hit = await cache.match(key || req, { ignoreSearch: true });
    if (hit) return hit;
    throw err;
  }
}

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== location.origin) return; // fonts, headshots: the browser's own cache handles them
  // every page load is the same single-page app, so keep one copy of it under "./"
  if (req.mode === "navigate") return e.respondWith(networkFirst(req, new URL("./", self.registration.scope).href));
  e.respondWith(networkFirst(req));
});
