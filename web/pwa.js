// Makes Patel Picks installable as a phone app: offline support, an "App" button where the
// browser can install it (Chrome, Edge, Android), and a one-time how-to on iPhone Safari.
(function () {
  if ("serviceWorker" in navigator) {
    addEventListener("load", () => navigator.serviceWorker.register("sw.js").catch(() => {}));
  }
  const store = {
    get: (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} },
  };
  const installed = matchMedia("(display-mode: standalone)").matches || navigator.standalone === true;
  if (installed) return;

  let deferred = null;
  function button() {
    let b = document.getElementById("getapp");
    if (b) return b;
    const spot = document.querySelector(".hright");
    if (!spot) return null;
    b = document.createElement("button");
    b.id = "getapp"; b.className = "getapp"; b.type = "button";
    b.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg><span>Get the app</span>';
    b.setAttribute("aria-label", "Install the Patel Picks app");
    b.addEventListener("click", async () => {
      if (!deferred) return;
      deferred.prompt();
      await deferred.userChoice.catch(() => null);
      deferred = null; b.classList.remove("show");
    });
    spot.insertBefore(b, document.getElementById("theme"));
    return b;
  }
  addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault(); deferred = e;
    const b = button(); if (b) b.classList.add("show");
  });
  addEventListener("appinstalled", () => { const b = document.getElementById("getapp"); if (b) b.classList.remove("show"); });

  // iPhone/iPad Safari has no install prompt: say how, once.
  const ua = navigator.userAgent;
  const iOS = /iPhone|iPad|iPod/.test(ua) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  const safari = /Safari/.test(ua) && !/CriOS|FxiOS|EdgiOS|OPiOS/.test(ua);
  if (iOS && safari && !store.get("pp-apptip")) {
    addEventListener("load", () => setTimeout(() => {
      const tip = document.createElement("div");
      tip.className = "apptip"; tip.setAttribute("role", "note");
      tip.innerHTML = 'Get Patel Picks as an app: tap <b>Share</b>, then <b>Add to Home Screen</b>.<button type="button" aria-label="Close">×</button>';
      tip.querySelector("button").addEventListener("click", () => { tip.remove(); store.set("pp-apptip", "1"); });
      document.body.appendChild(tip);
    }, 4000));
  }
})();
