// The motion layer for the whole page. app.py injects this once into the Streamlit page.
// It counts the big figures up, tweens the live scores in the header, draws the reading
// progress bar, moves Pinky and the Brain with the pointer, opens a page from the plan
// menu, and fades one page out and the next one in.
(() => {
  if (window.__pcFx) return;
  window.__pcFx = true;
  const d = document;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ease = t => 1 - Math.pow(1 - t, 3);

  // ---------- numbers: count the open figures up, tween the header scores when they change ----------
  const NUM = /^(-|−)?([\d,]*\.?\d+)(%?)$/;
  function parse(txt) {
    const m = NUM.exec((txt || '').trim());
    if (!m) return null;
    const raw = m[2], dec = raw.includes('.') ? raw.split('.')[1].length : 0;
    return {v: parseFloat(raw.replace(/,/g, '')) * (m[1] ? -1 : 1), dec, comma: raw.includes(','), suf: m[3], neg: m[1] || '-'};
  }
  function fmt(v, p) {
    const s = Math.abs(v).toLocaleString('en-US', {minimumFractionDigits: p.dec, maximumFractionDigits: p.dec, useGrouping: p.comma});
    return (v < 0 && Math.abs(v) >= Math.pow(10, -p.dec) / 2 ? p.neg : '') + s + p.suf;
  }
  function tween(node, from, to, p, dur) {
    const lead = node.nodeValue.match(/^\s*/)[0], tail = node.nodeValue.match(/\s*$/)[0];
    const target = node.nodeValue;
    let last = null;
    node.__pcTw = true;
    const t0 = performance.now();
    (function step(now) {
      // if Streamlit writes a new value in the middle, stop and let the next scan take it from there
      if (last !== null && node.nodeValue !== last) { node.__pcTw = false; return; }
      const k = Math.min(1, (now - t0) / dur);
      last = k < 1 ? lead + fmt(from + (to - from) * ease(k), p) + tail : target;
      node.nodeValue = last;
      if (k < 1) requestAnimationFrame(step); else node.__pcTw = false;
    })(t0);
  }
  function flash(el) {
    el.classList.remove('pc-flash');
    void el.offsetWidth;
    el.classList.add('pc-flash');
  }
  const liveMemory = {};
  function numbers() {
    d.querySelectorAll('.pc-figs .val, .pc-live b').forEach(el => {
      const node = [...el.childNodes].find(n => n.nodeType === 3 && n.nodeValue.trim());
      if (!node || node.__pcTw) return;
      const txt = node.nodeValue;
      if (node.__pcSeen === txt) return;
      const before = node.__pcSeen;
      node.__pcSeen = txt;
      const p = parse(txt);
      const live = el.matches('.pc-live b');
      let from = null;
      if (live) {
        const slot = 'live' + [...el.closest('.pc-live').querySelectorAll('b')].indexOf(el);
        const old = before !== undefined ? before : liveMemory[slot];
        liveMemory[slot] = txt;
        if (old === undefined || old === txt) return;
        const q = parse(old);
        if (!q) return;
        from = q.v;
        flash(el);
      } else {
        // a new figure counts up from zero, a changed one moves from its old value
        const q = before !== undefined ? parse(before) : null;
        from = q ? q.v : 0;
      }
      if (!p || reduce || from === p.v) return;
      tween(node, from, p.v, p, live ? 700 : 1000);
    });
  }

  // ---------- reading progress along the bottom of the sticky header ----------
  function scroller() {
    const m = d.querySelector('[data-testid="stMain"]');
    return m && m.scrollHeight > m.clientHeight + 2 ? m : d.scrollingElement;
  }
  function progress() {
    const tb = d.querySelector('.st-key-topbar');
    if (!tb) return;
    let bar = tb.querySelector(':scope > .pc-progress');
    if (!bar) { bar = d.createElement('div'); bar.className = 'pc-progress'; tb.appendChild(bar); }
    const s = scroller(), room = s.scrollHeight - s.clientHeight;
    bar.style.transform = `scaleX(${room > 0 ? Math.min(1, s.scrollTop / room) : 0})`;
  }
  let progQueued = false;
  d.addEventListener('scroll', () => {
    if (progQueued) return;
    progQueued = true;
    requestAnimationFrame(() => { progQueued = false; progress(); });
  }, true);

  // ---------- the heroes: Pinky and the Brain lean toward the pointer ----------
  let hero = null;
  d.addEventListener('pointermove', e => {
    const h = e.target.closest ? e.target.closest('.tt-hero, .pb-hero') : null;
    if (hero && h !== hero) { hero.style.setProperty('--mx', 0); hero.style.setProperty('--my', 0); }
    hero = h;
    if (!h || reduce) return;
    const r = h.getBoundingClientRect();
    h.style.setProperty('--mx', ((e.clientX - r.left) / r.width - .5).toFixed(3));
    h.style.setProperty('--my', ((e.clientY - r.top) / r.height - .5).toFixed(3));
  }, {passive: true});

  // ---------- page changes: fade the old page out, bring the new one in block by block ----------
  let leaveTimer = null;
  function leave() {
    if (reduce) return;
    d.documentElement.classList.add('pc-leaving');
    clearTimeout(leaveTimer);
    leaveTimer = setTimeout(() => d.documentElement.classList.remove('pc-leaving'), 2200);
  }
  function blocks(W) {
    // the blocks of the page that are on screen; a tall container is entered by its children
    const out = [];
    const visit = (el, depth) => {
      const r = el.getBoundingClientRect();
      if (r.height < 4 || r.top > innerHeight + 40) return;
      const inner = el.querySelector(':scope > [data-testid="stVerticalBlock"]');
      if (depth < 2 && r.height > innerHeight * 0.9 && inner) { [...inner.children].forEach(c => visit(c, depth + 1)); return; }
      out.push(el);
    };
    for (let n = W.nextElementSibling; n; n = n.nextElementSibling) visit(n, 0);
    return out;
  }
  window.__pcEnter = () => {
    const W = d.querySelector('[data-testid="stLayoutWrapper"]:has(> .st-key-topbar)');
    if (!W) { d.documentElement.classList.remove('pc-leaving'); return; }
    let quiet = null, done = false;
    const t0 = performance.now();
    const go = () => {
      if (done) return;
      done = true;
      mo.disconnect();
      clearTimeout(leaveTimer);
      if (!reduce) {
        blocks(W).slice(0, 10).forEach((el, i) => {
          // a block holding something pinned to the screen only fades, so the pin does not move
          const pinned = el.querySelector('.pb-corner, .pb-peek');
          const frames = pinned ? [{opacity: 0}, {opacity: 1}] : [{opacity: 0, transform: 'translateY(18px)'}, {opacity: 1, transform: 'none'}];
          el.animate(frames, {duration: 560, delay: i * 70, easing: 'cubic-bezier(.2,.8,.2,1)', fill: 'backwards'});
        });
      }
      d.documentElement.classList.remove('pc-leaving');
      progress();
    };
    // wait until the new page stops arriving (a short quiet moment), but never longer than 1.4 s
    const arm = () => { clearTimeout(quiet); quiet = setTimeout(go, 140); };
    const mo = new MutationObserver(() => (performance.now() - t0 > 1400 ? go() : arm()));
    mo.observe(W.parentElement, {childList: true, subtree: true});
    arm();
    setTimeout(go, 1400);
  };

  // ---------- clicks: the plan rows open their page; menu and pager clicks start the fade ----------
  function openRow(row) {
    const rows = [...row.parentElement.querySelectorAll(':scope > .row')];
    const b = d.querySelector(`.st-key-plango_${rows.indexOf(row)} button`);
    if (b) { leave(); b.click(); }
  }
  d.addEventListener('click', e => {
    const t = e.target.closest ? e.target : e.target.parentElement;
    if (!t) return;
    const row = t.closest('.pc-menu .row');
    if (row) { openRow(row); return; }
    const nav = t.closest('.st-key-nav button');
    if (nav && nav.getAttribute('aria-checked') !== 'true') { leave(); return; }
    if (t.closest('.st-key-pager button')) leave();
  }, true);
  d.addEventListener('keydown', e => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const row = e.target.closest && e.target.closest('.pc-menu .row');
    if (row) { e.preventDefault(); openRow(row); }
  });
  function rowsReady() {
    d.querySelectorAll('.pc-menu .row:not([tabindex])').forEach(r => {
      r.setAttribute('tabindex', '0');
      r.setAttribute('role', 'link');
    });
    d.querySelectorAll('.st-key-plango button:not([tabindex])').forEach(b => b.setAttribute('tabindex', '-1'));
  }

  // ---------- watch the page: run the scans whenever Streamlit changes something ----------
  let queued = false;
  const scan = () => { queued = false; numbers(); rowsReady(); progress(); };
  new MutationObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(scan);
  }).observe(d.body, {childList: true, subtree: true, characterData: true});
  scan();
})();
