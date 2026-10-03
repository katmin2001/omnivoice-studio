/* OmniVoice Studio — frontend. Vanilla JS, no build step.
   Single mode: one text → one take.  Line mode: a project of lines, each generated / retaken on its own, exported with timings. */
(() => {
  'use strict';

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  // ------------------------------------------------------------------ api
  async function api(url, opts = {}) {
    const res = await fetch(url, opts);
    if (!res.ok) {
      let msg = `${res.status} ${res.statusText}`;
      try { const j = await res.json(); if (j.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch {}
      throw new Error(msg);
    }
    return res.json();
  }
  const postJSON = (url, body, method = 'POST') =>
    api(url, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

  // ------------------------------------------------------------------ state
  const DEFAULTS = {
    text: '', mode: 'clone', voiceId: null, styleId: null, language: 'vi', model: 'base', speed: 1, speedMode: 'auto',
    num_step: 32, seed: '', guidance_scale: 2, denoise: true, design: {}, designExtra: '', scriptMode: 'single',
    takes: 1, qaAuto: true, readNumbers: true, showSpoken: false, projectId: null, gap: 0.3,
  };
  const state = Object.assign({}, DEFAULTS, loadState());
  function loadState() { try { return JSON.parse(localStorage.getItem('ovs-state') || '{}'); } catch { return {}; } }
  function persist() { try { localStorage.setItem('ovs-state', JSON.stringify(state)); } catch {} }

  let status = null;
  let voices = [];
  let history = [];
  let current = null;        // output meta shown in the result panel
  let busy = false;
  let batchStop = false;
  let peaks = null;          // waveform peaks of `current`
  let audioCtx = null;
  let proj = null;           // current line-mode project
  let projects = [];

  const WORDS_PER_SECOND = 3.6;   // Vietnamese speaking pace, for estimates only

  const STAGES = {
    loading: 'Đang nạp model vào GPU…',
    loading_asr: 'Đang nạp Whisper (lần đầu tải ~1,6 GB)…',
    transcribing: 'Whisper đang nghe đoạn mẫu…',
    encoding_ref: 'Đang mã hoá giọng mẫu…',
    generating: 'Đang tạo giọng…',
    qa: 'Whisper đang soát lỗi đọc…',
    aligning: 'Đang căn mốc thời gian từng chữ…',
    stretching: 'Đang kéo giãn tốc độ…',
    exporting: 'Đang xuất dự án…',
  };

  const TAGS = [
    ['[laughter]', 'cười'], ['[sigh]', 'thở dài'], ['[surprise-ah]', 'á!'], ['[surprise-oh]', 'ồ!'],
    ['[surprise-wa]', 'oa!'], ['[question-ah]', 'hả?'], ['[question-ei]', 'ơ?'],
    ['[dissatisfaction-hnn]', 'hừm'], ['[confirmation-en]', 'ừm'],
  ];

  const DESIGN = [
    { key: 'gender', label: 'Giới tính', opts: [['male', 'Nam'], ['female', 'Nữ']] },
    { key: 'age', label: 'Tuổi', opts: [['child', 'Trẻ em'], ['teenager', 'Thiếu niên'], ['young adult', 'Thanh niên'], ['middle-aged', 'Trung niên'], ['elderly', 'Cao tuổi']] },
    { key: 'pitch', label: 'Cao độ', opts: [['very low pitch', 'Rất trầm'], ['low pitch', 'Trầm'], ['moderate pitch', 'Vừa'], ['high pitch', 'Cao'], ['very high pitch', 'Rất cao']] },
    { key: 'style', label: 'Phong cách', opts: [['whisper', 'Thì thầm']] },
    { key: 'accent', label: 'Giọng Anh (chỉ khi đọc tiếng Anh)', opts: [['american accent', 'Mỹ'], ['british accent', 'Anh'], ['australian accent', 'Úc'], ['indian accent', 'Ấn'], ['japanese accent', 'Nhật'], ['korean accent', 'Hàn']] },
  ];

  // ------------------------------------------------------------------ helpers
  const fmtTime = (s) => { s = Math.max(0, s || 0); return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`; };
  const wordCount = (t) => t.trim() ? t.trim().split(/\s+/).length : 0;
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const initial = (name) => (name || '?').trim().replace(/^[^\p{L}\p{N}]+/u, '').charAt(0).toUpperCase() || '?';
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  const selectedVoice = () => voices.find((v) => v.id === state.voiceId) || null;
  const voiceStyle = (v, sid) => (v && (v.styles.find((s) => s.id === sid) || v.styles[0])) || null;

  function toast(msg, type = '', sfx) {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    const label = sfx ?? ({ ok: 'OK!', warn: 'Ơ…', err: 'ẶC!' }[type] || '');
    el.innerHTML = `${label ? `<span class="sfx">${esc(label)}</span>` : ''}<span>${esc(msg)}</span>`;
    $('#toasts').appendChild(el);
    setTimeout(() => { el.classList.add('is-out'); setTimeout(() => el.remove(), 220); }, type === 'err' ? 7000 : 3600);
  }

  function confirmDialog(message, yesLabel = 'XOÁ') {
    const d = $('#confirmModal');
    $('#confirmText').textContent = message;
    $('#confirmYes').textContent = yesLabel;
    return new Promise((resolve) => {
      const finish = (v) => { d.removeEventListener('close', onClose); $('#confirmYes').onclick = $('#confirmNo').onclick = null; if (d.open) d.close(); resolve(v); };
      const onClose = () => finish(false);
      $('#confirmYes').onclick = () => finish(true);
      $('#confirmNo').onclick = () => finish(false);
      d.addEventListener('close', onClose);
      d.showModal();
    });
  }

  // starburst clip-path for stickers
  function burst(n = 12, outer = 50, inner = 38) {
    const pts = [];
    for (let i = 0; i < n * 2; i++) {
      const a = (Math.PI * i) / n - Math.PI / 2;
      const r = i % 2 ? inner : outer;
      pts.push(`${(50 + r * Math.cos(a)).toFixed(1)}% ${(50 + r * Math.sin(a)).toFixed(1)}%`);
    }
    return `polygon(${pts.join(',')})`;
  }
  $$('.burst').forEach((el) => { el.style.clipPath = burst(el.classList.contains('pop') ? 14 : 12); });

  // ------------------------------------------------------------------ status
  async function pollStatus() {
    try {
      status = await api('/api/status');
      renderStatus();
      if (!$('#language').options.length) fillSelects();
    } catch {
      status = null;
      $('#statusDot').className = 'dot is-err';
      $('#statusText').textContent = 'Mất kết nối máy chủ — server.py còn chạy không?';
    }
    setTimeout(pollStatus, status && status.busy ? 700 : 2500);
  }

  function stageLabel() {
    if (!status || !status.stage) return '';
    return `${STAGES[status.stage] || 'Đang bận…'}${status.stage_detail ? ' ' + status.stage_detail : ''}`;
  }

  function renderStatus() {
    const dot = $('#statusDot'), txt = $('#statusText');
    const gpu = status.gpu ? status.gpu.replace('NVIDIA GeForce ', '') : 'CPU (chậm)';
    const modelLabel = status.model ? status.models[status.model].label : 'chưa nạp model';
    if (status.busy) {
      dot.className = 'dot is-busy';
      txt.textContent = `${stageLabel()} ${status.stage_seconds ? status.stage_seconds.toFixed(0) + 's' : ''}`;
    } else if (status.error) {
      dot.className = 'dot is-err';
      txt.textContent = `Lỗi nạp model: ${status.error}`;
    } else {
      dot.className = status.model ? 'dot is-ok' : 'dot';
      txt.textContent = `${gpu} · ${modelLabel}${status.model ? ' · sẵn sàng' : ''}${status.asr_loaded ? ' · Whisper' : ''}`;
    }
    if (status.vram_total_gb) {
      $('#vramFill').style.width = `${Math.min(100, (status.vram_used_gb / status.vram_total_gb) * 100)}%`;
      $('#vramText').textContent = `VRAM ${status.vram_used_gb.toFixed(1)} / ${status.vram_total_gb.toFixed(1)} GB`;
    } else {
      $('#vramText').textContent = 'Không có GPU';
    }
    if (busy && status.busy) $('#progressText').dataset.stage = stageLabel();
  }

  async function waitForModel() {
    // the server holds its lock while (re)loading; wait instead of bouncing off a 409
    const t0 = Date.now();
    while (Date.now() - t0 < 180000) {
      try { status = await api('/api/status'); } catch { break; }
      if (!status.busy || !['loading', 'loading_asr'].includes(status.stage)) return;
      $('#progressText').textContent = `${stageLabel()} ${status.stage_seconds.toFixed(0)}s`;
      await new Promise((r) => setTimeout(r, 1000));
    }
  }

  function fillSelects() {
    const lang = $('#language');
    lang.innerHTML = status.languages.map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join('');
    lang.value = state.language in Object.fromEntries(status.languages) ? state.language : 'vi';
    const model = $('#model');
    model.innerHTML = Object.entries(status.models).map(([k, m]) => `<option value="${k}">${esc(m.label)}</option>`).join('');
    model.value = state.model in status.models ? state.model : 'base';
    $('#modelNote').textContent = status.models[model.value].note;
  }

  // ------------------------------------------------------------------ script panel
  const textEl = $('#text');
  textEl.value = state.text || '';

  function updateCounter() {
    const w = wordCount(textEl.value);
    $('#wordCount').textContent = w;
    $('#estDur').textContent = Math.round(w / WORDS_PER_SECOND / (+state.speed || 1));
    $('#longHint').hidden = !(w > 80 && state.scriptMode === 'single');
  }

  const previewSpoken = debounce(async () => {
    if (!state.showSpoken) return;
    const text = textEl.value.trim();
    const box = $('#spoken');
    if (!text) { box.textContent = '(trống)'; return; }
    try {
      const { spoken } = await postJSON('/api/pronunciation/preview', { text, read_numbers: state.readNumbers });
      const orig = new Set(text.toLowerCase().replace(/\*+/g, '').split(/\s+/));
      box.innerHTML = spoken.split(/(\s+)/).map((tok) => /\s/.test(tok) || orig.has(tok.toLowerCase()) ? esc(tok) : `<b>${esc(tok)}</b>`).join('');
    } catch (e) { box.textContent = e.message; }
  }, 400);

  textEl.addEventListener('input', () => {
    state.text = textEl.value; updateCounter(); persist(); previewSpoken();
    if (state.scriptMode === 'lines') syncLinesDebounced();
  });
  $('#showSpoken').checked = !!state.showSpoken;
  $('#spoken').hidden = !state.showSpoken;
  $('#showSpoken').addEventListener('change', (e) => { state.showSpoken = e.target.checked; $('#spoken').hidden = !state.showSpoken; persist(); previewSpoken(); });

  $('#tags').innerHTML = TAGS.map(([tag, label]) => `<button type="button" data-tag="${tag}" title="${tag}">${esc(label)}</button>`).join('');
  $('#tags').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-tag]');
    if (!b) return;
    const { selectionStart: s, selectionEnd: en, value } = textEl;
    const before = value.slice(0, s), after = value.slice(en);
    const ins = `${before && !/\s$/.test(before) ? ' ' : ''}${b.dataset.tag}${after && !/^\s/.test(after) ? ' ' : ''}`;
    textEl.value = before + ins + after;
    textEl.focus();
    textEl.setSelectionRange(s + ins.length, s + ins.length);
    textEl.dispatchEvent(new Event('input'));
  });

  // ---- script mode (single | lines)
  function setScriptMode(mode) {
    state.scriptMode = mode;
    $$('#scriptMode button').forEach((b) => b.classList.toggle('is-active', b.dataset.v === mode));
    $('#linesWrap').hidden = mode !== 'lines';
    textEl.style.minHeight = mode === 'lines' ? '160px' : '';
    updateCounter(); updateGoLabel(); persist();
    if (mode === 'lines') ensureProject().then(() => { syncLines(); renderLines(); });
  }
  $('#scriptMode').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) setScriptMode(b.dataset.v); });

  function parseLines(text) {
    const out = [];
    for (const raw of text.split(/\r?\n/)) {
      let s = raw.trim();
      if (!s || s.startsWith('#')) continue;
      let label = null;
      const m = s.match(/^(b\d{1,3})\s*[:.\-–]\s*(.*)$/i);
      if (m) { label = m[1].toLowerCase(); s = m[2]; }
      else if (/^[A-ZĐÀ-Ỹ][A-ZĐÀ-Ỹ0-9 _()-]{1,28}:\s/.test(s)) continue;   // HOOK:, THUMB TOP: … header lines of script.txt
      s = s.replace(/\[sfx:[^\]]*\]/gi, '').replace(/\*+/g, '').replace(/\s+/g, ' ').trim();
      if (!s) continue;
      out.push({ n: out.length + 1, label, text: s });
    }
    if (out.length <= 1 && wordCount(text) > 40) {  // one long paragraph → sentences
      const sentences = text.replace(/\[sfx:[^\]]*\]/gi, '').replace(/\*+/g, '').split(/(?<=[.!?…])\s+/).map((s) => s.trim()).filter(Boolean);
      return sentences.map((t, i) => ({ n: i + 1, label: null, text: t }));
    }
    return out;
  }

  // ------------------------------------------------------------------ projects (line mode)
  async function saveProjectNow() {
    if (!proj) return;
    try { await postJSON(`/api/projects/${proj.id}`, proj, 'PUT'); } catch (e) { toast(`Không lưu được dự án: ${e.message}`, 'err'); }
  }
  const saveProject = debounce(saveProjectNow, 600);

  async function loadProjects() {
    try { projects = await api('/api/projects'); } catch { projects = []; }
    const sel = $('#projectSelect');
    sel.innerHTML = projects.map((p) => `<option value="${p.id}">${esc(p.name)} (${p.done}/${p.lines})</option>`).join('') || '<option value="">(chưa có dự án)</option>';
    if (proj) sel.value = proj.id;
  }

  async function newProjectFromText() {
    const text = textEl.value;
    const first = parseLines(text)[0];
    const name = first ? first.text.split(/\s+/).slice(0, 6).join(' ') : `Dự án ${new Date().toLocaleString('vi-VN')}`;
    proj = { id: `p-${Date.now().toString(36)}`, name, created: Date.now() / 1000, text, voice_id: state.voiceId, lines: [] };
    state.projectId = proj.id; persist();
    syncLines();
    await saveProjectNow();
    await loadProjects();
  }

  async function ensureProject() {
    await loadProjects();
    if (proj) return;
    if (state.projectId && projects.some((p) => p.id === state.projectId)) {
      await openProject(state.projectId, false);
    } else if (textEl.value.trim()) {
      await newProjectFromText();
    } else if (projects[0]) {
      await openProject(projects[0].id, true);
    }
  }

  async function openProject(id, loadText) {
    try {
      proj = await api(`/api/projects/${id}`);
      state.projectId = id; persist();
      if (loadText || !textEl.value.trim()) {
        textEl.value = proj.text || proj.lines.map((l) => (l.label ? `${l.label}: ` : '') + l.text).join('\n');
        state.text = textEl.value; updateCounter();
      }
      if (proj.voice_id && voices.some((v) => v.id === proj.voice_id)) { state.voiceId = proj.voice_id; renderVoices(); renderStyles(); }
      syncLines(); renderLines();
      $('#projectSelect').value = id;
      $('#exportResult').hidden = true;
    } catch (e) { toast(e.message, 'err'); }
  }

  function syncLines() {
    if (!proj) return;
    const parsed = parseLines(textEl.value);
    const old = proj.lines || [];
    const used = new Set();
    proj.lines = parsed.map((p) => {
      let prev = old.find((o, i) => !used.has(i) && o.text === p.text && (o.label || null) === (p.label || null));
      if (!prev) prev = old.find((o, i) => !used.has(i) && o.text === p.text);
      if (prev) used.add(old.indexOf(prev));
      return { ...p, style_id: prev?.style_id ?? null, output_id: prev?.output_id ?? null, seed: prev?.seed ?? null,
               qa: prev?.qa ?? null, duration: prev?.duration ?? null, error: null };
    });
    proj.text = textEl.value;
    proj.voice_id = state.voiceId;
    saveProject();
  }
  const syncLinesDebounced = debounce(() => { syncLines(); renderLines(); }, 500);

  function lineNeedsWork(l) { return !l.output_id || (l.qa && l.qa.errors > 0) || l.error; }

  function renderLines() {
    const ol = $('#lines');
    if (!proj) { ol.innerHTML = ''; $('#linesStats').textContent = ''; return; }
    const v = selectedVoice();
    const styleOpts = v && v.styles.length > 1 ? v.styles : null;
    const done = proj.lines.filter((l) => l.output_id).length;
    const errs = proj.lines.filter((l) => l.qa && l.qa.errors > 0).length;
    const total = proj.lines.reduce((s, l) => s + (l.duration || 0), 0) + Math.max(0, done - 1) * (+state.gap || 0);
    $('#linesStats').textContent = proj.lines.length ? `${done}/${proj.lines.length} dòng · ${errs ? errs + ' dòng có lỗi · ' : ''}~${fmtTime(total)}` : 'Dán kịch bản vào ô trên — mỗi dòng một câu.';
    ol.innerHTML = proj.lines.map((l) => {
      const label = l.label || String(l.n).padStart(2, '0');
      const qa = (l.qa ? (l.qa.errors ? `<span class="pill bad">${l.qa.errors} lỗi</span>` : `<span class="pill ok">✓ đọc đúng</span>`) + (l.qa.near ? `<span class="pill near">${l.qa.near} gần đúng</span>` : '') : '') + (l.noisy ? '<span class="pill near">rè</span>' : '');
      const stateCls = l.error ? 'is-error' : l.output_id ? 'is-done' : '';
      const cur = current && current.id === l.output_id ? 'is-current' : '';
      return `<li class="line ${stateCls} ${cur}" data-n="${l.n}">
        <span class="num">${esc(label)}</span>
        <div class="body">
          <div class="ltext">${esc(l.text)}</div>
          <div class="lmeta">${l.output_id ? `<span>${l.duration}s</span><span>seed ${l.seed}</span>` : '<span>chưa tạo</span>'}${qa}${l.error ? `<span class="pill bad">${esc(l.error)}</span>` : ''}</div>
        </div>
        <div class="lacts">
          ${styleOpts ? `<select class="input lstyle" title="Sắc thái">${styleOpts.map((s) => `<option value="${s.id}" ${(l.style_id || v.styles[0].id) === s.id ? 'selected' : ''}>${esc(s.label)}</option>`).join('')}</select>` : ''}
          <button type="button" class="lplay" title="Nghe" ${l.output_id ? '' : 'disabled'}>▶</button>
          <button type="button" class="lgen" title="${l.output_id ? 'Tạo lại dòng này' : 'Tạo dòng này'}">⟳</button>
          <button type="button" class="lretake" title="Thu lại với seed khác">🎲</button>
        </div></li>`;
    }).join('');
  }

  $('#lines').addEventListener('click', async (e) => {
    const li = e.target.closest('li.line'); if (!li || !proj) return;
    const line = proj.lines.find((l) => l.n === +li.dataset.n); if (!line) return;
    if (e.target.closest('.lplay')) { const m = await outputMeta(line.output_id); if (m) { showResult(m, true); } return; }
    if (e.target.closest('.lgen')) { await runBatch([line], {}); return; }
    if (e.target.closest('.lretake')) { await runBatch([line], { newSeed: true }); return; }
  });
  $('#lines').addEventListener('change', (e) => {
    const sel = e.target.closest('.lstyle'); if (!sel || !proj) return;
    const line = proj.lines.find((l) => l.n === +sel.closest('li').dataset.n);
    if (line) { line.style_id = sel.value; saveProject(); }
  });
  $('#projectSelect').addEventListener('change', (e) => { if (e.target.value) openProject(e.target.value, true); });
  $('#projectNew').addEventListener('click', async () => {
    if (!textEl.value.trim()) { toast('Gõ hoặc dán kịch bản trước đã.', 'warn'); return; }
    await newProjectFromText(); renderLines(); $('#exportResult').hidden = true; toast(`Dự án mới: ${proj.name}`, 'ok');
  });
  $('#projectDelete').addEventListener('click', async () => {
    if (!proj) return;
    if (!await confirmDialog(`Xoá dự án "${proj.name}"? Các bản ghi trong Lịch sử vẫn còn.`)) return;
    try { await api(`/api/projects/${proj.id}`, { method: 'DELETE' }); proj = null; state.projectId = null; persist(); await loadProjects(); if (projects[0]) await openProject(projects[0].id, true); else renderLines(); }
    catch (err) { toast(err.message, 'err'); }
  });
  $('#gap').value = state.gap;
  $('#gap').addEventListener('input', (e) => { state.gap = +e.target.value || 0; persist(); renderLines(); });
  $('#genMissing').addEventListener('click', () => proj && runBatch(proj.lines.filter(lineNeedsWork), {}));
  $('#stopBatch').addEventListener('click', () => { batchStop = true; $('#stopBatch').textContent = 'ĐANG DỪNG…'; });
  $('#exportProject').addEventListener('click', exportProject);

  async function outputMeta(id) {
    if (!id) return null;
    let m = history.find((x) => x.id === id);
    if (!m) { await loadHistory(); m = history.find((x) => x.id === id); }
    if (!m) toast('Bản ghi này không còn trong Lịch sử.', 'warn');
    return m || null;
  }

  async function exportProject() {
    if (!proj || busy) return;
    if (!proj.lines.some((l) => l.output_id)) { toast('Chưa có dòng nào được tạo.', 'warn'); return; }
    setBusy(true); $('#progressText').textContent = 'Đang căn chữ và xuất…';
    try {
      const r = await postJSON(`/api/projects/${proj.id}/export`, { gap: +state.gap || 0, align: true });
      const box = $('#exportResult');
      box.hidden = false;
      box.innerHTML = `<span>Đã xuất ${r.lines} dòng · ${fmtTime(r.duration)}</span>
        <a class="btn btn-sm" href="${r.url}" download="${esc(proj.name)}.zip">TẢI ZIP</a>
        <a class="btn btn-sm" href="${r.full_url}" target="_blank" rel="noopener">NGHE FULL</a>
        <code title="Thư mục trên máy">${esc(r.dir)}</code>${r.missing.length ? `<span>· thiếu: ${esc(r.missing.join(', '))}</span>` : ''}`;
      toast('Xuất xong: wav + mp3 từng dòng, full, timeline.json', 'ok', 'XONG!');
    } catch (e) { toast(e.message, 'err'); }
    finally { setBusy(false); }
  }

  // ------------------------------------------------------------------ voice panel
  function setMode(mode) {
    state.mode = mode;
    $$('#modeTabs .tab').forEach((t) => t.classList.toggle('is-active', t.dataset.mode === mode));
    $$('.panel-voice .mode').forEach((m) => m.classList.toggle('is-active', m.id === `mode-${mode}`));
    persist();
  }
  $('#modeTabs').addEventListener('click', (e) => { const t = e.target.closest('.tab'); if (t) setMode(t.dataset.mode); });
  setMode(state.mode);

  async function loadVoices() {
    voices = await api('/api/voices');
    if (state.voiceId && !voices.some((v) => v.id === state.voiceId)) state.voiceId = null;
    if (!state.voiceId && voices.length) state.voiceId = voices[0].id;
    const v = selectedVoice();
    if (v && !v.styles.some((s) => s.id === state.styleId)) state.styleId = v.styles[0].id;
    renderVoices(); renderStyles();
  }

  function renderVoices() {
    const grid = $('#voiceGrid');
    if (!voices.length) {
      grid.innerHTML = `<p class="voice-empty">Chưa có giọng nào. Thêm một đoạn ghi âm 3–10 giây của bạn để bắt đầu.</p>`;
      return;
    }
    grid.innerHTML = voices.map((v) => {
      const s0 = v.styles[0];
      return `<div class="voice ${v.id === state.voiceId ? 'is-selected' : ''}" data-id="${v.id}" role="button" tabindex="0" title="${esc(s0.ref_text)}">
        <span class="avatar" style="--c:${esc(v.color || '#4CC9F0')}">${esc(initial(v.name))}</span>
        <span class="meta">
          <span class="name">${esc(v.name)}</span>
          <span class="sub">${s0.duration ? s0.duration.toFixed(1) + 's' : ''}${v.styles.length > 1 ? ` · ${v.styles.length} sắc thái` : ''}${v.preset ? ' · mẫu' : ''}${s0.transcribed ? ' · whisper' : ''}</span>
        </span>
        <span class="acts">
          <button type="button" class="prev" title="Nghe mẫu" aria-label="Nghe mẫu">▶</button>
          <button type="button" class="edit" title="Sửa tên / lời" aria-label="Sửa">✎</button>
          <button type="button" class="del" title="Xoá giọng" aria-label="Xoá">×</button>
        </span></div>`;
    }).join('');
  }

  function renderStyles() {
    const v = selectedVoice();
    const box = $('#styles');
    box.hidden = !v;
    if (!v) return;
    const cur = voiceStyle(v, state.styleId);
    $('#styleChips').innerHTML = v.styles.map((s) => `<button type="button" class="chip ${s.id === cur.id ? 'is-on' : ''}" data-sid="${s.id}" title="${esc(s.ref_text)}">${esc(s.label)} <small class="mono">${s.duration ? s.duration.toFixed(1) + 's' : ''}${s.pace ? ` · ${s.pace.toFixed(1)} từ/s` : ''}</small>${s.pace > 4.6 ? '<span class="fast" title="Mẫu đọc nhanh: app tự hãm model về ≤4,6 từ/s để không nuốt chữ">NHANH</span>' : ''}${v.styles.length > 1 ? `<span class="x" data-del="${s.id}" title="Xoá sắc thái">×</span>` : ''}</button>`).join('');
    $('#asStyle').hidden = !current || !v;
    showSpeed();
  }

  $('#voiceGrid').addEventListener('click', async (e) => {
    const card = e.target.closest('.voice');
    if (!card) return;
    const id = card.dataset.id;
    const v = voices.find((x) => x.id === id);
    if (e.target.closest('.prev')) {
      const p = $('#previewAudio');
      const sid = id === state.voiceId ? voiceStyle(v, state.styleId).id : v.styles[0].id;
      if (!p.paused && p.dataset.key === id + sid) { p.pause(); return; }
      p.src = `/api/voices/${id}/audio?style=${encodeURIComponent(sid)}&t=${Date.now()}`;
      p.dataset.key = id + sid;
      p.play().catch(() => {});
      return;
    }
    if (e.target.closest('.edit')) { openModal('edit', { voice: v, style: id === state.voiceId ? voiceStyle(v, state.styleId) : v.styles[0] }); return; }
    if (e.target.closest('.del')) {
      if (!await confirmDialog(`Xoá giọng "${v.name}" và mọi sắc thái của nó?`)) return;
      try { await api(`/api/voices/${id}`, { method: 'DELETE' }); toast(`Đã xoá "${v.name}"`, 'ok', 'BÙM'); await loadVoices(); persist(); renderLines(); }
      catch (err) { toast(err.message, 'err'); }
      return;
    }
    state.voiceId = id;
    state.styleId = v.styles[0].id;
    renderVoices(); renderStyles(); persist();
    if (proj) { proj.voice_id = id; saveProject(); renderLines(); }
  });
  $('#voiceGrid').addEventListener('keydown', (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && e.target.classList.contains('voice')) { e.preventDefault(); e.target.click(); }
  });
  $('#addVoice').addEventListener('click', () => openModal('add'));
  $('#addStyle').addEventListener('click', () => { const v = selectedVoice(); if (v) openModal('add-style', { voice: v }); });
  $('#styleChips').addEventListener('click', async (e) => {
    const v = selectedVoice(); if (!v) return;
    const del = e.target.closest('[data-del]');
    if (del) {
      const s = voiceStyle(v, del.dataset.del);
      if (!await confirmDialog(`Xoá sắc thái "${s.label}" của ${v.name}?`)) return;
      try { await api(`/api/voices/${v.id}/styles/${s.id}`, { method: 'DELETE' }); await loadVoices(); renderLines(); } catch (err) { toast(err.message, 'err'); }
      return;
    }
    const chip = e.target.closest('.chip');
    if (chip) { state.styleId = chip.dataset.sid; renderStyles(); persist(); }
  });

  // design chips
  $('#designChips').innerHTML = DESIGN.map((g) => `
    <div class="chip-group" data-key="${g.key}"><span>${esc(g.label)}</span>
      <div class="chips">${g.opts.map(([v, l]) => `<button type="button" class="chip ${state.design[g.key] === v ? 'is-on' : ''}" data-v="${esc(v)}">${esc(l)}</button>`).join('')}</div>
    </div>`).join('');
  $('#designChips').addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    const key = chip.closest('.chip-group').dataset.key;
    state.design[key] = state.design[key] === chip.dataset.v ? null : chip.dataset.v;
    $$('.chip', chip.parentElement).forEach((c) => c.classList.toggle('is-on', c.dataset.v === state.design[key]));
    renderInstruct(); persist();
  });
  $('#instructExtra').value = state.designExtra || '';
  $('#instructExtra').addEventListener('input', (e) => { state.designExtra = e.target.value; renderInstruct(); persist(); });
  function buildInstruct() {
    const parts = DESIGN.map((g) => state.design[g.key]).filter(Boolean);
    const extra = (state.designExtra || '').trim();
    if (extra) parts.push(extra);
    return parts.join(', ');
  }
  function renderInstruct() { $('#instructPreview').textContent = buildInstruct() || '—'; }
  renderInstruct();

  // ------------------------------------------------------------------ settings
  $('#language').addEventListener('change', (e) => { state.language = e.target.value; persist(); });
  $('#model').addEventListener('change', (e) => { state.model = e.target.value; $('#modelNote').textContent = status.models[state.model].note; persist(); });
  const speedEl = $('#speed'); speedEl.value = state.speed;
  function showSpeed() {
    $('#speedOut').textContent = `${(+speedEl.value).toFixed(2)}×`;
    const s = +speedEl.value, hint = $('#speedHint');
    const v = selectedVoice();
    const pace = (state.mode === 'clone' && v && voiceStyle(v, state.styleId).pace) || null;
    const natural = pace || 3.8;
    const target = (natural * s).toFixed(1);
    if (state.speedMode === 'model') hint.textContent = `Model đọc ≈${target} từ/giây (mẫu ${natural.toFixed(1)}). Trên ~4,6 từ/giây model nuốt chữ, dưới ~3 ngắt quãng.`;
    else if (state.speedMode === 'stretch') hint.textContent = `Model đọc nhịp tự nhiên của mẫu (${natural.toFixed(1)} từ/giây), toàn bộ chênh lệch kéo giãn audio.`;
    else {
      const modelPace = Math.min(Math.max(natural * s, 3.0), 4.6);
      const stretch = (natural * s) / modelPace;
      hint.textContent = `Kết quả ≈${target} từ/giây${pace ? ` (mẫu ${natural.toFixed(1)})` : ''}. ${Math.abs(stretch - 1) > 0.02 ? `Model đọc ${modelPace.toFixed(1)} từ/giây rồi kéo giãn ${stretch.toFixed(2)}× — không nuốt chữ, không ngắt quãng.` : 'Model tự đọc đúng nhịp này.'}${pace > 4.6 && s >= 1 ? ' Mẫu này nói nhanh — muốn nghe thoải mái thử 0,8–0,85×.' : ''}`;
    }
  }
  speedEl.addEventListener('input', () => { state.speed = +speedEl.value; showSpeed(); updateCounter(); persist(); });
  $('#speedMode').value = state.speedMode || 'auto';
  $('#speedMode').addEventListener('change', (e) => { state.speedMode = e.target.value; showSpeed(); persist(); });
  showSpeed();
  $$('#steps button').forEach((b) => b.classList.toggle('is-active', +b.dataset.v === +state.num_step));
  $('#steps').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    state.num_step = +b.dataset.v;
    $$('#steps button').forEach((x) => x.classList.toggle('is-active', x === b));
    persist();
  });
  $$('#takes button').forEach((b) => b.classList.toggle('is-active', +b.dataset.v === +state.takes));
  $('#takes').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    state.takes = +b.dataset.v;
    $$('#takes button').forEach((x) => x.classList.toggle('is-active', x === b));
    if (state.takes > 1 && !state.qaAuto) { state.qaAuto = true; $('#qaAuto').checked = true; toast('Nhiều bản thử cần Whisper soát lỗi → đã bật.', 'warn'); }
    persist();
  });
  $('#qaAuto').checked = state.qaAuto !== false;
  $('#qaAuto').addEventListener('change', (e) => { state.qaAuto = e.target.checked; persist(); });
  $('#readNumbers').checked = state.readNumbers !== false;
  $('#readNumbers').addEventListener('change', (e) => { state.readNumbers = e.target.checked; persist(); previewSpoken(); });
  const seedEl = $('#seed'); seedEl.value = state.seed ?? '';
  seedEl.addEventListener('input', () => { state.seed = seedEl.value; persist(); });
  $('#seedDice').addEventListener('click', () => { seedEl.value = Math.floor(Math.random() * 2 ** 31); seedEl.dispatchEvent(new Event('input')); });
  $('#seedClear').addEventListener('click', () => { seedEl.value = ''; seedEl.dispatchEvent(new Event('input')); });
  const guidanceEl = $('#guidance'); guidanceEl.value = state.guidance_scale;
  const showGuidance = () => { $('#guidanceOut').textContent = (+guidanceEl.value).toFixed(1); };
  guidanceEl.addEventListener('input', () => { state.guidance_scale = +guidanceEl.value; showGuidance(); persist(); });
  showGuidance();
  $('#denoise').checked = state.denoise !== false;
  $('#denoise').addEventListener('change', (e) => { state.denoise = e.target.checked; persist(); });
  updateCounter();

  // ------------------------------------------------------------------ generate
  const goBtn = $('#generate');
  function updateGoLabel() {
    if (busy) return;
    $('.go-label', goBtn).textContent = state.scriptMode === 'lines' ? `TẠO TẤT CẢ${proj && proj.lines.length ? ` (${proj.lines.length} DÒNG)` : ''}` : 'TẠO GIỌNG!';
    $('.go-sub', goBtn).textContent = state.scriptMode === 'lines' ? 'Ctrl + Enter = chỉ dòng chưa có / lỗi' : 'Ctrl + Enter';
  }
  function setBusy(b, label) {
    busy = b;
    goBtn.classList.toggle('is-busy', b);
    goBtn.disabled = b;
    $('#exportProject').disabled = b; $('#genMissing').disabled = b;
    if (b) $('.go-label', goBtn).textContent = label || 'ĐANG TẠO…'; else updateGoLabel();
    $('#progress').hidden = !b;
    if (!b) { $('#stopBatch').hidden = true; $('#stopBatch').textContent = 'DỪNG'; }
  }

  let progressTimer = null;
  function startProgress(estimate) {
    const t0 = performance.now();
    const fill = $('#progressFill'), text = $('#progressText');
    fill.style.width = '0%'; text.dataset.stage = '';
    progressTimer = setInterval(() => {
      const el = (performance.now() - t0) / 1000;
      fill.style.width = `${Math.min(96, (1 - Math.exp(-el / Math.max(1, estimate) * 2.2)) * 100)}%`;
      text.textContent = `${text.dataset.stage || 'Đang xử lý…'} ${el.toFixed(1)}s / ~${Math.round(estimate)}s`;
    }, 120);
  }
  function stopProgress() { clearInterval(progressTimer); $('#progressFill').style.width = '100%'; }

  function estimateSeconds(text) {
    const audioSec = wordCount(text) / WORDS_PER_SECOND / (+state.speed || 1);
    const rtf = (status && status.last_rtf) || 0.25;
    let est = Math.max(2, audioSec * rtf * (state.num_step / 32) + 1.5);
    if (state.qaAuto || state.takes > 1) est += 1.2;
    if (!status || status.model !== state.model) est += 25; // model (re)load
    return est;
  }

  function genBody(text, extra = {}) {
    const v = selectedVoice();
    return {
      text, mode: state.mode, voice_id: state.mode === 'clone' ? state.voiceId : null,
      style_id: state.mode === 'clone' && v ? voiceStyle(v, state.styleId).id : null,
      instruct: state.mode === 'design' ? buildInstruct() : null,
      language: state.language, speed: +state.speed, tempo: 1, speed_mode: state.speedMode || 'auto',
      num_step: +state.num_step, seed: state.seed === '' || state.seed == null ? null : +state.seed,
      guidance_scale: +state.guidance_scale, denoise: !!state.denoise, model: state.model,
      read_numbers: !!state.readNumbers, takes: +state.takes || 1, qa: !!state.qaAuto, ...extra,
    };
  }

  function validateVoice() {
    if (state.mode === 'clone' && !state.voiceId) { toast('Chọn một giọng mẫu để nhân bản.', 'warn'); return false; }
    if (state.mode === 'design' && !buildInstruct()) { toast('Chọn ít nhất một thuộc tính giọng.', 'warn'); return false; }
    return true;
  }

  async function generate(extra = {}) {
    if (busy) return;
    const text = textEl.value.replace(/\*+/g, '').trim();
    if (!text) { toast('Gõ kịch bản đã chứ!', 'warn'); textEl.focus(); return; }
    if (!validateVoice()) return;
    setBusy(true);
    startProgress(estimateSeconds(text));
    try {
      await waitForModel();
      const meta = await postJSON('/api/generate', genBody(text, extra));
      history.unshift(meta);
      renderHistory();
      await showResult(meta, true);
      const q = meta.qa;
      toast(`${meta.duration}s audio trong ${meta.gen_seconds}s${q ? (q.errors ? ` · ${q.errors} chữ đọc sai` : ' · đọc đúng hết') : ''}${meta.takes && meta.takes.length > 1 ? ` · ${meta.takes.length} bản thử` : ''}`, q && q.errors ? 'warn' : 'ok', q && q.errors ? 'HỪM' : 'XONG!');
      const pop = $('#pop'); pop.classList.remove('is-on'); void pop.offsetWidth; pop.classList.add('is-on');
    } catch (err) {
      toast(err.message, 'err');
    } finally {
      stopProgress(); setBusy(false);
    }
  }

  async function runBatch(lines, { newSeed = false } = {}) {
    if (busy || !proj) return;
    if (!lines.length) { toast('Không có dòng nào cần tạo.', 'warn'); return; }
    if (!validateVoice()) return;
    batchStop = false;
    setBusy(true, `ĐANG TẠO 0/${lines.length}…`);
    $('#stopBatch').hidden = lines.length < 2;
    const fill = $('#progressFill'), text = $('#progressText');
    let done = 0, bad = 0;
    try {
      await waitForModel();
      for (const line of lines) {
        if (batchStop) break;
        const label = line.label || String(line.n).padStart(2, '0');
        $('.go-label', goBtn).textContent = `ĐANG TẠO ${done + 1}/${lines.length}…`;
        text.textContent = `${label}: ${line.text.slice(0, 60)}…`;
        const li = $(`#lines li[data-n="${line.n}"]`); if (li) li.classList.add('is-busy');
        const v = selectedVoice();
        try {
          const meta = await postJSON('/api/generate', genBody(line.text, {
            project_id: proj.id, line: line.n, seed: newSeed ? null : (line.seed ?? (state.seed === '' ? null : +state.seed)),
            style_id: state.mode === 'clone' && v ? (line.style_id || voiceStyle(v, state.styleId).id) : null,
          }));
          Object.assign(line, { output_id: meta.id, seed: meta.seed, duration: meta.duration, qa: meta.qa || null, noisy: !!(meta.quality && meta.quality.noisy), error: null });
          history.unshift(meta);
          if (lines.length === 1) await showResult(meta, true);
          if (meta.qa && meta.qa.errors) bad++;
        } catch (err) {
          line.error = err.message.slice(0, 80); bad++;
          if (/409|đang bận/i.test(err.message)) { toast('Máy đang bận — dừng loạt.', 'err'); break; }
        }
        done++;
        fill.style.width = `${(done / lines.length) * 100}%`;
        saveProject(); renderLines(); renderHistory();
      }
    } finally {
      setBusy(false); renderLines();
      toast(`${done}/${lines.length} dòng xong${bad ? ` · ${bad} dòng có lỗi, bấm ⟳ để thu lại` : ''}`, bad ? 'warn' : 'ok', bad ? 'HỪM' : 'XONG!');
    }
  }

  goBtn.addEventListener('click', () => state.scriptMode === 'lines' ? runBatch(proj ? proj.lines : [], {}) : generate());
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !$('#voiceModal').open && !$('#dictModal').open) {
      e.preventDefault();
      if (state.scriptMode === 'lines') runBatch(proj ? proj.lines.filter(lineNeedsWork) : [], {}); else generate();
    }
  });

  // ------------------------------------------------------------------ result + player
  const audio = $('#audio');
  const wave = $('#wave');

  async function showResult(meta, autoplay = false) {
    current = meta;
    $('#resultEmpty').hidden = true;
    $('#result').hidden = false;
    $('#download').href = meta.url;
    $('#download').download = `omnivoice-${meta.id}.wav`;
    $('#resultText').textContent = meta.text;
    $('#asStyle').hidden = !selectedVoice();
    const voiceLabel = meta.mode === 'clone' ? meta.voice_name : meta.mode === 'design' ? `thiết kế: ${meta.instruct}` : 'tự động';
    const speedLabel = (meta.model_speed != null && (Math.abs(meta.model_speed - meta.speed) > 0.001 || Math.abs((meta.tempo || 1) - 1) > 0.01)
      ? `${meta.speed}× (model ${meta.model_speed}× · giãn ${meta.tempo}×)` : `${meta.speed}×`)
      + (meta.ref_pace ? ` · mẫu ${meta.ref_pace} từ/s` : '') + (meta.duration ? ` · ra ${(wordCount(meta.spoken_text || meta.text) / meta.duration).toFixed(1)} từ/s` : '');
    const rows = [
      ['Giọng', voiceLabel], ['Thời lượng', `${meta.duration}s`], ['Tạo trong', `${meta.gen_seconds}s`],
      ['RTF', meta.rtf ?? '—'], ['VRAM đỉnh', meta.peak_vram_gb ? `${meta.peak_vram_gb} GB` : '—'],
      ['Seed', `${meta.seed} <button type="button" data-seed="${meta.seed}" title="Dùng lại seed này">dùng lại</button>`],
      ['Model', status && status.models[meta.model] ? status.models[meta.model].label : meta.model],
      ['Ngôn ngữ', meta.language], ['Tốc độ', speedLabel], ['Bước', meta.num_step],
    ];
    if (meta.quality && meta.quality.floor_db != null) rows.push(['Nền', `${meta.quality.floor_db} dB${meta.quality.noisy ? ' · hơi rè → thử SEED KHÁC' : ' · sạch'}`]);
    if (meta.takes && meta.takes.length > 1) rows.push(['Bản thử', meta.takes.map((t) => `${t.errors ?? '?'} lỗi${t.noisy ? ', rè' : ''}${t.chosen ? ' ✓' : ''}`).join(' · ')]);
    if (meta.project_id && meta.line) rows.push(['Dòng', `#${meta.line}`]);
    $('#stats').innerHTML = rows.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${k === 'Seed' ? v : esc(v)}</dd></div>`).join('');
    renderQA(meta);
    $$('#history li').forEach((li) => li.classList.toggle('is-current', li.dataset.id === meta.id));
    $$('#lines li').forEach((li) => li.classList.toggle('is-current', proj && (proj.lines.find((l) => l.n === +li.dataset.n) || {}).output_id === meta.id));
    audio.src = meta.url;
    peaks = null;
    drawWave(0);
    loadPeaks(meta.url).then(() => drawWave(audio.duration ? audio.currentTime / audio.duration : 0));
    if (autoplay) audio.play().catch(() => {});
  }

  function renderQA(meta) {
    const box = $('#qa');
    const alignPart = meta.alignment
      ? `<span class="pill ok">đã căn ${esc(meta.alignment.matched)} từ</span><button type="button" class="btn btn-sm" data-act="dl-align">TẢI TIMING .JSON</button>`
      : `<button type="button" class="btn btn-sm" data-act="align" title="Mốc thời gian từng chữ (Whisper) để làm caption">CĂN CHỮ</button>`;
    if (!meta.qa) {
      box.innerHTML = `<div class="verdict"><span>Chưa soát lỗi đọc.</span><div class="qa-acts"><button type="button" class="btn btn-sm" data-act="qa">SOÁT LỖI (WHISPER)</button>${alignPart}</div></div>`;
      return;
    }
    const q = meta.qa;
    const verdict = q.errors ? `<span class="pill bad">${q.errors} chữ sai / thiếu</span>` : `<span class="pill ok">ĐỌC ĐÚNG HẾT</span>`;
    const near = (q.near ? `<span class="pill near">${q.near} gần đúng</span>` : '') + (meta.quality && meta.quality.noisy ? `<span class="pill near" title="Nền nhiễu ${meta.quality.floor_db} dB — bản sinh này hơi rè, bấm SEED KHÁC">NỀN RÈ</span>` : '');
    box.innerHTML = `
      <div class="verdict">${verdict}${near}<span class="mono">CER ${(q.cer * 100).toFixed(1)}%</span>
        <div class="qa-acts"><button type="button" class="btn btn-sm" data-act="qa" title="Nghe lại bằng Whisper">SOÁT LẠI</button>${alignPart}</div></div>
      <div class="qa-words">${q.words.map((w) => `<span class="qa-w ${w.s}">${esc(w.w)}</span>`).join(' ')}</div>
      <div class="heard">Whisper nghe: “${esc(q.heard)}”${q.extra && q.extra.length ? ` · thừa: ${esc(q.extra.join(', '))}` : ''}</div>
      <div class="legend">đỏ = đọc sai hoặc bỏ sót · vàng = gần đúng (thường chỉ là cách Whisper viết: sét/xét, ổng/ông) · Whisper cũng có thể nghe nhầm — hãy nghe lại chữ đỏ.</div>`;
  }

  $('#qa').addEventListener('click', async (e) => {
    const b = e.target.closest('button[data-act]'); if (!b || !current || busy) return;
    const act = b.dataset.act;
    if (act === 'dl-align') {
      const blob = new Blob([JSON.stringify({ id: current.id, text: current.text, spoken_text: current.spoken_text, duration: current.duration,
        words: current.alignment.words, alignment: current.alignment.alignment }, null, 1)], { type: 'application/json' });
      const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `omnivoice-${current.id}.json`; a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
      return;
    }
    setBusy(true); $('#progressText').textContent = act === 'qa' ? 'Whisper đang nghe…' : 'Đang căn chữ…';
    try {
      const meta = await postJSON(`/api/${act}/${current.id}`, {});
      const i = history.findIndex((x) => x.id === meta.id); if (i >= 0) history[i] = meta;
      if (proj) { const ln = proj.lines.find((l) => l.output_id === meta.id); if (ln) { ln.qa = meta.qa || null; saveProject(); renderLines(); } }
      await showResult(meta, false);
      renderHistory();
    } catch (err) { toast(err.message, 'err'); }
    finally { setBusy(false); }
  });

  $('#stats').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-seed]'); if (!b) return;
    seedEl.value = b.dataset.seed; seedEl.dispatchEvent(new Event('input'));
    toast(`Seed ${b.dataset.seed} đã được khoá`, 'ok');
  });
  $('#reseed').addEventListener('click', async () => {
    if (!current || busy) return;
    if (current.project_id && proj && proj.id === current.project_id) {
      const line = proj.lines.find((l) => l.output_id === current.id);
      if (line) { await runBatch([line], { newSeed: true }); return; }
    }
    const keep = textEl.value;
    textEl.value = current.text;
    await generate({ seed: null });
    textEl.value = keep;
  });

  async function loadPeaks(url) {
    try {
      const buf = await (await fetch(url)).arrayBuffer();
      audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      const ab = await audioCtx.decodeAudioData(buf);
      const data = ab.getChannelData(0);
      const bars = Math.max(40, Math.floor(wave.clientWidth / 5));
      const step = Math.floor(data.length / bars) || 1;
      const out = new Float32Array(bars);
      let max = 0;
      for (let i = 0; i < bars; i++) {
        let m = 0;
        for (let j = i * step, e = Math.min(data.length, j + step); j < e; j += 2) { const a = Math.abs(data[j]); if (a > m) m = a; }
        out[i] = m; if (m > max) max = m;
      }
      if (max > 0) for (let i = 0; i < bars; i++) out[i] /= max;
      peaks = out;
    } catch { peaks = null; }
  }

  function drawWave(progress) {
    const dpr = window.devicePixelRatio || 1;
    const W = wave.clientWidth, H = 120;
    if (wave.width !== W * dpr) { wave.width = W * dpr; wave.height = H * dpr; }
    const ctx = wave.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(17,17,17,.08)';
    for (let y = 6; y < H; y += 10) for (let x = 6 + ((y / 10) % 2) * 5; x < W; x += 10) { ctx.beginPath(); ctx.arc(x, y, 1, 0, 7); ctx.fill(); }
    if (!peaks) {
      ctx.fillStyle = '#6b6b6b'; ctx.font = '700 13px "Space Mono", monospace'; ctx.textAlign = 'center';
      ctx.fillText('đang vẽ sóng…', W / 2, H / 2 + 4);
      return;
    }
    const n = peaks.length, bw = W / n, cut = Math.floor(progress * n);
    for (let i = 0; i < n; i++) {
      const h = Math.max(3, peaks[i] * (H - 16));
      ctx.fillStyle = i < cut ? '#FF4D8D' : '#111111';
      ctx.fillRect(i * bw + 1, (H - h) / 2, Math.max(1.5, bw - 2), h);
    }
    ctx.fillStyle = '#111111'; ctx.fillRect(progress * W - 1, 0, 2, H);
  }
  wave.addEventListener('click', (e) => {
    if (!audio.duration) return;
    const r = wave.getBoundingClientRect();
    audio.currentTime = ((e.clientX - r.left) / r.width) * audio.duration;
  });
  window.addEventListener('resize', debounce(() => { if (current) loadPeaks(current.url).then(() => drawWave(audio.duration ? audio.currentTime / audio.duration : 0)); }, 200));

  const playBtn = $('#play');
  playBtn.addEventListener('click', () => { if (audio.paused) audio.play().catch(() => {}); else audio.pause(); });
  const syncPlay = () => { playBtn.textContent = audio.paused ? '▶' : '❚❚'; playBtn.classList.toggle('is-playing', !audio.paused); };
  audio.addEventListener('play', syncPlay); audio.addEventListener('pause', syncPlay); audio.addEventListener('ended', syncPlay);
  audio.addEventListener('timeupdate', () => {
    $('#time').textContent = `${fmtTime(audio.currentTime)} / ${fmtTime(audio.duration)}`;
    drawWave(audio.duration ? audio.currentTime / audio.duration : 0);
  });
  audio.addEventListener('loadedmetadata', () => { $('#time').textContent = `0:00 / ${fmtTime(audio.duration)}`; });
  $('#asVoice').addEventListener('click', () => { if (current) openModal('from-output', { output: current }); });
  $('#asStyle').addEventListener('click', () => { const v = selectedVoice(); if (current && v) openModal('from-output-style', { output: current, voice: v }); });

  // ------------------------------------------------------------------ history
  async function loadHistory() { history = await api('/api/history'); renderHistory(); }
  function renderHistory() {
    const ol = $('#history');
    $('#historyEmpty').hidden = history.length > 0;
    ol.innerHTML = history.map((m, i) => {
      const voice = m.mode === 'clone' ? m.voice_name : m.mode === 'design' ? 'thiết kế' : 'tự động';
      const when = new Date(m.created * 1000).toLocaleString('vi-VN', { hour: '2-digit', minute: '2-digit', day: '2-digit', month: '2-digit' });
      const qa = m.qa ? (m.qa.errors ? `<span class="pill bad">${m.qa.errors} lỗi</span>` : `<span class="pill ok">✓</span>`) : '';
      return `<li data-id="${m.id}" class="${current && current.id === m.id ? 'is-current' : ''}">
        <span class="num">${history.length - i}</span>
        <span class="body"><span class="excerpt">${esc(m.text)}</span>
          <span class="sub"><span class="pill">${esc(voice)}</span>${qa}<span>${m.duration}s</span><span>seed ${m.seed}</span>${m.line ? `<span>dòng ${m.line}</span>` : ''}<span>${esc(when)}</span></span></span>
        <span class="acts">
          <a href="${m.url}" download="omnivoice-${m.id}.wav" title="Tải .wav" aria-label="Tải">⤓</a>
          <button type="button" class="del" title="Xoá" aria-label="Xoá">×</button>
        </span></li>`;
    }).join('');
  }
  $('#history').addEventListener('click', async (e) => {
    const li = e.target.closest('li'); if (!li) return;
    const m = history.find((x) => x.id === li.dataset.id);
    if (e.target.closest('a')) return;
    if (e.target.closest('.del')) {
      e.stopPropagation();
      if (!await confirmDialog('Xoá bản ghi này?')) return;
      try {
        await api(`/api/history/${m.id}`, { method: 'DELETE' });
        history = history.filter((x) => x.id !== m.id);
        if (proj) { const ln = proj.lines.find((l) => l.output_id === m.id); if (ln) { Object.assign(ln, { output_id: null, qa: null, duration: null }); saveProject(); renderLines(); } }
        if (current && current.id === m.id) { audio.pause(); audio.removeAttribute('src'); current = null; $('#result').hidden = true; $('#resultEmpty').hidden = false; }
        renderHistory();
      } catch (err) { toast(err.message, 'err'); }
      return;
    }
    showResult(m, true);
    $('.panel-result').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });

  // ------------------------------------------------------------------ voice modal (add / edit / from-output / add-style / from-output-style)
  const modal = $('#voiceModal');
  const form = $('#voiceForm');
  let modalMode = 'add', modalCtx = {}, pendingFile = null;

  function openModal(mode, ctx = {}) {
    modalMode = mode; modalCtx = ctx; setPendingFile(null);
    $('#vName').value = ''; $('#vLabel').value = ''; $('#vText').value = '';
    const needsAudio = mode === 'add' || mode === 'add-style';
    const styleMode = mode === 'add-style' || mode === 'from-output-style' || (mode === 'edit' && ctx.voice && ctx.voice.styles.length > 1);
    $('#vSource').hidden = !needsAudio;
    $('#fName').hidden = mode === 'add-style' || mode === 'from-output-style';
    $('#fLabel').hidden = !styleMode;
    $('#fText').hidden = mode === 'from-output' || mode === 'from-output-style';
    const titles = { add: 'THÊM GIỌNG MẪU', edit: 'SỬA GIỌNG', 'from-output': 'LƯU KẾT QUẢ LÀM GIỌNG', 'add-style': `SẮC THÁI MỚI · ${ctx.voice ? ctx.voice.name : ''}`, 'from-output-style': `SẮC THÁI MỚI · ${ctx.voice ? ctx.voice.name : ''}` };
    $('#modalTitle').textContent = titles[mode];
    $('#modalSave').textContent = mode === 'edit' ? 'LƯU' : mode.includes('style') ? 'LƯU SẮC THÁI' : 'LƯU GIỌNG';
    if (needsAudio) setSrc('file');
    if (mode === 'edit') {
      $('#vName').value = ctx.voice.name; $('#vLabel').value = ctx.style.label; $('#vText').value = ctx.style.ref_text || '';
    } else if (mode === 'from-output') {
      const o = ctx.output;
      $('#vName').value = o.mode === 'design' ? `Giọng thiết kế #${o.seed % 1000}` : `${(o.voice_name || 'Giọng').split(' · ')[0]} (bản ${o.seed % 1000})`;
    } else if (mode === 'from-output-style') {
      $('#vLabel').value = `Sắc thái ${ctx.voice.styles.length + 1}`;
    }
    modal.showModal();
    setTimeout(() => { const f = $('#fName').hidden ? $('#vLabel') : $('#vName'); f.focus(); f.select(); }, 50);
  }
  function closeModal() { stopRec(); modal.close(); }
  $('#modalClose').addEventListener('click', closeModal);
  $('#modalCancel').addEventListener('click', closeModal);
  modal.addEventListener('click', (e) => { if (e.target === modal) closeModal(); });
  modal.addEventListener('close', stopRec);

  function setSrc(which) {
    $$('#srcTabs .tab').forEach((t) => t.classList.toggle('is-active', t.dataset.src === which));
    $$('.src').forEach((s) => s.classList.toggle('is-active', s.id === `src-${which}`));
    if (which !== 'rec') stopRec();
  }
  $('#srcTabs').addEventListener('click', (e) => { const t = e.target.closest('.tab'); if (t) setSrc(t.dataset.src); });

  function setPendingFile(file) {
    pendingFile = file;
    const prev = $('#vPreview'), pa = $('#vPreviewAudio');
    if (pa.src && pa.src.startsWith('blob:')) URL.revokeObjectURL(pa.src);
    if (!file) { prev.hidden = true; pa.removeAttribute('src'); return; }
    pa.src = URL.createObjectURL(file);
    $('#vFileName').textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
    prev.hidden = false;
  }
  $('#vClear').addEventListener('click', () => setPendingFile(null));
  const drop = $('#drop');
  $('#vFile').addEventListener('change', (e) => { if (e.target.files[0]) setPendingFile(e.target.files[0]); e.target.value = ''; });
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('is-over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('is-over'); }));
  drop.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) setPendingFile(f); });

  // microphone recording (max 20 s, matches the server-side cap)
  let rec = null, recChunks = [], recTimer = null, recStream = null, recAnalyser = null, recRaf = null;
  async function startRec() {
    try {
      recStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: true, autoGainControl: true } });
    } catch { toast('Không mở được micro. Kiểm tra quyền trình duyệt.', 'err'); return; }
    const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'].find((m) => MediaRecorder.isTypeSupported(m)) || '';
    rec = new MediaRecorder(recStream, mime ? { mimeType: mime } : undefined);
    recChunks = [];
    rec.ondataavailable = (e) => { if (e.data.size) recChunks.push(e.data); };
    rec.onstop = () => {
      const blob = new Blob(recChunks, { type: rec.mimeType || 'audio/webm' });
      const ext = (rec.mimeType || 'audio/webm').includes('ogg') ? 'ogg' : (rec.mimeType || '').includes('mp4') ? 'm4a' : 'webm';
      setPendingFile(new File([blob], `ghi-am.${ext}`, { type: blob.type }));
      if (recStream) { recStream.getTracks().forEach((t) => t.stop()); recStream = null; }
    };
    rec.start(100);
    const t0 = performance.now();
    $('#recBtn').textContent = '■ DỪNG'; $('#recBtn').classList.add('is-rec');
    recTimer = setInterval(() => {
      const s = (performance.now() - t0) / 1000;
      $('#recTime').textContent = `${s.toFixed(1)}s`;
      if (s >= 20) stopRec();
    }, 100);
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const src = audioCtx.createMediaStreamSource(recStream);
    recAnalyser = audioCtx.createAnalyser(); recAnalyser.fftSize = 512; src.connect(recAnalyser);
    const buf = new Uint8Array(recAnalyser.fftSize);
    const tick = () => {
      recAnalyser.getByteTimeDomainData(buf);
      let sum = 0; for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
      $('#recLevel').style.width = `${Math.min(100, Math.sqrt(sum / buf.length) * 300)}%`;
      recRaf = requestAnimationFrame(tick);
    };
    tick();
  }
  function stopRec() {
    if (recTimer) { clearInterval(recTimer); recTimer = null; }
    if (recRaf) { cancelAnimationFrame(recRaf); recRaf = null; }
    $('#recLevel').style.width = '0%';
    $('#recBtn').textContent = '● GHI ÂM'; $('#recBtn').classList.remove('is-rec');
    if (rec && rec.state !== 'inactive') rec.stop();
    else if (recStream) { recStream.getTracks().forEach((t) => t.stop()); recStream = null; }
    rec = null;
  }
  $('#recBtn').addEventListener('click', () => { if (rec && rec.state === 'recording') stopRec(); else startRec(); });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = $('#vName').value.trim(), label = $('#vLabel').value.trim(), text = $('#vText').value.trim();
    const save = $('#modalSave');
    const needName = modalMode === 'add' || modalMode === 'from-output' || modalMode === 'edit';
    if (needName && !name) { toast('Đặt tên cho giọng đã.', 'warn'); $('#vName').focus(); return; }
    save.disabled = true;
    const prevLabel = save.textContent;
    try {
      let v;
      if (modalMode === 'add' || modalMode === 'add-style') {
        if (!pendingFile) { toast('Chọn file hoặc ghi âm một đoạn mẫu.', 'warn'); return; }
        const fd = new FormData();
        if (modalMode === 'add') fd.append('name', name); else fd.append('label', label || `Sắc thái ${modalCtx.voice.styles.length + 1}`);
        if (modalMode === 'add' && label) fd.append('label', label);
        fd.append('ref_text', text); fd.append('language', state.language); fd.append('file', pendingFile, pendingFile.name);
        if (!text) toast('Whisper đang nghe đoạn mẫu…', 'warn', '…');
        save.textContent = 'ĐANG LƯU…';
        v = await api(modalMode === 'add' ? '/api/voices' : `/api/voices/${modalCtx.voice.id}/styles`, { method: 'POST', body: fd });
      } else if (modalMode === 'edit') {
        await postJSON(`/api/voices/${modalCtx.voice.id}`, { name }, 'PATCH');
        v = await postJSON(`/api/voices/${modalCtx.voice.id}/styles/${modalCtx.style.id}`, { label: label || undefined, ref_text: text || undefined }, 'PATCH');
      } else if (modalMode === 'from-output') {
        save.textContent = 'ĐANG LƯU…';
        v = await postJSON('/api/voices/from-output', { output_id: modalCtx.output.id, name });
      } else {
        save.textContent = 'ĐANG LƯU…';
        v = await postJSON('/api/voices/from-output', { output_id: modalCtx.output.id, voice_id: modalCtx.voice.id, label: label || undefined });
      }
      state.voiceId = v.id;
      if (modalMode.includes('style')) state.styleId = v.styles[v.styles.length - 1].id;
      await loadVoices();
      setMode('clone'); persist(); renderLines();
      closeModal();
      const s = v.styles[v.styles.length - 1];
      toast(`Đã lưu ${modalMode.includes('style') ? `sắc thái "${s.label}"` : `giọng "${v.name}"`}${s.transcribed && modalMode !== 'edit' ? ` — Whisper nghe: “${s.ref_text}”` : ''}`, 'ok');
      if (v.note) toast(v.note, 'warn', 'Ơ…');
    } catch (err) {
      toast(err.message, 'err');
    } finally {
      save.disabled = false; save.textContent = prevLabel;
    }
  });

  // ------------------------------------------------------------------ pronunciation dictionary
  const dict = $('#dictModal');
  let rules = [], defaultRules = [];
  function renderDict() {
    $('#dictRows').innerHTML = rules.map((r, i) => `<tr data-i="${i}">
      <td><input class="input r-from" value="${esc(r.from)}"></td><td><input class="input r-to" value="${esc(r.to)}"></td>
      <td><label class="check check-sm"><input type="checkbox" class="r-on" ${r.enabled !== false ? 'checked' : ''}></label></td>
      <td><button type="button" class="del" title="Xoá">×</button></td></tr>`).join('') || '<tr><td colspan="4" class="hint">Chưa có quy tắc nào.</td></tr>';
  }
  function readDict() {
    rules = $$('#dictRows tr[data-i]').map((tr) => ({ from: $('.r-from', tr).value.trim(), to: $('.r-to', tr).value.trim(), enabled: $('.r-on', tr).checked, whole: true, regex: false })).filter((r) => r.from);
  }
  $('#openDict').addEventListener('click', async () => {
    try { const r = await api('/api/pronunciation'); rules = r.rules; defaultRules = r.defaults; renderDict(); dict.showModal(); }
    catch (e) { toast(e.message, 'err'); }
  });
  $('#dictClose').addEventListener('click', () => dict.close());
  dict.addEventListener('click', (e) => { if (e.target === dict) dict.close(); });
  $('#dictRows').addEventListener('click', (e) => { const b = e.target.closest('.del'); if (!b) return; readDict(); rules.splice(+b.closest('tr').dataset.i, 1); renderDict(); });
  $('#dictAdd').addEventListener('click', () => {
    const from = $('#dictFrom').value.trim(), to = $('#dictTo').value.trim();
    if (!from || !to) { toast('Điền cả hai ô đã.', 'warn'); return; }
    readDict(); rules.push({ from, to, enabled: true, whole: true, regex: false }); renderDict();
    $('#dictFrom').value = ''; $('#dictTo').value = ''; $('#dictFrom').focus();
  });
  $('#dictReset').addEventListener('click', () => { rules = defaultRules.map((r) => ({ ...r })); renderDict(); });
  $('#dictSave').addEventListener('click', async () => {
    readDict();
    try { const r = await postJSON('/api/pronunciation', { rules }, 'PUT'); rules = r.rules; dict.close(); toast(`Đã lưu ${rules.length} quy tắc phát âm`, 'ok'); previewSpoken(); }
    catch (e) { toast(e.message, 'err'); }
  });

  // ------------------------------------------------------------------ boot
  pollStatus();
  loadVoices().then(() => { if (state.scriptMode === 'lines') setScriptMode('lines'); else updateGoLabel(); }).catch((e) => toast(e.message, 'err'));
  loadHistory().then(() => { if (history[0]) showResult(history[0]); }).catch(() => {});
  previewSpoken();
})();
