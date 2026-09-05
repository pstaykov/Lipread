const MODEL_LABELS = {
  video_only: 'Video',
  audio_only: 'Audio',
  multimodal: 'Multimodal',
};

const MODEL_CHANNEL = {
  video_only: 'ch-video',
  audio_only: 'ch-audio',
  multimodal: 'ch-fusion',
};

let DATA = null;
let currentCondition = 'clean';

function pct(x) {
  return (x * 100).toFixed(1) + '%';
}

function markSpan(correct) {
  const cls = correct ? 'ok' : 'bad';
  const glyph = correct ? '✓' : '✗';
  return `<span class="mark ${cls}">${glyph}</span>`;
}

function modelColHtml(key, pred, truth) {
  const top1Prob = pred.top5[0].prob;
  const top5Items = pred.top5.map(item => {
    const hit = item.word === truth ? 'hit' : '';
    return `<li class="${hit}"><span>${item.word}</span><span>${pct(item.prob)}</span></li>`;
  }).join('');
  return `
    <h3>${MODEL_LABELS[key]}</h3>
    <div class="pred-top1">
      ${markSpan(pred.correct)}
      <span>${pred.top1}</span>
      <span class="prob">${pct(top1Prob)}</span>
    </div>
    <div class="bar-track"><div class="bar-fill" style="width:${(top1Prob * 100).toFixed(1)}%"></div></div>
    <div class="top5-status">Top 5: ${markSpan(pred.top5_correct)}</div>
    <ul class="top5">${top5Items}</ul>`;
}

function clipSrc(clip) {
  return `clips/${encodeURIComponent(clip.clip_files[currentCondition])}`;
}

function updateAudioCols(clip) {
  document.getElementById(`col-audio-${clip.id}`).innerHTML =
    modelColHtml('audio_only', clip.audio_only[currentCondition], clip.ground_truth);
  document.getElementById(`col-mm-${clip.id}`).innerHTML =
    modelColHtml('multimodal', clip.multimodal[currentCondition], clip.ground_truth);

  const video = document.getElementById(`video-${clip.id}`);
  const wasPlaying = !video.paused;
  const t = video.currentTime;
  video.src = clipSrc(clip);
  if (t) video.currentTime = t;
  if (wasPlaying) video.play().catch(() => {});
  document.getElementById(`audiotag-${clip.id}`).textContent = conditionLabel(currentCondition);
}

function renderClip(clip) {
  return `
    <article class="clip-card">
      <div>
        <video id="video-${clip.id}" controls autoplay muted loop playsinline preload="none" src="${clipSrc(clip)}"></video>
        <div class="clip-meta">
          <div class="truth">Wort: ${clip.ground_truth}</div>
          <div>${clip.source_file} <span class="audio-tag" id="audiotag-${clip.id}">${conditionLabel(currentCondition)}</span></div>
        </div>
      </div>
      <div class="model-cols">
        <div class="model-col ch-video">${modelColHtml('video_only', clip.video_only, clip.ground_truth)}</div>
        <div class="model-col ch-audio" id="col-audio-${clip.id}"></div>
        <div class="model-col ch-fusion" id="col-mm-${clip.id}"></div>
      </div>
    </article>`;
}

function renderClips() {
  document.getElementById('clips').innerHTML = DATA.clips.map(renderClip).join('');
  DATA.clips.forEach(updateAudioCols);
}

// Confidence -> red (0%) through yellow (50%) to green (100%), fixed saturation/
// lightness so white text stays readable across the whole ramp.
function confColor(t) {
  const hue = t <= 0.5 ? (t / 0.5) * 60 : 60 + ((t - 0.5) / 0.5) * 60;
  return `hsl(${hue.toFixed(0)}, 72%, 38%)`;
}

function heatCell(prob) {
  const pctText = pct(prob);
  return `<div class="hm-cell" style="background:${confColor(prob)}" title="${pctText}">${pctText}</div>`;
}

function renderHeatmap() {
  const rows = DATA.clips.map(clip => `
    <div class="hm-word">${clip.ground_truth}</div>
    ${heatCell(clip.video_only.true_word_prob)}
    ${heatCell(clip.audio_only[currentCondition].true_word_prob)}
    ${heatCell(clip.multimodal[currentCondition].true_word_prob)}
  `).join('');
  document.getElementById('heatmap').innerHTML = `
    <div class="hm-head"></div>
    <div class="hm-head ch-video">Video</div>
    <div class="hm-head ch-audio">Audio</div>
    <div class="hm-head ch-fusion">Multim.</div>
    ${rows}`;
}

function refreshAudioCols() {
  DATA.clips.forEach(updateAudioCols);
  renderHeatmap();
}

function entryFor(clip, key) {
  return key === 'video_only' ? clip.video_only : clip[key][currentCondition];
}

function sampleStats(key) {
  const n = DATA.clips.length;
  return {
    top1: DATA.clips.filter(c => entryFor(c, key).correct).length / n,
    top5: DATA.clips.filter(c => entryFor(c, key).top5_correct).length / n,
  };
}

function mtCell(x) {
  return x == null ? '<td class="mt-dash">–</td>' : `<td>${pct(x)}</td>`;
}

function renderMetricsTable() {
  const h = DATA.model_headline_accuracy;
  const rows = [
    { key: 'video_only', full: { top1: h.video_only.test_top1, top5: h.video_only.test_top5 } },
    { key: 'audio_only', full: { top1: h.audio_only.val_top1, top5: h.audio_only.val_top5 } },
    { key: 'multimodal', full: { top1: h.multimodal.test_top1, top5: null } },
  ];

  document.getElementById('metrics-tbody').innerHTML = rows.map(r => {
    const s = sampleStats(r.key);
    return `
      <tr>
        <td class="mt-model"><span class="ch-dot ${MODEL_CHANNEL[r.key]}"></span>${MODEL_LABELS[r.key]}</td>
        ${mtCell(r.full.top1)}${mtCell(r.full.top5)}${mtCell(s.top1)}${mtCell(s.top5)}
      </tr>`;
  }).join('');

  document.getElementById('mt-sample-head').textContent =
    `${DATA.clips.length} Clips (${conditionLabel(currentCondition)})`;
}

function conditionLabel(key) {
  const c = DATA.noise_conditions.find(c => c.key === key);
  return c ? c.label : key;
}

function setupNoisePicker() {
  const select = document.getElementById('noise-select');
  select.innerHTML = DATA.noise_conditions.map(
    c => `<option value="${c.key}">${c.label}</option>`
  ).join('');
  select.value = currentCondition;
  select.addEventListener('change', () => {
    currentCondition = select.value;
    refreshAudioCols();
    renderMetricsTable();
  });
}

// Live self-recording (needs server.py, not the plain static serve.py)
const RECORD_MS = 1800;
let ALL_CLASSES = [];
let camStream = null;

function pickTargetWord() {
  const word = ALL_CLASSES[Math.floor(Math.random() * ALL_CLASSES.length)];
  document.getElementById('target-word').textContent = word;
}

function pickMimeType() {
  const candidates = ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm'];
  return candidates.find(c => window.MediaRecorder && MediaRecorder.isTypeSupported(c)) || '';
}

async function ensureCamera() {
  if (camStream) return camStream;
  camStream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 }, audio: true });
  document.getElementById('cam-preview').srcObject = camStream;
  return camStream;
}

function renderRecordResult(data) {
  document.getElementById('result-video-skeleton').style.display = 'none';
  const preview = document.getElementById('result-preview');
  preview.style.display = 'block';
  preview.src = `clips/${encodeURIComponent(data.clip_file)}`;
  const cols = ['video_only', 'audio_only', 'multimodal']
    .map(k => `<div class="model-col ${MODEL_CHANNEL[k]}">${modelColHtml(k, data[k], data.target)}</div>`).join('');
  document.getElementById('record-cols').innerHTML = cols;

  const status = document.getElementById('record-status');
  const condNote = data.condition === 'clean' ? '' : ` — Audio: ${conditionLabel(data.condition)}`;
  if (data.detect_rate < 0.5) {
    status.textContent = `Gesicht nur in ${pct(data.detect_rate)} der Frames erkannt — Ergebnis evtl. unzuverlässig. Näher an die Kamera?${condNote}`;
    status.classList.add('warn');
  } else {
    status.textContent = `Fertig (Gesicht erkannt: ${pct(data.detect_rate)} der Frames).${condNote}`;
    status.classList.remove('warn');
  }
}

async function recordAndSend() {
  const btn = document.getElementById('record-btn');
  const status = document.getElementById('record-status');
  btn.disabled = true;
  status.classList.remove('warn');

  let stream;
  try {
    stream = await ensureCamera();
  } catch (err) {
    status.textContent = 'Kein Kamerazugriff — bitte in den Browsereinstellungen erlauben.';
    btn.disabled = false;
    return;
  }

  const mimeType = pickMimeType();
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  const chunks = [];
  recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };

  const countdownEl = document.getElementById('record-countdown');
  const viewfinder = document.querySelector('.viewfinder');
  for (let i = 3; i > 0; i--) {
    countdownEl.textContent = i;
    countdownEl.classList.add('show');
    status.textContent = `Start in ${i}…`;
    await new Promise(r => setTimeout(r, 700));
    countdownEl.classList.remove('show');
  }

  const recDot = document.getElementById('rec-dot');
  const stopped = new Promise(resolve => { recorder.onstop = resolve; });

  viewfinder.classList.add('recording');
  recorder.start();
  recDot.classList.add('rec');
  status.textContent = `Aufnahme läuft (${(RECORD_MS / 1000).toFixed(1)}s) — jetzt sprechen…`;
  setTimeout(() => recorder.stop(), RECORD_MS);
  await stopped;
  recDot.classList.remove('rec');
  viewfinder.classList.remove('recording');

  status.textContent = 'Wird ausgewertet…';
  const blob = new Blob(chunks, { type: mimeType || 'video/webm' });
  const form = new FormData();
  form.append('video', blob, 'recording.webm');
  form.append('target', document.getElementById('target-word').textContent);
  form.append('condition', currentCondition);

  try {
    const res = await fetch('/api/record', { method: 'POST', body: form });
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    renderRecordResult(await res.json());
  } catch (err) {
    status.textContent = `Fehler: ${err.message}`;
    status.classList.add('warn');
  } finally {
    btn.disabled = false;
  }
}

async function setupRecording() {
  const statusDot = document.getElementById('status-dot');
  const statusText = document.getElementById('status-text');
  try {
    const res = await fetch('/api/classes');
    if (!res.ok) throw new Error();
    ALL_CLASSES = await res.json();
  } catch {
    statusText.textContent = 'nur Beispielclips (server.py nicht aktiv)';
    return; // server.py not running (e.g. plain serve.py) -- record section stays hidden
  }
  statusDot.classList.add('live');
  statusText.textContent = 'live-aufnahme bereit';
  document.getElementById('record-section').hidden = false;
  pickTargetWord();
  document.getElementById('shuffle-word').addEventListener('click', pickTargetWord);
  document.getElementById('record-btn').addEventListener('click', recordAndSend);
}

async function main() {
  const res = await fetch('predictions.json');
  DATA = await res.json();

  setupNoisePicker();
  renderMetricsTable();
  renderHeatmap();
  renderClips();
  setupRecording();
}

main().catch(err => {
  document.getElementById('clips').textContent =
    'predictions.json konnte nicht geladen werden — Ordner per HTTP-Server öffnen (z. B. `python -m http.server`), nicht als file://.';
  console.error(err);
});
