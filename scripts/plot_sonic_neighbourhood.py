from pathlib import Path
import json, os, plistlib, re, colorsys, socket, subprocess, sys
from urllib.parse import urlparse, unquote
import numpy as np
import faiss
import hashlib
import plotly.graph_objects as go
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

LABELS_FILE = DATA / "music_clusters_raw_clap_12.npy"
EMBEDDINGS_FILE = DATA / "embeddings.npy"
MAPPING_FILE = DATA / "embedding_mapping.txt"
ITUNES_XML = Path(r"M:\Music\iTunes Library.xml")
COORDS_FILE = DATA / "music_3d_pca_direct.npy"
OUTPUT = DATA / "music_map_raw_clap_12_pca_direct_genre_ollama_verbose.html"
TEMP_OUTPUT = DATA / "music_map_raw_clap_12_pca_direct_genre_ollama_verbose.building.html"

LIVE_RE = re.compile(r"\blive\b|livealbum|concert|unplugged|bootleg|\bin concert\b|\blive at\b|\blive in\b", re.I)
MUSIC_PREFIX = "M:\\Music\\Music\\"
SONIC_NEIGHBOURS = 16   # CLAP neighbours stored per track; 8 are shown, extras replace any '+' folders you exclude
SONIC_GRAPH_FILE = DATA / "sonic_neighbourhood.npz"
PLUS_DIR_RE = re.compile(r"\+[\\/]")   # a folder name ending in '+', e.g. ...\Album+\Track.m4a

# --- Optional CLAP mood/sound tags (made once by make_clap_tags.py) ----------
TAGS_FILE = DATA / "clap_tag_scores.npy"
TAG_NAMES_FILE = DATA / "clap_tag_names.json"
TAG_RULES = {"mood": (2, 0.75), "sound": (2, 0.75), "voice": (1, 0.5)}   # group: (max tags shown, minimum z-score)
# --- Optional tags / audio features you already have in data\ -----------------
MUSICNN_FILE = DATA / "musicnn_tags_progress.jsonl"   # one {"path": ..., "tags": [...]} per line
MUSICNN_KEEP = 4                                      # musicnn tags kept per track
FEATURES_FILE = DATA / "features_60s.json"            # tempo / key / energy per track

# --- Playback options -------------------------------------------------------
PLAYER_HELPER = True   # True: "Play in default app" button, and play_server.py is launched for you
INLINE_AUDIO = False   # True: also show the browser's own audio player in the click panel
HELPER_SCRIPT = Path(__file__).resolve().parent / "play_server.py"
HELPER_PORT = 8765

def start_helper():
    if not PLAYER_HELPER:
        return
    with socket.socket() as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", HELPER_PORT)) == 0:
            print("Player helper already running.", flush=True)
            return
    if not HELPER_SCRIPT.is_file():
        print(f"Player helper not started: {HELPER_SCRIPT} not found.", flush=True)
        return
    flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    subprocess.Popen([sys.executable, str(HELPER_SCRIPT)], creationflags=flags, cwd=str(HELPER_SCRIPT.parent))
    print("Player helper started in its own window (close that window to stop it).", flush=True)

def norm_path(value):
    return os.path.normcase(os.path.normpath(str(value).strip()))

def itunes_location(location):
    if not location: return ""
    parsed = urlparse(location)
    if parsed.scheme.lower() == "file":
        path = unquote(parsed.path)
        if re.match(r"^/[A-Za-z]:", path): path = path[1:]
        if parsed.netloc and parsed.netloc.lower() != "localhost": path = "//" + parsed.netloc + path
        return path.replace("/", "\\")
    return location

def load_itunes():
    if not ITUNES_XML.is_file():
        print(f"iTunes XML not found: {ITUNES_XML}")
        return {}
    print(f"Reading iTunes metadata: {ITUNES_XML}", flush=True)
    try:
        with ITUNES_XML.open("rb") as f: library = plistlib.load(f)
    except Exception as exc:
        print(f"Standard plist/XML parsing failed: {exc}", flush=True)
        try: from lxml import etree
        except ImportError as e: raise RuntimeError("The iTunes XML is malformed and lxml is not installed. Install it in the musicrec environment. No files were changed.") from e
        print("Trying recovery parsing; source XML will not be modified...", flush=True)
        parser = etree.XMLParser(recover=True, huge_tree=True)
        tree = etree.parse(str(ITUNES_XML), parser)
        library = plistlib.loads(etree.tostring(tree, encoding="utf-8", xml_declaration=True))
        if parser.error_log: print(f"XML recovery reported {len(parser.error_log)} issue(s); metadata may be incomplete.", flush=True)
    result = {}
    for track in library.get("Tracks", {}).values():
        location = itunes_location(track.get("Location", ""))
        if location:
            result[norm_path(location)] = {"grouping": str(track.get("Grouping", "") or ""), "name": str(track.get("Name", "") or ""), "album": str(track.get("Album", "") or ""), "genre": str(track.get("Genre", "") or "").strip() or "Unknown / untagged"}
    print(f"iTunes tracks indexed: {len(result):,}", flush=True)
    return result

def classify(path, metadata):
    info = metadata.get(norm_path(path), {})
    text = " | ".join((path, info.get("name", ""), info.get("album", ""), info.get("grouping", "")))
    if LIVE_RE.search(text): return "Live / LiveAlbum"
    if re.search(r"\bstudio\b", info.get("grouping", ""), re.I): return "Studio"
    return "Other / unclassified"

def path_key(p):
    """Match paths from different files regardless of slash direction or case."""
    return str(p).replace("/", "\\").casefold()

def load_musicnn_tags(paths):
    """Per-track musicnn tags from the progress file (optional). Unreadable lines and unknown paths are skipped."""
    if not MUSICNN_FILE.is_file():
        print("musicnn tag file not found; skipping (optional).", flush=True)
        return [], [], []
    index = {path_key(p): i for i, p in enumerate(paths)}
    found = [None] * len(paths)
    bad = unmatched = 0
    with MUSICNN_FILE.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad += 1
                continue
            i = index.get(path_key(rec.get("path", ""))) if isinstance(rec, dict) else None
            if i is None:
                unmatched += 1
                continue
            tags = [str(t).strip() for t in rec.get("tags", []) if str(t).strip()][:MUSICNN_KEEP]
            if tags:
                found[i] = tags
    names = sorted({t for tags in found if tags for t in tags}, key=lambda s: s.casefold())
    ids = {t: j for j, t in enumerate(names)}
    track_tags = [[ids[t] for t in tags] if tags else [] for tags in found]
    print(f"musicnn tags: {sum(1 for t in track_tags if t):,} of {len(paths):,} tracks matched ({len(names)} different tags; {bad} unreadable lines; {unmatched:,} lines for tracks not in the library).", flush=True)
    return names, ["musicnn"] * len(names), track_tags

def merge_tag_sources(n, sources):
    """Combine tag tables from several files into one (names, groups, per-track tag ids)."""
    names, groups, merged = [], [], [[] for _ in range(n)]
    for s_names, s_groups, s_tracks in sources:
        if not s_names:
            continue
        offset = len(names)
        names += s_names
        groups += s_groups
        for i, tags in enumerate(s_tracks):
            merged[i].extend(offset + t for t in tags)
    return names, groups, merged

def load_audio_features(paths):
    """Tempo, key and an energy bucket (0 low / 1 mid / 2 high) per track from features_60s.json (optional)."""
    if not FEATURES_FILE.is_file():
        print("features_60s.json not found; tempo/key/energy will not be shown (optional).", flush=True)
        return [], [], [], []
    try:
        data = json.loads(FEATURES_FILE.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"features_60s.json could not be read ({exc}); tempo/key/energy skipped.", flush=True)
        return [], [], [], []
    n = len(paths)
    index = {path_key(p): i for i, p in enumerate(paths)}
    bpm = [0] * n
    key_ids, key_names = [-1] * n, []
    energy_raw = np.full(n, np.nan, dtype=np.float64)
    items = data.items() if isinstance(data, dict) else ((None, r) for r in data)
    matched = 0
    for k, rec in items:
        if not isinstance(rec, dict):
            continue
        i = index.get(path_key(rec.get("path") or k or ""))
        if i is None:
            continue
        matched += 1
        t = rec.get("tempo")
        if isinstance(t, (int, float)) and np.isfinite(t) and t > 0:
            bpm[i] = int(round(t))
        e = rec.get("energy")
        if isinstance(e, (int, float)) and np.isfinite(e):
            energy_raw[i] = float(e)
        key = rec.get("key")
        if isinstance(key, str) and key.strip():
            key = key.strip()
            if key not in key_names:
                key_names.append(key)
            key_ids[i] = key_names.index(key)
    have = np.isfinite(energy_raw)
    energy_bucket = [-1] * n
    if have.any():
        lo, hi = np.quantile(energy_raw[have], [1 / 3, 2 / 3])
        for i in np.flatnonzero(have):
            v = energy_raw[i]
            energy_bucket[int(i)] = 0 if v <= lo else (1 if v <= hi else 2)
    print(f"features_60s.json: {matched:,} of {n:,} tracks matched; tempo for {sum(1 for b in bpm if b):,}, key for {sum(1 for k in key_ids if k >= 0):,}.", flush=True)
    return bpm, key_names, key_ids, energy_bucket

def load_clap_tags(n):
    """Pick each track's strongest CLAP mood/sound/voice tags from the files made by make_clap_tags.py (optional)."""
    if not (TAGS_FILE.is_file() and TAG_NAMES_FILE.is_file()):
        print("CLAP tag files not found; run make_clap_tags.py to add mood/sound tags (optional).", flush=True)
        return [], [], []
    scores = np.load(TAGS_FILE, allow_pickle=False).astype(np.float32)
    meta = json.loads(TAG_NAMES_FILE.read_text(encoding="utf-8"))
    if scores.ndim != 2 or scores.shape != (n, len(meta)):
        print(f"CLAP tag file is {scores.shape} but the library has {n} tracks x {len(meta)} tags; tags skipped. Re-run make_clap_tags.py.", flush=True)
        return [], [], []
    names = [m["name"] for m in meta]
    groups = [m["group"] for m in meta]
    track_tags = [[] for _ in range(n)]
    for group, (keep, floor) in TAG_RULES.items():
        ids = np.array([j for j, g in enumerate(groups) if g == group], dtype=int)
        if not len(ids):
            continue
        sub = scores[:, ids]
        order = np.argsort(-sub, axis=1)[:, :keep]
        for row in range(n):
            for k in order[row]:
                if sub[row, k] >= floor:
                    track_tags[row].append(int(ids[k]))
    print(f"CLAP tags loaded: {len(names)} tags; {sum(1 for t in track_tags if t):,} tracks have at least one.", flush=True)
    return names, groups, track_tags

def save_sonic_neighbourhood(neighbour_ids, neighbour_scores, paths):
    """Persist the raw-512D CLAP graph for playlist generation."""
    ids = np.full((len(paths), SONIC_NEIGHBOURS), -1, dtype=np.int32)
    scores = np.full((len(paths), SONIC_NEIGHBOURS), np.nan, dtype=np.float32)
    for row, (row_ids, row_scores) in enumerate(zip(neighbour_ids, neighbour_scores)):
        count = min(SONIC_NEIGHBOURS, len(row_ids), len(row_scores))
        if count:
            ids[row, :count] = np.asarray(row_ids[:count], dtype=np.int32)
            scores[row, :count] = np.asarray(row_scores[:count], dtype=np.float32)

    mapping_json = json.dumps(paths, ensure_ascii=False, separators=(",", ":"))
    mapping_hash = hashlib.sha256(mapping_json.encode("utf-8")).hexdigest()
    temp_file = SONIC_GRAPH_FILE.with_name(SONIC_GRAPH_FILE.stem + ".building.npz")
    np.savez_compressed(
        temp_file, neighbour_ids=ids, neighbour_scores=scores,
        mapping_hash=np.asarray(mapping_hash),
        row_count=np.asarray(len(paths), dtype=np.int64),
        dimensions=np.asarray(512, dtype=np.int32),
        neighbour_count=np.asarray(SONIC_NEIGHBOURS, dtype=np.int32),
        method=np.asarray("normalised 512-D CLAP cosine via HNSW inner product"),
    )
    os.replace(temp_file, SONIC_GRAPH_FILE)
    print(f"Saved reusable sonic-neighbourhood graph: {SONIC_GRAPH_FILE}", flush=True)


def build_clap_context(embeddings, labels, genre_names):
    """Precompute full-512D CLAP neighbours, cluster-centre scores and genre summaries."""
    print("Building full-512D CLAP neighbour index (one-time step)...", flush=True)
    vectors = np.array(embeddings, dtype=np.float32, copy=True)
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 120
    index.hnsw.efSearch = 64
    index.add(vectors)
    distances, indices = index.search(vectors, SONIC_NEIGHBOURS + 1)  # self + nearest CLAP neighbours
    neighbour_ids, neighbour_scores = [], []
    for row, (row_ids, row_scores) in enumerate(zip(indices, distances)):
        pairs = [(int(i), float(sc)) for i, sc in zip(row_ids, row_scores) if i >= 0 and int(i) != row]
        pairs = pairs[:SONIC_NEIGHBOURS]
        neighbour_ids.append([i for i, _ in pairs])
        neighbour_scores.append([float(score) for _, score in pairs])
    print("Calculating similarities to the 12 CLAP cluster centres...", flush=True)
    centroids = []
    for cluster_id in range(12):
        members = vectors[labels == cluster_id]
        centre = members.mean(axis=0)
        norm = np.linalg.norm(centre)
        if norm:
            centre /= norm
        centroids.append(centre)
    centroid_scores = (vectors @ np.asarray(centroids, dtype=np.float32).T).astype(np.float32)
    cluster_genres = []
    for cluster_id in range(12):
        counts = {}
        for i in np.flatnonzero(labels == cluster_id):
            name = genre_names[int(i)]
            counts[name] = counts.get(name, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold()))[:5]
        cluster_genres.append([{"name": name, "count": count} for name, count in ranked])
    print("Full-CLAP sonic context ready.", flush=True)
    # float64 + rounding keeps the page payload small (float32.tolist() writes ~19 digits per value).
    return neighbour_ids, neighbour_scores, np.round(centroid_scores.astype(np.float64), 4).tolist(), cluster_genres

SEARCH_HTML = '<style>#pfSearchInput:focus{border-color:#6cb6ff!important}#pfSearchGrip:hover{color:#6cb6ff!important}</style><div id="pfSearch" style="position:absolute;top:10px;left:50%;transform:translateX(-50%);width:clamp(220px,36vw,460px);z-index:10001;font:14px Arial,sans-serif;color:#eee"><div style="position:relative"><span id="pfSearchGrip" title="Drag to move (double-click to reset)" style="position:absolute;left:0;top:0;bottom:0;width:24px;display:flex;align-items:center;justify-content:center;cursor:move;color:#888;font-size:15px;line-height:1;user-select:none;touch-action:none">&#8942;&#8942;</span><input id="pfSearchInput" type="text" placeholder="Search artist, album or track…   ( / )" autocomplete="off" spellcheck="false" style="box-sizing:border-box;width:100%;padding:8px 32px 8px 26px;background:#20242a;color:#eee;border:1px solid #777;border-radius:8px;font:14px Arial,sans-serif;outline:none;box-shadow:0 2px 10px rgba(0,0,0,.5)"><span id="pfSearchClear" title="Clear search" style="display:none;position:absolute;right:10px;top:50%;transform:translateY(-50%);cursor:pointer;color:#aaa;font-size:18px;line-height:1">&times;</span></div><div id="pfSearchDrop" style="display:none;margin-top:4px;background:#1e1e1e;border:1px solid #777;border-radius:8px;box-shadow:0 4px 18px rgba(0,0,0,.65);overflow:hidden"><ul id="pfSearchList" style="list-style:none;margin:0;padding:4px;max-height:min(55vh,460px);overflow-y:auto"></ul><div style="display:flex;justify-content:space-between;align-items:center;gap:10px;padding:6px 10px;border-top:1px solid #444;font-size:11px;color:#aaa"><span id="pfSearchInfo"></span><label style="white-space:nowrap;cursor:pointer"><input type="checkbox" id="pfSearchCentre" checked> Centre map</label></div></div></div>'
SEARCH_JS = r'''// ---- Track search (top centre) ----
(function () {
 const wrap = document.getElementById('pfSearch');
 const input = document.getElementById('pfSearchInput');
 const clearBtn = document.getElementById('pfSearchClear');
 const drop = document.getElementById('pfSearchDrop');
 const list = document.getElementById('pfSearchList');
 const info = document.getElementById('pfSearchInfo');
 const centreBox = document.getElementById('pfSearchCentre');
 const MAX_SHOWN = 20;
 let index = null, results = [], active = -1, timer = null, markerShown = false, centred = false;

 // Case- and accent-insensitive: "Beyonce" finds "Beyoncé".
 const fold = s => String(s).normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();

 function buildIndex() {
  if (index) return;
  const n = DATA.paths.length;
  index = {title: new Array(n), artist: new Array(n), album: new Array(n)};
  for (let i = 0; i < n; i++) {
   const t = trackParts(i);
   index.title[i] = fold(t.name);
   index.artist[i] = fold(t.artist);
   index.album[i] = fold(t.album);
  }
 }
 // Word-start hits score higher than hits in the middle of a word.
 function fieldScore(text, tok, atWord, inside) {
  const p = text.indexOf(tok);
  if (p < 0) return 0;
  return (p === 0 || text[p - 1] === ' ') ? atWord : inside;
 }
 function search(query) {
  const q = fold(query).trim();
  if (q.length < 2) return null;
  buildIndex();
  const toks = q.split(/\s+/);
  const cmp = (a, b) => b.score - a.score || index.title[a.i].length - index.title[b.i].length || a.i - b.i;
  let best = [], total = 0;
  for (let i = 0, n = DATA.paths.length; i < n; i++) {
   const ti = index.title[i], ar = index.artist[i], al = index.album[i];
   let score = 0, ok = true;
   for (let k = 0; k < toks.length; k++) {
    const s = fieldScore(ti, toks[k], 6, 3) + fieldScore(ar, toks[k], 4, 2) + fieldScore(al, toks[k], 2, 1);
    if (!s) { ok = false; break; }
    score += s;
   }
   if (!ok) continue;
   if (ti === q) score += 10;
   total++;
   best.push({i, score});
   if (best.length > 400) { best.sort(cmp); best.length = 60; }
  }
  best.sort(cmp);
  return {items: best.slice(0, MAX_SHOWN), total};
 }

 function setActive(k) {
  const rows = list.children;
  if (active >= 0 && rows[active]) rows[active].style.background = '';
  active = k;
  if (k >= 0 && rows[k]) { rows[k].style.background = '#34404f'; rows[k].scrollIntoView({block: 'nearest'}); }
 }
 function render(res, q) {
  list.innerHTML = '';
  active = -1;
  results = res ? res.items : [];
  if (!res) { drop.style.display = 'none'; return; }
  drop.style.display = 'block';
  if (!res.items.length) info.textContent = 'No tracks match \u201c' + q.trim() + '\u201d.';
  else info.textContent = res.total > res.items.length
   ? 'Showing ' + res.items.length + ' of ' + res.total.toLocaleString() + ' matches \u2014 keep typing to narrow down'
   : res.total + (res.total === 1 ? ' match' : ' matches');
  res.items.forEach(({i}, k) => {
   const t = trackParts(i);
   const hidden = !visibleTrackSet.has(i);
   const li = document.createElement('li');
   li.style.cssText = 'padding:5px 8px;margin:2px 0;border-left:4px solid ' + DATA.colours[DATA.labels[i]] + ';cursor:pointer;border-radius:3px;word-break:break-word';
   li.innerHTML = '<div style="font-weight:600">' + esc(t.name) + '</div><div style="font-size:11px;color:#aaa">'
    + esc(t.artist) + (t.album ? ' \u2014 ' + esc(t.album) : '') + ' \u00b7 ' + clusterTag(i)
    + (hidden ? ' \u00b7 <span style="color:#e8b86d">hidden by current filters</span>' : '') + '</div>';
   li.addEventListener('mousedown', e => e.preventDefault());
   li.addEventListener('mousemove', () => { if (active !== k) setActive(k); });
   li.addEventListener('click', () => choose(i));
   list.appendChild(li);
  });
 }
 function go() {
  timer = null;
  render(search(input.value), input.value);
 }

 // Move the 3D camera's look-at point onto a track (camera centre is in the scene's aspect-ratio-scaled units).
 function centreOn(i) {
  try {
   const s = plot._fullLayout && plot._fullLayout.scene;
   if (!s) return;
   const ar = s.aspectratio || {x: 1, y: 1, z: 1};
   const f = (v, ax, a) => { const r = ax && ax.range; return r && r[1] !== r[0] ? (v - (r[0] + r[1]) / 2) / (r[1] - r[0]) * a : 0; };
   centred = true;
   return Plotly.relayout(plot, {'scene.camera.center': {x: f(DATA.x[i], s.xaxis, ar.x), y: f(DATA.y[i], s.yaxis, ar.y), z: f(DATA.z[i], s.zaxis, ar.z)}});
  } catch (err) { console.log('Could not centre the map:', err); }
 }
 function mark(i) {
  markerShown = true;
  return Plotly.restyle(plot, {x: [[DATA.x[i]]], y: [[DATA.y[i]]], z: [[DATA.z[i]]], customdata: [[DATA.paths[i]]]}, [3])
   .then(() => { if (centreBox.checked) return centreOn(i); })
   .catch(err => { status.textContent = 'Could not mark the search result: ' + err; });
 }
 function choose(i) {
  const t = trackParts(i);
  input.value = t.name + (t.artist ? ' \u2014 ' + t.artist : '');
  clearBtn.style.display = 'block';
  drop.style.display = 'none';
  results = [];
  showPanel({customdata: DATA.paths[i]});   // opens the track panel and, if enabled, the sonic neighbours
  mark(i);
  if (!visibleTrackSet.has(i)) status.textContent = 'That track is hidden by the current filters; it is shown on the map as a yellow diamond.';
 }
 function clearSearch() {
  if (timer) { clearTimeout(timer); timer = null; }
  input.value = '';
  clearBtn.style.display = 'none';
  drop.style.display = 'none';
  results = [];
  if (markerShown) { markerShown = false; Plotly.restyle(plot, {x: [[]], y: [[]], z: [[]], customdata: [[]]}, [3]); }
  if (centred) { centred = false; Plotly.relayout(plot, {'scene.camera.center': {x: 0, y: 0, z: 0}}); }
  input.focus();
 }

 input.addEventListener('input', () => {
  clearBtn.style.display = input.value ? 'block' : 'none';
  if (timer) clearTimeout(timer);
  timer = setTimeout(go, 120);
 });
 input.addEventListener('focus', () => {
  buildIndex();
  if (results.length && input.value) drop.style.display = 'block';
 });
 input.addEventListener('keydown', e => {
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
   if (drop.style.display === 'none' && input.value) go();
   if (!results.length) return;
   e.preventDefault();
   const n = results.length;
   setActive(e.key === 'ArrowDown' ? (active + 1) % n : (active - 1 + n) % n);
  } else if (e.key === 'Enter') {
   if (timer) { clearTimeout(timer); go(); }
   if (!results.length) return;
   e.preventDefault();
   choose(results[active >= 0 ? active : 0].i);
  } else if (e.key === 'Escape') {
   if (drop.style.display !== 'none') drop.style.display = 'none'; else input.blur();
  }
 });

 // Drag the box by its grip. The first drag swaps the centred layout for explicit pixel positions.
 const grip = document.getElementById('pfSearchGrip');
 let dragFrom = null;
 function clampBox() {
  const par = wrap.offsetParent;
  if (!par || wrap.style.transform !== 'none') return;
  const maxL = Math.max(0, par.clientWidth - wrap.offsetWidth), maxT = Math.max(0, par.clientHeight - 40);
  wrap.style.left = Math.min(maxL, Math.max(0, wrap.offsetLeft)) + 'px';
  wrap.style.top = Math.min(maxT, Math.max(0, wrap.offsetTop)) + 'px';
 }
 grip.addEventListener('pointerdown', e => {
  const par = wrap.offsetParent;
  if (!par) return;
  const r = wrap.getBoundingClientRect(), pr = par.getBoundingClientRect();
  wrap.style.transform = 'none';
  wrap.style.width = r.width + 'px';
  wrap.style.left = (r.left - pr.left) + 'px';
  wrap.style.top = (r.top - pr.top) + 'px';
  dragFrom = {x: e.clientX, y: e.clientY, left: r.left - pr.left, top: r.top - pr.top};
  grip.setPointerCapture(e.pointerId);
  e.preventDefault();
 });
 grip.addEventListener('pointermove', e => {
  if (!dragFrom) return;
  const par = wrap.offsetParent;
  const maxL = Math.max(0, par.clientWidth - wrap.offsetWidth), maxT = Math.max(0, par.clientHeight - 40);
  wrap.style.left = Math.min(maxL, Math.max(0, dragFrom.left + e.clientX - dragFrom.x)) + 'px';
  wrap.style.top = Math.min(maxT, Math.max(0, dragFrom.top + e.clientY - dragFrom.y)) + 'px';
 });
 grip.addEventListener('pointerup', () => { dragFrom = null; });
 grip.addEventListener('pointercancel', () => { dragFrom = null; });
 grip.addEventListener('dblclick', () => {
  wrap.style.transform = 'translateX(-50%)';
  wrap.style.left = '50%';
  wrap.style.top = '10px';
  wrap.style.width = '';
 });
 window.addEventListener('resize', clampBox);
 clearBtn.addEventListener('click', clearSearch);
 // Click anywhere outside the search box closes the result list.
 document.addEventListener('mousedown', e => { if (!wrap.contains(e.target)) drop.style.display = 'none'; }, true);
 // "/" or Ctrl/Cmd+K jumps to the search box.
 document.addEventListener('keydown', e => {
  const tag = (document.activeElement && document.activeElement.tagName) || '';
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(tag);
  if ((e.key === '/' && !typing && !e.ctrlKey && !e.metaKey && !e.altKey) || ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k')) {
   e.preventDefault();
   input.focus();
   input.select();
  }
 });
})();
'''

def main():
    print("Loading existing coordinates, cluster labels and paths...", flush=True)
    coords = np.load(COORDS_FILE, allow_pickle=False)
    labels = np.load(LABELS_FILE, allow_pickle=False).astype(int)
    embeddings = np.load(EMBEDDINGS_FILE, mmap_mode="r", allow_pickle=False)
    paths = json.loads(MAPPING_FILE.read_text(encoding="utf-8"))
    n = len(paths)
    if coords.shape != (n, 3) or labels.shape != (n,) or embeddings.shape != (n, 512): raise ValueError(f"Data shape mismatch: coords={coords.shape}, labels={labels.shape}, embeddings={embeddings.shape}, paths={n}")
    if not np.isfinite(coords).all() or labels.min() < 0 or labels.max() >= 12: raise ValueError("Invalid coordinates or cluster labels.")
    # Precompute nearest neighbours once. Browser clicks then perform a simple lookup
    # instead of scanning and sorting the entire 70k-track library every time.
    print("Building 3D sonic-neighbour index (one-time step)...", flush=True)
    tree = cKDTree(np.asarray(coords, dtype=np.float32))
    _, queried_ids = tree.query(np.asarray(coords, dtype=np.float32), k=min(14, n), workers=-1)
    queried_ids = np.asarray(queried_ids)
    if queried_ids.ndim == 1:
        queried_ids = queried_ids[:, None]
    neighbour_ids = np.empty((n, min(12, max(0, n - 1))), dtype=np.int32)
    for row in range(n):
        candidates = queried_ids[row]
        candidates = candidates[candidates != row][:neighbour_ids.shape[1]]
        neighbour_ids[row, :len(candidates)] = candidates
    print(f"Nearest-neighbour table ready: {neighbour_ids.shape[0]:,} tracks × {neighbour_ids.shape[1]} neighbours.", flush=True)
    metadata = load_itunes()
    type_names = ["Studio", "Live / LiveAlbum", "Other / unclassified"]
    type_codes = {name: i for i, name in enumerate(type_names)}
    type_ids = [type_codes[classify(path, metadata)] for path in paths]
    plus = ["+" in path for path in paths]
    plus_dir = [bool(PLUS_DIR_RE.search(path)) for path in paths]
    print(f"Tracks in '+' folders: {sum(plus_dir):,}", flush=True)
    tag_names, tag_groups, track_tags = merge_tag_sources(n, [load_musicnn_tags(paths), load_clap_tags(n)])
    bpm, key_names, track_key, energy_bucket = load_audio_features(paths)
    genre_names = [metadata.get(norm_path(path), {}).get("genre", "Unknown / untagged") or "Unknown / untagged" for path in paths]
    neighbour_ids, neighbour_scores, centroid_scores, cluster_genres = build_clap_context(embeddings, labels, genre_names)
    save_sonic_neighbourhood(neighbour_ids, neighbour_scores, paths)
    genre_options = sorted((name for name in set(genre_names) if not name.endswith("+")), key=lambda value: (value.casefold() != "classical", value.casefold()))
    print(f"Genres available for filtering: {len(genre_options):,}", flush=True)
    initial = [i for i in range(n) if type_ids[i] == 0 and labels[i] == 0]
    print(f"Total rows: {n:,}; initial Studio + Cluster 00 points: {len(initial):,}", flush=True)
    # Stable, visually separated colours for all 12 clusters.
    cluster_colours = ["#%02x%02x%02x" % tuple(round(c * 255) for c in colorsys.hsv_to_rgb(i / 12.0, 0.72, 0.98)) for i in range(12)]
    relative_paths = []
    for path in paths:
        relative_paths.append(path[len(MUSIC_PREFIX):] if path.lower().startswith(MUSIC_PREFIX.lower()) else path)
    initial_colours = [cluster_colours[int(labels[i])] for i in initial]
    # Per-cluster distance from the coordinate-wise median, used only for an optional visual filter.
    outlier_distance = np.zeros(n, dtype=np.float32)
    outlier_cutoffs = {}
    for cluster_id in range(12):
        member_ids = np.flatnonzero(labels == cluster_id)
        if len(member_ids) == 0:
            outlier_cutoffs[str(cluster_id)] = {"99": 0, "97.5": 0, "95": 0}
            continue
        centre = np.median(coords[member_ids], axis=0)
        distances = np.linalg.norm(coords[member_ids] - centre, axis=1)
        outlier_distance[member_ids] = distances
        outlier_cutoffs[str(cluster_id)] = {str(q): float(np.percentile(distances, q)) for q in (99, 97.5, 95)}
    fig = go.Figure(data=[go.Scatter3d(x=coords[initial,0], y=coords[initial,1], z=coords[initial,2], mode="markers", name="Tracks", marker=dict(size=3, color=initial_colours, opacity=0.9), customdata=np.asarray(relative_paths, dtype=object)[initial], hovertemplate="<b>%{customdata}</b><extra></extra>"), go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", name="Playlist route", showlegend=False, line=dict(width=5, color=[], colorscale="Turbo", cmin=0, cmax=1), marker=dict(size=4, color=[], colorscale="Turbo", cmin=0, cmax=1, showscale=False), customdata=[], text=[], hovertemplate="<b>%{text}</b><extra></extra>"), go.Scatter3d(x=[], y=[], z=[], mode="markers", name="Sonic neighbourhood", showlegend=False, marker=dict(size=7, color=[], opacity=1.0, line=dict(width=1, color="#ffffff")), customdata=[], hovertemplate="<b>%{customdata}</b><extra></extra>"), go.Scatter3d(x=[], y=[], z=[], mode="markers", name="Search result", showlegend=False, marker=dict(size=12, color="#ffeb3b", symbol="diamond", opacity=1.0, line=dict(width=2, color="#000000")), customdata=[], hovertemplate="<b>%{customdata}</b><extra>Search result</extra>")])
    fig.update_layout(title="PulseFlow — raw CLAP 12-cluster map — Direct PCA<br><sup>512D → 3D</sup>", template="plotly_dark", height=850, scene=dict(xaxis_title="PC 1", yaxis_title="PC 2", zaxis_title="PC 3", aspectmode="data"), showlegend=False, margin=dict(l=0,r=0,t=75,b=0))
    html = fig.to_html(include_plotlyjs=True, full_html=True, div_id="pulseflow-map")
    payload = json.dumps({"x":coords[:,0].tolist(), "y":coords[:,1].tolist(), "z":coords[:,2].tolist(), "labels":labels.tolist(), "types":type_ids, "genres":genre_names, "genreNames":genre_options, "plus":plus, "plusDir":plus_dir, "tagNames":tag_names, "tagGroups":tag_groups, "trackTags":track_tags, "bpm":bpm, "keyNames":key_names, "trackKey":track_key, "energy":energy_bucket, "paths":relative_paths, "colours":cluster_colours, "outlierDistance":outlier_distance.tolist(), "outlierCutoffs":outlier_cutoffs, "prefix":MUSIC_PREFIX, "initialCount":len(initial), "neighbourIds":neighbour_ids, "neighbourScores":neighbour_scores, "centroidScores":centroid_scores, "clusterGenres":cluster_genres, "helper":PLAYER_HELPER, "inline":INLINE_AUDIO}, separators=(",",":"))
    clusters = "".join(f'<label style="display:flex;align-items:center;gap:7px"><input type="checkbox" class="cluster" value="{i}" {"checked" if i == 0 else ""}><span style="display:inline-block;width:11px;height:11px;border-radius:2px;background:{cluster_colours[i]}"></span>Cluster {i:02d}</label>' for i in range(12))
    genre_controls = "".join(
        f'<label style="display:flex;align-items:center;gap:6px"><input type="checkbox" class="genre" value="{i}" checked><span>{name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")}</span></label>'
        for i, name in enumerate(genre_options)
    )
    controls = f'''<div id="controls" style="box-sizing:border-box;width:270px;min-width:190px;max-width:42vw;height:100vh;overflow-y:auto;resize:horizontal;overflow-x:hidden;padding:14px;background:#20242a;color:#eee;font:14px Arial"><h3 style="margin:0 0 5px">Map filters</h3><div id="displayedCounter" style="font-size:12px;color:#9fd;margin:0 0 12px">Displayed points: {len(initial):,} / {n:,}</div><b>Recording type</b><div><label><input type="checkbox" class="type" value="0" checked> Studio</label></div><div><label><input type="checkbox" class="type" value="1"> Live / LiveAlbum</label></div><div><label><input type="checkbox" class="type" value="2"> Other / unclassified</label></div><hr style="border-color:#555"><b>iTunes Genre</b><div style="font-size:11px;color:#aaa;margin:4px 0 8px">Classical is listed separately and can be switched on or off independently.</div><div style="display:flex;gap:6px;margin-bottom:7px"><button id="allGenres">All genres</button> <button id="noGenres">Clear genres</button></div><div style="display:flex;flex-direction:column;gap:4px;max-height:240px;overflow-y:auto">{genre_controls}</div><div style="margin:10px 0 10px"><label><input type="checkbox" id="excludePlus"> Exclude paths containing +</label></div><div style="margin:8px 0 16px;padding:8px;border:1px solid #555;border-radius:5px"><label><input type="checkbox" id="hideOutliers"> Hide distant 3D points</label><div style="margin:8px 0 0 22px;font-size:12px;color:#ccc">Hide points beyond each cluster's distance percentile:</div><select id="outlierLevel" style="margin:6px 0 0 22px;max-width:205px"><option value="99">Very few (99th percentile)</option><option value="97.5" selected>Moderate (97.5th percentile)</option><option value="95">More (95th percentile)</option></select><div style="font-size:11px;color:#aaa;margin:6px 0 0 22px">Visual filter only; no tracks are deleted.</div></div><div style="margin:10px 0;padding:8px;border:1px solid #555;border-radius:5px"><label><input type="checkbox" id="showSonicPanel"> Show sonic-neighbour panel</label><div style="font-size:11px;color:#aaa;margin:4px 0 0 22px">Shows the nearest CLAP matches when you click a point.</div><div style="margin:6px 0 0 22px"><label><input type="checkbox" id="fadeOutsideSonic"> Fade tracks outside this sonic neighbourhood</label></div><div style="margin:6px 0 0 22px"><label><input type="checkbox" id="excludePlusSonic"> Exclude '+' folders from sonic neighbours</label></div><div style="margin:6px 0 0 22px"><label><input type="checkbox" id="sonicVerbose" checked> Detailed neighbour list</label><div style="font-size:11px;color:#aaa;margin:2px 0 0 22px">Adds album, genre, cluster, same artist/album, mood/sound tags, tempo and key where available.</div></div></div><hr style="border-color:#555"><b>Playlist route</b><div style="margin:8px 0"><input type="file" id="playlistFile" accept=".m3u,.m3u8,.txt" style="max-width:240px"></div><div><label><input type="checkbox" id="fadeOthers"> Fade other tracks</label></div><div style="margin:6px 0"><button id="clearRoute">Clear route</button></div><p id="playlistStatus" style="font-size:12px;color:#ccc;line-height:1.4;margin:4px 0 10px">Load an .m3u/.m3u8 file to draw its order as a route.</p><hr style="border-color:#555"><b>CLAP clusters</b><div style="display:flex;flex-direction:column;gap:5px;margin-top:8px">{clusters}</div><div style="position:sticky;bottom:0;background:#20242a;padding:10px 0"><button id="allClusters">Select all</button> <button id="noClusters">Clear all</button></div><p id="status" style="line-height:1.4">Preparing initial selection...</p><p id="sonicTiming" style="font-size:11px;color:#8a8;margin:0"></p></div>'''
    js = (r'''<script>
const DATA = __PULSEFLOW_DATA__;
const plot = document.getElementById('pulseflow-map');
const status = document.getElementById('status');
let filterVersion = 0;
let renderBusy = false;
let pendingRender = null;
let filterTimer = null;
function selectedValues(selector) { return [...document.querySelectorAll(selector+':checked')].map(e => Number(e.value)); }
const sonicToggle = document.getElementById('showSonicPanel');
const sonicFadeToggle = document.getElementById('fadeOutsideSonic');
const sonicExcludePlus = document.getElementById('excludePlusSonic');
const sonicVerbose = document.getElementById('sonicVerbose');
let sonicSelected = -1;
let visibleTrackIds = [];
let visibleTrackSet = new Set();
let sonicNeighbourSet = new Set();
let fadeBox = null;
let routeLen = 0;
function mainTrackOpacity() {
 // Fade the large background trace with one scalar value; never send a 70k opacity array.
 return (((sonicFadeToggle && sonicFadeToggle.checked && sonicToggle && sonicToggle.checked && sonicSelected >= 0)) || (fadeBox && fadeBox.checked && routeLen > 0)) ? 0.12 : 0.9;
}
// Any Plotly.restyle on a gl3d plot makes Plotly redo defaults/calc/WebGL buffers for EVERY trace,
// including the 70k-point background trace, even when only this 9-point overlay changed.
// So: (1) never redraw just to clear an overlay that is already empty, and (2) coalesce rapid
// clicks (map or sonic list) into a single redraw of the latest selection.
let overlayShown = false;
let sonicHighlightTimer = null;
const timingEl = document.getElementById('sonicTiming');
function timedOverlayRestyle(update) {
 const t0 = performance.now();
 return Plotly.restyle(plot, update, [2])
  .then(() => { if (timingEl) timingEl.textContent = 'Last highlight redraw: ' + Math.round(performance.now() - t0) + ' ms'; })
  .catch(err => { if (timingEl) timingEl.textContent = 'Highlight redraw failed: ' + err; });
}
function updateSonicHighlight() {
 if (sonicHighlightTimer) clearTimeout(sonicHighlightTimer);
 sonicHighlightTimer = setTimeout(() => { sonicHighlightTimer = null; applySonicHighlight(); }, 120);
 return Promise.resolve();
}
function applySonicHighlight() {
 const active = sonicFadeToggle && sonicFadeToggle.checked && sonicToggle && sonicToggle.checked && sonicSelected >= 0 && visibleTrackSet.size;
 if (!active) {
  if (!overlayShown) return Promise.resolve();
  overlayShown = false;
  return timedOverlayRestyle({x:[[]], y:[[]], z:[[]], customdata:[[]], 'marker.color':[[]]});
 }
 const ids = [sonicSelected, ...sonicNeighbours(sonicSelected, 8).map(n => n.id)].filter(id => visibleTrackSet.has(id));
 sonicNeighbourSet = new Set(ids);
 if (!ids.length && !overlayShown) return Promise.resolve();
 overlayShown = ids.length > 0;
 const xs = ids.map(id => DATA.x[id]);
 const ys = ids.map(id => DATA.y[id]);
 const zs = ids.map(id => DATA.z[id]);
 const paths = ids.map(id => DATA.paths[id]);
 const colours = ids.map(id => DATA.colours[DATA.labels[id]]);
 const sizes = ids.map(id => id === sonicSelected ? 10 : 7);
 return timedOverlayRestyle({x:[xs], y:[ys], z:[zs], customdata:[paths], 'marker.color':[colours], 'marker.size':[sizes]});
}
function updateTrackOpacity() {
 Plotly.restyle(plot, {'marker.opacity':[mainTrackOpacity()]}, [0]);
 updateSonicHighlight();
}
function updateMap() {
 const version = ++filterVersion;
 if (filterTimer) clearTimeout(filterTimer);
 filterTimer = setTimeout(() => buildFilteredArrays(version), 90);
}
function buildFilteredArrays(version) {
 const chosenTypes = new Set(selectedValues('.type'));
 const chosenClusters = new Set(selectedValues('.cluster'));
 const chosenGenres = new Set(selectedValues('.genre').map(i => DATA.genreNames[i]));
 const excludePlus = document.getElementById('excludePlus').checked;
 const hideOutliers = document.getElementById('hideOutliers').checked;
 const outlierLevel = document.getElementById('outlierLevel').value;
 const xs=[], ys=[], zs=[], paths=[], colours=[], sourceIds=[];
 let i = 0;
 status.textContent = 'Filtering tracks...';
 // Work in short batches so checkbox handling and chart interaction remain responsive.
 function batch() {
  if (version !== filterVersion) return;
  const stop = Math.min(i + 3500, DATA.paths.length);
  for (; i < stop; i++) {
   const cluster = DATA.labels[i];
   if (!chosenTypes.has(DATA.types[i]) || !chosenClusters.has(cluster) || !chosenGenres.has(DATA.genres[i]) || (excludePlus && DATA.plus[i])) continue;
   if (hideOutliers && DATA.outlierDistance[i] > DATA.outlierCutoffs[String(cluster)][outlierLevel]) continue;
   xs.push(DATA.x[i]); ys.push(DATA.y[i]); zs.push(DATA.z[i]); paths.push(DATA.paths[i]); colours.push(DATA.colours[cluster]); sourceIds.push(i);
  }
  if (i < DATA.paths.length) { setTimeout(batch, 0); return; }
  pendingRender = {version, xs, ys, zs, paths, colours, sourceIds};
  renderLatest();
 }
 batch();
}
function renderLatest() {
 if (renderBusy || !pendingRender) return;
 const job = pendingRender;
 pendingRender = null;
 if (job.version !== filterVersion) { renderLatest(); return; }
 renderBusy = true;
 visibleTrackIds = job.sourceIds;
 visibleTrackSet = new Set(job.sourceIds);
 status.textContent = 'Rendering ' + job.paths.length.toLocaleString() + ' of ' + DATA.paths.length.toLocaleString() + ' tracks...';
 Plotly.restyle(plot, {x:[job.xs], y:[job.ys], z:[job.zs], customdata:[job.paths], 'marker.color':[job.colours], 'marker.opacity':[mainTrackOpacity()]}, [0])
  .then(() => updateSonicHighlight())
  .then(() => {
   if (job.version === filterVersion) {
    status.textContent = 'Showing ' + job.paths.length.toLocaleString() + ' of ' + DATA.paths.length.toLocaleString() + ' tracks.';
    document.getElementById('displayedCounter').textContent = 'Displayed points: ' + job.paths.length.toLocaleString() + ' / ' + DATA.paths.length.toLocaleString();
   }
  })
  .catch(err => { status.textContent = 'Plot update failed: ' + err; })
  .finally(() => { renderBusy = false; if (pendingRender) renderLatest(); });
}
document.querySelectorAll('.type,.genre,.cluster,#excludePlus,#hideOutliers,#outlierLevel').forEach(e => e.addEventListener('change', updateMap));
document.getElementById('allGenres').addEventListener('click', () => { document.querySelectorAll('.genre').forEach(e => e.checked=true); updateMap(); });
document.getElementById('noGenres').addEventListener('click', () => { document.querySelectorAll('.genre').forEach(e => e.checked=false); updateMap(); });
document.getElementById('allClusters').addEventListener('click', () => { document.querySelectorAll('.cluster').forEach(e => e.checked=true); updateMap(); });
document.getElementById('noClusters').addEventListener('click', () => { document.querySelectorAll('.cluster').forEach(e => e.checked=false); updateMap(); });
status.textContent = 'Showing ' + DATA.initialCount.toLocaleString() + ' of ' + DATA.paths.length.toLocaleString() + ' tracks. Choose filters to update.';
// Start with the same selection Python drew (Studio + cluster 00) so the fade overlay works
// before any filter is touched; previously this set stayed empty until the first filter change.
visibleTrackIds = [];
for (let i = 0; i < DATA.paths.length; i++) if (DATA.types[i] === 0 && DATA.labels[i] === 0) visibleTrackIds.push(i);
visibleTrackSet = new Set(visibleTrackIds);
document.getElementById('displayedCounter').textContent = 'Displayed points: ' + DATA.initialCount.toLocaleString() + ' / ' + DATA.paths.length.toLocaleString();

// ---- Click a point: show clickable links to the track and its folder ----
const panel = document.createElement('div');
panel.style.cssText = 'position:fixed;left:290px;bottom:16px;right:auto;max-width:min(520px,calc(100vw - 320px));max-height:42vh;overflow:auto;padding:12px 14px;background:#1e1e1e;color:#eee;border:1px solid #777;border-radius:8px;font:14px/1.5 Arial,sans-serif;z-index:10000;display:none;box-shadow:0 4px 18px rgba(0,0,0,.65);resize:both';
document.body.appendChild(panel);
function esc(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function fullPath(p) { return /^([A-Za-z]:|\\\\)/.test(p) ? p : DATA.prefix + p; }
let pathIndex = null;
function idxOf(rel) {
 if (!pathIndex) { pathIndex = new Map(); for (let i = 0; i < DATA.paths.length; i++) pathIndex.set(DATA.paths[i], i); }
 const v = pathIndex.get(String(rel));
 return v === undefined ? -1 : v;
}
function clusterTag(i) { return 'C' + String(DATA.labels[i]).padStart(2, '0'); }
// Nearest CLAP neighbours of a track, optionally skipping tracks in '+' folders (the stored list is longer than 8 to allow for this).
function sonicNeighbours(index, limit) {
 const ids = DATA.neighbourIds[index] || [], sc = DATA.neighbourScores[index] || [];
 const out = [];
 for (let k = 0; k < ids.length && out.length < limit; k++) {
  if (sonicExcludePlus.checked && DATA.plusDir[ids[k]]) continue;
  out.push({id: ids[k], score: Number(sc[k] || 0)});
 }
 return out;
}
function trackParts(i) {
 const p = DATA.paths[i].split(/[\\/]/);
 return {name: p[p.length - 1].replace(/\.[^.]+$/, ''), album: p.length >= 2 ? p[p.length - 2] : '', artist: p.length >= 3 ? p[p.length - 3] : ''};
}
function chip(text, colour) {
 if (!text) return '';
 return '<span style="display:inline-block;margin:3px 4px 0 0;padding:1px 6px;border-radius:3px;font-size:10px;background:#343b45;color:#ddd' + (colour ? ';border-left:3px solid ' + colour : '') + '">' + esc(text) + '</span>';
}
// CLAP mood/sound/voice tags (from make_clap_tags.py). Tags a neighbour shares with the selected track are highlighted.
const TAG_BG = {mood: '#3a4a6a', sound: '#3d5a4b', voice: '#5a4a3a', musicnn: '#4a3f63'};
function tagChips(i, shared) {
 if (!DATA.tagNames.length) return '';
 return (DATA.trackTags[i] || []).map(t => {
  const on = !!shared && shared.has(t);
  return '<span style="display:inline-block;margin:3px 4px 0 0;padding:1px 6px;border-radius:3px;font-size:10px;background:' + (TAG_BG[DATA.tagGroups[t]] || '#343b45') + ';color:' + (on ? '#fff' : '#cfd6df') + (on ? ';font-weight:600;box-shadow:0 0 0 1px #9fe0c0 inset' : '') + '">' + esc(DATA.tagNames[t]) + '</span>';
 }).join('');
}
// Tempo / key / energy from features_60s.json. Key is a note name only (no major/minor), tempo estimates can be half or double.
const ENERGY_WORDS = ['low energy', 'mid energy', 'high energy'];
function soundLine(i) {
 if (!DATA.bpm.length) return '';
 const parts = [];
 if (DATA.bpm[i] > 0) parts.push(DATA.bpm[i] + ' BPM');
 if (DATA.trackKey[i] >= 0) parts.push('key ' + DATA.keyNames[DATA.trackKey[i]]);
 if (DATA.energy[i] >= 0) parts.push(ENERGY_WORDS[DATA.energy[i]]);
 return parts.join(' · ');
}
function tempoRelation(a, b) {
 if (!a || !b) return '';
 const r = a / b;
 if (Math.abs(r - 1) <= 0.04) return 'similar tempo';
 if (Math.abs(r - 2) <= 0.08 || Math.abs(r - 0.5) <= 0.02) return 'half/double tempo';
 return '';
}
function detailBlock(i, index) {
 let html = '';
 const selTags = new Set(DATA.tagNames.length ? (DATA.trackTags[index] || []) : []);
 const common = DATA.tagNames.length ? (DATA.trackTags[i] || []).filter(t => selTags.has(t)).map(t => DATA.tagNames[t]) : [];
 if (DATA.tagNames.length) html += '<div>' + tagChips(i, selTags) + '</div>';
 const line = soundLine(i);
 if (line) html += '<div style="font-size:11px;color:#aaa;margin-top:2px">' + esc(line) + '</div>';
 if (DATA.bpm.length) {
  if (DATA.trackKey[i] >= 0 && DATA.trackKey[i] === DATA.trackKey[index]) common.push('same key');
  const tr = tempoRelation(DATA.bpm[i], DATA.bpm[index]);
  if (tr) common.push(tr);
 }
 if (common.length) html += '<div style="font-size:11px;color:#9fe0c0;margin-top:2px">In common: ' + esc(common.join(', ')) + '</div>';
 return html;
}
function headerDetail(index) {
 let html = DATA.tagNames.length ? tagChips(index, null) : '';
 const line = soundLine(index);
 if (line) html += '<div style="font-size:11px;color:#aaa;margin-top:2px">' + esc(line) + '</div>';
 return html;
}
function similarityWords(score) {
 if (score >= 0.92) return 'Extremely close sonic match';
 if (score >= 0.88) return 'Very strong resemblance';
 if (score >= 0.84) return 'Strong shared sonic qualities';
 if (score >= 0.80) return 'Noticeable sonic overlap';
 if (score >= 0.75) return 'Some shared qualities';
 return 'Looser sonic connection';
}
function playViaHelper(full, ok, fail) {
 fetch('http://127.0.0.1:8765/play?p=' + encodeURIComponent(full), {mode: 'no-cors'}).then(ok).catch(fail);
}
function fileUrl(p) {
 let u = p.replace(/\\/g, '/');
 if (u.startsWith('//')) return 'file:' + encodeURI(u).replace(/#/g, '%23').replace(/\?/g, '%3F');
 return 'file:///' + encodeURI(u).replace(/#/g, '%23').replace(/\?/g, '%3F');
}
function showPanel(pt) {
 let cd = pt.customdata;
 if (cd === undefined) {
  const tr = pt.data || plot.data[pt.curveNumber || 0];
  const idx = pt.pointNumber !== undefined ? pt.pointNumber : pt.pointIndex;
  if (tr && tr.customdata) cd = tr.customdata[idx];
 }
 if (Array.isArray(cd)) cd = cd[0];
 if (!cd) { status.textContent = 'Click received, but no path was found for that point.'; return; }
 const selectedIndex = idxOf(String(cd));
 if (sonicToggle.checked && selectedIndex >= 0) showSonicNeighbours(selectedIndex);
 const full = fullPath(String(cd));
 const folder = full.replace(/[\\\/][^\\\/]*$/, '');
 const title = full.split(/[\\\/]/).pop().replace(/\.[^.]+$/, '');
 const parts = String(cd).split(/[\\\/]/);
 const sub = parts.length >= 3 ? parts[parts.length-3] + ' \u2014 ' + parts[parts.length-2] : parts.slice(0,-1).join(' / ');
 const ci = idxOf(cd);
 const clusterHtml = ci >= 0 ? '<div style="margin-top:2px"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;background:' + DATA.colours[DATA.labels[ci]] + '"></span>Cluster ' + String(DATA.labels[ci]).padStart(2, '0') + '</div>' : '';
 panel.innerHTML = '<div id="pf-drag" style="font-weight:600;font-size:15px;cursor:move;user-select:none;padding:2px 0 7px;border-bottom:1px solid #555;margin-bottom:6px">&#8942;&#8942; ' + esc(title) + ' <span style="float:right;color:#aaa;font-size:12px;font-weight:normal">drag panel</span></div><div style="color:#bbb">' + esc(sub) + '</div>' + clusterHtml + '<div style="margin-top:8px"><a href="' + fileUrl(full) + '" target="_blank" style="color:#6cb6ff">&#9654; Open track</a> &nbsp;|&nbsp; <a href="' + fileUrl(folder) + '" target="_blank" style="color:#6cb6ff">&#128193; Open folder</a> &nbsp;|&nbsp; <a href="#" id="pf-copy" style="color:#6cb6ff">Copy path</a> &nbsp;|&nbsp; <a href="#" id="pf-close" style="color:#888">Close</a></div>';
 panel.style.left = panel.style.left || '290px'; panel.style.right = 'auto';
 panel.style.display = 'block';
 const note = document.createElement('div');
 note.style.cssText = 'color:#e88;font-size:12px;margin-top:4px';
 panel.appendChild(note);
 if (DATA.inline) {
  const audio = document.createElement('audio');
  audio.controls = true; audio.autoplay = true; audio.style.cssText = 'width:100%;margin-top:8px';
  audio.addEventListener('error', () => { note.textContent = "This browser couldn't play the file inline."; });
  audio.src = fileUrl(full);
  panel.insertBefore(audio, note);
 }
 if (DATA.helper) {
 const play = document.createElement('a');
 play.href = '#'; play.textContent = '\u25B6 Play in default app';
 play.style.cssText = 'color:#6cb6ff;display:inline-block;margin-top:6px';
 play.onclick = e => {
  e.preventDefault();
  fetch('http://127.0.0.1:8765/play?p=' + encodeURIComponent(full), {mode: 'no-cors'})
   .then(() => { play.textContent = 'Sent to default player'; })
   .catch(() => { note.textContent = 'Helper not running - start play_server.py first.'; });
 };
 panel.appendChild(play);
 }
 const drag = document.getElementById('pf-drag');
 let dragStart = null;
 drag.onpointerdown = e => { dragStart = {x:e.clientX, y:e.clientY, left:panel.offsetLeft, top:panel.offsetTop}; drag.setPointerCapture(e.pointerId); };
 drag.onpointermove = e => { if (!dragStart) return; panel.style.left = Math.max(280, Math.min(window.innerWidth-panel.offsetWidth, dragStart.left + e.clientX-dragStart.x)) + 'px'; panel.style.top = Math.max(8, Math.min(window.innerHeight-panel.offsetHeight, dragStart.top + e.clientY-dragStart.y)) + 'px'; panel.style.bottom = 'auto'; };
 drag.onpointerup = () => { dragStart = null; };
 document.getElementById('pf-copy').onclick = function (e) { e.preventDefault(); if (navigator.clipboard) navigator.clipboard.writeText(full); this.textContent = 'Copied!'; };
 document.getElementById('pf-close').onclick = function (e) { e.preventDefault(); panel.style.display = 'none'; };
}
let lastHover = null, downAt = null;
plot.on('plotly_hover', ev => { lastHover = (ev.points && ev.points[0]) || null; });
plot.on('plotly_unhover', () => { lastHover = null; });
plot.on('plotly_click', ev => { if (ev.points && ev.points[0]) showPanel(ev.points[0]); });
plot.addEventListener('mousedown', e => { downAt = [e.clientX, e.clientY]; }, true);
plot.addEventListener('click', e => {
 if (!downAt || !lastHover) return;
 if (Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
 showPanel(lastHover);
}, true);
status.textContent += ' Click a point for track links.';

// ---- Sonic-neighbour panel (nearest points in the existing 3D PCA map) ----
const sonicPanel = document.getElementById('sonicPanel');
const rightPanel = document.getElementById('rightPanel');
function refreshRightPanel() {
 const playlistVisible = document.getElementById('playlistPanel').style.display !== 'none';
 const sonicVisible = sonicToggle.checked;
 const wasVisible = rightPanel.style.display !== 'none';
 rightPanel.style.display = (playlistVisible || sonicVisible) ? 'block' : 'none';
 sonicPanel.style.display = sonicVisible ? 'block' : 'none';
 // Resize only when the side panel opens/closes. Never resize during track clicks.
 if (wasVisible !== (playlistVisible || sonicVisible)) {
  requestAnimationFrame(() => Plotly.Plots.resize(plot));
 }
}
function showSonicNeighbours(index) {
 if (index < 0) return;
 sonicSelected = index;
 updateSonicHighlight();
 // These are true 512D CLAP neighbours, not merely nearby points in the 3D projection.
 const shown = sonicNeighbours(index, 8);
 const selectedName = DATA.paths[index].split(/[\\/]/).pop();
 const clusterId = DATA.labels[index];
 document.getElementById('sonicTitle').textContent = selectedName;
 document.getElementById('sonicMeta').textContent = 'Cluster ' + String(clusterId).padStart(2,'0') + (sonicVerbose.checked ? ' · ' + DATA.genres[index] : '') +' · nearest tracks by original 512D CLAP similarity' + (sonicExcludePlus.checked ? " · '+' folders hidden" : '');
 const selTagsEl = document.getElementById('sonicTags');
 if (selTagsEl) selTagsEl.innerHTML = sonicVerbose.checked ? headerDetail(index) : '';
 const list = document.getElementById('sonicList');
 list.innerHTML = '';
 shown.forEach(({id: i, score}) => {
  const li = document.createElement('li');
  li.style.cssText = 'padding:6px 4px;margin:3px 0;border-left:4px solid ' + DATA.colours[DATA.labels[i]] + ';cursor:pointer;word-break:break-word';
  const name = DATA.paths[i].split(/[\\/]/).pop().replace(/\.[^.]+$/, '');
  const artist = DATA.paths[i].split(/[\\/]/).slice(-3,-2)[0] || '';
  if (sonicVerbose.checked) {
   const t = trackParts(i), s = trackParts(index);
   const sameArtist = !!t.artist && t.artist.toLowerCase() === s.artist.toLowerCase();
   const sameAlbum = sameArtist && t.album.toLowerCase() === s.album.toLowerCase();
   const sameCluster = DATA.labels[i] === clusterId;
   li.innerHTML = '<div style="font-weight:600">' + esc(name) + '</div><div style="font-size:11px;color:#aaa">' + esc(t.artist) + (t.album ? ' — ' + esc(t.album) : '') + '</div><div style="font-size:11px;color:#aaa">' + similarityWords(score) + ' · ' + score.toFixed(3) + '</div><div>' + chip(clusterTag(i), DATA.colours[DATA.labels[i]]) + chip(sameCluster ? 'same cluster' : 'different cluster') + chip(DATA.genres[i]) + (sameAlbum ? chip('same album') : (sameArtist ? chip('same artist') : '')) + '</div>' + detailBlock(i, index);
  } else
  li.innerHTML = '<div style="font-weight:600">' + esc(name) + '</div><div style="font-size:11px;color:#aaa">' + esc(artist) + ' · ' + similarityWords(score) + ' · ' + score.toFixed(3) + '</div>';
  li.addEventListener('click', () => showPanel({customdata:DATA.paths[i]}));
  list.appendChild(li);
 });
 const clusterRanks = (DATA.centroidScores[index] || []).map((score,id) => ({id,score:Number(score)})).sort((a,b) => b.score-a.score).slice(0,3);
 const clusterBlock = document.createElement('div');
 clusterBlock.style.cssText = 'margin-top:10px;padding-top:8px;border-top:1px solid #444';
 clusterBlock.innerHTML = '<b>Other cluster centres</b><div style="margin-top:5px">' + clusterRanks.map(x => '<span style="display:inline-block;margin:2px 4px 2px 0;padding:3px 6px;border-radius:4px;background:#343b45">C' + String(x.id).padStart(2,'0') + ' · ' + similarityWords(x.score) + '</span>').join('') + '</div>';
 const genres = DATA.clusterGenres[clusterId] || [];
 const genreBlock = document.createElement('div');
 genreBlock.style.cssText = 'margin-top:9px;padding-top:7px;border-top:1px solid #444;font-size:12px;color:#bbb';
 genreBlock.innerHTML = '<b style="color:#eee">Common iTunes genres in this cluster</b><div style="margin-top:4px">' + (genres.length ? genres.map(g => esc(g.name) + ' (' + Number(g.count).toLocaleString() + ')').join(' · ') : 'No genre tags available') + '</div><div style="font-size:10px;color:#888;margin-top:6px">Neighbour descriptions use relative similarity bands from the original CLAP embeddings. Cluster genres are iTunes tags, not CLAP-inferred genres.</div>';
 list.appendChild(clusterBlock);
 list.appendChild(genreBlock);
 refreshRightPanel();
}
sonicToggle.addEventListener('change', () => { updateTrackOpacity(); refreshRightPanel(); });
sonicExcludePlus.addEventListener('change', () => { if (sonicSelected >= 0 && sonicToggle.checked) showSonicNeighbours(sonicSelected); });
sonicVerbose.addEventListener('change', () => { if (sonicSelected >= 0 && sonicToggle.checked) showSonicNeighbours(sonicSelected); });
sonicFadeToggle.addEventListener('change', () => { if (sonicFadeToggle.checked && !sonicToggle.checked) { sonicToggle.checked = true; refreshRightPanel(); } updateTrackOpacity(); });

// ---- Playlist route: load an m3u/m3u8 and draw it as a line through the map ----
const plInput = document.getElementById('playlistFile');
const plStatus = document.getElementById('playlistStatus');
fadeBox = document.getElementById('fadeOthers');
let lookup = null;
function normP(p) {
 let s = String(p).trim();
 if (/^file:/i.test(s)) { s = s.replace(/^file:(\/\/)?\/?/i, ''); try { s = decodeURIComponent(s); } catch (e) {} }
 return s.replace(/\\/g, '/').replace(/\/+/g, '/').toLowerCase();
}
function buildLookup() {
 const full = new Map(), tail3 = new Map(), name = new Map();
 for (let i = 0; i < DATA.paths.length; i++) {
  const p = /^([A-Za-z]:|\\\\)/.test(DATA.paths[i]) ? DATA.paths[i] : DATA.prefix + DATA.paths[i];
  const parts = normP(p).split('/');
  full.set(parts.join('/'), i);
  const t3 = parts.slice(-3).join('/');
  tail3.set(t3, tail3.has(t3) ? -1 : i);
  const nm = parts[parts.length - 1];
  name.set(nm, name.has(nm) ? -1 : i);
 }
 return {full, tail3, name};
}
function matchTrack(line, L) {
 const parts = normP(line).split('/');
 const f = L.full.get(parts.join('/'));
 if (f !== undefined) return f;
 const t3 = L.tail3.get(parts.slice(-3).join('/'));
 if (t3 !== undefined && t3 >= 0) return t3;
 const nm = L.name.get(parts[parts.length - 1]);
 if (nm !== undefined && nm >= 0) return nm;
 return -1;
}
function applyFade() { updateTrackOpacity(); }
function drawRoute(ids, title, missed) {
 const n = ids.length;
 const xs = [], ys = [], zs = [], txt = [], paths = [], col = [], mcol = [], size = [];
 ids.forEach((i, k) => {
  xs.push(DATA.x[i]); ys.push(DATA.y[i]); zs.push(DATA.z[i]);
  txt.push('#' + (k + 1) + '  [' + clusterTag(i) + ']  ' + DATA.paths[i]);
  paths.push(DATA.paths[i]);
  col.push(n > 1 ? k / (n - 1) : 0);
  mcol.push(DATA.colours[DATA.labels[i]]);
  size.push(k === 0 || k === n - 1 ? 9 : 4);
 });
 routeLen = n;
 Plotly.restyle(plot, {x: [xs], y: [ys], z: [zs], text: [txt], customdata: [paths], 'line.color': [col], 'marker.color': [mcol], 'marker.size': [size]}, [1]);
 applyFade();
 renderList(ids, title, missed);
}
const plPanel = document.getElementById('playlistPanel');
const plList = document.getElementById('plList');
let plSelected = null;
function renderList(ids, title, missed) {
 plList.innerHTML = '';
 plSelected = null;
 document.getElementById('plMissed').innerHTML = '';
 document.getElementById('plMsg').textContent = '';
 if (!ids.length && !(missed && missed.length)) {
  plPanel.style.display = 'none';
  refreshRightPanel();
  return;
 }
 document.getElementById('plTitle').textContent = title ? title.replace(/\.[^.]+$/, '') : 'Playlist';
 document.getElementById('plCount').textContent = ids.length + ' tracks on the map' + (missed && missed.length ? ' \u00b7 ' + missed.length + ' not found' : '');
 let listPos = 0;
 function appendListBatch() {
  const stop = Math.min(listPos + 180, ids.length);
  for (; listPos < stop; listPos++) {
   const i = ids[listPos];
  const rel = DATA.paths[i];
  const parts = rel.split(/[\\\/]/);
  const name = parts[parts.length - 1].replace(/\.[^.]+$/, '');
  const sub = parts.length >= 3 ? parts[parts.length - 3] + ' \u2014 ' + parts[parts.length - 2] : '';
  const li = document.createElement('li');
  const colour = DATA.colours[DATA.labels[i]];
  li.style.cssText = 'cursor:pointer;padding:3px 4px;border-radius:4px;margin-bottom:2px;border-left:4px solid ' + colour;
  li.innerHTML = '<div style="display:flex;align-items:center;gap:6px"><div style="flex:1;min-width:0"><div>' + esc(name) + '</div><div style="font-size:11px;color:#9aa">' + esc(sub) + ' &middot; <span style="color:' + colour + ';font-weight:600">' + clusterTag(i) + '</span></div></div>' + (DATA.helper ? '<a href="#" class="plplay" title="Play in default app" style="color:#6cb6ff;text-decoration:none;font-size:16px;padding:0 4px">&#9654;</a>' : '') + '</div>';
  const pb = li.querySelector('.plplay');
  if (pb) pb.addEventListener('click', e => {
   e.preventDefault(); e.stopPropagation();
   playViaHelper(fullPath(rel),
    () => { pb.innerHTML = '&#10003;'; document.getElementById('plMsg').textContent = ''; setTimeout(() => { pb.innerHTML = '&#9654;'; }, 1500); },
    () => { document.getElementById('plMsg').textContent = 'Helper not running - start play_server.py first.'; });
  });
  li.addEventListener('click', () => {
   if (plSelected) plSelected.style.background = '';
   li.style.background = '#34404f';
   plSelected = li;
   showPanel({customdata: rel});
  });
  plList.appendChild(li);
  }
  if (listPos < ids.length) { requestAnimationFrame(appendListBatch); }
  else document.getElementById('plMsg').textContent = '';
 }
 appendListBatch();
 if (missed && missed.length) {
  document.getElementById('plMissed').innerHTML = '<details><summary style="cursor:pointer">Not found (' + missed.length + ')</summary>' + missed.map(m => '<div style="margin:3px 0;word-break:break-all">' + esc(m) + '</div>').join('') + '</details>';
 }
 plPanel.style.display = 'block';
 refreshRightPanel();
}
let playlistLoadVersion = 0;
plInput.addEventListener('change', async () => {
 const file = plInput.files[0];
 if (!file) return;
 const version = ++playlistLoadVersion;
 plStatus.textContent = 'Reading playlist...';
 await new Promise(r => requestAnimationFrame(r));
 const buf = await file.arrayBuffer();
 let text;
 try { text = new TextDecoder('utf-8', {fatal: true}).decode(buf); } catch (e) { text = new TextDecoder('windows-1252').decode(buf); }
 const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/).map(l => l.trim()).filter(l => l && l[0] !== '#');
 if (!lookup) {
  plStatus.textContent = 'Indexing library paths (one-time)...';
  await new Promise(r => setTimeout(r, 20));
  lookup = buildLookup();
 }
 const ids = [], missed = [];
 let pos = 0;
 function matchBatch() {
  if (version !== playlistLoadVersion) return;
  const end = Math.min(pos + 1200, lines.length);
  for (; pos < end; pos++) {
   const i = matchTrack(lines[pos], lookup);
   if (i >= 0) ids.push(i); else missed.push(lines[pos]);
  }
  plStatus.textContent = 'Matching playlist... ' + pos.toLocaleString() + ' / ' + lines.length.toLocaleString();
  if (pos < lines.length) { setTimeout(matchBatch, 0); return; }
  plStatus.textContent = 'Drawing playlist route...';
  requestAnimationFrame(() => {
   if (version !== playlistLoadVersion) return;
   drawRoute(ids, file.name, missed);
   plStatus.textContent = file.name + ': ' + ids.length + ' of ' + lines.length + ' tracks matched.' + (missed.length ? ' ' + missed.length + ' not found (listed in the browser console).' : '');
   if (missed.length) console.log('Playlist entries not matched:', missed);
  });
 }
 matchBatch();
});
fadeBox.addEventListener('change', applyFade);
document.getElementById('clearRoute').addEventListener('click', () => {
 plInput.value = '';
 drawRoute([], '', []);
 plStatus.textContent = 'Route cleared.';
});
''' + SEARCH_JS + "</script>").replace("__PULSEFLOW_DATA__", payload)
    html = html.replace("<body>", "<body>\n<div id=\"appLayout\" style=\"display:flex;width:100%;height:100vh;overflow:hidden;gap:0\">\n" + controls + "\n<div id=\"chartColumn\" style=\"flex:1;min-width:260px;height:100vh;overflow:hidden;position:relative\">\n" + SEARCH_HTML, 1)
    right_panel = '<div id="rightPanel" style="display:none;box-sizing:border-box;width:330px;min-width:220px;max-width:48vw;height:100vh;overflow-y:auto;resize:horizontal;overflow-x:hidden;padding:12px;background:#20242a;color:#eee;font:13px Arial;border-left:1px solid #555"><section id="sonicPanel" style="display:none;margin-bottom:18px"><h3 style="margin:0 0 6px">Sonic neighbours</h3><div id="sonicTitle" style="font-weight:600;word-break:break-word">Click a track</div><div id="sonicMeta" style="font-size:11px;color:#aaa;margin:4px 0 8px">Nearest points in the 3D map</div><div id="sonicTags" style="margin:0 0 8px"></div><ol id="sonicList" style="padding-left:22px;line-height:1.35"></ol></section><section id="playlistPanel" style="display:none"><h3 id="plTitle" style="margin:0 0 4px;word-break:break-word">Playlist</h3><div id="plCount" style="color:#aaa;margin-bottom:6px"></div><div id="plMsg" style="color:#e88;font-size:12px;margin-bottom:6px"></div><ol id="plList" style="margin:0;padding-left:30px;line-height:1.4"></ol><div id="plMissed" style="margin-top:12px;font-size:12px;color:#aaa"></div></section></div>'
    html = html.replace("</body>", "</div>\n" + right_panel + "\n</div>\n" + js + "\n</body>", 1)
    TEMP_OUTPUT.write_text(html, encoding="utf-8")
    if not TEMP_OUTPUT.is_file() or TEMP_OUTPUT.stat().st_size == 0: raise RuntimeError("HTML output was not created correctly.")
    os.replace(TEMP_OUTPUT, OUTPUT)
    print(f"Map saved: {OUTPUT}", flush=True)
    print(f"HTML size: {OUTPUT.stat().st_size/(1024*1024):.1f} MB")
    print("Existing coordinates, labels, production map and library were not modified.")
    print("Sonic neighbours and similarity descriptions were precomputed from the original 512D CLAP embeddings; clicks now use cached lookups.")
    print("Sonic fade uses a uniformly faded background trace plus a separate 9-point CLAP-neighbour overlay; no full-library opacity arrays are sent on track clicks.")
    print("Displayed-points counter reflects the active map filters.")
    print("Optional 3D outlier filter uses per-cluster distance percentiles; it is visual only.")
    print("Point hover labels omit the M:\\Music\\Music\\ prefix where present.")
    print("Click a point to get links to the track and its folder.")
    start_helper()

if __name__ == "__main__": main()
