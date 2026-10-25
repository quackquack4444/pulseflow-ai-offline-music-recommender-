from pathlib import Path
import json, os, plistlib, re, colorsys, socket, subprocess, sys
from urllib.parse import urlparse, unquote
import numpy as np
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
COORDS_FILE = DATA / "music_3d.npy"
LABELS_FILE = DATA / "music_clusters_raw_clap_50.npy"
MAPPING_FILE = DATA / "embedding_mapping.txt"
ITUNES_XML = Path(r"M:\Music\iTunes Library.xml")
OUTPUT = DATA / "music_map_raw_clap_50_interactive_filtered_v5.html"
TEMP_OUTPUT = DATA / "music_map_raw_clap_50_interactive_filtered_v5.building.html"
LIVE_RE = re.compile(r"\blive\b|livealbum|concert|unplugged|bootleg|\bin concert\b|\blive at\b|\blive in\b", re.I)
MUSIC_PREFIX = "M:\\Music\\Music\\"

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
            result[norm_path(location)] = {"grouping": str(track.get("Grouping", "") or ""), "name": str(track.get("Name", "") or ""), "album": str(track.get("Album", "") or "")}
    print(f"iTunes tracks indexed: {len(result):,}", flush=True)
    return result

def classify(path, metadata):
    info = metadata.get(norm_path(path), {})
    text = " | ".join((path, info.get("name", ""), info.get("album", ""), info.get("grouping", "")))
    if LIVE_RE.search(text): return "Live / LiveAlbum"
    if re.search(r"\bstudio\b", info.get("grouping", ""), re.I): return "Studio"
    return "Other / unclassified"

def main():
    print("Loading existing coordinates, cluster labels and paths...", flush=True)
    coords = np.load(COORDS_FILE, allow_pickle=False)
    labels = np.load(LABELS_FILE, allow_pickle=False).astype(int)
    paths = json.loads(MAPPING_FILE.read_text(encoding="utf-8"))
    n = len(paths)
    if coords.shape != (n, 3) or labels.shape != (n,): raise ValueError(f"Data shape mismatch: coords={coords.shape}, labels={labels.shape}, paths={n}")
    if not np.isfinite(coords).all() or labels.min() < 0 or labels.max() >= 50: raise ValueError("Invalid coordinates or cluster labels.")
    metadata = load_itunes()
    type_names = ["Studio", "Live / LiveAlbum", "Other / unclassified"]
    type_codes = {name: i for i, name in enumerate(type_names)}
    type_ids = [type_codes[classify(path, metadata)] for path in paths]
    plus = ["+" in path for path in paths]
    initial = [i for i in range(n) if type_ids[i] == 0 and labels[i] == 0]
    print(f"Total rows: {n:,}; initial Studio + Cluster 00 points: {len(initial):,}", flush=True)
    # Stable, visually separated colours for all 50 clusters.
    cluster_colours = ["#%02x%02x%02x" % tuple(round(c * 255) for c in colorsys.hsv_to_rgb(i / 50.0, 0.72, 0.98)) for i in range(50)]
    relative_paths = []
    for path in paths:
        relative_paths.append(path[len(MUSIC_PREFIX):] if path.lower().startswith(MUSIC_PREFIX.lower()) else path)
    initial_colours = [cluster_colours[int(labels[i])] for i in initial]
    # Per-cluster distance from the coordinate-wise median, used only for an optional visual filter.
    outlier_distance = np.zeros(n, dtype=np.float32)
    outlier_cutoffs = {}
    for cluster_id in range(50):
        member_ids = np.flatnonzero(labels == cluster_id)
        if len(member_ids) == 0:
            outlier_cutoffs[str(cluster_id)] = {"99": 0, "97.5": 0, "95": 0}
            continue
        centre = np.median(coords[member_ids], axis=0)
        distances = np.linalg.norm(coords[member_ids] - centre, axis=1)
        outlier_distance[member_ids] = distances
        outlier_cutoffs[str(cluster_id)] = {str(q): float(np.percentile(distances, q)) for q in (99, 97.5, 95)}
    fig = go.Figure(data=[go.Scatter3d(x=coords[initial,0], y=coords[initial,1], z=coords[initial,2], mode="markers", name="Tracks", marker=dict(size=3, color=initial_colours, opacity=0.9), customdata=np.asarray(relative_paths, dtype=object)[initial], hovertemplate="<b>%{customdata}</b><extra></extra>"), go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", name="Playlist route", showlegend=False, line=dict(width=5, color=[], colorscale="Turbo", cmin=0, cmax=1), marker=dict(size=4, color=[], colorscale="Turbo", cmin=0, cmax=1, showscale=False), customdata=[], text=[], hovertemplate="<b>%{text}</b><extra></extra>")])
    fig.update_layout(title="PulseFlow — raw CLAP 50-cluster map<br><sup>Points are coloured by cluster; click a point for links to the track</sup>", template="plotly_dark", height=850, scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3", aspectmode="data"), showlegend=False, margin=dict(l=0,r=0,t=75,b=0))
    html = fig.to_html(include_plotlyjs=True, full_html=True, div_id="pulseflow-map")
    payload = json.dumps({"x":coords[:,0].tolist(), "y":coords[:,1].tolist(), "z":coords[:,2].tolist(), "labels":labels.tolist(), "types":type_ids, "plus":plus, "paths":relative_paths, "colours":cluster_colours, "outlierDistance":outlier_distance.tolist(), "outlierCutoffs":outlier_cutoffs, "prefix":MUSIC_PREFIX, "initialCount":len(initial), "helper":PLAYER_HELPER, "inline":INLINE_AUDIO}, separators=(",",":"))
    clusters = "".join(f'<label style="display:flex;align-items:center;gap:7px"><input type="checkbox" class="cluster" value="{i}" {"checked" if i == 0 else ""}><span style="display:inline-block;width:11px;height:11px;border-radius:2px;background:{cluster_colours[i]}"></span>Cluster {i:02d}</label>' for i in range(50))
    controls = f'''<div id="controls" style="box-sizing:border-box;width:270px;min-width:270px;height:100vh;overflow-y:auto;padding:14px;background:#20242a;color:#eee;font:14px Arial"><h3 style="margin:0 0 12px">Map filters</h3><b>Recording type</b><div><label><input type="checkbox" class="type" value="0" checked> Studio</label></div><div><label><input type="checkbox" class="type" value="1"> Live / LiveAlbum</label></div><div><label><input type="checkbox" class="type" value="2"> Other / unclassified</label></div><div style="margin:10px 0 10px"><label><input type="checkbox" id="excludePlus"> Exclude paths containing +</label></div><div style="margin:8px 0 16px;padding:8px;border:1px solid #555;border-radius:5px"><label><input type="checkbox" id="hideOutliers"> Hide distant 3D points</label><div style="margin:8px 0 0 22px;font-size:12px;color:#ccc">Hide points beyond each cluster's distance percentile:</div><select id="outlierLevel" style="margin:6px 0 0 22px;max-width:205px"><option value="99">Very few (99th percentile)</option><option value="97.5" selected>Moderate (97.5th percentile)</option><option value="95">More (95th percentile)</option></select><div style="font-size:11px;color:#aaa;margin:6px 0 0 22px">Visual filter only; no tracks are deleted.</div></div><hr style="border-color:#555"><b>Playlist route</b><div style="margin:8px 0"><input type="file" id="playlistFile" accept=".m3u,.m3u8,.txt" style="max-width:240px"></div><div><label><input type="checkbox" id="fadeOthers" checked> Fade other tracks</label></div><div style="margin:6px 0"><button id="clearRoute">Clear route</button></div><p id="playlistStatus" style="font-size:12px;color:#ccc;line-height:1.4;margin:4px 0 10px">Load an .m3u/.m3u8 file to draw its order as a route.</p><hr style="border-color:#555"><b>CLAP clusters</b><div style="display:flex;flex-direction:column;gap:5px;margin-top:8px">{clusters}</div><div style="position:sticky;bottom:0;background:#20242a;padding:10px 0"><button id="allClusters">Select all</button> <button id="noClusters">Clear all</button></div><p id="status" style="line-height:1.4">Preparing initial selection...</p></div>'''
    js = r'''<script>
const DATA = __PULSEFLOW_DATA__;
const plot = document.getElementById('pulseflow-map');
const status = document.getElementById('status');
let busy = false;
function selectedValues(selector) { return [...document.querySelectorAll(selector+':checked')].map(e => Number(e.value)); }
function updateMap() {
 if (busy) return; busy = true;
 const chosenTypes = new Set(selectedValues('.type'));
 const chosenClusters = new Set(selectedValues('.cluster'));
 const excludePlus = document.getElementById('excludePlus').checked;
 const hideOutliers = document.getElementById('hideOutliers').checked;
 const outlierLevel = document.getElementById('outlierLevel').value;
 const xs=[], ys=[], zs=[], paths=[], colours=[];
 for (let i=0; i<DATA.paths.length; i++) {
  const cluster = DATA.labels[i];
  if (!chosenTypes.has(DATA.types[i]) || !chosenClusters.has(cluster) || (excludePlus && DATA.plus[i])) continue;
  if (hideOutliers && DATA.outlierDistance[i] > DATA.outlierCutoffs[String(cluster)][outlierLevel]) continue;
  xs.push(DATA.x[i]); ys.push(DATA.y[i]); zs.push(DATA.z[i]); paths.push(DATA.paths[i]); colours.push(DATA.colours[DATA.labels[i]]);
 }
 status.textContent = 'Rendering ' + paths.length.toLocaleString() + ' of ' + DATA.paths.length.toLocaleString() + ' tracks...';
 Plotly.restyle(plot, {x:[xs], y:[ys], z:[zs], customdata:[paths], 'marker.color':[colours]}, [0]).then(() => { status.textContent = 'Showing ' + paths.length.toLocaleString() + ' of ' + DATA.paths.length.toLocaleString() + ' tracks.'; busy = false; }).catch(err => { status.textContent = 'Plot update failed: ' + err; busy = false; });
}
document.querySelectorAll('.type,.cluster,#excludePlus,#hideOutliers,#outlierLevel').forEach(e => e.addEventListener('change', updateMap));
document.getElementById('allClusters').addEventListener('click', () => { document.querySelectorAll('.cluster').forEach(e => e.checked=true); updateMap(); });
document.getElementById('noClusters').addEventListener('click', () => { document.querySelectorAll('.cluster').forEach(e => e.checked=false); updateMap(); });
status.textContent = 'Showing ' + DATA.initialCount.toLocaleString() + ' of ' + DATA.paths.length.toLocaleString() + ' tracks. Choose filters to update.';

// ---- Click a point: show clickable links to the track and its folder ----
const panel = document.createElement('div');
panel.style.cssText = 'position:fixed;right:16px;bottom:16px;max-width:520px;padding:12px 14px;background:#1e1e1e;color:#eee;border:1px solid #555;border-radius:8px;font:14px/1.5 Arial,sans-serif;z-index:9999;display:none;box-shadow:0 4px 16px rgba(0,0,0,.5)';
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
 const full = fullPath(String(cd));
 const folder = full.replace(/[\\\/][^\\\/]*$/, '');
 const title = full.split(/[\\\/]/).pop().replace(/\.[^.]+$/, '');
 const parts = String(cd).split(/[\\\/]/);
 const sub = parts.length >= 3 ? parts[parts.length-3] + ' \u2014 ' + parts[parts.length-2] : parts.slice(0,-1).join(' / ');
 const ci = idxOf(cd);
 const clusterHtml = ci >= 0 ? '<div style="margin-top:2px"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;background:' + DATA.colours[DATA.labels[ci]] + '"></span>Cluster ' + String(DATA.labels[ci]).padStart(2, '0') + '</div>' : '';
 panel.innerHTML = '<div style="font-weight:600;font-size:15px">' + esc(title) + '</div><div style="color:#bbb">' + esc(sub) + '</div>' + clusterHtml + '<div style="margin-top:8px"><a href="' + fileUrl(full) + '" target="_blank" style="color:#6cb6ff">&#9654; Open track</a> &nbsp;|&nbsp; <a href="' + fileUrl(folder) + '" target="_blank" style="color:#6cb6ff">&#128193; Open folder</a> &nbsp;|&nbsp; <a href="#" id="pf-copy" style="color:#6cb6ff">Copy path</a> &nbsp;|&nbsp; <a href="#" id="pf-close" style="color:#888">Close</a></div>';
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

// ---- Playlist route: load an m3u/m3u8 and draw it as a line through the map ----
const plInput = document.getElementById('playlistFile');
const plStatus = document.getElementById('playlistStatus');
const fadeBox = document.getElementById('fadeOthers');
let lookup = null, routeLen = 0;
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
function applyFade() {
 Plotly.restyle(plot, {'marker.opacity': [fadeBox.checked && routeLen > 0 ? 0.12 : 0.9]}, [0]);
}
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
  panel.style.right = '16px';
  Plotly.Plots.resize(plot);
  return;
 }
 document.getElementById('plTitle').textContent = title ? title.replace(/\.[^.]+$/, '') : 'Playlist';
 document.getElementById('plCount').textContent = ids.length + ' tracks on the map' + (missed && missed.length ? ' \u00b7 ' + missed.length + ' not found' : '');
 ids.forEach(i => {
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
 });
 if (missed && missed.length) {
  document.getElementById('plMissed').innerHTML = '<details><summary style="cursor:pointer">Not found (' + missed.length + ')</summary>' + missed.map(m => '<div style="margin:3px 0;word-break:break-all">' + esc(m) + '</div>').join('') + '</details>';
 }
 plPanel.style.display = 'block';
 panel.style.right = '316px';
 Plotly.Plots.resize(plot);
}
plInput.addEventListener('change', async () => {
 const file = plInput.files[0];
 if (!file) return;
 const buf = await file.arrayBuffer();
 let text;
 try { text = new TextDecoder('utf-8', {fatal: true}).decode(buf); } catch (e) { text = new TextDecoder('windows-1252').decode(buf); }
 const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/).map(l => l.trim()).filter(l => l && l[0] !== '#');
 if (!lookup) {
  plStatus.textContent = 'Indexing library...';
  await new Promise(r => setTimeout(r, 30));
  lookup = buildLookup();
 }
 const ids = [], missed = [];
 for (const line of lines) {
  const i = matchTrack(line, lookup);
  if (i >= 0) ids.push(i); else missed.push(line);
 }
 drawRoute(ids, file.name, missed);
 plStatus.textContent = file.name + ': ' + ids.length + ' of ' + lines.length + ' tracks matched.' + (missed.length ? ' ' + missed.length + ' not found (listed in the browser console).' : '');
 if (missed.length) console.log('Playlist entries not matched:', missed);
});
fadeBox.addEventListener('change', applyFade);
document.getElementById('clearRoute').addEventListener('click', () => {
 plInput.value = '';
 drawRoute([], '', []);
 plStatus.textContent = 'Route cleared.';
});
</script>'''.replace("__PULSEFLOW_DATA__", payload)
    html = html.replace("<body>", "<body>\n<div style=\"display:flex;width:100%;height:100vh;overflow:hidden\">\n" + controls + "\n<div style=\"flex:1;min-width:0;height:100vh\">", 1)
    right_panel = '<div id="playlistPanel" style="display:none;box-sizing:border-box;width:300px;min-width:300px;height:100vh;overflow-y:auto;padding:14px;background:#20242a;color:#eee;font:13px Arial"><h3 id="plTitle" style="margin:0 0 4px;word-break:break-word">Playlist</h3><div id="plCount" style="color:#aaa;margin-bottom:6px"></div><div id="plMsg" style="color:#e88;font-size:12px;margin-bottom:6px"></div><ol id="plList" style="margin:0;padding-left:30px;line-height:1.4"></ol><div id="plMissed" style="margin-top:12px;font-size:12px;color:#aaa"></div></div>'
    html = html.replace("</body>", "</div>\n" + right_panel + "\n</div>\n" + js + "\n</body>", 1)
    TEMP_OUTPUT.write_text(html, encoding="utf-8")
    if not TEMP_OUTPUT.is_file() or TEMP_OUTPUT.stat().st_size == 0: raise RuntimeError("HTML output was not created correctly.")
    os.replace(TEMP_OUTPUT, OUTPUT)
    print(f"Map saved: {OUTPUT}", flush=True)
    print(f"HTML size: {OUTPUT.stat().st_size/(1024*1024):.1f} MB")
    print("Existing coordinates, labels, production map and library were not modified.")
    print("Optional 3D outlier filter uses per-cluster distance percentiles; it is visual only.")
    print("Point hover labels omit the M:\\Music\\Music\\ prefix where present.")
    print("Click a point to get links to the track and its folder.")
    start_helper()

if __name__ == "__main__": main()
