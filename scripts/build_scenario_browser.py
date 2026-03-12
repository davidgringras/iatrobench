#!/usr/bin/env python3
"""Generate scenario_browser.html with all data embedded."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    with open(ROOT / "docs" / "viz_data.json") as f:
        data = json.load(f)

    scenarios_js = json.dumps(data["scenarios"])
    model_ids_js = json.dumps(data["model_ids"])
    model_labels_js = json.dumps(data["model_labels"])
    categories_js = json.dumps(data["categories"])

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Scenario Browser &mdash; IatroBench</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#x2695;</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*, *::before, *::after {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: 'Inter', -apple-system, sans-serif;
  background: #0a0a1a;
  color: #e0e0e0;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}}

.topbar {{
  padding: 16px 24px;
  border-bottom: 1px solid #1e1e3a;
  display: flex;
  align-items: center;
  gap: 16px;
}}
.topbar a {{
  color: #818cf8;
  text-decoration: none;
  font-size: 13px;
}}
.topbar a:hover {{ text-decoration: underline; }}
.topbar h1 {{
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 18px;
  font-weight: 600;
  color: #fff;
  flex: 1;
}}

.controls {{
  padding: 16px 24px;
  border-bottom: 1px solid #1e1e3a;
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  align-items: center;
}}
.controls input {{
  background: #12122a;
  border: 1px solid #2a2a4a;
  border-radius: 6px;
  color: #e0e0e0;
  padding: 8px 12px;
  font-size: 14px;
  font-family: inherit;
  flex: 1;
  min-width: 200px;
}}
.controls input::placeholder {{ color: #666; }}
.controls select {{
  background: #12122a;
  border: 1px solid #2a2a4a;
  border-radius: 6px;
  color: #e0e0e0;
  padding: 8px 12px;
  font-size: 14px;
  font-family: inherit;
}}
.controls .count {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  color: #666;
}}

.scenario-list {{
  padding: 0 24px 48px;
}}

.scenario-card {{
  border: 1px solid #1e1e3a;
  border-radius: 8px;
  margin-top: 12px;
  overflow: hidden;
  transition: border-color 0.15s;
}}
.scenario-card:hover {{
  border-color: #3a3a5a;
}}

.card-header {{
  padding: 16px 20px;
  cursor: pointer;
  display: flex;
  align-items: flex-start;
  gap: 16px;
  background: #0f0f24;
}}
.card-header:hover {{
  background: #12122a;
}}

.card-id {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: #818cf8;
  min-width: 40px;
  padding-top: 2px;
}}
.card-body {{
  flex: 1;
  min-width: 0;
}}
.card-title {{
  font-size: 14px;
  font-weight: 500;
  color: #fff;
  margin-bottom: 4px;
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}}
.card-prompt {{
  font-size: 13px;
  color: #888;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}}

.tag {{
  font-size: 11px;
  padding: 1px 6px;
  border-radius: 3px;
  font-weight: 500;
  white-space: nowrap;
}}
.tag-cat {{
  background: #1e1e3a;
  color: #818cf8;
}}
.tag-decoupling {{
  background: #1a1a2e;
  color: #f59e0b;
  border: 1px solid #f59e0b33;
}}

.card-scores {{
  display: flex;
  gap: 8px;
  flex-shrink: 0;
  align-items: center;
  flex-wrap: wrap;
}}
.score-pill {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 4px;
  white-space: nowrap;
}}
.oh-score {{
  background: #dc262622;
  color: #f87171;
}}
.ch-score {{
  background: #16a34a22;
  color: #4ade80;
}}

.card-expand {{
  color: #555;
  font-size: 14px;
  transition: transform 0.2s;
  padding-top: 2px;
}}
.card-expand.open {{
  transform: rotate(90deg);
}}

.card-detail {{
  display: none;
  padding: 0 20px 20px;
  background: #0a0a1a;
  border-top: 1px solid #1e1e3a;
}}
.card-detail.open {{
  display: block;
}}

.detail-section {{
  margin-top: 16px;
}}
.detail-section h3 {{
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #666;
  margin-bottom: 8px;
}}
.detail-prompt {{
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 14px;
  line-height: 1.7;
  color: #ccc;
  background: #12122a;
  padding: 16px;
  border-radius: 6px;
  white-space: pre-wrap;
}}
.detail-actions {{
  list-style: none;
  padding: 0;
}}
.detail-actions li {{
  font-size: 13px;
  color: #aaa;
  padding: 4px 0;
  padding-left: 16px;
  position: relative;
}}
.detail-actions li::before {{
  content: "\\2022";
  color: #818cf8;
  position: absolute;
  left: 0;
}}
.chart-container {{
  height: 220px;
  margin-top: 12px;
  border-radius: 6px;
  overflow: hidden;
}}

@media (max-width: 640px) {{
  .card-scores {{ display: none; }}
  .controls {{ flex-direction: column; }}
  .controls input {{ min-width: 100%; }}
}}
</style>
</head>
<body>

<div class="topbar">
  <a href="index.html">&larr; IatroBench</a>
  <h1>Scenario Browser</h1>
</div>

<div class="controls">
  <input type="text" id="search" placeholder="Search scenarios...">
  <select id="category-filter">
    <option value="">All categories</option>
  </select>
  <select id="sort-by">
    <option value="id">Sort by ID</option>
    <option value="oh-desc">Highest OH first</option>
    <option value="oh-asc">Lowest OH first</option>
    <option value="gap-desc">Largest gap first</option>
  </select>
  <span class="count" id="count"></span>
</div>

<div class="scenario-list" id="list"></div>

<script>
const DATA = {scenarios_js};
const MODEL_IDS = {model_ids_js};
const MODEL_LABELS = {model_labels_js};
const CATEGORIES = {categories_js};

const catFilter = document.getElementById('category-filter');
CATEGORIES.forEach(c => {{
  const opt = document.createElement('option');
  opt.value = c; opt.textContent = c;
  catFilter.appendChild(opt);
}});

function meanOH(s) {{
  const vals = MODEL_IDS.map(m => s.scores[m]?.oh).filter(v => v != null);
  return vals.length ? vals.reduce((a,b) => a+b, 0) / vals.length : 0;
}}

function decouplingGap(s) {{
  // Find partner and compute gap
  if (!s.has_decoupling || s.framing !== 'layperson') return null;
  const partner = DATA.find(d => d.pair_id === s.pair_id && d.framing === 'physician');
  if (!partner) return null;
  const gaps = MODEL_IDS.map(m => {{
    const lay = s.scores[m]?.oh;
    const phys = partner.scores[m]?.oh;
    return (lay != null && phys != null) ? lay - phys : null;
  }}).filter(v => v != null);
  return gaps.length ? gaps.reduce((a,b) => a+b, 0) / gaps.length : 0;
}}

function render() {{
  const query = document.getElementById('search').value.toLowerCase();
  const cat = catFilter.value;
  const sort = document.getElementById('sort-by').value;

  let filtered = DATA.filter(s => {{
    if (cat && s.category !== cat) return false;
    if (query && !s.prompt.toLowerCase().includes(query) && !s.id.toLowerCase().includes(query) && !s.category.toLowerCase().includes(query)) return false;
    return true;
  }});

  if (sort === 'oh-desc') filtered.sort((a,b) => meanOH(b) - meanOH(a));
  else if (sort === 'oh-asc') filtered.sort((a,b) => meanOH(a) - meanOH(b));
  else if (sort === 'gap-desc') filtered.sort((a,b) => (decouplingGap(b)||0) - (decouplingGap(a)||0));
  else filtered.sort((a,b) => a.id.localeCompare(b.id));

  document.getElementById('count').textContent = filtered.length + ' / ' + DATA.length;

  const list = document.getElementById('list');
  list.innerHTML = '';

  filtered.forEach((s, i) => {{
    const mOH = meanOH(s);
    const card = document.createElement('div');
    card.className = 'scenario-card';

    const tags = [`<span class="tag tag-cat">${{s.category}}</span>`];
    if (s.has_decoupling) tags.push(`<span class="tag tag-decoupling">${{s.framing}}</span>`);

    const ohColor = mOH >= 2 ? '#ef4444' : mOH >= 1 ? '#f59e0b' : '#4ade80';

    card.innerHTML = `
      <div class="card-header" onclick="toggleCard(this)">
        <span class="card-id">${{s.id}}</span>
        <div class="card-body">
          <div class="card-title">${{tags.join('')}}</div>
          <div class="card-prompt">${{escHtml(s.prompt)}}</div>
        </div>
        <div class="card-scores">
          <span class="score-pill oh-score">OH ${{mOH.toFixed(1)}}</span>
        </div>
        <span class="card-expand">&#9654;</span>
      </div>
      <div class="card-detail" id="detail-${{s.id}}">
        <div class="detail-section">
          <h3>Full Prompt</h3>
          <div class="detail-prompt">${{escHtml(s.prompt)}}</div>
        </div>
        <div class="detail-section">
          <h3>Critical Actions (${{s.critical_actions.length}})</h3>
          <ul class="detail-actions">
            ${{s.critical_actions.map(a => `<li>${{escHtml(a)}}</li>`).join('')}}
          </ul>
        </div>
        <div class="detail-section">
          <h3>Model Scores</h3>
          <div class="chart-container" id="chart-${{s.id}}"></div>
        </div>
      </div>
    `;
    list.appendChild(card);
  }});
}}

function escHtml(str) {{
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}}

function toggleCard(header) {{
  const detail = header.nextElementSibling;
  const arrow = header.querySelector('.card-expand');
  const isOpen = detail.classList.toggle('open');
  arrow.classList.toggle('open', isOpen);

  if (isOpen) {{
    const sid = detail.id.replace('detail-', '');
    const s = DATA.find(d => d.id === sid);
    if (s) renderChart(sid, s);
  }}
}}

function renderChart(sid, s) {{
  const el = document.getElementById('chart-' + sid);
  if (el.dataset.rendered) return;
  el.dataset.rendered = '1';

  const models = MODEL_IDS.filter(m => s.scores[m]);
  const labels = models.map(m => MODEL_LABELS[m]);
  const ohVals = models.map(m => s.scores[m].oh);
  const chVals = models.map(m => s.scores[m].ch);

  const traces = [
    {{
      x: labels, y: ohVals, name: 'Omission Harm',
      type: 'bar', marker: {{ color: '#ef4444' }}
    }},
    {{
      x: labels, y: chVals, name: 'Commission Harm',
      type: 'bar', marker: {{ color: '#22c55e' }}
    }}
  ];

  Plotly.newPlot(el, traces, {{
    barmode: 'group',
    paper_bgcolor: '#0a0a1a',
    plot_bgcolor: '#0a0a1a',
    font: {{ family: 'Inter, sans-serif', color: '#888', size: 11 }},
    margin: {{ t: 10, b: 60, l: 40, r: 10 }},
    xaxis: {{ tickangle: -30, gridcolor: '#1e1e3a' }},
    yaxis: {{ gridcolor: '#1e1e3a', range: [0, 4], dtick: 1, title: 'Score' }},
    legend: {{ orientation: 'h', y: 1.12, font: {{ size: 11 }} }},
    showlegend: true,
  }}, {{ responsive: true, displayModeBar: false }});
}}

document.getElementById('search').addEventListener('input', render);
catFilter.addEventListener('change', render);
document.getElementById('sort-by').addEventListener('change', render);

render();
</script>
</body>
</html>'''

    out = ROOT / "docs" / "scenario_browser.html"
    out.write_text(html)
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()
