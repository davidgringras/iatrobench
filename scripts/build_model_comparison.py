#!/usr/bin/env python3
"""Generate model_comparison.html with all data embedded."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    with open(ROOT / "docs" / "viz_data.json") as f:
        data = json.load(f)

    oh_dists_js = json.dumps(data["model_oh_distributions"])
    ch_dists_js = json.dumps(data["model_ch_distributions"])
    model_ids_js = json.dumps(data["model_ids"])
    model_labels_js = json.dumps(data["model_labels"])
    h1_js = json.dumps(data["h1"])
    h2_js = json.dumps(data["h2"])
    heatmap_js = json.dumps(data["category_heatmap"])

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Comparison &mdash; IatroBench</title>
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
}}

.dashboard {{
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}}

.panel {{
  background: #0f0f24;
  border: 1px solid #1e1e3a;
  border-radius: 8px;
  padding: 20px;
}}
.panel.full {{
  grid-column: 1 / -1;
}}
.panel h2 {{
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 16px;
  font-weight: 600;
  color: #fff;
  margin-bottom: 4px;
}}
.panel .subtitle {{
  font-size: 12px;
  color: #666;
  margin-bottom: 16px;
}}
.chart {{
  width: 100%;
  height: 350px;
}}
.chart.tall {{
  height: 400px;
}}

.stats-grid {{
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 1px;
  background: #1e1e3a;
  border-radius: 6px;
  overflow: hidden;
  margin-top: 16px;
}}
.stat-cell {{
  background: #0f0f24;
  padding: 12px 8px;
  text-align: center;
}}
.stat-model {{
  font-size: 11px;
  color: #888;
  margin-bottom: 4px;
}}
.stat-value {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 16px;
  font-weight: 400;
}}
.stat-label {{
  font-size: 10px;
  color: #555;
  margin-top: 2px;
}}
.stat-oh {{ color: #ef4444; }}
.stat-ch {{ color: #4ade80; }}

@media (max-width: 800px) {{
  .dashboard {{ grid-template-columns: 1fr; }}
  .stats-grid {{ grid-template-columns: repeat(3, 1fr); }}
}}
</style>
</head>
<body>

<div class="topbar">
  <a href="index.html">&larr; IatroBench</a>
  <h1>Model Comparison</h1>
</div>

<div class="dashboard">

  <div class="panel full">
    <h2>Summary Statistics</h2>
    <p class="subtitle">Mean scores from clinician audit (structured evaluation, N=785). Distributions below use primary judge scores.</p>
    <div class="stats-grid" id="stats-grid"></div>
  </div>

  <div class="panel">
    <h2>Omission Harm Distribution</h2>
    <p class="subtitle">All individual response scores (primary judge, N=600 per model)</p>
    <div class="chart" id="oh-dist"></div>
  </div>

  <div class="panel">
    <h2>Commission Harm Distribution</h2>
    <p class="subtitle">All individual response scores (primary judge, N=600 per model)</p>
    <div class="chart" id="ch-dist"></div>
  </div>

  <div class="panel">
    <h2>Decoupling Gap by Model</h2>
    <p class="subtitle">Layperson OH minus Physician OH (positive = worse for laypersons)</p>
    <div class="chart" id="gap-chart"></div>
  </div>

  <div class="panel">
    <h2>Category Heatmap</h2>
    <p class="subtitle">Mean OH by model and clinical category (primary judge)</p>
    <div class="chart tall" id="heatmap"></div>
  </div>

</div>

<script>
const OH_DISTS = {oh_dists_js};
const CH_DISTS = {ch_dists_js};
const MODEL_IDS = {model_ids_js};
const MODEL_LABELS = {model_labels_js};
const H1 = {h1_js};
const H2 = {h2_js};
const HEATMAP = {heatmap_js};

const PLOT_BG = '#0f0f24';
const PAPER_BG = '#0f0f24';
const GRID = '#1a1a30';
const FONT = {{ family: 'Inter, sans-serif', color: '#888', size: 11 }};
const MODEL_COLORS = {{
  opus: '#818cf8',
  gpt52: '#f472b6',
  gemini3pro: '#34d399',
  llama4: '#fb923c',
  deepseek: '#38bdf8',
  mistral: '#fbbf24',
}};

// Summary stats
const grid = document.getElementById('stats-grid');
MODEL_IDS.forEach(m => {{
  const d = H1[m];
  grid.innerHTML += `
    <div class="stat-cell">
      <div class="stat-model">${{MODEL_LABELS[m].split(' ')[0]}}</div>
      <div class="stat-value stat-oh">${{d.mean_oh.toFixed(2)}}</div>
      <div class="stat-label">mean OH</div>
      <div class="stat-value stat-ch" style="margin-top:8px">${{d.mean_ch.toFixed(2)}}</div>
      <div class="stat-label">mean CH</div>
    </div>
  `;
}});

// OH Distribution (violin/box)
const ohTraces = MODEL_IDS.map(m => ({{
  y: OH_DISTS[m],
  name: MODEL_LABELS[m].split(' ').slice(0,2).join(' '),
  type: 'violin',
  box: {{ visible: true }},
  meanline: {{ visible: true }},
  line: {{ color: MODEL_COLORS[m] }},
  fillcolor: MODEL_COLORS[m] + '33',
  points: false,
  span: [0, 4],
  spanmode: 'hard',
}}));
Plotly.newPlot('oh-dist', ohTraces, {{
  paper_bgcolor: PAPER_BG, plot_bgcolor: PLOT_BG, font: FONT,
  margin: {{ t: 10, b: 40, l: 40, r: 10 }},
  yaxis: {{ gridcolor: GRID, title: 'OH Score', range: [-0.5, 4.5], dtick: 1 }},
  xaxis: {{ tickangle: -20 }},
  showlegend: false,
  violinmode: 'group',
}}, {{ responsive: true, displayModeBar: false }});

// CH Distribution
const chTraces = MODEL_IDS.map(m => ({{
  y: CH_DISTS[m],
  name: MODEL_LABELS[m].split(' ').slice(0,2).join(' '),
  type: 'violin',
  box: {{ visible: true }},
  meanline: {{ visible: true }},
  line: {{ color: MODEL_COLORS[m] }},
  fillcolor: MODEL_COLORS[m] + '33',
  points: false,
  span: [0, 3],
  spanmode: 'hard',
}}));
Plotly.newPlot('ch-dist', chTraces, {{
  paper_bgcolor: PAPER_BG, plot_bgcolor: PLOT_BG, font: FONT,
  margin: {{ t: 10, b: 40, l: 40, r: 10 }},
  yaxis: {{ gridcolor: GRID, title: 'CH Score', range: [-0.5, 3.5], dtick: 1 }},
  xaxis: {{ tickangle: -20 }},
  showlegend: false,
  violinmode: 'group',
}}, {{ responsive: true, displayModeBar: false }});

// Decoupling Gap bar chart
const gapModels = MODEL_IDS.filter(m => H2.per_model[m]);
const gapVals = gapModels.map(m => H2.per_model[m].mean_gap);
const gapColors = gapVals.map(v => v > 0 ? '#ef4444' : '#38bdf8');

Plotly.newPlot('gap-chart', [{{
  x: gapModels.map(m => MODEL_LABELS[m]),
  y: gapVals,
  type: 'bar',
  marker: {{ color: gapColors }},
  text: gapVals.map(v => (v > 0 ? '+' : '') + v.toFixed(2)),
  textposition: 'outside',
  textfont: {{ family: 'JetBrains Mono', size: 12 }},
}}], {{
  paper_bgcolor: PAPER_BG, plot_bgcolor: PLOT_BG, font: FONT,
  margin: {{ t: 40, b: 80, l: 50, r: 10 }},
  yaxis: {{ gridcolor: GRID, title: 'Gap (OH points)', zeroline: true, zerolinecolor: '#444', range: [-0.75, 0.85] }},
  xaxis: {{ tickangle: -30 }},
  showlegend: false,
  shapes: [{{
    type: 'line', x0: -0.5, x1: gapModels.length - 0.5,
    y0: H2.overall_gap, y1: H2.overall_gap,
    line: {{ color: '#ef444488', dash: 'dash', width: 1 }},
  }}],
  annotations: [{{
    x: gapModels.length - 1, y: H2.overall_gap,
    text: 'Overall +' + H2.overall_gap.toFixed(2),
    showarrow: false, font: {{ size: 10, color: '#ef4444' }},
    yshift: 12
  }}],
}}, {{ responsive: true, displayModeBar: false }});

// Category Heatmap
const cats = Object.keys(HEATMAP);
const z = cats.map(c => MODEL_IDS.map(m => HEATMAP[c][m]));
const heatText = z.map(row => row.map(v => v != null ? v.toFixed(2) : ''));

Plotly.newPlot('heatmap', [{{
  z: z,
  x: MODEL_IDS.map(m => MODEL_LABELS[m].split(' ').slice(0,2).join(' ')),
  y: cats,
  type: 'heatmap',
  colorscale: [
    [0, '#0a0a1a'],
    [0.25, '#1e3a5f'],
    [0.5, '#f59e0b'],
    [0.75, '#ef4444'],
    [1, '#dc2626'],
  ],
  zmin: 0, zmax: 3,
  text: heatText,
  texttemplate: '%{{text}}',
  textfont: {{ size: 11, family: 'JetBrains Mono' }},
  hovertemplate: '%{{y}}<br>%{{x}}: OH %{{z:.2f}}<extra></extra>',
  colorbar: {{ title: 'OH', titlefont: {{ size: 11 }}, tickfont: {{ size: 10 }} }},
}}], {{
  paper_bgcolor: PAPER_BG, plot_bgcolor: PLOT_BG, font: FONT,
  margin: {{ t: 10, b: 60, l: 120, r: 60 }},
  xaxis: {{ tickangle: -30, side: 'bottom' }},
  yaxis: {{ autorange: 'reversed' }},
}}, {{ responsive: true, displayModeBar: false }});
</script>
</body>
</html>'''

    out = ROOT / "docs" / "model_comparison.html"
    out.write_text(html)
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()
