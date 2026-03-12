#!/usr/bin/env python3
"""Generate decoupling_viz.html with all data embedded."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    with open(ROOT / "docs" / "viz_data.json") as f:
        data = json.load(f)

    pairs_js = json.dumps(data["decoupling_pairs"])
    model_ids_js = json.dumps(data["model_ids"])
    model_labels_js = json.dumps(data["model_labels"])
    h2_js = json.dumps(data["h2"])

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Decoupling Evaluation &mdash; IatroBench</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#x2695;</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
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

.hero {{
  max-width: 860px;
  margin: 0 auto;
  padding: 32px 24px 16px;
}}
.hero h2 {{
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 28px;
  font-weight: 700;
  color: #fff;
  margin-bottom: 8px;
}}
.hero p {{
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 16px;
  color: #999;
  line-height: 1.7;
  max-width: 700px;
}}
.hero .stat {{
  font-family: 'JetBrains Mono', monospace;
  color: #ef4444;
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
  height: 450px;
}}
.chart.wide {{
  height: 500px;
}}

.model-selector {{
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}}
.model-btn {{
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 4px;
  border: 1px solid #2a2a4a;
  background: #12122a;
  color: #888;
  cursor: pointer;
  transition: all 0.15s;
}}
.model-btn.active {{
  border-color: #818cf8;
  color: #818cf8;
  background: #818cf822;
}}
.model-btn:hover {{
  border-color: #555;
}}

@media (max-width: 800px) {{
  .dashboard {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>

<div class="topbar">
  <a href="index.html">&larr; IatroBench</a>
  <h1>Decoupling Evaluation</h1>
</div>

<div class="hero">
  <h2>The Identity Gap</h2>
  <p>
    Matched clinical scenarios presented in layperson vs. physician framing reveal systematic identity-contingent withholding.
    All five testable models provide better guidance to physicians
    (<span class="stat">gap +0.38, p&nbsp;=&nbsp;0.003</span>),
    with the most safety-trained model showing the largest gap
    (<span class="stat">Opus +0.65</span>).
  </p>
</div>

<div class="dashboard">

  <div class="panel full">
    <h2>Per-Model Decoupling Gap</h2>
    <p class="subtitle">Mean OH difference (layperson minus physician) across 22 matched scenario pairs</p>
    <div class="chart" id="gap-overview" style="height:300px"></div>
  </div>

  <div class="panel full">
    <h2>Slope Chart: Layperson vs. Physician OH by Scenario</h2>
    <p class="subtitle">Each line connects a scenario pair. Lines sloping upward to the right indicate higher layperson OH. Select a model to view.</p>
    <div class="model-selector" id="model-selector"></div>
    <div class="chart wide" id="slope-chart"></div>
  </div>

  <div class="panel">
    <h2>Gap Distribution</h2>
    <p class="subtitle">Per-pair gap across all models (positive = worse for laypersons)</p>
    <div class="chart" id="gap-hist"></div>
  </div>

  <div class="panel">
    <h2>Layperson vs. Physician Scatter</h2>
    <p class="subtitle">Each point is one scenario pair for one model. Above diagonal = gap favours physician.</p>
    <div class="chart" id="scatter"></div>
  </div>

</div>

<script>
const PAIRS = {pairs_js};
const MODEL_IDS = {model_ids_js};
const MODEL_LABELS = {model_labels_js};
const H2 = {h2_js};

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

// 1. Per-model gap overview
const testable = MODEL_IDS.filter(m => H2.per_model[m]);
const gapVals = testable.map(m => H2.per_model[m].mean_gap);
const gapColors = gapVals.map((v, i) => {{
  if (v > 0) return testable[i] === 'opus' ? '#818cf8' : '#ef4444';
  return '#38bdf8';
}});
const nPos = testable.map(m => H2.per_model[m].n_positive);
const nPairs = testable.map(m => H2.per_model[m].n_pairs);

Plotly.newPlot('gap-overview', [{{
  x: testable.map(m => MODEL_LABELS[m]),
  y: gapVals,
  type: 'bar',
  marker: {{ color: gapColors }},
  text: gapVals.map((v, i) => (v > 0 ? '+' : '') + v.toFixed(2) + ' (' + nPos[i] + '/' + nPairs[i] + ')'),
  textposition: 'outside',
  textfont: {{ family: 'JetBrains Mono', size: 12 }},
  customdata: nPos.map((n,i) => n + '/' + nPairs[i]),
  hovertemplate: '%{{x}}<br>Gap: %{{y:.2f}}<br>Positive pairs: %{{customdata}}<extra></extra>',
}}], {{
  paper_bgcolor: PAPER_BG, plot_bgcolor: PLOT_BG, font: FONT,
  margin: {{ t: 30, b: 80, l: 50, r: 10 }},
  yaxis: {{ gridcolor: GRID, title: 'Gap (OH points)', zeroline: true, zerolinecolor: '#555', zerolinewidth: 2 }},
  xaxis: {{ tickangle: -20 }},
  showlegend: false,
  shapes: [{{
    type: 'line', x0: -0.5, x1: testable.length - 0.5,
    y0: H2.overall_gap, y1: H2.overall_gap,
    line: {{ color: '#ef444466', dash: 'dash', width: 1 }},
  }}],
  annotations: [{{
    x: testable.length - 1, y: H2.overall_gap + 0.04,
    text: 'Overall +' + H2.overall_gap.toFixed(2) + ' (p=0.003)',
    showarrow: false, font: {{ size: 10, color: '#ef4444' }},
  }}],
}}, {{ responsive: true, displayModeBar: false }});

// 2. Slope chart (select by model)
let activeModel = 'opus';

function buildSlopeChart(modelId) {{
  const traces = [];
  const pairsWithModel = PAIRS.filter(p => p.models[modelId]);

  pairsWithModel.forEach((p, i) => {{
    const d = p.models[modelId];
    const gap = d.gap;
    const color = gap > 0 ? '#ef444488' : gap < 0 ? '#38bdf888' : '#55555588';
    const width = Math.abs(gap) > 0.5 ? 2 : 1;

    traces.push({{
      x: ['Physician', 'Layperson'],
      y: [d.phys_oh, d.lay_oh],
      mode: 'lines+markers',
      line: {{ color, width }},
      marker: {{ size: 6 }},
      name: p.pair_id.replace(/_/g, ' '),
      hovertemplate: p.pair_id.replace(/_/g, ' ') + '<br>Phys OH: ' + d.phys_oh + '<br>Lay OH: ' + d.lay_oh + '<br>Gap: ' + (gap > 0 ? '+' : '') + gap.toFixed(2) + '<extra></extra>',
      showlegend: false,
    }});
  }});

  // Mean line
  const pm = H2.per_model[modelId];
  if (pm) {{
    traces.push({{
      x: ['Physician', 'Layperson'],
      y: [pm.phys_oh_mean, pm.lay_oh_mean],
      mode: 'lines+markers',
      line: {{ color: MODEL_COLORS[modelId], width: 3 }},
      marker: {{ size: 10, symbol: 'diamond' }},
      name: 'Mean',
      showlegend: true,
    }});
  }}

  Plotly.newPlot('slope-chart', traces, {{
    paper_bgcolor: PAPER_BG, plot_bgcolor: PLOT_BG, font: FONT,
    margin: {{ t: 10, b: 40, l: 50, r: 30 }},
    yaxis: {{ gridcolor: GRID, title: 'Mean OH Score', range: [-0.2, 4.2], dtick: 1 }},
    xaxis: {{ fixedrange: true }},
    showlegend: true,
    legend: {{ x: 1, xanchor: 'right', y: 1, font: {{ size: 11 }} }},
  }}, {{ responsive: true, displayModeBar: false }});
}}

const selector = document.getElementById('model-selector');
MODEL_IDS.forEach(m => {{
  const btn = document.createElement('button');
  btn.className = 'model-btn' + (m === activeModel ? ' active' : '');
  btn.textContent = MODEL_LABELS[m];
  btn.style.borderColor = MODEL_COLORS[m] + '88';
  btn.addEventListener('click', () => {{
    activeModel = m;
    selector.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    buildSlopeChart(m);
  }});
  selector.appendChild(btn);
}});
buildSlopeChart(activeModel);

// 3. Gap distribution histogram (all models combined)
const allGaps = [];
PAIRS.forEach(p => {{
  MODEL_IDS.forEach(m => {{
    if (p.models[m]) allGaps.push(p.models[m].gap);
  }});
}});

Plotly.newPlot('gap-hist', [{{
  x: allGaps,
  type: 'histogram',
  marker: {{ color: '#818cf8', line: {{ color: '#818cf8', width: 1 }} }},
  opacity: 0.7,
  xbins: {{ size: 0.25 }},
}}], {{
  paper_bgcolor: PAPER_BG, plot_bgcolor: PLOT_BG, font: FONT,
  margin: {{ t: 10, b: 40, l: 40, r: 10 }},
  xaxis: {{ gridcolor: GRID, title: 'Gap (lay OH - phys OH)', zeroline: true, zerolinecolor: '#ef4444', zerolinewidth: 2 }},
  yaxis: {{ gridcolor: GRID, title: 'Count' }},
  shapes: [{{
    type: 'line', x0: H2.overall_gap, x1: H2.overall_gap, y0: 0, y1: 1, yref: 'paper',
    line: {{ color: '#ef4444', dash: 'dash', width: 1 }},
  }}],
  annotations: [{{
    x: H2.overall_gap, y: 1, yref: 'paper',
    text: 'Mean +' + H2.overall_gap.toFixed(2),
    showarrow: true, arrowhead: 0, ax: 40, ay: -20,
    font: {{ size: 10, color: '#ef4444' }},
  }}],
}}, {{ responsive: true, displayModeBar: false }});

// 4. Scatter: lay OH vs phys OH
const scatterTraces = MODEL_IDS.map(m => {{
  const xs = [], ys = [], texts = [];
  PAIRS.forEach(p => {{
    if (p.models[m]) {{
      xs.push(p.models[m].phys_oh);
      ys.push(p.models[m].lay_oh);
      texts.push(p.pair_id.replace(/_/g, ' '));
    }}
  }});
  return {{
    x: xs, y: ys, text: texts,
    mode: 'markers',
    marker: {{ color: MODEL_COLORS[m], size: 6, opacity: 0.7 }},
    name: MODEL_LABELS[m].split(' ').slice(0,2).join(' '),
    hovertemplate: '%{{text}}<br>Phys: %{{x}}<br>Lay: %{{y}}<extra>' + MODEL_LABELS[m] + '</extra>',
  }};
}});

Plotly.newPlot('scatter', scatterTraces, {{
  paper_bgcolor: PAPER_BG, plot_bgcolor: PLOT_BG, font: FONT,
  margin: {{ t: 10, b: 50, l: 50, r: 10 }},
  xaxis: {{ gridcolor: GRID, title: 'Physician OH', range: [-0.3, 4.3], dtick: 1 }},
  yaxis: {{ gridcolor: GRID, title: 'Layperson OH', range: [-0.3, 4.3], dtick: 1 }},
  showlegend: true,
  legend: {{ x: 0.02, y: 0.98, font: {{ size: 10 }} }},
  shapes: [{{
    type: 'line', x0: 0, x1: 4, y0: 0, y1: 4,
    line: {{ color: '#555', dash: 'dash', width: 1 }},
  }}],
  annotations: [{{
    x: 3.5, y: 3.8, text: 'Lay > Phys', showarrow: false,
    font: {{ size: 10, color: '#ef4444' }},
  }}, {{
    x: 3.8, y: 3.5, text: 'Phys > Lay', showarrow: false,
    font: {{ size: 10, color: '#38bdf8' }},
  }}],
}}, {{ responsive: true, displayModeBar: false }});
</script>
</body>
</html>'''

    out = ROOT / "docs" / "decoupling_viz.html"
    out.write_text(html)
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()
