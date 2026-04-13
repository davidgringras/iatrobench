#!/usr/bin/env python3
"""Score the IatroBench paper prose against Pangram AI detection API.

Usage:
    python scripts/pangram_score.py                    # Full paper score
    python scripts/pangram_score.py --segments         # Per-segment sweep
    python scripts/pangram_score.py --section 59 76    # Score lines 59-76

Requires PANGRAM_API_KEY env var.
"""
import re, json, urllib.request, sys, time, os

API_KEY = os.environ.get('PANGRAM_API_KEY', '')
if not API_KEY:
    raise RuntimeError("Set PANGRAM_API_KEY in your environment")
PAPER_PATH = os.path.join(os.path.dirname(__file__), '..', 'paper', 'main.tex')


def convert_math(text):
    """Convert simple LaTeX math to plain text instead of deleting."""
    def math_to_text(m):
        expr = m.group(1)
        for old, new in [('\\geq', '>='), ('\\leq', '<='), ('\\kappa', 'kappa'),
                         ('\\rho', 'rho'), ('\\times', 'x'), ('\\text{', ''),
                         ('_{', '_'), ('^{', '^')]:
            expr = expr.replace(old, new)
        expr = re.sub(r'\\[a-zA-Z]+', '', expr)
        expr = expr.replace('{', '').replace('}', '').strip()
        return expr if expr else ''
    return re.sub(r'\$([^$]+)\$', math_to_text, text)


def extract_prose_only(text):
    """Strip LaTeX to clean prose for AI detection scoring."""
    # Remove entire environment contents (tikz, tabular, axis)
    text = re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{axis\}.*?\\end\{axis\}', '', text, flags=re.DOTALL)
    # Remove environment markers (keep surrounding prose)
    for env in ['enumerate', 'table', 'figure']:
        text = re.sub(rf'\\begin\{{{env}\}}(\[[^\]]*\])?', '', text)
        text = re.sub(rf'\\end\{{{env}\}}', '', text)
    for cmd in ['centering', 'small', 'footnotesize', 'noindent']:
        text = re.sub(rf'\\{cmd}\b', '', text)
    text = re.sub(r'\\caption\{[^}]*\}', '', text)
    # Abbreviation spacing
    for abbr in ['vs', 'al', 'i.e', 'e.g']:
        text = re.sub(rf'{re.escape(abbr)}\.\\\\\s', f'{abbr}. ', text)
    text = re.sub(r'et~al\.', 'et al.', text)
    # Citations to author-year
    def cite_to_author(m):
        key = m.group(1)
        match = re.match(r'([a-z]+)(\d{4})', key)
        return f'{match.group(1).capitalize()} et al. ({match.group(2)})' if match else ''
    def citep_to_author(m):
        key = m.group(1)
        match = re.match(r'([a-z]+)(\d{4})', key)
        return f'({match.group(1).capitalize()} et al., {match.group(2)})' if match else ''
    text = re.sub(r'\\citet\{([^}]+)\}', cite_to_author, text)
    text = re.sub(r'\\citealt\{([^}]+)\}', cite_to_author, text)
    text = re.sub(r'\\citep\{([^}]+)\}', citep_to_author, text)
    # Math to plain text
    text = convert_math(text)
    # Formatting commands
    for cmd in ['textbf', 'emph', 'textit', 'textsc', 'texttt']:
        text = re.sub(rf'\\{cmd}\{{([^}}]+)\}}', r'\1', text)
    # References
    text = re.sub(r'\\S\\ref\{[^}]+\}', '', text)
    text = re.sub(r'Table~\\ref\{[^}]+\}', 'Table', text)
    text = re.sub(r'\\ref\{[^}]+\}', '', text)
    text = re.sub(r'\\url\{[^}]+\}', '', text)
    # Quotes and special chars
    text = text.replace('``', '"').replace("''", '"')
    text = re.sub(r'Table~', 'Table ', text)
    text = re.sub(r'\\,', ' ', text)
    text = text.replace('---', '\u2014').replace('--', '\u2013')
    text = re.sub(r'\\%', '%', text)
    text = re.sub(r'~', ' ', text)
    # Section headers
    text = re.sub(r'\\paragraph\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\subsection\*?\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\section\*?\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\label\{[^}]+\}', '', text)
    text = re.sub(r'\\begin\{[^}]+\}.*', '', text)
    text = re.sub(r'\\end\{[^}]+\}', '', text)
    text = re.sub(r'\\item', '', text)
    # Remaining LaTeX
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'%%.*', '', text)
    text = re.sub(r'[{}]', '', text)
    text = re.sub(r'\\[\s]', ' ', text)
    text = re.sub(r'\\$', '', text)
    text = re.sub(r'  +', ' ', text)
    # Filter non-prose lines
    lines = text.split('\n')
    filtered = []
    for l in lines:
        s = l.strip()
        if not s:
            filtered.append('')
            continue
        if len(s) < 25:
            continue
        if '&' in s:
            continue
        if any(x in s.lower() for x in ['coordinates', 'axis cs:', 'mark size', 'draw=', 'fill=', 'color=']):
            continue
        if re.match(r'^[\d\s.,%\-/]+$', s):
            continue
        alpha_ratio = sum(1 for c in s if c.isalpha()) / max(len(s), 1)
        if alpha_ratio < 0.4:
            continue
        filtered.append(s)
    text = '\n'.join(filtered)
    text = re.sub(r'\n\s*\n+', '\n\n', text).strip()
    return text


def score_pangram(text):
    """Score text against Pangram API. Returns (fraction_ai, prediction)."""
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        'https://text.api.pangram.com/v3',
        data=data,
        headers={'Content-Type': 'application/json', 'x-api-key': API_KEY}
    )
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    return result.get('fraction_ai', -1), result.get('prediction_short', '?')


SEGMENTS = [
    ("Abstract", 40, 54),
    ("Intro vignette", 58, 96),
    ("Intro analysis+equity", 96, 170),
    ("Contributions", 120, 170),
    ("RW Safety+Medical", 171, 197),
    ("RW SpecGaming+RLHF", 197, 214),
    ("RW Frameworks+Judge", 214, 230),
    ("Benchmark Design", 231, 275),
    ("Scoring Axes", 276, 300),
    ("Decoupling Eval", 301, 312),
    ("Scoring Architecture", 313, 343),
    ("Experimental Setup", 344, 416),
    ("Results H1-H2", 417, 567),
    ("Results H3-H5", 568, 644),
    ("Results H6-H8", 645, 704),
    ("GPT-5.2", 705, 740),
    ("Summary", 741, 765),
    ("Disc Goodhart+SpecGaming", 766, 819),
    ("Three Failure Modes", 820, 861),
    ("Eval Blind Spot", 862, 899),
    ("Clinical+Policy", 900, 935),
    ("Limitations", 936, 956),
    ("Broader Impact", 957, 970),
    ("Conclusion", 971, 996),
]


def main():
    with open(PAPER_PATH) as f:
        full = f.read()
    lines = full.split('\n')

    if '--segments' in sys.argv:
        print("=== PER-SEGMENT PANGRAM SWEEP ===\n")
        for name, start, end in SEGMENTS:
            chunk = '\n'.join(lines[start-1:end])
            prose = extract_prose_only(chunk)
            if len(prose.strip()) < 200:
                print(f"  {name} (L{start}-{end}): SKIP ({len(prose.strip())} chars)")
                continue
            frac, pred = score_pangram(prose)
            status = "PASS" if frac < 0.20 else ("WARN" if frac < 0.50 else "FAIL")
            print(f"  {name} (L{start}-{end}): {frac:.3f} ({pred}) [{status}] ({len(prose)} chars)")
            time.sleep(0.3)
        print("\n=== DONE ===")

    elif '--section' in sys.argv:
        idx = sys.argv.index('--section')
        start, end = int(sys.argv[idx+1]), int(sys.argv[idx+2])
        chunk = '\n'.join(lines[start-1:end])
        prose = extract_prose_only(chunk)
        print(f"Lines {start}-{end}: {len(prose)} chars")
        if len(prose.strip()) < 50:
            print("Too short for reliable scoring")
            return
        frac, pred = score_pangram(prose)
        print(f"fraction_ai: {frac:.4f} ({pred})")

    else:
        start_idx = full.find('\\begin{abstract}')
        end_idx = full.find('\\bibliographystyle')
        body = full[start_idx:end_idx]
        prose = extract_prose_only(body)
        print(f"Prose length: {len(prose)} chars")
        frac, pred = score_pangram(prose)
        print(f"fraction_ai: {frac:.4f} ({pred})")


if __name__ == '__main__':
    main()
