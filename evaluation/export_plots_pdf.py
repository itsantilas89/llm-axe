import json
import os
import img2pdf

plots_dir = os.path.join('output', 'evaluation', 'plots')
vs_path = os.path.join(plots_dir, 'visual_summary.json')

if not os.path.exists(vs_path):
    raise SystemExit(f'visual_summary.json not found at {vs_path}')

with open(vs_path, 'r', encoding='utf-8') as f:
    vs = json.load(f)

charts = vs.get('charts', [])
if not charts:
    raise SystemExit('No charts listed in visual_summary.json')

paths = []
for name in charts:
    path = os.path.join(plots_dir, name)
    if not os.path.exists(path):
        print(f'Warning: missing {path}, skipping')
        continue
    paths.append(path)

if not paths:
    raise SystemExit('No images found to combine')

out_pdf = os.path.join(plots_dir, 'evaluation_plots.pdf')
with open(out_pdf, 'wb') as f:
    f.write(img2pdf.convert(paths))

print(out_pdf)
