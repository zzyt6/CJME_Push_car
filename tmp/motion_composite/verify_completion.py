"""Read-only audit of sources, native chart workbooks, and PowerPoint export."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from lxml import etree as E

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(Path(__file__).parent / 'qa_deps')]
import pymupdf
from plot_motion_ability import load_trial

NS = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
      's': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
MM = 25.4 / 72


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(actual, expected, tolerance=1e-12):
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)


def color_matches(actual, expected):
    rgb = tuple(int(expected.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    return actual is not None and np.max(np.abs(np.array(actual)-rgb)) < .001


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deck', type=Path, required=True)
    parser.add_argument('--render', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = ROOT / 'logs/Picture/motion_composite'
    spec = json.loads((root / 'data/composition.json').read_text('utf-8'))
    manifest = json.loads((root / 'manifest.json').read_text('utf-8'))
    for path, expected in manifest['source_hashes'].items():
        assert digest(path) == expected, path
    assert digest(manifest['reference_pptx']) == manifest['reference_sha256']
    for name, expected in manifest['outputs'].items():
        assert digest(root / name) == expected, name

    records = []
    with ZipFile(args.deck) as deck:
        chart_names = [n for n in deck.namelist() if n.startswith('ppt/charts/chart') and n.endswith('.xml')]
        books = [n for n in deck.namelist() if n.endswith('.xlsx')]
        assert len(chart_names) == len(books) == 6
        slide = E.fromstring(deck.read('ppt/slides/slide1.xml'))
        assert len(slide.findall('.//p:pic', NS)) == 7
        bands = slide.findall('.//a:custGeom', NS)
        assert len(bands) == 12
        assert {n.get('typeface') for n in slide.findall('.//a:latin', NS)} == {'Arial'}

        for i, panel in enumerate(spec['panels'], 1):
            engine = 'isaac' if i <= 3 else 'mujoco'
            raw_root = ROOT / 'logs/Picture/motion_ability'
            if engine == 'mujoco': raw_root /= 'mujoco'
            axis = panel['axis']
            for method, series, band in zip(('CWC', 'baseline'), panel['series'][1:], panel['bands']):
                trial = load_trial(raw_root, axis, method)
                # Independent masks recompute population SD in each local window.
                values = [trial.signal[(np.abs(trial.t-t) <= panel['mean_window_s']/2+1e-8)
                                       & (trial.stage_index == stage)]
                          for t, stage in zip(trial.t, trial.stage_index)]
                mean = np.array([v.mean() for v in values], dtype=np.float64)
                sd = np.array([v.std(ddof=0) for v in values], dtype=np.float64)
                offset = panel['initialization_display_hold_s']
                x = trial.t + offset
                lower, upper = mean-sd, mean+sd
                csv = np.loadtxt(root / f'data/{engine}_{axis}_{method}.csv', delimiter=',', skiprows=1)
                same(csv, np.column_stack([trial.t, x, trial.signal, mean, lower, upper]))
                if trial.initial_velocity is not None:
                    initial = trial.initial_velocity[('vx', 'vy', 'wz').index(axis)]
                    initial_times = [0., offset] if offset else [trial.initial_t]
                    x = np.r_[initial_times, x]
                    mean, lower, upper = [np.r_[[initial]*len(initial_times), v] for v in (mean, lower, upper)]
                same(series['x'], x)
                same(series['y'], mean)
                expected_vertices = np.vstack([[x[0], upper[0]], np.column_stack([x, lower]),
                                               [x[-1], upper[-1]], np.column_stack([x[::-1], upper[::-1]]),
                                               [x[0], upper[0]]])
                same(band['vertices'], expected_vertices)
                assert min(lower) >= panel['ylim'][0] and max(upper) <= panel['ylim'][1]
                vertices = np.asarray(band['vertices'])
                _, _, w, h = panel['plot_rect_mm']
                lo_x, hi_x = panel['xlim']; lo_y, hi_y = panel['ylim']
                expected_emu = np.rint(np.column_stack([(vertices[:,0]-lo_x)/(hi_x-lo_x)*w,
                                                       (hi_y-vertices[:,1])/(hi_y-lo_y)*h])*36000)
                index = (i-1)*2 + (method == 'baseline')
                pts = bands[index].findall('.//a:pt', NS)
                actual_emu = [[int(p.get('x')), int(p.get('y'))] for p in pts]
                same(actual_emu, expected_emu, 0)
                records.append({'engine': engine, 'axis': axis, 'method': method, 'raw_samples': len(trial.t)})

            command = panel['series'][0]
            steps = trial.stages
            edges = [s['start_s']+offset for s in steps] + [steps[-1]['end_s']+offset]
            heights = [s['value'] for s in steps]
            if offset: edges, heights = [0.]+edges, [0.]+heights
            # Matplotlib's step path duplicates the first and last endpoints.
            expected_steps = [[edges[0], heights[0]]]
            for j, height in enumerate(heights): expected_steps.extend([[edges[j], height], [edges[j+1], height]])
            expected_steps.append([edges[-1], heights[-1]])
            same(np.column_stack([command['x'], command['y']]), expected_steps)

            chart = E.fromstring(deck.read(f'ppt/charts/chart{i}.xml'))
            assert {n.get('typeface') for n in chart.findall('.//a:latin', NS)} == {'Arial'}
            axes = chart.findall('.//c:valAx', NS)
            for dim, ax in enumerate(axes):
                same([float(ax.find('c:scaling/c:min', NS).get('val')), float(ax.find('c:scaling/c:max', NS).get('val'))],
                     panel['xlim' if dim == 0 else 'ylim'])
            with ZipFile(io.BytesIO(deck.read(f'ppt/embeddings/chartData{i}.xlsx'))) as book:
                sheet = E.fromstring(book.read('xl/worksheets/sheet1.xml'))
            assert not sheet.findall('.//s:f', NS)
            cells = {c.get('r'): float(c.find('s:v', NS).text) for c in sheet.findall('.//s:c', NS) if c.find('s:v', NS) is not None}
            for j, (node, series) in enumerate(zip(chart.findall('.//c:scatterChart/c:ser', NS), panel['series'])):
                assert node.find('c:tx/c:v', NS).text == series['name']
                for k, tag in enumerate(('xVal', 'yVal')):
                    cached = [float(p.find('c:v', NS).text) for p in node.findall(f'c:{tag}/c:numRef/c:numCache/c:pt', NS)]
                    expected = series[('x', 'y')[k]]
                    workbook = [cells[f'{chr(65+2*j+k)}{r+2}'] for r in range(len(expected))]
                    same(cached, expected, 0)
                    same(workbook, expected, 0)

    render = pymupdf.open(args.render)
    paper = pymupdf.open(root / 'motion_ability_composite.pdf')
    assert len(render) == len(paper) == 1
    for doc in (render, paper):
        same(np.array(doc[0].rect)[2:]*MM, [180, 130], .02)
        assert len(doc[0].get_image_info()) == 7
        fonts = doc[0].get_fonts()
        assert all('Arial' in font[3] for font in fonts)
        if doc is paper:
            for font in fonts: assert doc.extract_font(font[0])[3], 'paper font is not embedded'
    drawings = render[0].get_drawings()
    grey = [d for d in drawings if color_matches(d['color'], '#555555')]
    vertical = [d for d in grey if len(d['items']) == 1 and d['rect'].height > 80 and d['rect'].width < 1]
    horizontal = [d for d in grey if len(d['items']) == 1 and d['rect'].width > 100 and d['rect'].height < 1]
    assert len(vertical) == len(horizontal) == 6
    errors = []
    curve_errors = []
    for panel, v, h in zip(spec['panels'], vertical, horizontal):
        box = [v['rect'].x0+v['rect'].width/2, v['rect'].y0, h['rect'].x1-h['rect'].x0,
               v['rect'].height]
        box = np.array(box)*MM
        errors.append(float(np.max(np.abs(box-panel['plot_rect_mm']))))
        same(box, panel['plot_rect_mm'], .003)
        # The rendered translucent polygons retain the source vertex extents.
        for band in panel['bands']:
            pts = np.asarray(band['vertices'])
            x,y,w,height = panel['plot_rect_mm']
            lx,hx = panel['xlim']; ly,hy = panel['ylim']
            mapped = np.column_stack([x+(pts[:,0]-lx)/(hx-lx)*w, y+(hy-pts[:,1])/(hy-ly)*height])
            expected = np.r_[mapped.min(axis=0), mapped.max(axis=0)]
            candidates = [d for d in drawings if color_matches(d['fill'], band['color'])
                          and abs(d['fill_opacity']-.18) < .001
                          and abs(d['rect'].x0*MM-expected[0]) < .01
                          and abs(d['rect'].y0*MM-expected[1]) < .01]
            assert len(candidates) == 1
            same(np.array(candidates[0]['rect'])*MM, expected, .003)
        for series in panel['series'][1:]:
            pts = np.column_stack([series['x'], series['y']])
            x,y,w,height = panel['plot_rect_mm']
            lx,hx = panel['xlim']; ly,hy = panel['ylim']
            mapped = np.column_stack([x+(pts[:,0]-lx)/(hx-lx)*w, y+(hy-pts[:,1])/(hy-ly)*height])
            expected = np.r_[mapped.min(axis=0), mapped.max(axis=0)]
            candidates = [d for d in drawings if color_matches(d['color'], series['color'])
                          and len(d['items']) > 100
                          and abs(d['rect'].x0*MM-expected[0]) < .01
                          and abs(d['rect'].y0*MM-expected[1]) < .025]
            assert len(candidates) == 1
            curve_errors.append(float(np.max(np.abs(np.array(candidates[0]['rect'])*MM-expected))))
            # PowerPoint rounds native chart path coordinates during PDF export.
            # 0.025 mm is below one tenth of the 0.9 pt curve width.
            same(np.array(candidates[0]['rect'])*MM, expected, .025)
    application = json.loads(args.render.with_suffix('.application.json').read_text('utf-8-sig'))
    assert application['Application'] == 'Microsoft PowerPoint'
    assert 'Microsoft Office' in application['ApplicationPath']
    for actual, panel in zip(application['Charts'], spec['panels']):
        same([actual['XMinimum'], actual['XMaximum']], panel['xlim'])
        same([actual['YMinimum'], actual['YMaximum']], panel['ylim'])
        assert actual['SeriesCount'] == 3
    result = {'status': 'pass', 'deck': str(args.deck.resolve()), 'deck_sha256': digest(args.deck),
              'paper_pdf_sha256': digest(root / 'motion_ability_composite.pdf'),
              'render_sha256': digest(args.render), 'source_files_verified': len(manifest['source_hashes']),
              'native_charts': 6, 'embedded_workbooks': 6, 'series_checked': 18, 'editable_sd_bands': 12,
              'axis_rectangle_max_error_mm': max(errors), 'axis_rectangle_errors_mm': errors,
              'rendered_curve_extent_max_error_mm': max(curve_errors),
              'trials': records, 'application': application['Application'],
              'application_path': application['ApplicationPath'], 'application_version': application['Version'],
              'runtime': 'Existing OOXML builder retained because @oai/artifact-tool is unavailable on this host.'}
    args.report.write_text(json.dumps(result, indent=2), 'utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'trials'}, indent=2))


if __name__ == '__main__': main()
