"""Small project-local layout checks, independent of any desktop skill package.

These checks cover finite axes and clipped visible text, not semantic plot review.
"""
import math


def audit_layout(figure):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    issues = []
    for index, axis in enumerate(figure.axes):
        if not all(math.isfinite(v) for v in (*axis.get_xlim(), *axis.get_ylim())):
            issues.append(('FAIL', f'Axis {index} has nonfinite limits'))
    # Axis-limit tick labels outside the displayed interval are not rendered.
    texts = list(figure.texts)
    for axis in figure.axes:
        texts.extend([axis.title, axis.xaxis.label, axis.yaxis.label])
        texts.extend(axis.texts)
        legend = axis.get_legend()
        if legend:
            texts.extend(legend.get_texts())
    for legend in figure.legends:
        texts.extend(legend.get_texts())
    for text in texts:
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < canvas.x0-2 or box.y0 < canvas.y0-2 or box.x1 > canvas.x1+2 or box.y1 > canvas.y1+2:
            issues.append(('FAIL', 'Visible title/label/annotation extends outside canvas'))
    return issues
