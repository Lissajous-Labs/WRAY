import argparse
import math
import pathlib

from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath

RED = "#ff5c4a"
TRACKING = 0.02
MUTED = "#8a8a90"
INK = {"dark": "#f2f2f2", "light": "#1a1a1d"}
TAGLINE = "fast, verified math for audio DSP"
WIDTH, HEIGHT = 1280, 320
WAVE_CYCLES = 6
WAVE_DECAY = 0.80
WAVE_AMPLITUDE = 10.0
WAVE_WIDTH = 7.0
POINTS = 400


def path_data(text, size, prop, x, y):
    tp = TextPath((0, 0), text, size=size, prop=prop)
    extents = tp.get_extents()
    x -= extents.x0
    point = lambda v: f"{x + v[0]:.2f},{y - v[1]:.2f}"
    out, i, verts, codes = [], 0, tp.vertices, tp.codes
    while i < len(codes):
        code = codes[i]
        if code == Path.MOVETO:
            out.append("M" + point(verts[i])); i += 1
        elif code == Path.LINETO:
            out.append("L" + point(verts[i])); i += 1
        elif code == Path.CURVE3:
            out.append("Q" + point(verts[i]) + " " + point(verts[i + 1])); i += 2
        elif code == Path.CURVE4:
            out.append("C" + point(verts[i]) + " " + point(verts[i + 1]) + " " + point(verts[i + 2])); i += 3
        elif code == Path.CLOSEPOLY:
            out.append("Z"); i += 1
        else:
            i += 1
    return "".join(out), extents.width


def tracked(text, size, prop, x, y):
    def advance(ch):
        with_ch = TextPath((0, 0), "H" + ch + "H", size=size, prop=prop).get_extents().width
        return with_ch - TextPath((0, 0), "HH", size=size, prop=prop).get_extents().width

    parts, cursor = [], x
    for ch in text:
        glyph = TextPath((0, 0), ch, size=size, prop=prop)
        parts.append(path_data(ch, size, prop, cursor + glyph.get_extents().x0, y)[0])
        cursor += advance(ch) + TRACKING * size
    return "".join(parts), cursor - x - TRACKING * size


def outline(centre, widths, cap_segments=12):
    last = len(centre) - 1
    tangents = []
    for i in range(last + 1):
        (ax, ay), (bx, by) = centre[max(i - 1, 0)], centre[min(i + 1, last)]
        length = math.hypot(bx - ax, by - ay) or 1.0
        tangents.append(((bx - ax) / length, (by - ay) / length))

    def cap(p, n, t, half):
        return [(p[0] + half * (math.cos(a) * n[0] + math.sin(a) * t[0]),
                 p[1] + half * (math.cos(a) * n[1] + math.sin(a) * t[1]))
                for a in (math.pi * j / cap_segments for j in range(cap_segments + 1))]

    left, right = [], []
    for (px, py), (tx, ty), w in zip(centre, tangents, widths):
        left.append((px - ty * w / 2, py + tx * w / 2))
        right.append((px + ty * w / 2, py - tx * w / 2))
    end_t = tangents[last]
    start_t = tangents[0]
    end = cap(centre[last], (-end_t[1], end_t[0]), end_t, widths[last] / 2)
    start = cap(centre[0], (start_t[1], -start_t[0]), (-start_t[0], -start_t[1]), widths[0] / 2)
    points = left + end + right[::-1] + start
    return "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in points) + " Z"


def header(clash, mono, ink):
    _, word_width = tracked("WRAY", 150, clash, 0, 0)
    x0 = (WIDTH - word_width) / 2
    word, _ = tracked("WRAY", 150, clash, x0, 190)
    _, tag_width = path_data(TAGLINE, 26, mono, 0, 0)
    tag, _ = path_data(TAGLINE, 26, mono, (WIDTH - tag_width) / 2, 286)
    decay = [WAVE_DECAY ** (2 * WAVE_CYCLES * i / POINTS) for i in range(POINTS + 1)]
    wave = [(x0 + word_width * i / POINTS, 228 + WAVE_AMPLITUDE * math.sin(2 * math.pi * WAVE_CYCLES * i / POINTS) * decay[i])
            for i in range(POINTS + 1)]
    underline = outline(wave, [WAVE_WIDTH * d for d in decay])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}">'
            f'<title>WRAY</title><path d="{word}" fill="{ink}"/><path d="{underline}" fill="{RED}"/>'
            f'<path d="{tag}" fill="{MUTED}"/></svg>\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--font", required=True)
    parser.add_argument("--mono", default="/System/Library/Fonts/Menlo.ttc")
    parser.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parents[2] / "docs" / "brand"))
    args = parser.parse_args()
    clash, mono = FontProperties(fname=args.font), FontProperties(fname=args.mono)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for theme, ink in INK.items():
        target = out / f"wray-{theme}.svg"
        target.write_text(header(clash, mono, ink))
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
