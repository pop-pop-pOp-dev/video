#!/usr/bin/env python3
"""Render the NC-RTED method architecture figure without external packages."""
from pathlib import Path
from xml.sax.saxutils import escape


OUT = Path(__file__).with_name("nc_rted_method_architecture_v1.svg")
W, H = 1600, 1120


def box(x, y, w, h, title, lines, fill, stroke="#243247", dash=""):
    attrs = f'fill="{fill}" stroke="{stroke}" stroke-width="2" rx="10"'
    if dash:
        attrs += f' stroke-dasharray="{dash}"'
    line_h = 22
    total_height = 23 + 8 + len(lines) * line_h
    top = y + (h - total_height) / 2
    text = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" {attrs}/>',
            f'<text x="{x + w/2}" y="{top + 18}" class="title">{escape(title)}</text>']
    for i, line in enumerate(lines):
        text.append(f'<text x="{x + w/2}" y="{top + 48 + i*line_h}" class="body">{escape(line)}</text>')
    return "\n".join(text)


def arrow(x1, y1, x2, y2, label="", color="#334155"):
    middle_x, middle_y = (x1 + x2) / 2, (y1 + y2) / 2
    label_svg = "" if not label else f'<text x="{middle_x}" y="{middle_y - 8}" class="edge">{escape(label)}</text>'
    return f'<path d="M {x1} {y1} L {x2} {y2}" class="arrow" stroke="{color}"/>{label_svg}'


def render_raster():
    """Create review/export rasters when Pillow is available in the project venv."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return
    image = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        body, title, head = (ImageFont.truetype(font_path, 16), ImageFont.truetype(bold_path, 19),
                             ImageFont.truetype(bold_path, 28))
    except OSError:
        body = title = head = ImageFont.load_default()

    def centered(text, x, y, font, fill="#182433"):
        draw.text((x, y), text, font=font, fill=fill, anchor="mm")

    def rect(x, y, width, height, name, lines, fill, dashed=False):
        draw.rounded_rectangle((x, y, x + width, y + height), 10, fill=fill, outline="#243247", width=2)
        total_height = 23 + 8 + len(lines) * 21
        top = y + (height - total_height) / 2
        centered(name, x + width / 2, top + 14, title)
        for index, line in enumerate(lines):
            centered(line, x + width / 2, top + 43 + index * 21, body)
        if dashed:
            for dx in range(x + 8, x + width - 8, 14):
                draw.line((dx, y, min(dx + 8, x + width - 8), y), fill="#243247", width=2)

    def line(x1, y1, x2, y2, label="", color="#334155"):
        draw.line((x1, y1, x2, y2), fill=color, width=3)
        import math
        angle = math.atan2(y2 - y1, x2 - x1)
        wing = 12
        for offset in (2.6, 3.68):
            draw.line((x2, y2, x2 + wing * math.cos(angle + offset), y2 + wing * math.sin(angle + offset)), fill=color, width=3)
        if label:
            centered(label, (x1 + x2) / 2, (y1 + y2) / 2 - 12, body, color)

    frozen, trainable, teacher, neutral = "#dcecf8", "#dff1e3", "#fae7c5", "#f7f9fc"
    centered("NC-RTED: source-disjoint teacher construction and deployed student", 800, 35, head)
    centered("Method schematic only; no experimental results are implied.", 800, 65, body, "#425466")
    draw.rounded_rectangle((32, 96, 1568, 536), 14, outline="#91a4b7", width=2)
    draw.rounded_rectangle((32, 568, 1568, 1078), 14, outline="#91a4b7", width=2)
    draw.text((60, 118), "Training only: source-disjoint teacher", font=head, fill="#182433")
    draw.text((60, 590), "Student training and deployment", font=head, fill="#182433")
    top = [(72, 170, 225, 132, "Five-fold split", ["target Q = fold k", "C = (k + 1) mod 5", "R = other 3 folds"]),
           (350, 150, 245, 172, "Normal references", ["3 source-disjoint", "normal reference families", "static-context compatible", "no target-process matching"]),
           (654, 150, 240, 172, "Relation-time d[e,t]", ["monotone 4-step alignment", "robust normal scaling", "d[e,t] and M = max d"]),
           (954, 150, 245, 172, "Normal calibration", ["compatible normals in C", "min. support required", "empirical tail p"]),
           (1258, 150, 240, 172, "Teacher target", ["a = clipped tail quality", "pi = masked softmax(d)", "reject: aux loss masked", "empty mass: 1 - a"])]
    for item in top: rect(*item, teacher)
    for start, end in ((297, 350), (595, 654), (894, 954), (1199, 1258)): line(start, 236, end, 236)
    centered("Teacher reference identities, calibration sources, and labels do not enter the deployed student.", 800, 380, body, "#425466")
    centered("Insufficient reference/calibration support rejects distillation; it is not encoded as normal evidence.", 800, 412, body, "#425466")
    bottom = [(72, 674, 210, 135, "Observed media", ["VAD: causal latest 8 s", "frozen Fast trigger", "existing SigLIP patches"], frozen),
              (325, 650, 250, 184, "Frozen visual path", ["patch pooling", "frozen projector", "existing memory path", "causal observations"], frozen),
              (620, 650, 260, 184, "Added evidence module", ["relation-time encoder", "2 layers, width 384", "a_hat and pi_hat", "no-candidate bypass"], trainable),
              (925, 650, 255, 184, "Visual evidence tokens", ["pi_hat weights; a_hat gate", "<=16 visual tokens", "captions: all observed", "8 s blocks", "fixed 16-query aggregation"], trainable),
              (1225, 650, 275, 184, "Existing Slow pathway", ["frozen backbone + memory", "trainable existing Slow LoRA", "direct original + added tokens"], neutral)]
    for item in bottom: rect(*item[:-1], item[-1])
    for x1, x2 in ((282, 325), (575, 620), (880, 925), (1180, 1225)): line(x1, 741, x2, 741)
    line(1362, 834, 1362, 890)
    draw.line((1378, 322, 1520, 540, 1520, 875, 1220, 875, 1220, 900), fill="#a56500", width=3)
    draw.polygon(((1220, 900), (1214, 888), (1226, 888)), fill="#a56500")
    centered("teacher targets", 1460, 585, body, "#a56500")
    line(880, 785, 980, 900, "a_hat / pi_hat", "#2f6e44")
    draw.line((450, 834, 450, 860, 1200, 860, 1225, 800), fill="#334155", width=3)
    draw.polygon(((1225, 800), (1213, 794), (1213, 806)), fill="#334155")
    rect(980, 880, 240, 88, "Training loss only", ["KL targets: a and pi", "rejected: auxiliary masked"], teacher)
    rect(250, 920, 650, 82, "Auxiliary supervision variants", ["A: none | U: strength-matched global quality", "S: calibrated a only | F: calibrated a + relation-time pi"], neutral, True)
    rect(1260, 890, 220, 90, "Outputs", ["VAD detection", "description generation"], neutral)
    draw.text((80, 890), "Deployment excludes teacher, source identities, calibration data, and test labels.", font=body, fill="#425466")
    for x, fill, text in ((92, frozen, "frozen inherited component"), (390, trainable, "trainable added module / Slow LoRA"), (790, teacher, "training-only teacher material")):
        draw.rectangle((x, 1035, x + 18, 1053), fill=fill, outline="#243247")
        draw.text((x + 28, 1044), text, font=body, fill="#182433", anchor="lm")
    png = OUT.with_suffix(".png")
    image.save(png, optimize=True)
    image.save(OUT.with_suffix(".pdf"), "PDF", resolution=144.0)


def main():
    frozen, trainable, teacher, neutral = "#dcecf8", "#dff1e3", "#fae7c5", "#f7f9fc"
    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#334155"/></marker></defs>
<style>
text {{ font-family: Arial, Helvetica, sans-serif; fill: #182433; text-anchor: middle; }}
.head {{ font-size: 28px; font-weight: 700; }} .sub {{ font-size: 17px; fill: #425466; }}
.title {{ font-size: 18px; font-weight: 700; }} .body {{ font-size: 15px; }} .edge {{ font-size: 13px; fill: #425466; }}
.arrow {{ fill: none; stroke-width: 2.4; marker-end: url(#arrow); }} .panel {{ fill: #ffffff; stroke: #91a4b7; stroke-width: 2; rx: 14; }}
</style><rect width="100%" height="100%" fill="#ffffff"/>''']
    parts += ['<text x="800" y="42" class="head">NC-RTED: source-disjoint teacher construction and deployed student</text>',
              '<text x="800" y="70" class="sub">Method schematic only; no experimental results are implied.</text>',
              '<rect x="32" y="96" width="1536" height="440" class="panel"/>',
              '<text x="60" y="130" class="head" style="text-anchor:start">Training only: source-disjoint teacher</text>',
              '<rect x="32" y="568" width="1536" height="510" class="panel"/>',
              '<text x="60" y="602" class="head" style="text-anchor:start">Student training and deployment</text>']

    parts += [box(72, 170, 225, 132, "Five-fold split", ["target Q = fold k", "C = (k + 1) mod 5", "R = other 3 folds"], teacher),
              box(350, 150, 245, 172, "Normal references", ["3 source-disjoint", "normal reference families", "static-context compatible", "no target-process matching"], teacher),
              box(654, 150, 240, 172, "Relation-time d[e,t]", ["monotone 4-step alignment", "robust normal scaling", "d[e,t] and M = max d"], teacher),
              box(954, 150, 245, 172, "Normal calibration", ["compatible normals in C", "min. support required", "empirical tail p"], teacher),
              box(1258, 150, 240, 172, "Teacher target", ["a = clipped tail quality", "pi = masked softmax(d)", "reject: aux loss masked", "empty mass: 1 - a"], teacher)]
    parts += [arrow(297, 236, 350, 236), arrow(595, 236, 654, 236), arrow(894, 236, 954, 236), arrow(1199, 236, 1258, 236),
              '<text x="800" y="380" class="sub">Teacher reference identities, calibration sources, and labels do not enter the deployed student.</text>',
              '<text x="800" y="412" class="sub">Insufficient reference/calibration support rejects distillation; it is not encoded as normal evidence.</text>']

    parts += [box(72, 674, 210, 135, "Observed media", ["VAD: causal latest 8 s", "frozen Fast trigger", "existing SigLIP patches"], frozen),
              box(325, 650, 250, 184, "Frozen visual path", ["patch pooling", "frozen projector", "existing memory path", "causal observations"], frozen),
              box(620, 650, 260, 184, "Added evidence module", ["relation-time encoder", "2 layers, width 384", "a_hat and pi_hat", "no-candidate bypass"], trainable),
              box(925, 650, 255, 184, "Visual evidence tokens", ["pi_hat weights; a_hat gate", "<=16 visual tokens", "captions: all observed", "8 s blocks", "fixed 16-query aggregation"], trainable),
              box(1225, 650, 275, 184, "Existing Slow pathway", ["frozen backbone + memory", "trainable existing Slow LoRA", "direct original + added tokens"], neutral),
              box(1260, 890, 220, 90, "Outputs", ["VAD detection", "description generation"], neutral)]
    parts += [arrow(282, 741, 325, 741), arrow(575, 741, 620, 741), arrow(880, 741, 925, 741), arrow(1180, 741, 1225, 741), arrow(1362, 834, 1362, 890),
              arrow(880, 785, 980, 900, "a_hat / pi_hat", "#2f6e44"),
              '<path d="M 1378 322 L 1520 540 L 1520 875 L 1220 875 L 1220 900" class="arrow" stroke="#a56500"/><text x="1460" y="585" class="edge" fill="#a56500">teacher targets</text>',
              '<path d="M 450 834 L 450 860 L 1200 860 L 1225 800" class="arrow"/>']
    parts += [box(980, 880, 240, 88, "Training loss only", ["KL targets: a and pi", "rejected: auxiliary masked"], teacher),
              box(250, 920, 650, 82, "Auxiliary supervision variants", ["A: none | U: strength-matched global quality", "S: calibrated a only | F: calibrated a + relation-time pi"], neutral, dash="5 4"),
              '<text x="80" y="890" class="sub" style="text-anchor:start">Deployment excludes teacher, source identities, calibration data, and test labels.</text>',
              '<rect x="92" y="1035" width="18" height="18" fill="#dcecf8" stroke="#243247"/><text x="120" y="1050" class="body" style="text-anchor:start">frozen inherited component</text>',
              '<rect x="390" y="1035" width="18" height="18" fill="#dff1e3" stroke="#243247"/><text x="418" y="1050" class="body" style="text-anchor:start">trainable added module / Slow LoRA</text>',
              '<rect x="790" y="1035" width="18" height="18" fill="#fae7c5" stroke="#243247"/><text x="818" y="1050" class="body" style="text-anchor:start">training-only teacher material</text>',
              '</svg>']
    OUT.write_text("\n".join(parts), encoding="utf-8")
    render_raster()
    print(OUT)


if __name__ == "__main__":
    main()
