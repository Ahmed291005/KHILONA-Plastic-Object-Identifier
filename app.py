# KHILONA - elegant light-theme web page for the trained model (Gradio)
import os
import time
import html
import numpy as np
import tensorflow as tf
import gradio as gr

COLORS = ["Blue", "Yellow", "Purple"]
BELTS = ["Conveyor Belt A", "Conveyor Belt B", "Conveyor Belt C"]

SOFT = ["#eff6ff", "#fefce8", "#f5f3ff"]          # light card background per class
ACCENT = ["#2563eb", "#b45309", "#7c3aed"]        # text / border colour per class
CHIP_BG = ["#2563eb", "#facc15", "#7c3aed"]       # belt label background
CHIP_TEXT = ["#ffffff", "#E9EC1F", "#ffffff"]     # belt label text
BAR = ["#3b82f6", "#facc15", "#8b5cf6"]           # confidence bar colours

model = tf.keras.models.load_model("khilona_model.keras")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

body {
  background:
    radial-gradient(circle at 10% 15%, rgba(147,197,253,0.50) 0, transparent 36%),
    radial-gradient(circle at 90% 10%, rgba(253,224,71,0.38) 0, transparent 32%),
    radial-gradient(circle at 85% 90%, rgba(196,181,253,0.50) 0, transparent 38%),
    radial-gradient(circle at 8% 88%, rgba(253,230,138,0.30) 0, transparent 30%),
    linear-gradient(180deg, #f8fafc 0%, #f5f3ff 100%) !important;
  background-attachment: fixed !important;
  font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
}
gradio-app, .gradio-container {
  background: transparent !important;
  --body-background-fill: transparent;
}
.gradio-container { max-width: 1080px !important; margin: auto; }

.header {
  text-align: center; padding: 36px 20px 28px 20px; margin: 10px 0 18px 0;
  background: rgba(255,255,255,0.72); backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.95); border-radius: 28px;
  box-shadow: 0 12px 40px rgba(99,102,241,0.14);
}
.logo i { display: inline-block; width: 14px; height: 14px; border-radius: 50%; margin: 0 4px; }
.header h1 {
  font-family: 'Playfair Display', Georgia, 'Times New Roman', serif;
  font-size: 3.1rem; font-weight: 700; letter-spacing: 8px; color: #1e293b;
  margin: 12px 0 2px 0; padding-left: 8px;
}
.tagline { font-size: 0.85rem; letter-spacing: 5px; color: #64748b; font-weight: 600; }
.header .line {
  width: 110px; height: 3px; margin: 16px auto; border-radius: 4px;
  background: linear-gradient(90deg, #2563eb, #facc15, #7c3aed);
}
.header p { margin: 0 auto; max-width: 560px; color: #475569; font-size: 1.02rem; line-height: 1.6; }

.steps { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; margin-bottom: 18px; }
.steps span {
  background: rgba(255,255,255,0.8); border: 1px solid #e2e8f0; border-radius: 999px;
  padding: 7px 16px; font-size: 0.88rem; color: #475569; font-weight: 500;
}
.steps b { color: #4f46e5; margin-right: 6px; }

.panel {
  background: rgba(255,255,255,0.80) !important; border-radius: 24px !important;
  border: 1px solid rgba(255,255,255,0.95) !important; padding: 18px !important;
  box-shadow: 0 10px 30px rgba(100,116,139,0.14) !important;
}

.card { border-radius: 22px; padding: 30px 20px; text-align: center; }
.card .small { font-size: 0.8rem; letter-spacing: 4px; color: #64748b; font-weight: 600; }
.card .big {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: 2.9rem; font-weight: 700; margin: 8px 0 14px 0; letter-spacing: 3px;
}
.card .belt {
  display: inline-block; font-size: 1.2rem; font-weight: 600;
  border-radius: 999px; padding: 11px 28px; box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}
.waiting {
  background: rgba(255,255,255,0.65); border: 2px dashed #cbd5e1; color: #64748b;
}
.waiting .big { font-size: 1.7rem; color: #475569; letter-spacing: 1px; }

.bars { margin-top: 22px; padding: 0 6px; }
.row { display: flex; align-items: center; gap: 12px; margin: 11px 0; color: #334155; }
.row span { width: 70px; text-align: left; font-weight: 600; font-size: 0.95rem; }
.track { flex: 1; height: 12px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
.fill { height: 100%; border-radius: 999px; }
.row b { width: 64px; text-align: right; font-size: 0.95rem; }

.stage-title { text-align: center; font-family: 'Playfair Display', Georgia, serif; font-size: 1.5rem;
  font-weight: 700; color: #1e293b; margin: 22px 0 10px 0; letter-spacing: 2px; }
.stage { width: 100%; height: 620px; border: 0; border-radius: 24px; display: block;
  box-shadow: 0 10px 30px rgba(100,116,139,0.18); background: #eef2ff; }

.footer { text-align: center; color: #64748b; margin: 18px 0 8px 0; font-size: 0.85rem; letter-spacing: 1px; }
"""

# forces the light theme even if the computer is in dark mode
FORCE_LIGHT = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url);
  }
}
"""

HEADER = """
<div class="header">
  <div class="logo">
    <i style="background:#2563eb"></i><i style="background:#facc15"></i><i style="background:#7c3aed"></i>
  </div>
  <h1>KHILONA</h1>
  <div class="tagline">PLASTIC OBJECT IDENTIFIER</div>
  <div class="line"></div>
  <p>Show us a blue, yellow or purple object and we will point it to the right conveyor belt.</p>
</div>
<div class="steps">
  <span><b>1</b>Upload a photo</span>
  <span><b>2</b>AI identifies the colour</span>
  <span><b>3</b>Belt is assigned</span>
</div>
"""

WAITING = """
<div class="card waiting">
  <div class="small">READY</div>
  <div class="big">Waiting for an image</div>
  <div>Upload or capture a photo to begin</div>
</div>
"""

FOOTER = '<div class="footer">CNN TRAINED FROM SCRATCH ON OUR OWN DATASET</div>'



SCENE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene.html")


def scene_html(index):
    """3D factory animation. index = 0/1/2 plays the story, -1 shows the idle factory."""
    with open(SCENE_FILE, encoding="utf-8") as f:
        page = f.read().replace("__CLASS_INDEX__", str(index))
    page += "<!-- run %f -->" % time.time()          # new text each time, so the iframe always reloads
    return '<iframe class="stage" srcdoc="%s"></iframe>' % html.escape(page, quote=True)


def classify(image):
    try:
        img = np.array(image.convert("RGB"))
    except AttributeError:                       # no image yet
        return WAITING, scene_html(-1)

    img = tf.image.resize(img, (128, 128))             # same resize as training
    img = np.expand_dims(img.numpy(), axis=0)          # add batch dimension

    probabilities = model.predict(img, verbose=0)[0]
    index = int(np.argmax(probabilities))              # highest probability

    bars = ""
    for i in range(3):
        percent = probabilities[i] * 100
        bars += (
            '<div class="row"><span>' + COLORS[i] + '</span>'
            '<div class="track"><div class="fill" style="width:{:.1f}%;background:{}"></div></div>'
            '<b>{:.1f}%</b></div>'
        ).format(percent, BAR[i], percent)

    card = (
        '<div class="card" style="background:{soft};border:2px solid {accent}44">'
        '<div class="small">DETECTED COLOUR</div>'
        '<div class="big" style="color:{accent}">{color}</div>'
        '<div class="belt" style="background:{chip};color:{chip_text}">Send to {belt}</div>'
        '</div>'
        '<div class="bars">{bars}</div>'
    ).format(soft=SOFT[index], accent=ACCENT[index], chip=CHIP_BG[index],
             chip_text=CHIP_TEXT[index], color=COLORS[index].upper(),
             belt=BELTS[index], bars=bars)
    return card, scene_html(index)


def clear():
    return None, WAITING, scene_html(-1)


with gr.Blocks(title="KHILONA Object Identifier") as page:
    gr.HTML(HEADER)
    with gr.Row():
        with gr.Column(elem_classes="panel"):
            photo = gr.Image(type="pil", label="Object photo", height=380,
                             sources=["upload", "webcam", "clipboard"])
            with gr.Row():
                identify = gr.Button("Identify Object", variant="primary", size="lg")
                reset = gr.Button("Clear", size="lg")
        with gr.Column(elem_classes="panel"):
            output = gr.HTML(WAITING)
    gr.HTML('<div class="stage-title">Live 3D sorting line</div>')
    stage = gr.HTML(scene_html(-1))
    gr.HTML(FOOTER)

    identify.click(classify, inputs=photo, outputs=[output, stage])
    photo.upload(classify, inputs=photo, outputs=[output, stage])     # also runs on upload
    reset.click(clear, outputs=[photo, output, stage])

page.launch(
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    css=CSS,
    js=FORCE_LIGHT,
)