"""
Gradio demo — public showcase for the DREAM pipeline.

Deploy to HuggingFace Spaces
-----------------------------
1. Create a new Space (Gradio SDK).
2. Push this repo.  Set these Space secrets:
       DREAM_CHECKPOINT = checkpoints/dream_best.pt   (or upload the file directly)
       DREAM_BACKBONE   = sentence-transformers/LaBSE
3. Done.  Your Space URL is the live demo link for your CV.

Run locally
-----------
    python api/demo.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dream.pipeline import DREAMPipeline

# ---------------------------------------------------------------------------
# Load pipeline once
# ---------------------------------------------------------------------------

_CHECKPOINT = os.getenv("DREAM_CHECKPOINT", "checkpoints/dream_best.pt")
_BACKBONE   = os.getenv("DREAM_BACKBONE",   "sentence-transformers/LaBSE")
_DEVICE     = os.getenv("DREAM_DEVICE",     None)

pipe = DREAMPipeline.from_pretrained(_CHECKPOINT, backbone_name=_BACKBONE, device=_DEVICE)


# ---------------------------------------------------------------------------
# Handler functions
# ---------------------------------------------------------------------------

def compare_pair(text_a: str, text_b: str) -> tuple[float, str]:
    """Handler for the pairwise similarity tab."""
    if not text_a.strip() or not text_b.strip():
        return 0.0, "⚠️ Please enter two sentences."
    score = pipe.similarity(text_a.strip(), text_b.strip())
    if score > 0.85:
        label = f"🟢 Very similar ({score:.4f})"
    elif score > 0.5:
        label = f"🟡 Somewhat similar ({score:.4f})"
    else:
        label = f"🔴 Dissimilar ({score:.4f})"
    return round(score, 4), label


def compute_matrix(sentences_raw: str):
    """Handler for the similarity matrix tab → Plotly heatmap."""
    import plotly.graph_objects as go

    sentences = [s.strip() for s in sentences_raw.strip().splitlines() if s.strip()]
    if len(sentences) < 2:
        return go.Figure()

    matrix = pipe.similarity_matrix(sentences)
    labels = [s[:35] + "…" if len(s) > 35 else s for s in sentences]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="RdYlGn",
            zmin=-1,
            zmax=1,
            text=np.round(matrix, 3),
            texttemplate="%{text}",
            hovertemplate="(%{x}, %{y})<br>similarity = %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Cross-lingual semantic similarity matrix",
        height=max(420, 80 * len(sentences)),
        xaxis_tickangle=-35,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

_DESCRIPTION = """
## DREAM: Language-Agnostic Sentence Embeddings

Implementation of [**Tiyajamorn et al., EMNLP 2021**](https://aclanthology.org/2021.emnlp-main.612) —
*"Language-Agnostic Representation from Multilingual Sentence Encoders for Cross-Lingual Similarity Estimation"*.

Enter sentences in **any language** — the model strips language identity and compares pure meaning.
"""

_PAIR_EXAMPLES = [
    ["The cat sat on the mat.",        "Le chat était assis sur le tapis."],
    ["I love artificial intelligence.", "Tôi yêu trí tuệ nhân tạo."],
    ["The weather is beautiful today.", "Das Wetter ist heute schön."],
    ["She reads books every night.",    "She watches TV every night."],   # same language, different meaning
]

_MATRIX_PLACEHOLDER = (
    "The cat sat on the mat.\n"
    "Le chat était assis sur le tapis.\n"
    "El gato estaba sentado en la alfombra.\n"
    "She reads books every night.\n"
    "Er liest jeden Abend Bücher."
)

with gr.Blocks(title="DREAM Embedding Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(_DESCRIPTION)

    with gr.Tab("Pairwise similarity"):
        gr.Markdown("Compare any two sentences — they can be in different languages.")
        with gr.Row():
            txt_a = gr.Textbox(label="Sentence A", placeholder="The cat sat on the mat.")
            txt_b = gr.Textbox(label="Sentence B", placeholder="Le chat était assis sur le tapis.")
        btn_pair = gr.Button("Compare", variant="primary")
        with gr.Row():
            score_out = gr.Number(label="Cosine similarity", precision=4)
            label_out = gr.Markdown()
        btn_pair.click(compare_pair, inputs=[txt_a, txt_b], outputs=[score_out, label_out])
        gr.Examples(_PAIR_EXAMPLES, inputs=[txt_a, txt_b])

    with gr.Tab("Similarity matrix"):
        gr.Markdown("Enter one sentence per line (mix languages freely).")
        sentences_in = gr.Textbox(
            label="Sentences (one per line)",
            lines=6,
            placeholder=_MATRIX_PLACEHOLDER,
        )
        btn_matrix = gr.Button("Compute matrix", variant="primary")
        plot_out = gr.Plot()
        btn_matrix.click(compute_matrix, inputs=[sentences_in], outputs=[plot_out])

    gr.Markdown(
        "---\n"
        "*Backbone: LaBSE · DREAM head: 2× single-layer MLP · "
        "Trained on Tatoeba parallel corpora (7 language pairs)*"
    )

if __name__ == "__main__":
    demo.launch(share=False)
