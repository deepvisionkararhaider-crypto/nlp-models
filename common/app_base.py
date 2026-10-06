"""
Shared Streamlit application body for every model.

Each ``<folder>/app.py`` is a real Streamlit entry point that simply points this
module at its own model. Keeping the body here means all 15 apps behave and look
identically, and the honest-labelling logic lives in exactly one place.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common import nlp_common as nc  # noqa: E402
from common import ui  # noqa: E402

# Unseen example posts (written by us, one per class, none from the corpus).
EXAMPLES = {
    "hockey": "The goaltender stopped 40 shots and the team clinched a playoff spot "
              "after a dramatic overtime win in the Stanley Cup series.",
    "space": "The telescope captured a new image of a distant galaxy while the rover "
             "drilled samples on the surface of Mars.",
    "graphics": "Rendering a complex 3D scene requires optimizing the polygon count "
                "and texture resolution for the graphics card.",
    "politics": "The committee debated the new healthcare bill and the upcoming vote "
                "on electoral reform.",
}


@st.cache_resource(show_spinner=False)
def _load_bundle(folder: str):
    return nc.load_bundle(folder)


def _predict(folder: str, text: str, bundle):
    return nc.predict_text(folder, text, bundle=bundle)


def _word_similarity_panel(bundle, meta):
    st.markdown("### 🔎 Word-similarity explorer")
    st.caption(
        "Because this model learns dense word vectors, you can inspect the "
        "neighbourhood of any word in its vocabulary. Cosine similarity is shown - "
        "it is a *similarity score* (higher = closer in meaning), not a probability."
    )
    word = st.text_input("Word", value="hockey", key="sim_word")
    if word.strip():
        neigh = nc.nearest_words(bundle, word, topn=10)
        if not neigh:
            st.info(
                "That word is not in this model's vocabulary (trained on "
                "20-Newsgroups). Try: hockey, space, computer, graphics, nasa, "
                "team, encryption, government."
            )
        else:
            import pandas as pd
            df = pd.DataFrame(neigh, columns=["neighbour", "cosine similarity"])
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.caption("Similar neighbours confirm the vectors captured topical "
                       "structure from the corpus.")


def _layer_panel(bundle, meta):
    st.markdown("### 🧱 Representation layers")
    st.caption(
        "This model stacks three representation levels, mimicking ELMo's "
        "multi-layer idea. The weights below are the fixed weights actually used."
    )
    for layer, weight in zip(bundle["layers"], bundle["weights"]):
        st.markdown(f"- **{layer['name']}** - weight `{weight}`")
    st.info(
        "This is a TF-IDF + SVD feature stack, not a trained bidirectional "
        "language model. See the 'What this model actually is' note above.",
        icon="ℹ️",
    )


def run_app(folder: str) -> None:
    meta = dict(nc.MODEL_REGISTRY[folder])
    meta["folder"] = folder
    ui.configure_page(meta)
    ui.render_hero(meta)
    ui.render_family_badges(meta)
    st.write("")

    artifact_path = nc.artifact_dir(folder) / "model.joblib"
    try:
        bundle = _load_bundle(folder)
    except nc.ArtifactMissingError as exc:
        ui.sidebar(meta)
        ui.artifact_missing_ui(exc)
        return

    metrics = nc.load_metrics(folder)
    ui.sidebar(meta, bundle)

    tab_predict, tab_about = st.tabs(["🔮 Predict", "📊 Model & evaluation"])

    with tab_predict:
        col_in, col_out = st.columns([1.05, 1.0], gap="large")
        with col_in:
            st.markdown("### ✍️ Enter text")
            text = ui.example_picker(EXAMPLES, key="input_text")
            run = st.button("Predict topic", type="primary",
                            use_container_width=True, key="predict_btn")
            st.caption("Input format: any English text (a sentence or a full post).")
        with col_out:
            st.markdown("### 🎯 Result")
            if run:
                if not text or not text.strip():
                    st.warning("Please enter some text before predicting.", icon="✏️")
                else:
                    with st.spinner("Running the model..."):
                        try:
                            result = _predict(folder, text, bundle)
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
                            result = None
                    if result:
                        ui.render_prediction(result, meta)
            else:
                st.info("Enter text and click **Predict topic** to see the real "
                        "model output.", icon="👈")

        if meta["needs_gensim"]:
            st.divider()
            _word_similarity_panel(bundle, meta)
        if bundle["kind"] == "stacked_doc_clf":
            st.divider()
            _layer_panel(bundle, meta)

    with tab_about:
        ui.render_metrics(metrics)
        st.divider()
        ui.render_honesty_note(meta)
        ui.render_model_info(meta, bundle, artifact_path)
        ui.render_limitations(meta)

    st.divider()
    st.caption(
        "Built for classroom demonstration. Input → tokenise/vectorise → trained "
        "model → decoded result. No fabricated predictions."
    )
