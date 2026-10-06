"""
Shared Streamlit UI components - one consistent design system for all 15 NLP
apps, plus the honesty primitives that keep every screen truthful about what the
model really is and what its numbers really mean.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make ``common`` importable when Streamlit runs ``<folder>/app.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common import nlp_common as nc  # noqa: E402

_CSS = """
<style>
  .nlp-hero {padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1.1rem;
             background: linear-gradient(120deg, {accent}18, {accent}05);
             border: 1px solid {accent}33; border-left: 6px solid {accent};}
  .nlp-hero h1 {margin: 0 0 .25rem 0; font-size: 1.85rem; color: {accent};
                font-weight: 700; letter-spacing: -.02em;}
  .nlp-hero p  {margin: 0; color: #475569; font-size: .98rem;}
  .nlp-badge {display:inline-block; padding:.15rem .55rem; border-radius:999px;
              font-size:.72rem; font-weight:600; letter-spacing:.02em;}
  .nlp-badge-ok {background:#dcfce7; color:#166534;}
  .nlp-badge-warn {background:#fef3c7; color:#92400e;}
  .nlp-badge-info {background:#e0e7ff; color:#3730a3;}
  .nlp-card {border:1px solid #e2e8f0; border-radius:12px; padding:1rem 1.15rem;
             background:#fff; box-shadow:0 1px 2px rgba(15,23,42,.04);}
  .nlp-result {border:1px solid {accent}44; border-left:5px solid {accent};
               border-radius:12px; padding:1rem 1.2rem; background:{accent}0d;}
  .nlp-result .label {font-size:.74rem; text-transform:uppercase; letter-spacing:.08em;
                      color:#64748b; font-weight:600;}
  .nlp-result .value {font-size:1.7rem; font-weight:700; color:{accent}; line-height:1.15;}
  .nlp-metric {text-align:center; padding:.55rem .3rem; border-radius:10px;
               background:#f8fafc; border:1px solid #eef2f7;}
  .nlp-metric .v {font-size:1.15rem; font-weight:700; color:#0f172a;}
  .nlp-metric .k {font-size:.7rem; color:#64748b; text-transform:uppercase;
                  letter-spacing:.05em;}
  .nlp-bar {height:9px; border-radius:999px; background:#eef2f7; overflow:hidden;
            margin-top:3px;}
  .nlp-bar > span {display:block; height:100%; border-radius:999px;}
  .nlp-note {font-size:.82rem; color:#64748b;}
  section.main > div {padding-top: .6rem;}
</style>
"""


def configure_page(meta: dict) -> None:
    st.set_page_config(
        page_title=f"{meta['display']} | NLP Portfolio",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS.format(accent=meta["accent"]), unsafe_allow_html=True)


def render_hero(meta: dict) -> None:
    st.markdown(
        f"""
        <div class="nlp-hero">
          <h1>{meta['num']:02d}. {meta['display']}</h1>
          <p>{meta['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_family_badges(meta: dict) -> None:
    """State plainly whether the folder name matches the real technique."""
    if meta["faithful_family"]:
        st.markdown(
            '<span class="nlp-badge nlp-badge-ok">Technique matches name</span> '
            f'<span class="nlp-badge nlp-badge-info">{meta["family"]}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="nlp-badge nlp-badge-warn">Name vs. implementation</span> '
            f'<span class="nlp-badge nlp-badge-info">{meta["family"]}</span>',
            unsafe_allow_html=True,
        )


def render_honesty_note(meta: dict) -> None:
    """The 'what this really is' callout - shown on every model's page."""
    with st.expander("🔍 What this model actually is (read this)", expanded=False):
        st.markdown(f"**Implementation in this repository:** {meta['actual']}")
        st.markdown(f"**Framework:** {meta['framework']}")
        if meta.get("caveat"):
            st.warning(meta["caveat"], icon="⚠️")
        else:
            st.info(
                "The folder name describes the technique that is actually "
                "implemented here - nothing is misrepresented.",
                icon="✅",
            )
        st.caption(
            "Every prediction on this page comes from a real trained model run on "
            "your input. No outputs are hard-coded, simulated or randomised."
        )


_SCORE_TYPE_TEXT = {
    "probability": "The bars below are real **probabilities** from the model "
                   "(`predict_proba`); they sum to 100%.",
    "decision_function_softmax": "This classifier has no native probability output. "
                                 "The bars are a **softmax of the model's real "
                                 "`decision_function` scores** - they show relative "
                                 "preference between classes and are *not* calibrated "
                                 "probabilities.",
}


def render_prediction(result: dict, meta: dict) -> None:
    top = result["scores"][0]
    st.markdown(
        f"""
        <div class="nlp-result">
          <div class="label">Predicted topic</div>
          <div class="value">{top['short']}</div>
          <div class="nlp-note">{nc.CATEGORY_DESCRIPTIONS.get(top['label'], top['label'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(_SCORE_TYPE_TEXT.get(result["score_type"], ""))
    st.caption(f"score type: `{result['score_type']}`")
    for row in result["scores"]:
        pct = 100.0 * row["score"]
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;font-size:.85rem'>"
            f"<span><b>{row['short']}</b> &nbsp;<span class='nlp-note'>{row['label']}</span></span>"
            f"<span>{pct:5.1f}%</span></div>"
            f"<div class='nlp-bar'><span style='width:{max(pct,0):.1f}%;"
            f"background:{meta['accent']}'></span></div>",
            unsafe_allow_html=True,
        )


def render_metrics(metrics: dict | None) -> None:
    if not metrics:
        return
    st.markdown("#### Held-out test metrics")
    cols = st.columns(4)
    items = [
        ("Test accuracy", f"{metrics['accuracy'] * 100:.2f}%"),
        ("F1 (macro)", f"{metrics['f1_macro'] * 100:.2f}%"),
        ("Precision (macro)", f"{metrics['precision_macro'] * 100:.2f}%"),
        ("Recall (macro)", f"{metrics['recall_macro'] * 100:.2f}%"),
    ]
    for col, (k, v) in zip(cols, items):
        col.markdown(f"<div class='nlp-metric'><div class='v'>{v}</div>"
                     f"<div class='k'>{k}</div></div>", unsafe_allow_html=True)
    st.caption(
        f"Measured on {metrics['n_test']} documents the model never saw during "
        f"training (20 Newsgroups test split, 4 categories). "
        f"Training-set accuracy is **not** shown - it would overstate performance."
    )
    with st.expander("Per-class breakdown & confusion matrix"):
        import pandas as pd
        rows = []
        for name in nc.SHORT_NAMES:
            pc = metrics["per_class"][name]
            rows.append({"class": name, "precision": round(pc["precision"], 3),
                         "recall": round(pc["recall"], 3), "f1": round(pc["f1"], 3),
                         "support": pc["support"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Rows = true class, columns = predicted class.")
        st.dataframe(
            pd.DataFrame(metrics["confusion_matrix"],
                         index=nc.SHORT_NAMES, columns=nc.SHORT_NAMES),
            use_container_width=True,
        )


def render_model_info(meta: dict, bundle: dict | None, artifact_path: Path) -> None:
    with st.expander("⚙️ Model & artifact details"):
        st.markdown(f"**Display name:** {meta['display']}")
        st.markdown(f"**Folder:** `{meta.get('folder', '')}`")
        st.markdown(f"**Real architecture:** {meta['actual']}")
        st.markdown(f"**Framework:** {meta['framework']}")
        if bundle and bundle.get("hyperparams"):
            st.markdown("**Hyper-parameters**")
            st.json(bundle["hyperparams"])
        st.markdown(f"**Artifact:** `{artifact_path.relative_to(nc.REPO_ROOT)}` "
                    f"({nc.human_bytes(artifact_path.stat().st_size)})")
        if bundle and bundle.get("trained_at"):
            st.caption(f"trained_at={bundle['trained_at']} | "
                       f"sklearn={bundle.get('sklearn_version')} | "
                       f"numpy={bundle.get('numpy_version')}")


def render_limitations(meta: dict) -> None:
    with st.expander("⚠️ Limitations & usage notes"):
        st.markdown(
            f"- Trained only on **4 categories** of the 20-Newsgroups corpus "
            f"({', '.join(nc.SHORT_NAMES)}). Inputs about other topics will still be "
            f"assigned to one of these four.\n"
            f"- Output classes are fixed; the model cannot invent a new topic.\n"
            f"- Posts are informal 1990s Usenet text, so very short or modern inputs "
            f"are harder.\n"
            f"- This is a **classification** demo: it predicts a topic label, it does "
            f"not summarise, translate or generate text.\n"
            f"- At inference the vectorizer truncates/weights internally; extremely "
            f"long input still works but only the most informative features matter."
        )


def example_picker(examples: dict[str, str], key: str) -> str:
    """Render example buttons and return the chosen text (possibly edited)."""
    st.caption("Pick an example, or paste your own text below.")
    if key not in st.session_state:
        st.session_state[key] = ""
    cols = st.columns(len(examples))
    for col, (name, text) in zip(cols, examples.items()):
        if col.button(name, use_container_width=True, key=f"{key}_{name}"):
            st.session_state[key] = text
    return st.text_area(
        "Your text",
        key=key,
        height=150,
        placeholder="Type or paste any text here...",
    )


def artifact_missing_ui(exc: nc.ArtifactMissingError) -> None:
    """Clear setup error instead of a fake prediction."""
    st.error(
        "This app needs a trained model artifact before it can predict. "
        "It will **not** return a fake result.",
        icon="🗂️",
    )
    st.markdown(exc.message)
    st.markdown(
        "**Deploying on Streamlit Community Cloud?** The artifact must be committed "
        "to the repository (it is small - a few MB - see this model's README) or "
        "downloaded at startup from the location given in the README."
    )


def sidebar(meta: dict, bundle: dict | None = None) -> None:
    with st.sidebar:
        st.markdown("### 🧠 NLP Model Portfolio")
        st.markdown(
            f"**Model {meta['num']} of 15**  \n{meta['display']}"
        )
        st.divider()
        st.markdown("**Task**")
        st.markdown(nc.TASK_TOPIC_CLASSIFICATION)
        st.markdown("**Technique**")
        st.caption(meta["actual"])
        if meta.get("needs_gensim"):
            st.markdown("**Runtime note**")
            st.caption("Loads a gensim embedding model on first prediction "
                       "(a few seconds).")
        st.divider()
        st.caption("Part of a 15-model NLP teaching portfolio. "
                   "Every demo runs a real trained model on your input.")
