"""
NLP Model Portfolio - Streamlit dashboard for all 15 models.

This is the front page of the portfolio. It lists every model with honest
metadata (task, framework, real architecture, verified held-out accuracy,
artifact status) and points to each model's own Streamlit entry point.

Deploy this on Streamlit Community Cloud with main file path: app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import nlp_common as nc  # noqa: E402

GITHUB_REPO = "https://github.com/deepvisionkararhaider-crypto/nlp-models"

st.set_page_config(page_title="NLP Model Portfolio", page_icon="🧠",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      .card {border:1px solid #e2e8f0;border-radius:14px;padding:1rem 1.15rem;
             background:#fff;height:100%;box-shadow:0 1px 3px rgba(15,23,42,.05);}
      .card h4 {margin:.1rem 0 .35rem 0;font-size:1.05rem;}
      .chip {display:inline-block;padding:.12rem .5rem;border-radius:999px;
             font-size:.7rem;font-weight:600;margin:.12rem .18rem .12rem 0;}
      .chip-task{background:#e0e7ff;color:#3730a3;}
      .chip-fw{background:#f1f5f9;color:#334155;}
      .chip-ok{background:#dcfce7;color:#166534;}
      .chip-warn{background:#fef3c7;color:#92400e;}
      .chip-miss{background:#fee2e2;color:#991b1b;}
      .acc {font-size:1.25rem;font-weight:700;color:#0f172a;}
      .acc-k {font-size:.68rem;color:#64748b;text-transform:uppercase;
              letter-spacing:.05em;}
      .hero {padding:1.6rem 1.8rem;border-radius:16px;margin-bottom:1.1rem;
             background:linear-gradient(120deg,#2563eb14,#7c3aed08);
             border:1px solid #dbeafe;border-left:6px solid #2563eb;}
      .hero h1 {margin:0 0 .3rem 0;color:#1d4ed8;font-size:2rem;}
      .hero p {margin:0;color:#475569;max-width:70ch;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🧠 NLP Model Portfolio</h1>
      <p>Fifteen natural-language-processing projects, each with a real trained
      model behind a Streamlit front end. Enter new text, run the model, and see
      the actual output. Task: multi-class topic classification on four
      20-Newsgroups categories (hockey, space, graphics, politics).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

models = nc.list_models()

# ---- summary metrics -------------------------------------------------------
n_ready = sum(1 for m in models if nc.artifacts_available(nc.folder_of(m["num"])))
n_faithful = sum(1 for m in models if m["faithful_family"])
accs = []
for m in models:
    met = nc.load_metrics(nc.folder_of(m["num"]))
    if met:
        accs.append(met["accuracy"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Models", "15")
c2.metric("Artifacts ready", f"{n_ready}/15")
c3.metric("Best test accuracy", f"{max(accs) * 100:.1f}%" if accs else "n/a")
c4.metric("Mean test accuracy", f"{sum(accs) / len(accs) * 100:.1f}%" if accs else "n/a")

st.info(
    "**Honesty note.** Folders **07-15** are *named* after transformer "
    "architectures (BERT, RoBERTa, T5, ...) but the repository implements them as "
    "TF-IDF + SVD + classical scikit-learn classifiers - no transformer is trained "
    "or downloaded. Each card states the **real** architecture. "
    f"({n_faithful} of 15 models have a name that matches their implementation.)",
    icon="ℹ️",
)

# ---- agreement ("consensus") panel ----------------------------------------
with st.expander("🤝 Do the models agree? (inter-model consensus)"):
    st.caption(
        "Paste a sentence and compare every model's prediction side by side. All 15 "
        "models share the same four output labels, so disagreement is meaningful - "
        "it highlights genuinely ambiguous text."
    )
    demo_text = st.text_area(
        "Text to classify", value="NASA launched a new satellite to study the "
        "atmosphere of Mars.", height=90, key="consensus_input")
    if st.button("Compare all 15 models", key="consensus_btn"):
        if not demo_text.strip():
            st.warning("Enter some text first.")
        else:
            rows = []
            for m in models:
                folder = nc.folder_of(m["num"])
                if not nc.artifacts_available(folder):
                    rows.append({"#": m["num"], "model": m["display"],
                                 "prediction": "(artifact missing)", "score": None})
                    continue
                try:
                    r = nc.predict_text(folder, demo_text)
                    rows.append({"#": m["num"], "model": m["display"],
                                 "prediction": r["predicted_short"],
                                 "score": round(r["scores"][0]["score"], 3)})
                except Exception as exc:  # noqa: BLE001
                    rows.append({"#": m["num"], "model": m["display"],
                                 "prediction": f"error: {exc}", "score": None})
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)
            counts = df["prediction"].value_counts().to_dict()
            st.markdown("**Prediction distribution:** " +
                        ", ".join(f"{k}: {v}" for k, v in counts.items()))

st.divider()
st.markdown("## The 15 models")

cols = st.columns(3)
for i, m in enumerate(models):
    folder = nc.folder_of(m["num"])
    met = nc.load_metrics(folder)
    ready = nc.artifacts_available(folder)
    col = cols[i % 3]
    with col:
        chip_fam = ("chip-ok", "Technique matches name") if m["faithful_family"] \
            else ("chip-warn", "Name ≠ implementation")
        chip_art = ("chip-ok", "Artifact ready") if ready else ("chip-miss", "Artifact missing")
        acc_html = (f"<div class='acc'>{met['accuracy'] * 100:.1f}%</div>"
                    f"<div class='acc-k'>test accuracy</div>") if met else \
                   "<div class='acc'>-</div><div class='acc-k'>not trained yet</div>"
        f1_html = f"F1 {met['f1_macro'] * 100:.1f}%" if met else ""
        st.markdown(
            f"""
            <div class="card">
              <div class="acc-k">Model {m['num']:02d}</div>
              <h4>{m['display']}</h4>
              <div>
                <span class="chip chip-task">{nc.TASK_TOPIC_CLASSIFICATION.split(' (')[0]}</span>
                <span class="chip chip-fw">{m['framework']}</span>
                <span class="chip {chip_fam[0]}">{chip_fam[1]}</span>
                <span class="chip {chip_art[0]}">{chip_art[1]}</span>
              </div>
              <p style="font-size:.85rem;color:#475569;margin:.5rem 0;">{m['description']}</p>
              <p style="font-size:.78rem;color:#64748b;margin:.4rem 0 0 0;"><b>Really:</b>
              {m['actual']}</p>
              <div style="margin-top:.5rem;display:flex;gap:1rem;align-items:flex-end;">
                <div>{acc_html}</div>
                <div class="acc-k" style="margin-bottom:.35rem;">{f1_html}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Entry point:** `{folder}/app.py`  \n"
            f"[Source folder]({GITHUB_REPO}/tree/main/{folder})",
            unsafe_allow_html=False,
        )
        st.write("")

st.divider()
st.markdown("## Deploying each model (free, Streamlit Community Cloud)")
st.markdown(
    f"""
1. Go to **share.streamlit.io** → *New app* → **Deploy from GitHub**.
2. Repository: `deepvisionkararhaider-crypto/nlp-models`, branch `main`.
3. Set **Main file path** to one of the entry points below.
4. Deploy. Repeat for the other models (or use the *workspace* to host several).

| # | Model | Main file path |
|---|-------|----------------|
"""
)
rows = "\n".join(f"| {m['num']:02d} | {m['display']} | `{nc.folder_of(m['num'])}/app.py` |"
                 for m in models)
st.markdown(rows)
st.caption(
    "This dashboard itself deploys with main file path `app.py`. "
    "See `DEPLOYMENT.md` in the repository for the full checklist."
)

st.divider()
st.caption(
    "All predictions shown anywhere in this portfolio come from real trained "
    "models run on your input. No hard-coded, simulated or randomised outputs."
)
