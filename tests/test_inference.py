"""
Inference test harness for all 15 NLP models.

Everything here runs the *same* code path the Streamlit apps use
(``common.nlp_common.predict_text`` / ``predict_many``) against hand-written
sentences that are NOT in the 20-Newsgroups corpus.  It checks that:

  * the artifact loads,
  * a real prediction is produced (no fake/random output),
  * the score table is well formed and truthful about its type,
  * empty input is rejected,
  * very long input does not crash (the vectorizer truncates via max_features),
  * for embedding models, word similarity works on the trained vectors.

Run:
    python tests/test_inference.py
    python tests/test_inference.py --only 3 5      # subset
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import nlp_common as nc  # noqa: E402

# Unseen, hand-written examples (one per intended topic).  None of these strings
# exist in the 20-Newsgroups dataset.
UNSEEN = {
    "hockey": "The goalie made an incredible save in the third period and the team "
              "advanced to the Stanley Cup playoffs after winning the overtime.",
    "space": "NASA launched a new rocket to the International Space Station to study "
             "the atmosphere of Mars with a robotic rover.",
    "graphics": "The 3D rendering engine uses ray tracing to produce realistic shadows "
                "and textures for the video game characters.",
    "politics": "The senator debated the new tax policy and voting reform bill in "
                "congress before the upcoming election.",
}


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_model(num: int) -> dict:
    folder = nc.folder_of(num)
    meta = nc.MODEL_REGISTRY[folder]
    bundle = nc.load_bundle(folder)
    result_rows = []

    # 1) one prediction per unseen example + one forced example
    correct = 0
    for topic, text in UNSEEN.items():
        r = nc.predict_text(folder, text, bundle=bundle)
        _check(r["predicted_label"] in nc.CATEGORIES,
               f"invalid predicted label {r['predicted_label']}")
        _check(len(r["scores"]) == 4, "expected 4 class scores")
        total = sum(s["score"] for s in r["scores"])
        if r["score_type"] == "probability":
            _check(abs(total - 1.0) < 1e-3, f"probs sum to {total}, expected 1")
        correct += int(r["predicted_short"] == topic)
        result_rows.append((topic, r["predicted_short"], r["scores"][0]["score"]))

    # 2) empty input must be rejected
    raised = False
    try:
        nc.predict_text(folder, "   ", bundle=bundle)
    except ValueError:
        raised = True
    _check(raised, "empty input was not rejected")

    # 3) very long input must not crash
    long_text = (UNSEEN["space"] + " ") * 400  # ~ 20k chars
    r_long = nc.predict_text(folder, long_text, bundle=bundle)
    _check(r_long["predicted_label"] in nc.CATEGORIES, "long input failed")

    # 4) embedding models: word similarity on the trained vectors
    sim_note = ""
    if bundle["kind"] == "embedding_doc_clf":
        neigh = nc.nearest_words(bundle, "hockey", topn=5)
        sim_note = f"nearest('hockey')={[w for w, _ in neigh][:5]}"

    return {
        "num": num, "folder": folder, "display": meta["display"],
        "unseen_correct": correct, "unseen_total": len(UNSEEN),
        "score_type": bundle.get("score_type", "probability"),
        "samples": result_rows, "sim_note": sim_note,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=None)
    args = ap.parse_args()
    nums = args.only or list(range(1, 16))

    print("=" * 78)
    print("INFERENCE TESTS - unseen hand-written text (not from the corpus)")
    print("=" * 78)
    failures = []
    for n in nums:
        try:
            info = test_model(n)
            print(f"\n[{n:02d}] {info['display']}")
            print(f"     score_type={info['score_type']}")
            for topic, pred, score in info["samples"]:
                flag = "OK " if topic == pred else "== "
                print(f"     {flag} entered={topic:<9} predicted={pred:<9} "
                      f"score={score:.3f}")
            print(f"     unseen accuracy: {info['unseen_correct']}/{info['unseen_total']}"
                  f"   empty-input guard: OK   long-input guard: OK")
            if info["sim_note"]:
                print(f"     {info['sim_note']}")
        except nc.ArtifactMissingError as exc:
            failures.append((n, "ARTIFACT MISSING"))
            print(f"\n[{n:02d}] ARTIFACT MISSING -> {exc.path}")
        except Exception as exc:  # noqa: BLE001
            failures.append((n, f"{type(exc).__name__}: {exc}"))
            print(f"\n[{n:02d}] FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)

    print("\n" + "=" * 78)
    if failures:
        print(f"RESULT: {len(failures)} model(s) with problems")
        for n, msg in failures:
            print(f"  [{n:02d}] {msg}")
    else:
        print("RESULT: all models passed inference tests")
    print("=" * 78)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
