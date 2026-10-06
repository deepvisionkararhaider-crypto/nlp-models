"""
Build the training dataset for the 15 NLP models.

The original notebooks/scripts use ``sklearn.datasets.fetch_20newsgroups`` with
4 categories and ``remove=('headers', 'footers', 'quotes')``.  In many sandboxes
(and on some hosts) the figshare mirror that scikit-learn downloads from returns
HTTP 403, so we make the dataset build reproducible from the *original* 20
Newsgroups archive (``20news-bydate.tar.gz``) instead.

The header / footer / quote stripping below mirrors scikit-learn's
``twenty_newsgroups.py`` implementation so the resulting corpus is equivalent to
``fetch_20newsgroups(remove=('headers','footers','quotes'))``.

Usage
-----
    python training/build_dataset.py --archive /path/to/20news-bydate.tar.gz

If ``--archive`` is omitted, the script tries, in order:
    1. an already extracted directory (``--extracted``),
    2. ``scikit-learn``'s own download (works when the mirror is reachable),
    3. a bundled archive at ``data/raw/20news-bydate.tar.gz``.

Output
------
    data/20newsgroups_4cat/train.csv   columns: text, label, category
    data/20newsgroups_4cat/test.csv
    data/20newsgroups_4cat/meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path

import pandas as pd

# The four categories used by every model in this repository (order matters:
# it defines the integer label -> class mapping used across all 15 models).
CATEGORIES = [
    "rec.sport.hockey",
    "sci.space",
    "comp.graphics",
    "talk.politics.misc",
]
SHORT_NAMES = ["hockey", "space", "graphics", "politics"]

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "20newsgroups_4cat"

# --- scikit-learn compatible stripping -------------------------------------
# Ref: sklearn/datasets/twenty_newsgroups.py
_QUOTE_RE = re.compile(
    r"(writes in|writes:|wrote:|says:|said:"
    r"|^In article|^Quoted from|^\||^>)"
)


def strip_newsgroup_header(text: str) -> str:
    """Remove everything up to and including the first blank line."""
    _before, _blankline, after = text.partition("\n\n")
    return after


def strip_newsgroup_quoting(text: str) -> str:
    """Drop lines that look like quoted material."""
    good_lines = [line for line in text.split("\n") if not _QUOTE_RE.search(line)]
    return "\n".join(good_lines)


def strip_newsgroup_footer(text: str) -> str:
    """Drop a trailing .signature block.

    scikit-learn considers the footer to start at the last line consisting of a
    repeated separator character (``-``, ``+`` or ``*``), provided at least 3
    lines follow it.  We reproduce that heuristic, but guard the "no separator
    found" case: without the guard the function would return an empty string for
    any document longer than 3 lines (which would silently destroy the corpus).
    """
    lines = text.strip().split("\n")
    separator_line = None
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip("-+*") == "" and len(line) > 5:
            separator_line = line_num
            break
    if separator_line is None:
        return text
    if len(lines) - separator_line > 3:
        return "\n".join(lines[:separator_line])
    return text


def clean(text: str) -> str:
    text = strip_newsgroup_header(text)
    text = strip_newsgroup_quoting(text)
    text = strip_newsgroup_footer(text)
    return text


def _load_split_from_dir(root: Path, subset: str) -> list[tuple[str, int]]:
    """Read ``20news-bydate-<subset>/<category>/<id>`` files for CATEGORIES."""
    base = root / f"20news-bydate-{subset}"
    if not base.is_dir():
        raise FileNotFoundError(f"missing split directory: {base}")
    rows: list[tuple[str, int]] = []
    for label, cat in enumerate(CATEGORIES):
        cat_dir = base / cat
        if not cat_dir.is_dir():
            raise FileNotFoundError(f"missing category directory: {cat_dir}")
        for path in sorted(cat_dir.iterdir()):
            if not path.is_file():
                continue
            try:
                raw = path.read_text(encoding="latin-1")
            except OSError:
                continue
            rows.append((clean(raw), label))
    return rows


def _extract_archive(archive: Path, tmpdir: Path) -> Path:
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(tmpdir)  # noqa: S202 - trusted, user-provided archive
    if (tmpdir / "20news-bydate-train").is_dir():
        return tmpdir
    # archive may contain a single top-level folder
    for child in tmpdir.iterdir():
        if (child / "20news-bydate-train").is_dir():
            return child
    raise FileNotFoundError("could not locate 20news-bydate-train in archive")


def _from_sklearn() -> tuple[list, list]:
    from sklearn.datasets import fetch_20newsgroups

    train = fetch_20newsgroups(
        subset="train", categories=CATEGORIES,
        remove=("headers", "footers", "quotes"), random_state=42,
    )
    test = fetch_20newsgroups(
        subset="test", categories=CATEGORIES,
        remove=("headers", "footers", "quotes"), random_state=42,
    )
    tr = list(zip(train.data, train.target))
    te = list(zip(test.data, test.target))
    return tr, te


def build(archive: Path | None, extracted: Path | None) -> None:
    tmpdir = None
    try:
        if extracted is not None:
            root = extracted
            print(f"[data] using extracted directory: {root}")
        elif archive is not None:
            tmpdir = Path(tempfile.mkdtemp(prefix="20news-"))
            root = _extract_archive(archive, tmpdir)
            print(f"[data] extracted archive to: {root}")
        else:
            bundled = REPO_ROOT / "data" / "raw" / "20news-bydate.tar.gz"
            if bundled.exists():
                tmpdir = Path(tempfile.mkdtemp(prefix="20news-"))
                root = _extract_archive(bundled, tmpdir)
                print(f"[data] extracted bundled archive to: {root}")
            else:
                print("[data] no archive provided - trying scikit-learn download")
                tr_rows, te_rows = _from_sklearn()
                _write(tr_rows, te_rows, source="sklearn.fetch_20newsgroups")
                return

        tr_rows = _load_split_from_dir(root, "train")
        te_rows = _load_split_from_dir(root, "test")
        _write(tr_rows, te_rows, source=f"20news-bydate archive ({root})")
    finally:
        if tmpdir is not None:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


def _write(tr_rows: list, te_rows: list, source: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def to_df(rows):
        return pd.DataFrame(
            {
                "text": [r[0] for r in rows],
                "label": [int(r[1]) for r in rows],
                "category": [CATEGORIES[int(r[1])] for r in rows],
            }
        )

    train_df, test_df = to_df(tr_rows), to_df(te_rows)
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)
    meta = {
        "source": source,
        "categories": CATEGORIES,
        "short_names": SHORT_NAMES,
        "label_map": {str(i): c for i, c in enumerate(CATEGORIES)},
        "remove": ["headers", "footers", "quotes"],
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[data] wrote {len(train_df)} train / {len(test_df)} test rows -> {OUT_DIR}")
    print(f"[data] per-class train counts: "
          f"{train_df['category'].value_counts().to_dict()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=None,
                    help="path to 20news-bydate.tar.gz")
    ap.add_argument("--extracted", type=Path, default=None,
                    help="path to an already extracted 20news-bydate directory")
    args = ap.parse_args()
    build(args.archive, args.extracted)


if __name__ == "__main__":
    main()
