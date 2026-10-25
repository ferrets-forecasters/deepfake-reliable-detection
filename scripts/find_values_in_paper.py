"""Check that every number in configs/paper_tables.yaml appears in the paper text as the paper writes it (decimal comma, thousands dot).
Run after verify_paper_tables.py: that script checks the values against the results, this one checks the paper prints those values."""
import argparse
import re
import sys
import zipfile
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "paper_tables.yaml"
SKIP = {"seeds", "folds"}
# Checked against the results by verify_paper_tables.py but not printed in the paper yet: remove a key when the paper prints it.
PENDING: set[str] = {"sides", "clips", "partner_elsewhere_share", "openforensics", "still_frame"}  # no longer printed after the Keterbatasan cuts of 2026-10-24


def paper_text(docx: Path) -> str:
    with zipfile.ZipFile(docx) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    paragraphs = re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)
    return "\n".join("".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p)) for p in paragraphs)


def leaves(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in PENDING:
                continue
            yield from leaves(value, path + (str(key),))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from leaves(value, path + (str(index),))
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield "/".join(path), node


def spellings(value) -> set[str]:
    if isinstance(value, int):
        return {f"{value:,}".replace(",", ".")}
    forms = set()
    for places in (4, 3, 2, 1):
        text = f"{abs(value):.{places}f}".replace(".", ",")
        forms.add(text)
        forms.add(text.rstrip("0").rstrip(",") if "," in text else text)
    return forms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx", type=Path)
    args = parser.parse_args()
    text = paper_text(args.docx).replace("−", "-")
    paper = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    missing = total = 0
    for path, value in leaves(paper):
        total += 1
        precise = max(spellings(value), key=len)
        if not any(re.search(rf"(?<![\d,.]){re.escape(form)}(?![\d])", text) for form in spellings(value)):
            missing += 1
            print(f"not in the paper: {path} = {value} (looked for {precise})")
    print(f"{total} values, {missing} not found")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
