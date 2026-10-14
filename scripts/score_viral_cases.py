from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.inference.factory import build_pipeline
from deepfake_detection.utils.watchdog import hang_watchdog

MANIFEST_FIELDS = ["case_id", "title", "date", "platform", "source_url", "claim", "basis_url", "language", "license_note", "file"]
SUMMARY_FIELDS = ["case_id", "title", "date", "claim", "video_verdict", "video_p_fake", "faces", "p_fake_per_face", "alpha_per_face", "duration_s",
                  "width", "height", "fps", "agrees_with_claim"]


def read_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in MANIFEST_FIELDS if rows and c not in rows[0]]
    if missing:
        raise SystemExit(f"{path} lacks the columns {missing}")
    return rows


def video_facts(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    facts = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "fps": round(cap.get(cv2.CAP_PROP_FPS), 2)}
    cap.release()
    return facts


def summary_row(case: dict, payload: dict, facts: dict) -> dict:
    """claim is 'fake' or 'real' from the fact-check named in basis_url, or '-' when none exists; the verdict agrees when it equals the claim."""
    faces = sorted(payload["subjects"], key=lambda s: s["center_x"])
    verdict = "fake" if faces and payload["video_p_fake"] >= payload.get("threshold", 0.5) else "real" if faces else "no_face"
    claim = case["claim"].strip().lower()
    return {
        "case_id": case["case_id"], "title": case["title"], "date": case["date"], "claim": claim or "-", "video_verdict": verdict,
        "video_p_fake": round(payload["video_p_fake"], 4) if faces else "-", "faces": len(faces),
        "p_fake_per_face": ";".join(f"{s['p_fake']:.4f}" for s in faces) or "-",
        "alpha_per_face": ";".join("-" if s["alpha"] is None else f"{s['alpha']:.3f}" for s in faces) or "-",
        "duration_s": round(payload["duration"], 1), **facts,
        "agrees_with_claim": "-" if claim not in ("fake", "real") or verdict == "no_face" else str(verdict == claim).lower(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scores the viral deepfake cases listed in datasets/viral_cases/labels/cases.csv with an exported model, one JSON per case, resumable, and writes a summary table. The cases and their claim (fake or real, with the fact-check that says so) are filled in by hand.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", type=Path, default=None, help="default: datasets/viral_cases")
    parser.add_argument("--model", default="moe_combined_retrain")
    parser.add_argument("--diarizer", default="ecapa", choices=["ecapa", "single"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--force", action="store_true", help="score again even when a result file exists")
    parser.add_argument("--claims", nargs="+", default=None, help="score only the cases with these claims, for example fake real (the cut stretches); default every case")
    args = parser.parse_args()

    config = load_config(args.config)
    root = args.root or config.paths.datasets_root / "viral_cases"
    cases = read_manifest(root / "labels" / "cases.csv")
    out_dir = config.paths.predictions_dir(f"{args.model}_on_viral_cases")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = config.paths.analysis_dir("viral_cases")
    summary_dir.mkdir(parents=True, exist_ok=True)

    todo = [c for c in cases if (root / "raw" / c["file"]).exists() and (args.force or not (out_dir / f"{c['case_id']}.json").exists())
            and (not args.claims or c["claim"].strip().lower() in args.claims)]
    missing = [c["case_id"] for c in cases if not (root / "raw" / c["file"]).exists()]
    log = open(out_dir / "progress.log", "a", encoding="utf-8")
    log.write(f"{time.strftime('%F %T')} {len(cases)} cases, {len(todo)} to score, {len(missing)} without a file: {missing}\n")
    if todo:
        pipeline = build_pipeline(config, args.model, args.diarizer, threshold=args.threshold)
        for case in tqdm(todo, desc="cases", unit="video"):
            started = time.time()
            with hang_watchdog(300, log, case["case_id"]):
                result = pipeline.run(root / "raw" / case["file"])
            payload = {"case_id": case["case_id"], "video_p_fake": result.video_p_fake, "threshold": result.threshold, "duration": result.duration,
                       "turns": [asdict(t) for t in result.turns], "subjects": [asdict(s) for s in result.subjects]}
            (out_dir / f"{case['case_id']}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log.write(f"{time.strftime('%T')} {case['case_id']} p_fake {result.video_p_fake:.4f} faces {len(result.subjects)} in {time.time() - started:.1f} s\n")
            log.flush()

    rows = []
    for case in cases:
        path = out_dir / f"{case['case_id']}.json"
        if path.exists():
            rows.append(summary_row(case, json.loads(path.read_text(encoding="utf-8")), video_facts(root / "raw" / case["file"])))
    with open(summary_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    known = [r for r in rows if r["agrees_with_claim"] != "-"]
    stats = {"model": args.model, "cases_in_manifest": len(cases), "scored": len(rows), "without_file": missing, "no_face": [r["case_id"] for r in rows if r["video_verdict"] == "no_face"],
             "with_known_claim": len(known), "agree": sum(1 for r in known if r["agrees_with_claim"] == "true"),
             "fake_claims_caught": sum(1 for r in known if r["claim"] == "fake" and r["agrees_with_claim"] == "true"), "fake_claims": sum(1 for r in known if r["claim"] == "fake"),
             "real_claims_flagged": sum(1 for r in known if r["claim"] == "real" and r["agrees_with_claim"] == "false"), "real_claims": sum(1 for r in known if r["claim"] == "real")}
    (summary_dir / "summary.yaml").write_text(yaml.safe_dump(stats, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(stats, sort_keys=False))


if __name__ == "__main__":
    main()
