from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.evaluation.reproduce import dataset_records
from deepfake_detection.reference import load_reference_repo

PREDICTIONS = "moe_combined_retrain_on_combined_test_seed42"
FFPP_FAKE = re.compile(r"^(deepfakes|face2face|faceswap|neuraltextures|faceshifter)_(\d{3})_(\d{3})$")
POLYGLOT_FAKE = re.compile(r"^fake_([a-z]{2})_(\d+)_to_")
CELEB_FAKE = re.compile(r"^id(\d+)_id(\d+)_\d+$")
HIDF_FAKE = re.compile(r"^(c\d+)_(c\d+)$")
FAKEAV_DONOR = re.compile(r"_(id\d+)_")


def fakeavceleb_identity_stems(config) -> dict[str, set[str]]:
    root = config.raw_dir("fakeavceleb") / "FakeAVCeleb_v1.2" / "RealVideo-RealAudio"
    out: dict[str, set[str]] = defaultdict(set)
    for path in root.glob("*/*/*/*.mp4"):
        out[path.parts[-2]].add(path.stem)
    return out


def partners(dataset: str, name: str, config, identity_stems) -> list[tuple[str, str]]:
    """The real video or reals whose face or voice a fake was built from or with, other than the clip its own subject key already follows:
    list of (kind, subject key or sample name of the partner)."""
    if dataset == "polyglotfake" and (m := POLYGLOT_FAKE.match(name)):
        return [("source real", f"{m[1]}_{m[2]}")]
    if dataset == "ffpp" and (m := FFPP_FAKE.match(name)):
        return [("donor identity", m[3])]
    if dataset == "celebdfv2" and (m := CELEB_FAKE.match(name)):
        return [("donor identity", f"celeb_id{m[2]}")]
    if dataset == "hidf" and (m := HIDF_FAKE.match(name)):
        return [("second identity", m[2])]
    if dataset == "fakeavceleb" and (m := FAKEAV_DONOR.search(name)):
        return [("donor identity", f"fac_{stem}") for stem in sorted(identity_stems.get(m[1], ()))]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="For every fake of every dataset: is the real video or identity it was made from or with in another partition of the subject-level split than the fake itself, and does the verdict of the retrained model differ between those fakes and the rest.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    reference = load_reference_repo(_project_root() / repro["reference_repo"])
    seed = args.seed if args.seed is not None else repro["split"]["seed"]
    identity_stems = fakeavceleb_identity_stems(config)

    with open(config.paths.predictions_dir(PREDICTIONS) / "predictions.csv", encoding="utf-8", newline="") as f:
        predicted = {re.sub(r"_seg\d+$", "", Path(r["video"]).stem): (float(r["p_fake_best_f1"]), int(float(r["label"])), r["dataset"]) for r in csv.DictReader(f)}

    rows, summary = [], {"seed": seed, "datasets": {}}
    for dataset in config.dataset_names():
        records = dataset_records(reference, config, dataset)
        train, val, test = reference.subject_split(records, repro["split"]["val_ratio"], seed)
        part_of_sample, part_of_subject = {}, {}
        for part, items in (("train", train), ("validation", val), ("test", test)):
            for r in items:
                part_of_sample[r[0][0].stem.rsplit("_seg", 1)[0]] = part
                part_of_subject[r[2]] = part
        counts, leaky_test = Counter(), []
        for sample, part in part_of_sample.items():
            kinds = partners(dataset, sample, config, identity_stems)
            if not kinds:
                continue
            other = {part_of_subject.get(key) or part_of_sample.get(key) for _, key in kinds}
            other.discard(None)
            different = sorted(other - {part})
            status = "partner not embedded" if not other else ("partner in another partition" if different else "same partition")
            counts[(part, status)] += 1
            rows.append({"dataset": dataset, "fake": sample, "partition": part, "partner_kind": kinds[0][0], "partner_partitions": "+".join(sorted(other)), "status": status})
            if part == "test" and sample in predicted:
                leaky_test.append((sample, status == "partner in another partition"))
        entry = {f"{p} / {s}": v for (p, s), v in sorted(counts.items())}
        test_fakes = {s: leaky for s, leaky in leaky_test}
        reals = [v[0] for k, v in predicted.items() if v[2] == dataset and v[1] == 0]
        effect = {}
        for name, selected in (("fakes with a partner in another partition", [s for s, l in test_fakes.items() if l]),
                               ("fakes without", [s for s, l in test_fakes.items() if not l])):
            fakes = [predicted[s][0] for s in selected]
            if fakes and reals:
                effect[name] = {"n": len(fakes), "flagged_at_0.5": round(float(np.mean(np.array(fakes) >= 0.5)), 4),
                                "auc_against_all_test_reals": round(float(roc_auc_score([1] * len(fakes) + [0] * len(reals), fakes + reals)), 4)}
        summary["datasets"][dataset] = {"fakes_by_partition_and_status": entry, "test_fakes_effect": effect}

    out = config.paths.analysis_dir("split_leakage_audit")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "fakes.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False, width=140), encoding="utf-8")
    print(yaml.safe_dump(summary, sort_keys=False, width=140))


if __name__ == "__main__":
    main()
