"""Recompute the numbers of the paper's result tables from the saved predictions and compare them with
the values the paper prints (configs/paper_tables.yaml). Prints one line per number: ok or MISMATCH."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
INFER = ROOT / "results" / "inferences"
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "paper_tables.yaml"
SEEDS = {42: "", 43: "", 44: ""}


def scores(frame: pd.DataFrame, column: str = "p_fake_best_f1") -> dict:
    label, p = frame["label"].to_numpy(), frame[column].to_numpy()
    # videos at exactly 0.5 (half precision) fall on either side in the training evaluation, so both are kept
    return {"f1": f1_score(label, p >= 0.5, average="macro"), "f1_strict": f1_score(label, p > 0.5, average="macro"),
            "ap": average_precision_score(label, p), "auc": roc_auc_score(label, p)}


def predictions(folder: str) -> pd.DataFrame:
    return pd.read_csv(INFER / folder / "predictions.csv")


def seed_folder(model: str, seed: int) -> str:
    if seed == 42:
        return f"{model}_combined_retrain_on_combined_test_seed42"
    return f"{model}_combined_seed{seed}_on_combined_test_seed{seed}"


def mean_sd(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


class Report:
    def __init__(self, places: int = 4):
        self.places, self.lines, self.bad = places, [], 0

    def check(self, name: str, got: float, paper: float, places: int | None = None, alternative: float | None = None) -> None:
        places = self.places if places is None else places
        low, high = sorted([got, got if alternative is None else alternative])
        ok = round(low, places) <= round(paper, places) <= round(high, places)
        self.bad += not ok
        self.lines.append(f"{'ok      ' if ok else 'MISMATCH'} {name}: paper {paper:.{places}f}, recomputed {got:.{places}f}")


def tabel_in_distribution(report: Report, paper: dict) -> None:
    for model, key in (("moe", "gate"), ("concat", "static")):
        per_seed = [scores(predictions(seed_folder(model, s))) for s in SEEDS]
        for metric in ("f1", "ap", "auc"):
            mean, sd = mean_sd([r[metric] for r in per_seed])
            report.check(f"in-distribution {key} {metric} mean", mean, paper["seeds"][key][metric][0])
            report.check(f"in-distribution {key} {metric} sd", sd, paper["seeds"][key][metric][1])
        folds = [scores(predictions(f"{model}_cv_fold{i}_on_cv_fold{i}_test")) for i in range(5)]
        for metric in ("f1", "ap", "auc"):
            mean, sd = mean_sd([r[metric] for r in folds])
            report.check(f"five folds {key} {metric} mean", mean, paper["folds"][key][metric][0])
            report.check(f"five folds {key} {metric} sd", sd, paper["folds"][key][metric][1])
    pooled = {m: scores(pd.concat([predictions(f"{m}_cv_fold{i}_on_cv_fold{i}_test") for i in range(5)])) for m in ("moe", "concat")}
    report.check("pooled out-of-fold videos", len(pd.concat([predictions(f"moe_cv_fold{i}_on_cv_fold{i}_test") for i in range(5)])),
                 paper["pooled_videos"])
    for metric in ("f1", "auc", "ap"):
        report.check(f"pooled gate minus static {metric}", pooled["moe"][metric] - pooled["concat"][metric], paper["pooled_difference"][metric])


def tabel_leave_one_out(report: Report, paper: dict) -> None:
    for dataset, row in paper["leave_one_out"].items():
        frame = predictions(f"moe_leave_{dataset}_retrain_on_{dataset}")
        got = scores(frame)
        for metric, value in zip(("f1", "ap", "auc"), row[:3]):
            report.check(f"leave-one-out {dataset} {metric}", got[metric], value,
                         alternative=got["f1_strict"] if metric == "f1" else None)
        report.check(f"leave-one-out {dataset} alpha", float(frame["alpha_best_f1"].mean()), row[3], places=3)
    combined = seed42("moe_combined_retrain")
    for dataset, value in paper["combined_alpha_seed42"].items():
        report.check(f"combined model {dataset} alpha", float(combined[combined.dataset == dataset].alpha_best_f1.mean()), value, places=3)


def tabel_two_people(report: Report, paper: dict) -> None:
    frame = predictions("moe_combined_retrain_on_fakeavceleb_tiled_own_test_lossless")
    label = frame["label"].to_numpy()
    expected = paper["two_people_first_segment"]
    for column, key in (("single_p_fake", "single"), ("multi_p_fake", "per_person")):
        p = frame[column].to_numpy()
        report.check(f"two people video auc {key}", roc_auc_score(label, p), expected["video_auc"][key])
        report.check(f"two people accuracy {key}", float(((p >= 0.5) == label).mean()), expected["accuracy"][key])
    fake, real = [], []
    for _, row in frame.iterrows():
        for side in ("left", "right"):
            (fake if row["fake_side"] == side else real).append(row[f"{side}_p_fake"])
    faces = np.array(fake + real), np.array([1] * len(fake) + [0] * len(real))
    report.check("two people face auc", roc_auc_score(faces[1], faces[0]), expected["face_auc"])
    report.check("two people fake faces", len(fake), expected["faces"][0])
    report.check("two people real faces", len(real), expected["faces"][1])


def tabel_calibration(report: Report, paper: dict) -> None:
    saved = yaml.safe_load((ROOT / "results" / "metrics" / "calibration_retrain" / "calibration.yaml").read_text(encoding="utf-8"))
    for name, row in paper["calibration"].items():
        got = saved[name]
        report.check(f"calibration {name} f1 at 0.5", got["macro_f1_at_0.5"], row[0], places=3)
        report.check(f"calibration {name} 20 videos", got["by_labelled_videos"][20]["macro_f1_threshold"][0], row[1])
        report.check(f"calibration {name} 500 videos", got["by_labelled_videos"][500]["macro_f1_threshold"][0], row[2])
        report.check(f"calibration {name} ece", got["ece"], row[3])


def seed42(name: str) -> pd.DataFrame:
    return predictions(f"{name}_on_combined_test_seed42")


def text_ablation(report: Report, paper: dict) -> None:
    for name, row in paper["ablation_seed42"].items():
        got = scores(seed42(name))
        for metric, value in zip(("f1", "auc", "ap"), row):
            if value is not None:
                report.check(f"ablation {name} {metric}", got[metric], value, alternative=got["f1_strict"] if metric == "f1" else None)


def text_per_dataset(report: Report, paper: dict) -> None:
    gate, static = seed42("moe_combined_retrain"), seed42("concat_combined_retrain")
    for dataset, (gf1, gauc, sf1, sauc) in paper["per_dataset_seed42"].items():
        g, s = scores(gate[gate.dataset == dataset]), scores(static[static.dataset == dataset])
        report.check(f"{dataset} gate f1", g["f1"], gf1, alternative=g["f1_strict"])
        report.check(f"{dataset} gate auc", g["auc"], gauc)
        report.check(f"{dataset} static f1", s["f1"], sf1, alternative=s["f1_strict"])
        report.check(f"{dataset} static auc", s["auc"], sauc)
    video_only_model = seed42("video_only_pure")
    for dataset, difference in paper["gate_minus_video_only_f1_seed42"].items():
        g = scores(gate[gate.dataset == dataset])["f1"]
        v = scores(video_only_model[video_only_model.dataset == dataset])["f1"]
        report.check(f"{dataset} gate minus video-only f1", g - v, difference, places=3)


def text_alpha(report: Report, paper: dict) -> None:
    gate, expected = seed42("moe_combined_retrain"), paper["alpha_seed42"]
    fakes = gate[gate.label == 1]
    report.check("alpha mean video-audio fakes", float(fakes[fakes.manip_type == "audio_video"].alpha_best_f1.mean()), expected["audio_video"], places=3)
    report.check("alpha mean video-only fakes", float(fakes[fakes.manip_type == "video_only"].alpha_best_f1.mean()), expected["video_only"], places=3)
    report.check("alpha mean hidf fakes", float(fakes[fakes.dataset == "hidf"].alpha_best_f1.mean()), expected["hidf"], places=3)
    report.check("videos video-audio fakes", int((fakes.manip_type == "audio_video").sum()), expected["audio_video_n"])
    report.check("videos video-only fakes", int((fakes.manip_type == "video_only").sum()), expected["video_only_n"])
    no_audio = fakes[fakes.dataset.isin(["celebdfv2", "ffpp"])]
    report.check("fakes without audio", len(no_audio), expected["no_audio_n"])
    gate["error"] = (gate.p_fake_best_f1 >= 0.5).astype(int) != gate.label
    for (low, high), (n, error) in zip(((0, 0.2), (0.2, 0.4), (0.4, 1.01)), expected["error_by_alpha"]):
        group = gate[(gate.alpha_best_f1 >= low) & (gate.alpha_best_f1 < high)]
        report.check(f"alpha {low} to {high} videos", len(group), n)
        report.check(f"alpha {low} to {high} error", float(group["error"].mean()), error, places=3)
    for name, (audio_video, video_only) in paper["alpha_other_runs"].items():
        run = seed42(name)
        run = run[run.label == 1]
        report.check(f"{name} alpha video-audio", float(run[run.manip_type == "audio_video"].alpha_best_f1.mean()), audio_video, places=3)
        report.check(f"{name} alpha video-only", float(run[run.manip_type == "video_only"].alpha_best_f1.mean()), video_only, places=3)


def multi_person_sets(report: Report, expected: dict) -> None:
    """Clip reuse, speaker count, every-segment scores and real-real flags of the two own-test two-person sets, from the saved JSON."""
    import json
    for name, key in (("tiled", "tiled"), ("conversation", "conversation")):
        manifest = pd.read_csv(ROOT / "datasets" / f"fakeavceleb_{name}_own_test_lossless" / "labels" / "manifest.csv")
        folder = INFER / f"moe_combined_retrain_on_fakeavceleb_{name}_own_test_lossless"
        sides = list(manifest.left_clip) + list(manifest.right_clip)
        report.check(f"{key} sides", len(sides), expected[key]["sides"])
        report.check(f"{key} distinct clips", len(set(sides)), expected[key]["clips"])
        results = {row.sample_id: json.loads((folder / f"{row.sample_id}.json").read_text(encoding="utf-8")) for row in manifest.itertuples()}
        video = np.array([results[s]["video_p_fake"] for s in manifest.sample_id])
        report.check(f"{key} video auc, every segment", roc_auc_score(manifest.label, video), expected[key]["video_auc"])
        real_real = manifest.label.to_numpy() == 0
        report.check(f"{key} real-real videos", int(real_real.sum()), expected[key]["real_real"][1])
        report.check(f"{key} real-real flagged", int((video[real_real] >= 0.5).sum()), expected[key]["real_real"][0])
        if key == "conversation":
            speakers = [len({t["speaker"] for t in results[s]["turns"]}) for s in manifest.sample_id]
            report.check("conversation videos with two speakers", sum(n == 2 for n in speakers), expected[key]["two_speakers"])
            starts = [sorted({t["end"] for t in results[s]["turns"] if t["speaker"] == "speaker_0"})[-1] for s in manifest.sample_id if len({t["speaker"] for t in results[s]["turns"]}) == 2]
            report.check("conversation first turn ends, median", float(np.median(starts)), expected[key]["turn_end_median"], places=1)
        fake_p, real_p = [], []
        for row in manifest.itertuples():
            subjects = sorted(results[row.sample_id]["subjects"], key=lambda s: s["center_x"])
            for side, subject in zip(("left", "right"), subjects):
                (fake_p if row.fake_side == side else real_p).append(subject["p_fake"])
        face_auc = roc_auc_score([1] * len(fake_p) + [0] * len(real_p), fake_p + real_p)
        report.check(f"{key} face auc, every segment", face_auc, expected[key]["face_auc"])
        report.check(f"{key} fake faces", len(fake_p), expected[key]["faces"][0])
        report.check(f"{key} real faces", len(real_p), expected[key]["faces"][1])


def text_gate_vs_static(report: Report, paper: dict) -> None:
    """Tabel VII and the sentences around it: the gated model against the static one per dataset (results/analyses/gate_vs_static, written by
    analyze_gate_vs_static.py), the branch-alone runs, the muted-audio test and the review-set comparison."""
    expected = paper["gate_vs_static"]
    folder = ROOT / "results" / "analyses" / "gate_vs_static"
    pooled = pd.read_csv(folder / "per_dataset.csv").set_index("group")
    for key, row in expected["pooled"].items():
        group = pooled.loc["all videos" if key == "all" else key]
        for name, got in (("auc diff", group.auc_diff), ("auc low", group.auc_low), ("auc high", group.auc_high), ("f1 diff", group.f1_diff)):
            report.check(f"{key} pooled {name}", float(got), row[name])
        report.check(f"{key} gate right only", int(group.gate_right_only), row["only"][0])
        report.check(f"{key} static right only", int(group.static_right_only), row["only"][1])
        report.check(f"{key} mcnemar p", float(group.mcnemar_p), row["p"], places=3)
    seeds = pd.read_csv(folder / "per_seed.csv")
    audio = seeds[seeds.group.isin(["hidf", "fakeavceleb", "polyglotfake"])]
    report.check("audio dataset and seed pairs with a higher gated AUC", int((audio.auc_gate > audio.auc_static).sum()), expected["pairs"]["auc_higher"])
    report.check("audio dataset and seed pairs with a higher gated F1", int((audio.f1_gate.round(4) > audio.f1_static.round(4)).sum()), expected["pairs"]["f1_higher"])
    report.check("audio dataset and seed pairs with an equal F1", int((audio.f1_gate.round(4) == audio.f1_static.round(4)).sum()), expected["pairs"]["f1_equal"])
    names = {"gate": "moe_combined_retrain", "visual": "visual_branch_alone", "audio_visual": "audio_visual_branch_alone", "static": "concat_combined_retrain"}
    for dataset, row in expected["branches"].items():
        for model, (auc, f1) in row.items():
            frame = seed42(names[model])
            got = scores(frame[frame.dataset == dataset])
            report.check(f"{dataset} {model} auc", got["auc"], auc)
            report.check(f"{dataset} {model} f1", got["f1"], f1, alternative=got["f1_strict"])
    muted = {}
    for model, folder_name in (("gate", "expert_tests_moe_combined_retrain"), ("static", "expert_tests_concat_combined_retrain")):
        scope = yaml.safe_load((ROOT / "results" / "metrics" / folder_name / "metrics.yaml").read_text(encoding="utf-8"))["scopes"]["datasets with audio"]
        muted[model] = (scope["baseline"]["f1_macro"], scope["audio_zeroed"]["f1_macro"])
        report.check(f"{model} F1 with the audio muted, datasets with audio", muted[model][1], expected["muted"][f"{model}_after"])
        report.check(f"{model} F1 drop with the audio muted, datasets with audio", muted[model][0] - muted[model][1], expected["muted"][f"{model}_drop"], places=3)
    triage = pd.read_csv(folder / "triage.csv")
    for column, name in (("review_share", "review share"), ("share_of_gate_errors_in_alpha_set", "gate errors in the alpha set"),
                         ("share_of_static_errors_in_least_confident_static", "static errors in the least confident")):
        report.check(f"{name} lowest, percent", float(triage[column].min() * 100), expected["triage"][column][0], places=0)
        report.check(f"{name} highest, percent", float(triage[column].max() * 100), expected["triage"][column][1], places=0)


def text_other_checks(report: Report, paper: dict) -> None:
    """Counts and scores of the Hasil text outside the tables: face tracking, image control, overlap, deduplicated leave-one-out, size."""
    expected = paper["other"]
    tracks = pd.read_csv(ROOT / "results" / "analyses" / "forgerynet_test_faces" / "face_counts.csv")
    for name, got in (("videos", len(tracks)), ("no face", int((tracks.tracks == 0).sum())), ("one person", int((tracks.tracks == 1).sum())),
                      ("two or more", int((tracks.tracks >= 2).sum())), ("second person in half the detections", int((tracks.second_share >= 0.5).sum()))):
        report.check(f"forgerynet {name}", got, expected["forgerynet"][name])
    faces = pd.read_csv(INFER / "moe_combined_retrain_on_openforensics_val" / "faces.csv")
    found = faces[faces.matched == 1]
    report.check("openforensics annotated faces", len(faces), expected["openforensics"]["annotated"])
    report.check("openforensics faces found", len(found), expected["openforensics"]["found"])
    report.check("openforensics face auc", roc_auc_score(found.label, found.p_fake), expected["openforensics"]["auc"])
    still = pd.read_csv(INFER / "moe_combined_retrain_on_fakeavceleb_still_frames" / "clips.csv")
    report.check("still-frame clips", len(still), expected["still_frame"]["clips"])
    report.check("still-frame auc", roc_auc_score(still.label, still.p_fake), expected["still_frame"]["auc"])
    dedup = yaml.safe_load((ROOT / "results" / "metrics" / "leave_one_out_fakeavceleb_dedup" / "metrics.yaml").read_text(encoding="utf-8"))
    for key, row in expected["dedup"].items():
        got = dedup[key]
        report.check(f"dedup {key} auc", got["auc"], row["auc"])
        report.check(f"dedup {key} real videos", got["real"], row["real"])
        report.check(f"dedup {key} interval low", got["auc_interval_95"][0], row["interval"][0])
        report.check(f"dedup {key} interval high", got["auc_interval_95"][1], row["interval"][1])
    for key, value in (("identical_pairs", 134), ("train", 77), ("validation", 32)):
        report.check(f"dedup copies {key}", dedup["copies"][key], value)
    overlap = yaml.safe_load((ROOT / "results" / "analyses" / "split_leakage_audit" / "summary.yaml").read_text(encoding="utf-8"))["datasets"]
    for dataset, share in expected["partner_elsewhere_share"].items():
        parts = overlap[dataset]["fakes_by_partition_and_status"]
        elsewhere, same = parts["test / partner in another partition"], parts["test / same partition"]
        report.check(f"{dataset} test fakes whose real video sits in another split", elsewhere / (elsewhere + same), share, places=2)
    multi_person_sets(report, expected["multi_person"])
    prediction = pd.read_csv(ROOT / "results" / "analyses" / "gate_branch_scores" / "held_out_alpha_prediction.csv").set_index("held_out")
    for dataset, (predicted, observed) in expected["alpha_prediction"].items():
        report.check(f"{dataset} alpha predicted from the nearest seen datasets", float(prediction.loc[dataset, "predicted_alpha"]), predicted, places=3)
        report.check(f"{dataset} alpha observed in leave-one-out", float(prediction.loc[dataset, "observed_alpha"]), observed, places=3)
    largest_f1 = largest_auc = 0.0
    shares = []
    for seed in SEEDS:
        plan = yaml.safe_load((ROOT / "results" / "analyses" / "split_leakage_audit" / f"plan_b_seed{seed}.yaml").read_text(encoding="utf-8"))
        shares.append(plan["excluded_union"] / plan["test_videos"])
        for model in ("gated", "static"):
            before, after = plan["tests"]["all test videos"][model], plan["tests"]["without the excluded videos"][model]
            largest_f1, largest_auc = max(largest_f1, abs(after["macro_f1"] - before["macro_f1"])), max(largest_auc, abs(after["auc"] - before["auc"]))
    report.check("excluded share of the test set, lowest", min(shares), expected["filtered"]["share"][0], places=2)
    report.check("excluded share of the test set, highest", max(shares), expected["filtered"]["share"][1], places=2)
    report.check("largest macro-F1 change after exclusion", largest_f1, expected["filtered"]["f1"])
    report.check("largest AUC change after exclusion", largest_auc, expected["filtered"]["auc"])
    size = yaml.safe_load((ROOT / "results" / "metrics" / "complexity" / "complexity.yaml").read_text(encoding="utf-8"))
    report.check("videomae parameters (M)", size["VideoMAE-base encoder, frozen"]["parameters_M"], expected["size"]["videomae_m"], places=1)
    report.check("whisper parameters (M)", size["Whisper-medium encoder, frozen"]["parameters_M"], expected["size"]["whisper_m"], places=1)
    report.check("classifier parameters (M)", size["moe_combined"]["classifier_parameters_M"], expected["size"]["classifier_m"], places=2)
    report.check("videomae gflop per segment", size["VideoMAE-base encoder, frozen"]["gflops_per_2s_segment"], expected["size"]["videomae_gflop"], places=0)
    report.check("whisper gflop per segment", size["Whisper-medium encoder, frozen"]["gflops_per_2s_segment"], expected["size"]["whisper_gflop"], places=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["in_distribution", "leave_one_out", "two_people", "calibration", "ablation", "per_dataset", "alpha", "other", "gate_vs_static"], nargs="*")
    args = parser.parse_args()
    paper = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    report = Report()
    steps = {"in_distribution": tabel_in_distribution, "leave_one_out": tabel_leave_one_out,
             "two_people": tabel_two_people, "calibration": tabel_calibration,
             "ablation": text_ablation, "per_dataset": text_per_dataset, "alpha": text_alpha, "other": text_other_checks, "gate_vs_static": text_gate_vs_static}
    for name, step in steps.items():
        if not args.only or name in args.only:
            step(report, paper)
    print("\n".join(report.lines))
    print(f"{report.bad} mismatches")
    return 1 if report.bad else 0


if __name__ == "__main__":
    sys.exit(main())
