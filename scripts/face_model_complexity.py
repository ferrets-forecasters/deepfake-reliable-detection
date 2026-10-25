from __future__ import annotations

import argparse
import zipfile
from math import prod
from pathlib import Path

import tflite
import yaml
from tqdm import tqdm

from deepfake_detection.config import load_config

BUILTIN = {getattr(tflite.BuiltinOperator, n): n for n in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "TRANSPOSE_CONV")}
DETECTIONS_PER_SEGMENT = 16  # 8 detections per second in a 2 second segment


def count(buffer: bytes) -> dict:
    """Parameters and multiply-adds of the convolution and dense layers of one TFLite model, for the input size the model file fixes."""
    model = tflite.Model.GetRootAsModel(buffer, 0)
    graph = model.Subgraphs(0)
    params = macs = 0
    for i in range(graph.OperatorsLength()):
        op = graph.Operators(i)
        code = model.OperatorCodes(op.OpcodeIndex())
        kind = BUILTIN.get(max(code.BuiltinCode(), code.DeprecatedBuiltinCode()))
        if kind is None:
            continue
        shape = lambda index: [int(x) for x in graph.Tensors(index).ShapeAsNumpy()]
        inputs = [op.Inputs(k) for k in range(op.InputsLength())]
        if kind == "TRANSPOSE_CONV":  # inputs: output shape, filter, input, bias
            weight, source, bias = shape(inputs[1]), shape(inputs[2]), (shape(inputs[3]) if len(inputs) > 3 and inputs[3] >= 0 else [])
            macs += source[1] * source[2] * source[3] * weight[1] * weight[2] * weight[0]
        else:
            source_index, weight_index = inputs[0], inputs[1]
            weight = shape(weight_index)
            bias = shape(inputs[2]) if len(inputs) > 2 and inputs[2] >= 0 else []
            out = shape(op.Outputs(0))
            if kind == "CONV_2D":
                macs += out[1] * out[2] * weight[0] * weight[1] * weight[2] * weight[3]
            elif kind == "DEPTHWISE_CONV_2D":
                macs += out[1] * out[2] * out[3] * weight[1] * weight[2]
            else:
                macs += prod(weight)
        params += prod(weight) + (prod(bias) if bias else 0)
    return {"parameters": params, "gflop_per_pass": round(2 * macs / 1e9, 5), "input": [int(x) for x in graph.Tensors(graph.Inputs(0)).ShapeAsNumpy()]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameters and GFLOP of the MediaPipe face models the pipeline runs: the face detector once and the landmark model once per face, on each of the 16 detection frames of a 2 second segment.")
    parser.add_argument("--task", default=str(Path(__file__).resolve().parents[3] / "external" / "mediapipe" / "face_landmarker.task"))
    parser.add_argument("--faces", type=int, default=1)
    args = parser.parse_args()
    config = load_config()
    report = {}
    with zipfile.ZipFile(args.task) as bundle:
        for name in tqdm(["face_detector", "face_landmarks_detector"], desc="models"):
            report[name] = count(bundle.read(f"{name}.tflite"))
    per_detection = report["face_detector"]["gflop_per_pass"] + args.faces * report["face_landmarks_detector"]["gflop_per_pass"]
    report["pipeline"] = {"parameters": report["face_detector"]["parameters"] + report["face_landmarks_detector"]["parameters"], "faces": args.faces,
                          "gflop_per_detection": round(per_detection, 5), "gflop_per_segment": round(DETECTIONS_PER_SEGMENT * per_detection, 4)}
    out_dir = config.paths.metrics_dir("complexity") if hasattr(config.paths, "metrics_dir") else config.paths.results_dir("complexity")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "face_model.yaml").write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
