from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config

import concurrent.futures
import json
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch
import tqdm
from insightface.app import FaceAnalysis
from prepare.utils import get_visual_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resize_face(face, size, maintain_aspect_ratio=True):
    if not isinstance(face, np.ndarray) or face.size == 0:
        return face
    target_h, target_w = size
    if not maintain_aspect_ratio:
        return cv2.resize(face, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    h, w = face.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(face, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, *face.shape[2:]), dtype=face.dtype)
    top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def create_face_app(det_score_thresh=0.55, fallback_score_thresh=0.4):
    if not 0 < fallback_score_thresh <= det_score_thresh <= 1:
        raise ValueError("阈值必须满足 0 < 补检阈值 <= 主检测阈值 <= 1")
    cv2.setNumThreads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    app = FaceAnalysis(allowed_modules=["detection"], providers=providers, sess_options=options)
    app.prepare(ctx_id=0 if providers[0] == "CUDAExecutionProvider" else -1, det_size=(640, 640), det_thresh=fallback_score_thresh)
    actual_providers = app.det_model.session.get_providers()
    print(f"人脸检测推理后端: {actual_providers}")
    if "CUDAExecutionProvider" not in actual_providers:
        print("当前使用 CPU；启用 GPU 需要支持 CUDA 的 ONNX Runtime。")
    return app


def box_iou(first, second):
    intersection = np.maximum(0, np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2])).prod()
    union = (first[2:] - first[:2]).prod() + (second[2:] - second[:2]).prod() - intersection
    return float(intersection / union) if union > 0 else 0.0


def is_strong_face(candidate, det_score_thresh):
    threshold = max(det_score_thresh, 0.75) if candidate["method"].startswith("rotated") else det_score_thresh
    return candidate["score"] >= threshold


def detect_faces(app, frame, target_size, min_face_size=24, method="direct"):
    turns = {"direct": 0, "large": 0, "rotated180": 2, "rotated90": 1, "rotated270": 3}[method]
    image = np.ascontiguousarray(np.rot90(frame, turns)) if turns else frame
    size = (1024, 1024) if method == "large" else (640, 640)
    boxes, landmarks = app.det_model.detect(image, input_size=size)
    h, w = image.shape[:2]
    candidates = []
    for index, detection in enumerate(boxes):
        if not np.isfinite(detection).all():
            continue
        raw_box = detection[:4]
        box = np.clip(raw_box, [0, 0, 0, 0], [w, h, w, h])
        width, height = box[2:] - box[:2]
        raw_area = np.maximum(0, raw_box[2:] - raw_box[:2]).prod()
        if min(width, height) < min_face_size or width * height < 0.8 * raw_area:
            continue
        if landmarks is not None:
            points = landmarks[index]
            if not np.isfinite(points).all() or np.any(points < 0) or np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h):
                continue
            if np.linalg.norm(points[1] - points[0]) < 0.1 * width or np.linalg.norm(points[4] - points[3]) < 0.06 * width:
                continue
        padding = np.array([width, height, width, height]) * np.array([-0.08, -0.08, 0.08, 0.08])
        x1, y1, x2, y2 = np.rint(np.clip(box + padding, [0, 0, 0, 0], [w, h, w, h])).astype(int)
        crop = cv2.cvtColor(resize_face(image[y1:y2, x1:x2], target_size), cv2.COLOR_BGR2RGB)
        original_h, original_w = frame.shape[:2]
        x1, y1, x2, y2 = box
        if turns == 1:
            box = np.array([original_w - y2, x1, original_w - y1, x2])
        elif turns == 2:
            box = np.array([original_w - x2, original_h - y2, original_w - x1, original_h - y1])
        elif turns == 3:
            box = np.array([y1, original_h - x2, y2, original_h - x1])
        candidates.append({"crop": crop, "box": box, "score": float(detection[4]), "method": method})
    return candidates


def read_sample_frames(visual_path, num_samples):
    cap = cv2.VideoCapture(str(visual_path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    try:
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {visual_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if total_frames <= 0:
            raise ValueError(f"视频没有有效帧: {visual_path}")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"视频帧率无效: {visual_path}, FPS={fps}")
        starts = np.rint(np.linspace(0, total_frames - 1, num_samples)).astype(int)
        groups = []
        for i, start in enumerate(starts):
            end = max(start, starts[i + 1] - 1) if i + 1 < len(starts) else start
            end = min(end, start + max(1, round(0.2 * fps)))
            groups.append(sorted({int(start), int((start + end) // 2), int(end)}))
        indices = {index for group in groups for index in group}
        indices.update(max(0, min(total_frames - 1, index + offset)) for index in tuple(indices) for offset in (-1, 1))
        records = {}
        next_frame, scene, previous = 0, 0, None
        for index in sorted(indices):
            while next_frame < index:
                if not cap.grab():
                    raise RuntimeError(f"视频无法解码: {visual_path}, 帧 {next_frame}")
                next_frame += 1
            ok, frame = cap.read()
            next_frame += 1
            if not ok or frame is None:
                raise RuntimeError(f"视频采样区间无法解码: {visual_path}, 帧 {index}")
            thumbnail = cv2.resize(frame, (48, 32)).astype(np.float32) / 255
            if previous is not None and np.abs(thumbnail - previous).mean() > 0.22:
                scene += 1
            records[index] = {"frame": frame, "scene": scene, "time": index / fps}
            previous = thumbnail
        return records, groups, fps
    finally:
        cap.release()


def supported_candidates(records, detections, det_score_thresh, temporal_window):
    strong = {index: [candidate for candidate in candidates if is_strong_face(candidate, det_score_thresh)] for index, candidates in detections.items()}
    anchors = {}
    for index, candidates in strong.items():
        neighbors = [
            other
            for other in strong
            if other != index
            and records[other]["scene"] == records[index]["scene"]
            and abs(records[other]["time"] - records[index]["time"]) <= temporal_window
        ]
        anchors[index] = [
            candidate
            for candidate in candidates
            if (len(records) == 1 and candidate["score"] >= max(0.85, det_score_thresh))
            or any(box_iou(candidate["box"], other["box"]) >= 0.3 for neighbor in neighbors for other in strong[neighbor])
        ]
    return anchors


def choose_faces(records, groups, detections, det_score_thresh, fallback_score_thresh, temporal_window, retry_detection):
    anchors = supported_candidates(records, detections, det_score_thresh, temporal_window)
    chosen, previous = [], None
    for group in groups:
        start = group[0]
        candidates = []
        for index in group:
            if records[index]["scene"] != records[start]["scene"]:
                break
            for candidate in detections.get(index, []):
                reliable = any(candidate is anchor for anchor in anchors.get(index, []))
                if not reliable and retry_detection and fallback_score_thresh <= candidate["score"] and not is_strong_face(candidate, det_score_thresh):
                    reliable = any(
                        records[other]["scene"] == records[index]["scene"]
                        and abs(records[other]["time"] - records[index]["time"]) <= temporal_window
                        and box_iou(candidate["box"], anchor["box"]) >= 0.3
                        for other, boxes in anchors.items()
                        for anchor in boxes
                    )
                if not reliable:
                    continue
                continuous = (
                    previous is not None
                    and records[previous[0]]["scene"] == records[index]["scene"]
                    and abs(records[index]["time"] - records[previous[0]]["time"]) <= temporal_window
                    and box_iou(candidate["box"], previous[1]["box"]) >= 0.3
                )
                area = float(np.prod(candidate["box"][2:] - candidate["box"][:2]))
                candidates.append(((continuous, -abs(index - start), area * candidate["score"]), index, candidate))
        best = max(candidates, key=lambda item: item[0]) if candidates else None
        selected = (best[1], best[2]) if best is not None else None
        chosen.append(selected)
        if selected is not None:
            previous = selected
    return chosen


def extract_face(
    app,
    visual_path,
    target_size,
    num_samples=32,
    det_score_thresh=0.55,
    fallback_score_thresh=0.4,
    retry_detection=True,
    return_info=False,
    min_face_size=24,
    max_fill_gap=0.25,
    temporal_window=0.5,
):
    if num_samples <= 0 or len(target_size) != 2 or min(target_size) <= 0:
        raise ValueError("采样数和目标图像尺寸必须为正数")
    if not 0 < fallback_score_thresh <= det_score_thresh <= 1:
        raise ValueError("阈值必须满足 0 < 补检阈值 <= 主检测阈值 <= 1")
    if (
        not all(np.isfinite(value) for value in (min_face_size, max_fill_gap, temporal_window))
        or min_face_size <= 0
        or max_fill_gap < 0
        or temporal_window <= 0
    ):
        raise ValueError("最小脸尺寸、核验时间必须为正数，填充时间必须非负且均为有限值")

    records, groups, fps = read_sample_frames(visual_path, num_samples)
    detections, attempted = {}, set()
    chosen = [None] * num_samples
    methods_to_try = ["direct", "large", "rotated180", "rotated90", "rotated270"] if retry_detection else ["direct"]
    for method in methods_to_try:
        for i, group in enumerate(groups):
            if chosen[i] is not None:
                continue
            for index in group:
                if records[index]["scene"] != records[group[0]]["scene"]:
                    break
                if (index, method) not in attempted:
                    detections.setdefault(index, []).extend(detect_faces(app, records[index]["frame"], target_size, min_face_size, method))
                    attempted.add((index, method))
                if any(is_strong_face(candidate, det_score_thresh) and candidate["method"] == method for candidate in detections[index]):
                    break
        anchors = supported_candidates(records, detections, det_score_thresh, temporal_window)
        unsupported = [
            index
            for index, candidates in detections.items()
            if any(is_strong_face(candidate, det_score_thresh) for candidate in candidates) and not anchors.get(index)
        ]
        for index in unsupported:
            neighbors = sorted(
                (
                    other
                    for other in records
                    if other != index
                    and records[other]["scene"] == records[index]["scene"]
                    and abs(records[other]["time"] - records[index]["time"]) <= temporal_window
                ),
                key=lambda other: abs(other - index),
            )[:2]
            for other in neighbors:
                if (other, method) not in attempted:
                    detections.setdefault(other, []).extend(detect_faces(app, records[other]["frame"], target_size, min_face_size, method))
                    attempted.add((other, method))
        chosen = choose_faces(records, groups, detections, det_score_thresh, fallback_score_thresh, temporal_window, retry_detection)

    valid_indices = [i for i, candidate in enumerate(chosen) if candidate is not None]
    missing_indices = [i for i, candidate in enumerate(chosen) if candidate is None]
    frame_indices = [candidate[0] if candidate else None for candidate in chosen]
    methods = [(candidate[1]["method"] if is_strong_face(candidate[1], det_score_thresh) else "temporal") if candidate else "missing" for candidate in chosen]
    faces = [candidate[1]["crop"] if candidate else np.zeros((*target_size, 3), dtype=np.uint8) for candidate in chosen]
    filled_from, fill_distances = {}, {}
    for i in missing_indices:
        filled_from[str(i)] = None
        left = next((j for j in reversed(valid_indices) if j < i), None)
        right = next((j for j in valid_indices if j > i), None)
        if left is None or right is None or max_fill_gap == 0:
            continue
        before, after = records[chosen[left][0]], records[chosen[right][0]]
        current = records[groups[i][0]]
        if (
            before["scene"] == current["scene"] == after["scene"]
            and before["time"] <= current["time"] <= after["time"]
            and after["time"] - before["time"] <= max_fill_gap
            and box_iou(chosen[left][1]["box"], chosen[right][1]["box"]) >= 0.3
        ):
            nearest = min((left, right), key=lambda j: abs(chosen[j][0] - groups[i][0]))
            faces[i], methods[i] = faces[nearest], "filled"
            filled_from[str(i)] = nearest
            fill_distances[str(i)] = abs(records[chosen[nearest][0]]["time"] - current["time"])
    recovered_indices = [i for i in valid_indices if methods[i] != "direct"]
    rejected_indices = [i for i in missing_indices if any(detections.get(index) for index in groups[i])]
    if missing_indices:
        print(f"人脸缺失: {visual_path}, 采样位置 {missing_indices}，短缺口填充 {len(fill_distances)} 个")
    info = {
        "missing_indices": missing_indices,
        "recovered_indices": recovered_indices,
        "rejected_indices": rejected_indices,
        "filled_indices": [int(i) for i in fill_distances],
        "filled_from": filled_from,
        "fill_distance_seconds": fill_distances,
        "valid_mask": [candidate is not None for candidate in chosen],
        "frame_indices": frame_indices,
        "methods": methods,
        "detection_methods": [candidate[1]["method"] if candidate else None for candidate in chosen],
        "source_frame_indices": [frame_indices[filled_from[str(i)]] if filled_from.get(str(i)) is not None else frame_indices[i] for i in range(num_samples)],
        "sample_timestamps": [group[0] / fps for group in groups],
        "detection_timestamps": [index / fps if index is not None else None for index in frame_indices],
        "scores": [candidate[1]["score"] if candidate else None for candidate in chosen],
        "bboxes": [candidate[1]["box"].tolist() if candidate else None for candidate in chosen],
        "scene_ids": [records[group[0]]["scene"] for group in groups],
        "fps": fps,
    }
    tensor = torch.from_numpy(np.stack(faces))
    return (tensor, info) if return_info else tensor


def cache_face_multithread(
    dataset,
    num_samples,
    det_score_thresh,
    target_size,
    num_workers,
    fallback_score_thresh=0.4,
    retry_detection=True,
    min_face_size=24,
    max_fill_gap=0.25,
    temporal_window=0.5,
):
    visual_root = PROJECT_ROOT.parent / "datasets" / dataset / "visual"
    cache_root = PROJECT_ROOT / "data_cache" / dataset
    visual_paths = get_visual_paths(visual_root)
    if not visual_paths:
        raise ValueError(f"没有找到视频: {visual_root}")
    app = create_face_app(det_score_thresh, fallback_score_thresh)

    def process_single_visual(visual_path):
        relative_path = Path(visual_path).relative_to(visual_root)
        face_path = cache_root / "visual_face" / relative_path.with_suffix(".pt")
        metadata_path = cache_root / "visual_face_metadata" / relative_path.with_suffix(".json")
        tensor, info = extract_face(
            app,
            visual_path,
            target_size,
            num_samples,
            det_score_thresh,
            fallback_score_thresh,
            retry_detection,
            return_info=True,
            min_face_size=min_face_size,
            max_fill_gap=max_fill_gap,
            temporal_window=temporal_window,
        )
        face_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, face_path)
        metadata_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        return info

    print(f"开始多线程提取人脸数据\n数据集: {dataset}\n每视频采样数: {num_samples}")
    print(f"检测阈值: {det_score_thresh}\n补检阈值: {fallback_score_thresh}\n增强补检: {retry_detection}")
    print(f"原图最小脸边长: {min_face_size} 像素\n最长填充缺口: {max_fill_gap} 秒\n跨帧核验窗口: {temporal_window} 秒")
    print(f"目标尺寸: {target_size}\n线程数: {num_workers}\n总视频数: {len(visual_paths)}")
    report = {
        "algorithm_version": 2,
        "dataset": dataset,
        "num_samples": num_samples,
        "det_score_thresh": det_score_thresh,
        "fallback_score_thresh": fallback_score_thresh,
        "retry_detection": retry_detection,
        "min_face_size": min_face_size,
        "rotation_score_thresh": max(det_score_thresh, 0.75),
        "max_fill_gap": max_fill_gap,
        "temporal_window": temporal_window,
        "target_size": target_size,
        "workers": num_workers,
        "providers": app.det_model.session.get_providers(),
        "success": 0,
        "missing_positions": 0,
        "recovered_positions": 0,
        "filled_positions": 0,
        "rejected_positions": 0,
        "clips": {},
        "errors": {},
    }
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_visual, path): path for path in visual_paths}
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理视频进度"):
            path = str(Path(futures[future]).relative_to(visual_root))
            try:
                info = future.result()
            except Exception as exc:
                report["errors"][path] = str(exc)
                tqdm.tqdm.write(f"处理失败: {path}: {exc}")
                continue
            report["success"] += 1
            for name in ("missing", "recovered", "filled", "rejected"):
                report[f"{name}_positions"] += len(info[f"{name}_indices"])
            if info["missing_indices"] or info["recovered_indices"]:
                report["clips"][path] = info
    report["seconds"] = round(time.perf_counter() - start, 3)
    cache_root.mkdir(parents=True, exist_ok=True)
    report_path = cache_root / "visual_face_report"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"完成 {report['success']}/{len(visual_paths)}；补检恢复 {report['recovered_positions']} 个位置；"
        f"未检出 {report['missing_positions']} 个位置（其中短缺口填充 {report['filled_positions']} 个）；"
        f"耗时 {report['seconds']:.1f} 秒"
    )
    print(f"检测记录: {report_path}")
    if report["errors"]:
        raise RuntimeError(f"{len(report['errors'])} 个视频处理失败，详见 {report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='从视频提取人脸，生成缓存及检测质量报告。')
    parser.add_argument("--dataset", choices=("MOSI", "MOSEI", "SIMS"), default="SIMS")
    config = Config(parser.parse_args().dataset)
    cache_face_multithread(config.dataset, num_samples=32, det_score_thresh=0.55, target_size=(224, 224), num_workers=config.num_workers)
