import argparse
import concurrent.futures
import json
import math
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_video_timing(visual_path):
    command = [
        "ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time,duration_time,pkt_duration_time:stream=start_time,duration:format=start_time,duration",
        "-of", "json", str(visual_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    frames = data.get("frames", [])
    if not frames:
        raise ValueError(f"视频没有可用的解码帧: {visual_path}")
    timestamps = [float(frame["best_effort_timestamp_time"]) for frame in frames]
    if any(not math.isfinite(value) for value in timestamps):
        raise ValueError(f"视频包含非有限 PTS: {visual_path}")
    if any(after < before for before, after in zip(timestamps, timestamps[1:])):
        raise ValueError(f"视频解码 PTS 不是按时间排列: {visual_path}")

    ends = []
    for key in ("duration_time", "pkt_duration_time"):
        value = frames[-1].get(key)
        if value not in (None, "N/A"):
            duration = float(value)
            if math.isfinite(duration) and duration > 0:
                ends.append(timestamps[-1] + duration)
    for info in data.get("streams", []) + [data.get("format", {})]:
        duration = info.get("duration")
        start = info.get("start_time", timestamps[0])
        if duration not in (None, "N/A") and start not in (None, "N/A"):
            duration, start = float(duration), float(start)
            if math.isfinite(duration) and duration > 0 and math.isfinite(start):
                ends.append(start + duration)
    if not ends or not math.isfinite(max(ends)) or max(ends) <= 0 or max(ends) <= timestamps[-1]:
        raise ValueError(f"无法确定视频的有效结束时间: {visual_path}")
    return timestamps, max(ends)


def build_visual_face_timing(metadata, frame_timestamps, visual_duration):
    sources = metadata["source_frame_indices"]
    samples = metadata["sample_timestamps"]
    fps = float(metadata["fps"])
    timestamps = [float(value) for value in frame_timestamps]
    visual_duration = float(visual_duration)
    if not sources or len(sources) != len(samples):
        raise ValueError("人脸来源索引与目标采样时间的数量必须相同且非空")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("人脸元数据 FPS 必须为有限正数")
    if not timestamps or any(not math.isfinite(value) for value in timestamps):
        raise ValueError("视频帧 PTS 必须为非空的有限数值序列")
    if any(after < before for before, after in zip(timestamps, timestamps[1:])):
        raise ValueError("视频帧 PTS 必须按时间排列")
    if not math.isfinite(visual_duration) or visual_duration <= 0 or visual_duration <= max(timestamps):
        raise ValueError("视频结束时间必须为有限正数且晚于最后一帧 PTS")

    frame_times, observed = [], []
    for source, sample in zip(sources, samples):
        sample = float(sample)
        if not math.isfinite(sample):
            raise ValueError("目标采样时间必须为有限数值")
        sample_index = round(sample * fps)
        if not 0 <= sample_index < len(timestamps):
            raise ValueError(f"目标采样帧索引越界: {sample_index}, 视频帧数 {len(timestamps)}")
        if source is not None and (type(source) is not int or not 0 <= source < len(timestamps)):
            raise ValueError(f"人脸来源帧索引无效: {source}, 视频帧数 {len(timestamps)}")
        frame_times.append(timestamps[sample_index if source is None else source])
        observed.append(source is not None)
    return {
        "frame_timestamps": frame_times,
        "frame_observed": observed,
        "visual_duration": visual_duration,
        "source_frame_indices": sources,
        "source": "decoded_video_pts",
    }


def cache_face_timing(dataset, workers=4):
    if workers < 1:
        raise ValueError("workers 必须为正整数")
    cache_root = PROJECT_ROOT / "data_cache" / dataset
    face_root = cache_root / "visual_face"
    metadata_root = cache_root / "visual_face_metadata"
    face_ids = {path.relative_to(face_root).with_suffix("") for path in face_root.glob("*/*.pt")}
    metadata_ids = {path.relative_to(metadata_root).with_suffix("") for path in metadata_root.glob("*/*.json")}
    if not face_ids or face_ids != metadata_ids:
        missing = sorted(str(path) for path in face_ids - metadata_ids)
        extra = sorted(str(path) for path in metadata_ids - face_ids)
        raise ValueError(
            f"{dataset} 人脸缓存与来源元数据不完整: 人脸 {len(face_ids)}, 元数据 {len(metadata_ids)}, "
            f"缺元数据 {len(missing)} {missing[:3]}, 缺人脸 {len(extra)} {extra[:3]}。\n"
            f"请先运行 prepare/cache_face.py 中的 cache_face_multithread 为 {dataset} "
            "生成带来源记录的人脸缓存，再重跑本命令。\n"
            "再运行（在仓库根目录）: uv run python -c "
            f"'from prepare.cache_face_timing import cache_face_timing; cache_face_timing(\"{dataset}\")'"
        )

    def cache_one(clip_id):
        metadata = json.loads((metadata_root / clip_id.with_suffix(".json")).read_text(encoding="utf-8"))
        visual_path = PROJECT_ROOT.parent / "datasets" / dataset / "visual" / clip_id.with_suffix(".mp4")
        timestamps, duration = read_video_timing(visual_path)
        timing = build_visual_face_timing(metadata, timestamps, duration)
        output_path = cache_root / "visual_face_timing" / clip_id.with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")

    success, failed = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(cache_one, clip_id): clip_id for clip_id in sorted(face_ids)}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failed += 1
                print(f"时间缓存失败 {dataset}/{futures[future]}: {exc}", flush=True)
            else:
                success += 1
            if (success + failed) % 100 == 0:
                print(f"{dataset}: 已处理 {success + failed}/{len(face_ids)}", flush=True)
    print(f"{dataset} 时间缓存完成: 成功 {success}, 失败 {failed}", flush=True)
    if failed:
        raise RuntimeError(f"{dataset} 有 {failed} 个时间缓存失败，修复后重新运行")
    return {"success": success, "failed": failed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='按原视频解码 PTS 为已有脸部缓存生成独立时间坐标，不改动人脸张量。')
    parser.add_argument("--dataset", choices=("MOSI", "MOSEI", "SIMS"), default="SIMS")
    cache_face_timing(parser.parse_args().dataset, workers=4)
