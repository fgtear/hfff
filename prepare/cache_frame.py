from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config
from prepare.cache_face_timing import read_video_timing

import torch
import json
import math
import os
import pandas as pd
import tqdm
from torchvision import transforms
from torchcodec.decoders import VideoDecoder
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple


def extract_uniform_frames(visual_path: str, num_frames: int, output_size: Tuple[int, int] = None, return_timing=False):
    visual_path = Path(visual_path)
    if not visual_path.is_file():
        raise FileNotFoundError(f"视频文件未找到: {visual_path}")
    if num_frames <= 0:
        raise ValueError("num_frames 必须为正整数")

    try:
        decoder = VideoDecoder(str(visual_path), dimension_order="NCHW", num_ffmpeg_threads=1)
        total_frames = len(decoder)
        if total_frames == 0:
            raise ValueError(f"视频 '{visual_path}' 为空或无法读取帧。")

        indices = torch.linspace(0, total_frames - 1, num_frames).long()
        sampled = decoder.get_frames_at(indices=indices)
        sampled_frames = sampled.data
    except RuntimeError:
        print(f"读取或解码视频失败: {visual_path}")
        raise

    if output_size is not None:
        resize_transform = transforms.Resize(output_size, antialias=True)
        sampled_frames = resize_transform(sampled_frames)

    if not return_timing:
        return sampled_frames

    _, visual_duration = read_video_timing(visual_path)
    frame_timestamps = sampled.pts_seconds.tolist()
    if len(frame_timestamps) != num_frames or any(
        not math.isfinite(value) or value < 0 or value >= visual_duration for value in frame_timestamps
    ):
        raise ValueError(f"采样帧时间无效或超出视频范围: {visual_path}")
    timing = {
        "frame_timestamps": frame_timestamps,
        "frame_observed": [True] * num_frames,
        "visual_duration": visual_duration,
        "source_frame_indices": indices.tolist(),
        "source": "decoded_video_pts",
    }
    return sampled_frames, timing


def process_dataset(dataset, num_frames, output_size, num_workers):
    def process_single_visual(row):
        visual_id = row["video_id"]
        clip_id = row["clip_id"]
        timing_path = Path(f"data_cache/{dataset}/visual_frame_timing/{visual_id}/{clip_id}.json")
        try:
            timing_path.unlink(missing_ok=True)

            visual_path = f"../datasets/{dataset}/visual/{visual_id}/{clip_id}.mp4"

            if not os.path.exists(visual_path):
                print(f"视频文件不存在: {visual_path}")
                return False

            visual_tensor, timing = extract_uniform_frames(
                visual_path, num_frames, output_size=output_size, return_timing=True
            )
            tensor_path = f"data_cache/{dataset}/visual_frame/{visual_id}/{clip_id}.pt"
            os.makedirs(os.path.dirname(tensor_path), exist_ok=True)
            torch.save(visual_tensor, tensor_path)
            timing_path.parent.mkdir(parents=True, exist_ok=True)
            timing_path.write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
            return True
        except Exception as e:
            timing_path.unlink(missing_ok=True)
            print(f"处理视频时出错 {row.get('video_id', 'unknown')}/{row.get('clip_id', 'unknown')}: {str(e)}")
            return False

    if dataset == "SIMS":
        df = pd.read_csv(
            f"../datasets/{dataset}/label.csv",
            names=["video_id", "clip_id", "text", "label", "label_T", "label_A", "label_V", "annotation", "mode"],
            dtype={"video_id": str, "clip_id": str},
        )
    elif dataset in ["MOSI", "MOSEI", "SIMSV2"]:
        df = pd.read_csv(f"../datasets/{dataset}/label.csv", dtype={"video_id": str, "clip_id": str})
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    success_count = 0
    failure_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_visual, row) for index, row in df.iterrows()]

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing items"):
            result = future.result()
            if result:
                success_count += 1
            else:
                failure_count += 1

    print("\n处理完成:")
    print(f"成功: {success_count}")
    print(f"失败: {failure_count}")
    print(f"总计: {len(df)}")
    if failure_count:
        raise RuntimeError(f"{dataset} 有 {failure_count} 个整帧缓存失败，修复后重新运行")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='为本实验缓存均匀采样整帧及其实际解码时间。')
    parser.add_argument("--dataset", choices=("MOSI", "MOSEI", "SIMS"), default="SIMS")
    config = Config(parser.parse_args().dataset)
    os.chdir(Path(__file__).resolve().parents[1])
    process_dataset(config.dataset, num_frames=32, output_size=(224, 224), num_workers=config.num_workers)
