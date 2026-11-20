from pathlib import Path

import subprocess
import os
import concurrent.futures
import tqdm

def split_video2av(video_path, audio_output_path, visual_output_path, sample_rate=16000):
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        print(f"错误：在 {video_path_obj} 未找到视频文件")
        return

    audio_command = [
        "ffmpeg",
        "-i",
        str(video_path_obj),
        "-vn",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        "-y",
        str(audio_output_path),
    ]

    visual_command = [
        "ffmpeg",
        "-i",
        str(video_path_obj),
        "-an",
        "-y",
        str(visual_output_path),
    ]

    try:
        subprocess.run(audio_command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"提取音频时出错: {e}")
        print(f"ffmpeg stdout: {e.stdout}")
        print(f"ffmpeg stderr: {e.stderr}")
        return

    try:
        subprocess.run(visual_command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"创建无声视频时出错: {e}")
        print(f"ffmpeg stdout: {e.stdout}")
        print(f"ffmpeg stderr: {e.stderr}")
        return

    return str(audio_output_path), str(visual_output_path)


def split_dataset_multithread(directory_path, num_workers, output_root):
    directory_path = Path(directory_path)
    output_root = Path(output_root)

    def process_one(video_path):
        relative_path = Path(video_path).relative_to(directory_path)
        audio_output_path = output_root / "audio" / relative_path.with_suffix(".wav")
        visual_output_path = output_root / "visual" / relative_path
        os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(visual_output_path), exist_ok=True)
        return split_video2av(video_path, audio_output_path, visual_output_path, sample_rate=16000)

    video_files = []
    for folder in os.listdir(directory_path):
        if folder == ".DS_Store":
            continue
        folder_path = os.path.join(directory_path, folder)
        if not os.path.isdir(folder_path):
            continue

        for file_name in os.listdir(folder_path):
            if file_name == ".DS_Store" or not file_name.endswith(".mp4"):
                continue
            fpath = os.path.join(folder_path, file_name)
            video_files.append(fpath)

    if not video_files:
        raise ValueError(f"未找到 MP4 片段: {directory_path}")

    failed_paths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one, video_path): video_path for video_path in video_files}
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="分离进度"):
            video_path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"分离失败: {video_path}: {exc}")
                result = None
            if result is None:
                failed_paths.append(video_path)

    print(f"分离完成: 总计 {len(video_files)}, 成功 {len(video_files) - len(failed_paths)}, 失败 {len(failed_paths)}")
    if failed_paths:
        raise RuntimeError("以下视频分离失败:\n" + "\n".join(failed_paths))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='为各实验共享提取 16 kHz 音频到 datasets/<dataset>/audio/，无声视频到 visual/。')
    parser.add_argument("--dataset", choices=("MOSI", "MOSEI", "SIMS"), default="SIMS")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2] / "datasets" / args.dataset
    split_dataset_multithread(root / "Raw", 8, root)
