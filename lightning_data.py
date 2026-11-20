import json
import wave
from pathlib import Path

import lightning as L
import torch
import torchaudio
import math
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
import pandas as pd
import typing
import sys
import os
from transformers import AutoModel, AutoTokenizer, BertTokenizer
from transformers import Wav2Vec2FeatureExtractor, VivitImageProcessor, Wav2Vec2Processor

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class AudioLengthBatchSampler(BatchSampler):

    def __init__(self, sampler, audio_lengths, batch_size, bucket_size=100, drop_last=False):
        super().__init__(sampler, batch_size, drop_last)
        if bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        self.audio_lengths = audio_lengths
        self.bucket_size = math.ceil(bucket_size / batch_size) * batch_size

    def __iter__(self):
        indices = list(self.sampler)
        batches = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = sorted(indices[start : start + self.bucket_size], key=self.audio_lengths.__getitem__)
            batches.extend(bucket[offset : offset + self.batch_size] for offset in range(0, len(bucket), self.batch_size))
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        for index in torch.randperm(len(batches)).tolist():
            yield batches[index]


class DatasetMSA(torch.utils.data.Dataset):
    def __init__(self, dataset, mode: typing.Literal["train", "test", "valid"], config):
        self.config = config
        self.visual_input = getattr(config, "visual_input", "face")
        if self.visual_input not in ("face", "frame"):
            raise ValueError(f"Unsupported visual_input: {self.visual_input}")
        self.data_root = f"../datasets/{dataset}"
        self.cache_root = f"data_cache/{dataset}"
        if dataset in ["SIMS"]:
            df = pd.read_csv(
                f"{self.data_root}/label.csv",
                names=["video_id", "clip_id", "text", "label", "label_T", "label_A", "label_V", "annotation", "mode"],
                dtype={"video_id": str, "clip_id": str},
            )
        elif dataset in ["MOSI", "MOSEI"]:
            df = pd.read_csv(f"{self.data_root}/label.csv", dtype={"video_id": str, "clip_id": str})
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        self.df = df[df["mode"] == mode].sort_values(by=["video_id", "clip_id"]).reset_index()

        if config.dataset in ["MOSI", "MOSEI"]:
            self.tokenizer = AutoTokenizer.from_pretrained(config.text_extractor)
        elif config.dataset in ["SIMS"]:
            self.tokenizer = BertTokenizer.from_pretrained(config.text_extractor, clean_up_tokenization_spaces=True)
        self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(config.audio_extractor)
        self.image_processor = VivitImageProcessor.from_pretrained(config.visual_extractor)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        file_name = f"{row['video_id']}/{row['clip_id']}"
        dataset = self.config.dataset

        if dataset in ["SIMS"]:
            t_score = row["label_T"]
            a_score = row["label_A"]
            v_score = row["label_V"]
        else:
            t_score = row["label"]
            a_score = row["label"]
            v_score = row["label"]


        if self.config.dataset == "MOSI":
            text = row["text"].lower()
        else:
            text = row["text"]
        with torch.no_grad():
            text_token = self.tokenizer(
                text,
                return_attention_mask=True,
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True,
                return_tensors="pt",
            )
        text_input_ids = text_token["input_ids"].squeeze(0)
        text_attention_mask = text_token["attention_mask"].squeeze(0)

        audio_path = f"{self.data_root}/audio/{file_name}.wav"
        audio, sample_rate = torchaudio.load(audio_path)
        audio = torch.mean(audio, dim=0, keepdim=False)
        if sample_rate != self.audio_processor.sampling_rate:
            raise ValueError(f"采样率不匹配: {audio_path}: {sample_rate}")
        audio_duration = audio.numel() / sample_rate
        with torch.no_grad():
            audio_token = self.audio_processor(
                audio.cpu().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt",
                truncation=False,
                return_attention_mask=True,
            )
        audio_input_values = audio_token.input_values.squeeze(0)
        audio_attention_mask = audio_token.attention_mask.squeeze(0).to(torch.bool)

        timing_dir = "visual_face_timing" if self.visual_input == "face" else "visual_frame_timing"
        timing_path = Path(self.cache_root) / timing_dir / f"{file_name}.json"
        if not timing_path.is_file():
            prepare_hint = (
                "调用 prepare.cache_face_timing.cache_face_timing 生成对应数据集的人脸时间缓存"
                if self.visual_input == "face"
                else "运行 prepare/cache_frame.py 为对应数据集同时生成整帧和时间缓存"
            )
            raise FileNotFoundError(
                f"缺少实际帧时间: {timing_path}；请先{prepare_hint}（数据集 {dataset}）"
            )
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        if self.visual_input == "face":
            metadata_path = Path(self.cache_root) / "visual_face_metadata" / f"{file_name}.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if timing["source_frame_indices"] != metadata["source_frame_indices"]:
                raise ValueError(f"人脸来源已变更，需重建时间缓存: {timing_path}")
        visual_frame_times = torch.tensor(timing["frame_timestamps"], dtype=torch.float32)
        visual_frame_observed = torch.tensor(timing["frame_observed"], dtype=torch.bool)
        visual_duration = float(timing["visual_duration"])
        if not math.isfinite(visual_duration) or visual_duration <= 0 or audio_duration <= 0:
            raise ValueError(f"无效音视频时长: {file_name}")
        clip_duration = max(audio_duration, visual_duration)
        visual_tensor = torch.load(f"{self.cache_root}/visual_{self.visual_input}/{file_name}.pt", weights_only=True)
        expected_shape = (32, 224, 224, 3) if self.visual_input == "face" else (32, 3, 224, 224)
        if tuple(visual_tensor.shape) != expected_shape:
            raise ValueError(f"视觉缓存形状不匹配: {file_name}: {tuple(visual_tensor.shape)}, 需要 {expected_shape}")
        if visual_frame_times.shape != (visual_tensor.size(0),) or visual_frame_observed.shape != visual_frame_times.shape:
            raise ValueError(f"图像与时间缓存长度不一致: {file_name}")
        if not torch.isfinite(visual_frame_times).all() or (visual_frame_times < 0).any() or (visual_frame_times > clip_duration).any():
            raise ValueError(f"帧时间超出片段范围: {timing_path}")
        if self.visual_input == "face":
            visual_tensor = visual_tensor.permute(0, 3, 1, 2)
        visual_tensor = visual_tensor / 127.5 - 1.0

        item = {
            "file_name": file_name,
            "text": text,
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask.to(torch.bool),
            "audio_input_values": audio_input_values,
            "audio_attention_mask": audio_attention_mask,
            "audio_sample_rate": sample_rate,
            "clip_duration": clip_duration,
            "visual_frame_times": visual_frame_times,
            "visual_frame_observed": visual_frame_observed,
            "visual_tensor": visual_tensor,
            "label_M": row["label"],
            "label_T": t_score,
            "label_A": a_score,
            "label_V": v_score,
        }
        return item

    def __len__(self):
        return len(self.df)


class LightningData(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        pass

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_data = DatasetMSA(self.config.dataset, "train", config=self.config)
            self.val_data = DatasetMSA(self.config.dataset, "valid", config=self.config)
        elif stage == "validate":
            self.val_data = DatasetMSA(self.config.dataset, "valid", config=self.config)
        elif stage == "test" or stage == "predict":
            self.test_data = DatasetMSA(self.config.dataset, "test", config=self.config)
        else:
            raise ValueError("Stage must be fit, test or predict")

    def train_dataloader(self):
        audio_lengths = []
        for row in self.train_data.df.itertuples(index=False):
            audio_path = Path(self.train_data.data_root) / "audio" / row.video_id / f"{row.clip_id}.wav"
            with wave.open(str(audio_path), "rb") as audio:
                audio_lengths.append(audio.getnframes())
        batch_sampler = AudioLengthBatchSampler(
            sampler=RandomSampler(self.train_data),
            audio_lengths=audio_lengths,
            batch_size=self.config.batch_size_train,
            bucket_size=self.config.audio_bucket_size,
        )
        return DataLoader(
            self.train_data,
            batch_sampler=batch_sampler,
            num_workers=self.config.num_workers,
            persistent_workers=True if self.config.num_workers > 0 else False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.config.batch_size_eval,
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=True if self.config.num_workers > 0 else False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.config.batch_size_eval,
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=True if self.config.num_workers > 0 else False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=True if self.config.num_workers > 0 else False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        file_name = [i["file_name"] for i in batch]
        text = [i["text"] for i in batch]
        text_input_ids = [i["text_input_ids"] for i in batch]
        text_attention_mask = [i["text_attention_mask"] for i in batch]
        audio_input_values = [i["audio_input_values"] for i in batch]
        audio_attention_mask = [i["audio_attention_mask"] for i in batch]
        visual = [i["visual_tensor"] for i in batch]
        label_M = torch.tensor([i["label_M"] for i in batch])
        label_T = torch.tensor([i["label_T"] for i in batch])
        label_A = torch.tensor([i["label_A"] for i in batch])
        label_V = torch.tensor([i["label_V"] for i in batch])

        text_input_ids = torch.nn.utils.rnn.pad_sequence(list(text_input_ids), batch_first=True, padding_value=0)
        text_attention_mask = torch.nn.utils.rnn.pad_sequence(list(text_attention_mask), batch_first=True, padding_value=0)
        text_length = text_attention_mask.sum(1)

        audio_input_values = torch.nn.utils.rnn.pad_sequence(list(audio_input_values), batch_first=True, padding_value=0)
        audio_attention_mask = torch.nn.utils.rnn.pad_sequence(list(audio_attention_mask), batch_first=True, padding_value=0).bool()

        visual_tensor = torch.stack(visual)
        audio_sample_rate = torch.tensor([i["audio_sample_rate"] for i in batch], dtype=torch.float32)
        clip_duration = torch.tensor([i["clip_duration"] for i in batch], dtype=torch.float32)
        visual_frame_times = torch.stack([i["visual_frame_times"] for i in batch])
        visual_frame_observed = torch.stack([i["visual_frame_observed"] for i in batch])

        return {
            "file_name": file_name,
            "text": text,
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "text_length": text_length,
            "audio_input_values": audio_input_values,
            "audio_attention_mask": audio_attention_mask,
            "audio_sample_rate": audio_sample_rate,
            "clip_duration": clip_duration,
            "visual_frame_times": visual_frame_times,
            "visual_frame_observed": visual_frame_observed,
            "visual_tensor": visual_tensor,
            "label_M": label_M,
            "label_T": label_T,
            "label_A": label_A,
            "label_V": label_V,
        }


if __name__ == "__main__":
    from config import Config

    config = Config(dataset="MOSI")
    dataset = DatasetMSA(config.dataset, "train", config)

    lightning_data = LightningData(config)

    dataloader = DataLoader(
        dataset,
        batch_size=100,
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
        pin_memory=True,
        collate_fn=lightning_data.collate_fn,
    )

    label = []
    for batch in dataloader:
        label.append(batch["label_V"])

    print(label)
