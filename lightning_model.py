import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
import numpy as np
import pandas as pd

from transformers import BertModel, RobertaModel, VivitModel
from transformers import HubertModel
from transformers.optimization import get_cosine_schedule_with_warmup

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vis_tsne import tsne_visualization
from utils.metricsTop import MetricsTop


def set_train_mode_for_trainable_module(module):
    if any(parameter.requires_grad for parameter in module.parameters()):
        module.train()
    else:
        module.eval()


class LightningModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.get_hparams())
        self.metrics_evaluator = MetricsTop(train_mode="regression")
        self.get_metrics = self.metrics_evaluator.getMetics(config.dataset)

        self.val_predictions = []
        self.val_labels = []
        self.test_predictions = []
        self.test_labels = []
        self.lr_group_names = ["backbone", "task"]

        if config.dataset in ["SIMS"]:
            self.text_extractor = BertModel.from_pretrained(
                config.text_extractor,
                output_hidden_states=False,
                add_pooling_layer=False,
            )
        elif config.dataset in ["MOSI", "MOSEI"]:
            self.text_extractor = RobertaModel.from_pretrained(
                config.text_extractor,
                output_hidden_states=False,
                add_pooling_layer=False,
            )
        self.audio_extractor = HubertModel.from_pretrained(
            config.audio_extractor,
            output_hidden_states=False,
        )
        self.visual_extractor = VivitModel.from_pretrained(
            config.visual_extractor,
            output_hidden_states=False,
            add_pooling_layer=False,
        )
        if getattr(config, "gradient_checkpointing", False):
            for extractor in (self.text_extractor, self.audio_extractor, self.visual_extractor):
                extractor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self._sync_extractor_train_modes()

        mode = config.ablation_mode
        text_dim = self.text_extractor.config.hidden_size
        audio_dim = self.audio_extractor.config.hidden_size
        visual_dim = self.visual_extractor.config.hidden_size
        self.audio_receptive_field = 1
        self.audio_stride = 1
        for kernel, stride in zip(self.audio_extractor.config.conv_kernel, self.audio_extractor.config.conv_stride):
            self.audio_receptive_field += (kernel - 1) * self.audio_stride
            self.audio_stride *= stride
        self.visual_tubelet_frames = self.visual_extractor.config.tubelet_size[0]

        need_text_query = mode in ("full", "wo_av_query", "wo_aggregation", "wo_temporal_anchor")
        need_av_query = mode in ("full", "wo_text_query", "wo_aggregation", "wo_temporal_anchor")
        need_aggregation = mode in ("full", "direct_fusion", "wo_text_query", "wo_av_query", "wo_temporal_anchor")
        need_proj_v2t = mode in ("direct_fusion", "wo_av_query")

        if need_text_query:
            self.text_query_attention = TextQueryAttention(
                num_query=8,
                num_layer=3,
                query_dim=text_dim,
                dropout=config.dropout,
            )
        if need_av_query:
            self.av_query_attention = AVQueryAttention(
                num_query=16,
                num_layer=3,
                query_dim=text_dim,
                audio_dim=audio_dim,
                visual_dim=visual_dim,
                dropout=config.dropout,
                visual_num_frames=self.visual_extractor.config.num_frames,
                visual_image_size=self.visual_extractor.config.image_size,
                visual_tubelet_size=self.visual_extractor.config.tubelet_size,
            )
            self.av_query_attention.set_temporal_anchor_config(
                enable_temporal_anchor=config.enable_temporal_anchor and mode != "wo_temporal_anchor",
                temporal_anchor_sigma=config.temporal_anchor_sigma,
                temporal_anchor_strength=config.temporal_anchor_strength,
                temporal_anchor_mode=getattr(config, "temporal_anchor_mode", "pooled"),
            )
        if need_aggregation:
            self.aggregation = Aggregation(text_dim, num_layers=2, dropout=config.dropout)
        if need_proj_v2t:
            self.proj_v2t = nn.Linear(visual_dim, text_dim)

        head_input_dim = {
            "full": text_dim,
            "text_only": text_dim,
            "audio_only": audio_dim,
            "visual_only": visual_dim,
            "cls_concat": text_dim + audio_dim + visual_dim,
            "direct_fusion": text_dim,
            "wo_text_query": text_dim,
            "wo_av_query": text_dim,
            "wo_aggregation": text_dim * 2,
            "wo_temporal_anchor": text_dim,
        }[mode]
        self.predict_head = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(head_input_dim, 1),
        )
        self.output_activation = nn.Tanh() if config.dataset in ["SIMS", "SIMSV2"] else nn.Identity()

    def on_load_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {key.replace("video", "visual"): value for key, value in checkpoint["state_dict"].items()}

    def forward(self, batch):
        text_input_ids = batch["text_input_ids"]
        text_attention_mask = batch["text_attention_mask"]
        text_feature = self.text_extractor(text_input_ids, attention_mask=text_attention_mask).last_hidden_state
        audio_input_values = batch["audio_input_values"]
        audio_attention_mask = batch["audio_attention_mask"]
        audio_lengths = audio_attention_mask.sum(dim=1)
        if (audio_lengths < self.audio_receptive_field).any():
            raise ValueError("音频短于 HuBERT 卷积感受野，无法生成有效 token")
        audio_sequence_length = (audio_input_values.size(1) - self.audio_receptive_field) // self.audio_stride + 1
        mask_time_indices = None
        if (
            self.audio_extractor.training
            and self.audio_extractor.config.mask_time_prob > 0
            and audio_sequence_length < self.audio_extractor.config.mask_time_length
        ):
            mask_time_indices = torch.zeros(
                (audio_input_values.size(0), audio_sequence_length), dtype=torch.bool, device=audio_input_values.device
            )
        audio_feature = self.audio_extractor(
            audio_input_values, attention_mask=audio_attention_mask, mask_time_indices=mask_time_indices
        ).last_hidden_state
        visual_tensor = batch["visual_tensor"]
        visual_feature = self.visual_extractor(visual_tensor).last_hidden_state

        audio_indices = torch.arange(audio_feature.size(1), device=audio_feature.device, dtype=torch.float32)
        audio_output_lengths = torch.div(audio_lengths - self.audio_receptive_field, self.audio_stride, rounding_mode="floor") + 1
        audio_output_attention_mask = audio_indices.unsqueeze(0) >= audio_output_lengths.unsqueeze(1)
        clip_duration = batch["clip_duration"].float()
        audio_seconds = audio_indices * self.audio_stride + (self.audio_receptive_field - 1) / 2
        audio_positions = audio_seconds.unsqueeze(0) / batch["audio_sample_rate"].float().unsqueeze(1) / clip_duration.unsqueeze(1)

        batch_size = visual_feature.size(0)
        frame_times = batch["visual_frame_times"].float().reshape(batch_size, -1, self.visual_tubelet_frames)
        observed = batch["visual_frame_observed"].reshape_as(frame_times)
        observed_count = observed.sum(dim=-1)
        tubelet_times = (frame_times * observed).sum(dim=-1) / observed_count.clamp(min=1)
        tubelet_times = torch.where(observed_count > 0, tubelet_times, frame_times.mean(dim=-1))
        spatial_count = (visual_feature.size(1) - 1) // tubelet_times.size(1)
        visual_positions = (tubelet_times / clip_duration.unsqueeze(1)).repeat_interleave(spatial_count, dim=1)
        visual_patch_mask = (observed_count == 0).repeat_interleave(spatial_count, dim=1)
        visual_output_attention_mask = torch.cat([visual_patch_mask.all(dim=1, keepdim=True), visual_patch_mask], dim=1)
        visual_valid = (~visual_output_attention_mask).unsqueeze(-1).to(dtype=visual_feature.dtype)
        visual_pool = (visual_feature * visual_valid).sum(dim=1) / visual_valid.sum(dim=1).clamp(min=1)
        av_time_inputs = {
            "audio_positions": audio_positions,
            "visual_positions": visual_positions,
            "visual_mask": visual_patch_mask,
        }

        mode = self.config.ablation_mode
        aux_outputs = {}

        if mode in ("full", "wo_temporal_anchor"):
            text_outputs = self.text_query_attention(text_feature, text_mask=~text_attention_mask)
            av_outputs = self.av_query_attention(audio_feature, visual_feature, audio_mask=audio_output_attention_mask, **av_time_inputs)
            mm_feature = torch.cat([text_outputs["query"], av_outputs["audio_query"], av_outputs["visual_query"]], dim=1)
            mm_feature = self.aggregation(mm_feature)
            aux_outputs = {
                "text_attention_map": text_outputs["cross_attn_map"],
                "audio_attention_map": av_outputs["audio_attn_map"],
                "visual_attention_map": av_outputs["visual_attn_map"],
                "anchor_centers": av_outputs["anchor_centers"],
                "audio_positions": audio_positions,
                "visual_positions": visual_positions,
                "audio_output_attention_mask": audio_output_attention_mask,
                "visual_output_attention_mask": visual_output_attention_mask,
            }

        elif mode == "text_only":
            mm_feature = text_feature[:, 0, :]

        elif mode == "audio_only":
            valid_mask = (~audio_output_attention_mask).unsqueeze(-1).to(dtype=audio_feature.dtype)
            mm_feature = (audio_feature * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)

        elif mode == "visual_only":
            mm_feature = visual_pool

        elif mode == "cls_concat":
            audio_pool = (~audio_output_attention_mask).unsqueeze(-1).to(dtype=audio_feature.dtype)
            audio_pool = (audio_feature * audio_pool).sum(dim=1) / audio_pool.sum(dim=1).clamp(min=1.0)
            mm_feature = torch.cat([text_feature[:, 0, :], audio_pool, visual_pool], dim=-1)

        elif mode == "direct_fusion":
            visual_proj = self.proj_v2t(visual_feature)
            mm_feature = torch.cat([text_feature, audio_feature, visual_proj], dim=1)
            mm_mask = torch.cat(
                [
                    ~text_attention_mask,
                    audio_output_attention_mask,
                    visual_output_attention_mask,
                ],
                dim=1,
            )
            mm_feature = self.aggregation(mm_feature, mask=mm_mask)

        elif mode == "wo_text_query":
            av_outputs = self.av_query_attention(audio_feature, visual_feature, audio_mask=audio_output_attention_mask, **av_time_inputs)
            mm_feature = torch.cat([text_feature[:, 0:1, :], av_outputs["audio_query"], av_outputs["visual_query"]], dim=1)
            mm_feature = self.aggregation(mm_feature)

        elif mode == "wo_av_query":
            text_outputs = self.text_query_attention(text_feature, text_mask=~text_attention_mask)
            audio_pool = (~audio_output_attention_mask).unsqueeze(-1).to(dtype=audio_feature.dtype)
            audio_pool = (audio_feature * audio_pool).sum(dim=1) / audio_pool.sum(dim=1).clamp(min=1.0)
            visual_pool = self.proj_v2t(visual_pool)
            mm_feature = torch.cat([text_outputs["query"], audio_pool.unsqueeze(1), visual_pool.unsqueeze(1)], dim=1)
            mm_feature = self.aggregation(mm_feature)

        elif mode == "wo_aggregation":
            text_outputs = self.text_query_attention(text_feature, text_mask=~text_attention_mask)
            av_outputs = self.av_query_attention(audio_feature, visual_feature, audio_mask=audio_output_attention_mask, **av_time_inputs)
            text_summary = text_outputs["query"].mean(dim=1)
            av_summary = av_outputs["av_query"].mean(dim=1)
            mm_feature = torch.cat([text_summary, av_summary], dim=-1)

        else:
            raise ValueError(f"Unknown ablation_mode: {mode}")

        logits = self.output_activation(self.predict_head(mm_feature))
        return logits, (mm_feature, aux_outputs)

    def compute_loss(self, logits, label, aux_outputs=None):
        predictions = logits.squeeze(-1)
        loss = F.l1_loss(predictions, label, reduction="none").mean()
        return {"loss": loss}

    def _sync_extractor_train_modes(self):
        set_train_mode_for_trainable_module(self.text_extractor)
        set_train_mode_for_trainable_module(self.audio_extractor)
        set_train_mode_for_trainable_module(self.visual_extractor)

    def _parameters_with_grad(self, module):
        return [param for param in module.parameters() if param.requires_grad]

    def _get_grad_norm_terms(self, parameters):
        grad_norm_terms = []
        for parameter in parameters:
            grad = parameter.grad
            if grad is not None:
                grad_norm_terms.append(torch.norm(grad.detach(), 2.0))
        return grad_norm_terms

    def _get_optimizer_group_lrs(self):
        lr_by_group = {name: 0.0 for name in self.lr_group_names}
        try:
            optimizer = self.optimizers()
            if isinstance(optimizer, list):
                optimizer = optimizer[0]
            for group in optimizer.param_groups:
                group_name = group.get("name")
                if group_name in lr_by_group:
                    lr_by_group[group_name] = group["lr"]
        except Exception:
            lr_by_group = {
                "backbone": self.config.backbone_learning_rate,
                "task": self.config.task_learning_rate,
            }
        return lr_by_group

    def configure_optimizers(self):
        backbone_modules = [
            self.text_extractor,
            self.audio_extractor,
            self.visual_extractor,
        ]
        task_modules = [self.predict_head]
        for attr in ("text_query_attention", "av_query_attention", "aggregation", "proj_v2t"):
            if hasattr(self, attr):
                task_modules.append(getattr(self, attr))

        backbone_params = []
        for module in backbone_modules:
            backbone_params.extend(self._parameters_with_grad(module))

        task_decay_params = []
        task_no_decay_params = []
        for module in task_modules:
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if name.endswith("bias") or "norm" in name:
                    task_no_decay_params.append(param)
                else:
                    task_decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"name": "backbone", "params": backbone_params, "lr": self.config.backbone_learning_rate, "weight_decay": 0.0},
                {"name": "task", "params": task_decay_params, "lr": self.config.task_learning_rate, "weight_decay": self.config.weight_decay},
                {"name": "task_no_decay", "params": task_no_decay_params, "lr": self.config.task_learning_rate, "weight_decay": 0.0},
            ],
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.trainer.estimated_stepping_batches * 0.1),
            num_training_steps=int(self.trainer.estimated_stepping_batches),
        )
        lr_scheduler = {"scheduler": scheduler, "name": "learning_rate", "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def on_after_backward111(self):
        lr_by_group = self._get_optimizer_group_lrs()
        for group_name, group_lr in lr_by_group.items():
            self.log(f"learning_rate/{group_name}", group_lr, on_step=True, on_epoch=False)

        params_with_grads = [p for p in self.parameters() if p.grad is not None]
        grad_global_norm = torch.norm(torch.stack(self._get_grad_norm_terms(params_with_grads)), 2.0)
        self.log("grad_global_norm", grad_global_norm, on_step=True, on_epoch=False)

        weight_norm = torch.norm(torch.stack([torch.norm(p.data.detach(), 2.0) for p in self.parameters()]), 2.0)
        self.log("weight_global_norm", weight_norm, on_step=True, on_epoch=False)

        _group_prefixes = {
            "backbone": ["text_extractor", "audio_extractor", "visual_extractor"],
            "task": ["text_query_attention", "av_query_attention", "aggregation", "predict_head"],
        }
        for group_name, prefixes in _group_prefixes.items():
            grads = [
                torch.norm(p.grad.detach(), 2.0) for n, p in self.named_parameters() if any(n.startswith(pfx) for pfx in prefixes) and p.grad is not None
            ]
            weights = [torch.norm(p.data.detach(), 2.0) for n, p in self.named_parameters() if any(n.startswith(pfx) for pfx in prefixes)]
            if grads and weights:
                g_norm = torch.norm(torch.stack(grads), 2.0)
                w_norm = torch.norm(torch.stack(weights), 2.0)
                self.log(f"{group_name}_grad_norm", g_norm, on_step=True, on_epoch=False)
                self.log(f"{group_name}_weight_norm", w_norm, on_step=True, on_epoch=False)
                self.log(f"{group_name}_update_ratio", (lr_by_group[group_name] * g_norm) / (w_norm + 1e-9), on_step=True, on_epoch=False)

    def training_step(self, batch, batch_idx):
        logits, (_, aux_outputs) = self(batch)
        losses = self.compute_loss(logits, batch["label_M"], aux_outputs=aux_outputs)
        self.log_dict(
            {f"losses/{k}_train": v for k, v in losses.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch["label_M"].size(0),
        )
        return losses

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        params_with_grads = [p for p in self.parameters() if p.grad is not None]
        if not params_with_grads:
            return
        self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)

    def on_train_epoch_end(self):
        pass

    def on_fit_start(self):
        self._sync_extractor_train_modes()

    def validation_step(self, batch, batch_idx):
        labels = batch["label_M"]
        logits, (_, aux_outputs) = self(batch)
        losses = self.compute_loss(logits, labels, aux_outputs=aux_outputs)
        predictions = logits.squeeze(-1)

        self.val_predictions.append(predictions.detach().cpu())
        self.val_labels.append(labels.detach().cpu())
        self.log_dict(
            {f"losses/{k}_val": v for k, v in losses.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch["label_M"].size(0),
        )

    def on_validation_epoch_end(self):
        if self.val_predictions and self.val_labels:
            val_pred = torch.cat(self.val_predictions, dim=0)
            val_true = torch.cat(self.val_labels, dim=0)
            val_pred = val_pred.to(torch.float32)
            val_true = val_true.to(torch.float32)
            val_metrics = self.get_metrics(val_pred, val_true)

            self.log_dict(
                {f"metrics/{k}_val": v for k, v in val_metrics.items()},
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        self.val_predictions.clear()
        self.val_labels.clear()

    def on_test_start(self):
        pass
        self.test_feature = []
        self.test_file_names = []

    def test_step(self, batch):
        labels = batch["label_M"]
        logits, (features, aux_outputs) = self(batch)
        predictions = logits.squeeze(-1)

        self.test_predictions.append(predictions.detach().cpu())
        self.test_labels.append(labels.detach().cpu())

        self.test_feature.append(features.detach().cpu())
        self.test_file_names.extend(batch["file_name"])

    def get_artifact_dir(self):
        experiment_id = getattr(self.logger, "experiment_id", None)
        run_id = getattr(self.logger, "run_id", None)
        if experiment_id is None or run_id is None:
            return os.path.join("hfff", "analysis")
        return os.path.join("mlruns", str(experiment_id), str(run_id), "artifacts")

    def save_test_predictions_csv(self, prediction):
        label_path = f"../datasets/{self.config.dataset}/label.csv"
        if self.config.dataset == "SIMS":
            df = pd.read_csv(
                label_path,
                names=["video_id", "clip_id", "text", "label", "label_T", "label_A", "label_V", "annotation", "mode"],
                dtype={"video_id": str, "clip_id": str},
            )
            write_header = False
        else:
            df = pd.read_csv(label_path, dtype={"video_id": str, "clip_id": str})
            write_header = True

        df["file_name"] = df["video_id"].astype(str) + "/" + df["clip_id"].astype(str)
        prediction_by_file = dict(zip(self.test_file_names, prediction.tolist()))
        previous_values = df["prediction"] if "prediction" in df.columns else pd.Series(np.nan, index=df.index)
        df["prediction"] = df["file_name"].map(prediction_by_file).combine_first(previous_values)
        df = df.drop(columns=["file_name"])

        artifact_dir = os.path.join(self.get_artifact_dir(), "test_outputs")
        os.makedirs(artifact_dir, exist_ok=True)
        output_path = os.path.join(artifact_dir, "label_with_predictions.csv")
        df.to_csv(output_path, index=False, header=write_header)
        return artifact_dir

    def on_test_epoch_end(self):
        if self.test_predictions and self.test_labels:
            test_pred = torch.cat(self.test_predictions, dim=0)
            test_true = torch.cat(self.test_labels, dim=0)
            test_pred = test_pred.to(torch.float32)
            test_true = test_true.to(torch.float32)
            metrics = self.get_metrics(test_pred, test_true)

            self.log_dict(
                {"metrics/" + k: v for k, v in metrics.items()},
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        prediction = torch.cat(self.test_predictions, dim=0).float().numpy()
        label = torch.cat(self.test_labels, dim=0).float().numpy()
        feature = torch.cat(self.test_feature, dim=0).float().numpy()
        output_dir = self.save_test_predictions_csv(prediction)
        np.save(os.path.join(output_dir, "feature.npy"), feature)
        fig = tsne_visualization(feature, label)
        fig.savefig(os.path.join(output_dir, "tsne_visualization.png"))
        self.test_predictions.clear()
        self.test_labels.clear()
        self.test_feature.clear()
        self.test_file_names.clear()


    def predict_step(self, batch):
        pass


class TextQueryAttentionLayer(nn.Module):
    def __init__(self, query_dim, dropout):
        super(TextQueryAttentionLayer, self).__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim),
        )
        self.norm_query = nn.LayerNorm(query_dim)
        self.norm_feature = nn.LayerNorm(query_dim)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, feature, text_mask):
        query_norm = self.norm_query(query)
        feature_norm = self.norm_feature(feature)
        query_output, cross_attn_output_weights = self.cross_attention(query_norm, feature_norm, feature_norm, key_padding_mask=text_mask)
        query = query + self.dropout(query_output)

        query_norm = self.norm1(query)
        query_output, self_attn_output_weights = self.self_attention(query_norm, query_norm, query_norm)
        query = query + self.dropout1(query_output)

        query_norm = self.norm2(query)
        query_output = self.ffn(query_norm)
        query = query + self.dropout2(query_output)

        return query, cross_attn_output_weights


class TextQueryAttention(nn.Module):
    def __init__(self, num_query, num_layer, query_dim, dropout):
        super(TextQueryAttention, self).__init__()
        self.num_query = num_query
        self.hidden_dim = query_dim
        self.dropout = dropout
        self.query_generator = nn.Sequential(
            nn.Linear(query_dim, num_query * query_dim),
        )
        self.query_attention_layer_list = nn.ModuleList([TextQueryAttentionLayer(query_dim, dropout) for _ in range(num_layer)])

    def forward(self, text_feature, text_mask):
        batch_size = text_feature.size(0)
        query = self.query_generator(text_feature[:, 0, :]).view(batch_size, self.num_query, self.hidden_dim)
        cross_attn_maps = []

        for layer in self.query_attention_layer_list:
            query, cross_attn_map = layer(query, text_feature[:, 1:, :], text_mask[:, 1:])
            cross_attn_maps.append(cross_attn_map)
        mean_cross_attn_map = torch.stack(cross_attn_maps, dim=0).mean(dim=0)
        return {
            "query": query,
            "cross_attn_map": mean_cross_attn_map,
        }


class AVQueryAttentionLayer(nn.Module):
    def __init__(self, query_dim, audio_dim, visual_dim, dropout):
        super(AVQueryAttentionLayer, self).__init__()
        self.num_heads = 8

        self.audio_cross_attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=audio_dim,
            vdim=audio_dim,
        )
        self.visual_cross_attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=visual_dim,
            vdim=visual_dim,
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim),
        )

        self.norm_audio_query = nn.LayerNorm(query_dim)
        self.norm_visual_query = nn.LayerNorm(query_dim)
        self.norm_audio_feature = nn.LayerNorm(audio_dim)
        self.norm_visual_feature = nn.LayerNorm(visual_dim)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.norm4 = nn.LayerNorm(query_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def _merge_attention_masks(self, attn_bias, key_padding_mask, query_len, key_len, target_dtype):
        if key_padding_mask is None:
            return None if attn_bias is None else attn_bias.to(dtype=target_dtype)
        batch_size = key_padding_mask.size(0)
        padding_mask = key_padding_mask[:, None, None, :].expand(batch_size, self.num_heads, query_len, key_len)
        padding_mask = padding_mask.reshape(batch_size * self.num_heads, query_len, key_len)
        if attn_bias is None:
            attn_bias = torch.zeros(padding_mask.shape, device=key_padding_mask.device, dtype=target_dtype)
        return attn_bias.to(dtype=target_dtype).masked_fill(padding_mask, -torch.inf)

    def forward(self, audio_query, visual_query, audio_feature, visual_feature, audio_mask, audio_attn_bias=None, visual_attn_bias=None, visual_mask=None):
        audio_query_norm = self.norm_audio_query(audio_query)
        audio_feature_norm = self.norm_audio_feature(audio_feature)
        audio_attn_mask = self._merge_attention_masks(
            attn_bias=audio_attn_bias,
            key_padding_mask=audio_mask,
            query_len=audio_query_norm.size(1),
            key_len=audio_feature_norm.size(1),
            target_dtype=audio_query_norm.dtype,
        )
        audio_query_output, audio_cross_attn_output_weights = self.audio_cross_attention(
            audio_query_norm,
            audio_feature_norm,
            audio_feature_norm,
            attn_mask=audio_attn_mask,
        )
        audio_query = audio_query + self.dropout1(audio_query_output)

        visual_query_norm = self.norm_visual_query(visual_query)
        visual_feature_norm = self.norm_visual_feature(visual_feature)
        visual_missing = None
        if visual_mask is not None:
            visual_missing = visual_mask.all(dim=1)
            visual_mask = visual_mask.clone()
            visual_mask[:, 0] &= ~visual_missing
        visual_attn_mask = self._merge_attention_masks(
            visual_attn_bias,
            visual_mask,
            visual_query_norm.size(1),
            visual_feature_norm.size(1),
            visual_query_norm.dtype,
        )
        visual_query_output, visual_cross_attn_output_weights = self.visual_cross_attention(
            visual_query_norm,
            visual_feature_norm,
            visual_feature_norm,
            attn_mask=visual_attn_mask,
        )
        if visual_missing is not None:
            visual_query_output = visual_query_output.masked_fill(visual_missing[:, None, None], 0)
            visual_cross_attn_output_weights = visual_cross_attn_output_weights.masked_fill(visual_missing[:, None, None], 0)
        visual_query = visual_query + self.dropout2(visual_query_output)

        av_query = torch.cat([audio_query, visual_query], dim=1)
        av_query_norm = self.norm1(av_query)
        av_query_output, self_attn_output_weights = self.self_attention(av_query_norm, av_query_norm, av_query_norm)
        av_query = av_query + self.dropout3(av_query_output)

        av_query_norm = self.norm2(av_query)
        av_query_output = self.ffn(av_query_norm)
        av_query = av_query + self.dropout4(av_query_output)
        audio_query, visual_query = torch.chunk(av_query, 2, dim=1)
        return audio_query, visual_query, audio_cross_attn_output_weights, visual_cross_attn_output_weights


class AVQueryAttention(nn.Module):
    def __init__(
        self, num_query, num_layer, query_dim, audio_dim, visual_dim, dropout, visual_num_frames=32, visual_image_size=224, visual_tubelet_size=(2, 16, 16)
    ):
        super(AVQueryAttention, self).__init__()
        self.num_query = num_query
        self.hidden_dim = query_dim
        self.dropout = dropout
        self.num_heads = 8
        self.enable_temporal_anchor = None
        self.temporal_anchor_sigma = None
        self.temporal_anchor_strength = None
        self.temporal_anchor_mode = "pooled"
        image_size = (visual_image_size, visual_image_size) if isinstance(visual_image_size, int) else visual_image_size
        temporal_size, patch_height, patch_width = visual_tubelet_size
        temporal_count = visual_num_frames // temporal_size
        spatial_count = (image_size[0] // patch_height) * (image_size[1] // patch_width)
        frame_positions = torch.linspace(0.0, 1.0, steps=visual_num_frames)
        temporal_positions = frame_positions[: temporal_count * temporal_size].reshape(temporal_count, temporal_size).mean(dim=1)
        self.register_buffer("visual_positions", temporal_positions.repeat_interleave(spatial_count), persistent=False)
        self.anchor_condition_norm = nn.LayerNorm(audio_dim + visual_dim)
        self.audio_query_embeddings = nn.Parameter(torch.empty(num_query, query_dim))
        self.visual_query_embeddings = nn.Parameter(torch.empty(num_query, query_dim))
        nn.init.trunc_normal_(self.audio_query_embeddings, std=0.02)
        nn.init.trunc_normal_(self.visual_query_embeddings, std=0.02)
        self.anchor_predictor = nn.Sequential(
            nn.Linear(audio_dim + visual_dim, query_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim, num_query),
        )

        self.query_attention_layer_list = nn.ModuleList([AVQueryAttentionLayer(query_dim, audio_dim, visual_dim, dropout) for _ in range(num_layer)])

    def set_temporal_anchor_config(self, enable_temporal_anchor, temporal_anchor_sigma, temporal_anchor_strength, temporal_anchor_mode="pooled"):
        if temporal_anchor_mode not in ("pooled", "uniform"):
            raise ValueError(f"Unknown temporal_anchor_mode: {temporal_anchor_mode}")
        self.temporal_anchor_mode = temporal_anchor_mode
        self.enable_temporal_anchor = enable_temporal_anchor
        self.temporal_anchor_sigma = temporal_anchor_sigma
        self.temporal_anchor_strength = temporal_anchor_strength
        use_learned_anchor = enable_temporal_anchor and temporal_anchor_mode == "pooled"
        self.anchor_condition_norm.requires_grad_(use_learned_anchor)
        self.anchor_predictor.requires_grad_(use_learned_anchor)

    def _build_temporal_attn_bias(self, anchor_centers, seq_len, key_padding_mask=None, positions=None):
        anchor_centers = anchor_centers.float()
        if positions is not None:
            positions = positions.to(device=anchor_centers.device, dtype=torch.float32).reshape(-1, 1, seq_len)
        elif key_padding_mask is not None:
            actual_lengths = (~key_padding_mask).sum(dim=1, keepdim=True).float().clamp(min=2)
            indices = torch.arange(seq_len, device=anchor_centers.device, dtype=anchor_centers.dtype).unsqueeze(0)
            positions = indices / (actual_lengths - 1)
            positions = positions.unsqueeze(1)
        else:
            positions = torch.linspace(0.0, 1.0, steps=seq_len, device=anchor_centers.device, dtype=anchor_centers.dtype)
            positions = positions.view(1, 1, seq_len)
        distances = positions - anchor_centers.unsqueeze(-1)
        sigma = max(self.temporal_anchor_sigma, 1e-3)
        bias = -self.temporal_anchor_strength * distances.pow(2) / (2.0 * sigma * sigma)
        batch_size, num_query, _ = bias.shape
        bias = bias.unsqueeze(1).expand(batch_size, self.num_heads, num_query, seq_len)
        return bias.reshape(batch_size * self.num_heads, num_query, seq_len)

    def _masked_mean_pool(self, feature, padding_mask=None):
        if padding_mask is None:
            return feature.mean(dim=1)

        valid_mask = (~padding_mask).unsqueeze(-1).to(dtype=feature.dtype)
        valid_count = valid_mask.sum(dim=1).clamp(min=1.0)
        return (feature * valid_mask).sum(dim=1) / valid_count

    def forward(self, audio_feature, visual_feature, audio_mask, audio_positions=None, visual_positions=None, visual_mask=None):
        if visual_feature.size(1) != self.visual_positions.numel() + 1:
            raise ValueError("ViViT tokens must contain CLS followed by the configured tubelet grid")
        visual_feature = visual_feature[:, 1:, :]
        batch_size = audio_feature.size(0)
        audio_query = self.audio_query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        visual_query = self.visual_query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        anchor_centers = None
        audio_attn_bias = None
        visual_attn_bias = None
        if self.enable_temporal_anchor:
            if self.temporal_anchor_mode == "uniform":
                anchor_centers = (torch.arange(self.num_query, device=audio_feature.device, dtype=audio_feature.dtype) + 0.5) / self.num_query
                anchor_centers = anchor_centers.unsqueeze(0).expand(batch_size, -1)
            else:
                audio_global_feature = self._masked_mean_pool(audio_feature, padding_mask=audio_mask)
                visual_global_feature = self._masked_mean_pool(visual_feature, padding_mask=visual_mask)
                anchor_feature = torch.cat([audio_global_feature, visual_global_feature], dim=-1)
                anchor_feature = self.anchor_condition_norm(anchor_feature)
                anchor_centers = torch.sigmoid(self.anchor_predictor(anchor_feature))
            audio_attn_bias = self._build_temporal_attn_bias(
                anchor_centers,
                audio_feature.size(1),
                key_padding_mask=audio_mask,
                positions=audio_positions,
            )
            visual_attn_bias = self._build_temporal_attn_bias(
                anchor_centers,
                visual_feature.size(1),
                positions=self.visual_positions if visual_positions is None else visual_positions,
            )

        audio_attn_maps = []
        visual_attn_maps = []

        for layer in self.query_attention_layer_list:
            audio_query, visual_query, audio_attn_map, visual_attn_map = layer(
                audio_query,
                visual_query,
                audio_feature,
                visual_feature,
                audio_mask,
                audio_attn_bias=audio_attn_bias,
                visual_attn_bias=visual_attn_bias,
                visual_mask=visual_mask,
            )
            audio_attn_maps.append(audio_attn_map)
            visual_attn_maps.append(visual_attn_map)
        query = torch.cat([audio_query, visual_query], dim=1)
        return {
            "audio_query": audio_query,
            "visual_query": visual_query,
            "av_query": query,
            "anchor_centers": anchor_centers,
            "audio_attn_map": torch.stack(audio_attn_maps, dim=0).mean(dim=0),
            "visual_attn_map": F.pad(torch.stack(visual_attn_maps, dim=0).mean(dim=0), (1, 0)),
        }


class Aggregation(nn.Module):
    def __init__(self, d_model, num_layers, dropout):
        super(Aggregation, self).__init__()
        self.cls_embeddings = nn.Parameter(torch.empty(1, d_model))
        nn.init.trunc_normal_(self.cls_embeddings, std=0.02)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, features, mask=None):
        batch_size = features.size(0)
        cls_token = self.cls_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        features = torch.cat([cls_token, features], dim=1)
        if mask is not None:
            cls_mask = torch.zeros((batch_size, 1), dtype=mask.dtype, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        for i, layer in enumerate(self.layers):
            features = layer(features, src_key_padding_mask=mask)

        return self.norm(features[:, 0, :])


if __name__ == "__main__":
    from config import Config
    from lightning_data import LightningData

    config = Config("SIMS")

    config.batch_size_eval = 1
    dm = LightningData(config)
    dm.setup(stage="validate")
    for batch in dm.val_dataloader():
        print(len(batch))
        print(batch)
        break

    model = LightningModel(config)
    model.eval()
