import datetime
import copy


class Config:
    def __init__(self, dataset):
        self.dataset = dataset

        self.strategy = "auto"
        self.accelerator = "gpu"
        self.num_workers = 8
        self.precision = "bf16-mixed"
        self.gradient_checkpointing = False
        self.fast_dev_run = 0
        self.enable_profiler = False

        self.save_model = True
        self.monitor = "metrics/MAE_val"
        self.monitor_mode = "min"

        self.max_epochs = 3 if dataset == "MOSEI" else 16
        self.early_stopping_patience = 6
        self.seed = 0
        self.batch_size_train = 1 if dataset == "MOSEI" else 2
        self.accumulate_grad_batches = 32 if dataset == "MOSEI" else 8
        self.audio_bucket_size = 100
        self.batch_size_eval = 8
        self.backbone_learning_rate = 1e-5
        self.task_learning_rate = 5e-5
        self.weight_decay = 0.01
        self.dropout = 0.2
        self.gradient_clip_val = 1

        self.enable_temporal_anchor = True
        self.temporal_anchor_mode = "pooled"
        self.temporal_anchor_sigma = 0.25
        self.temporal_anchor_strength = 4.5
        self.ablation_mode = "full"
        self.visual_input = "face"
        self.data_time = datetime.datetime.now().strftime("%m%d%H%M")

        if self.dataset in ["MOSI", "MOSEI"]:
            self.text_extractor = "FacebookAI/roberta-large"
            self.audio_extractor = "facebook/hubert-large-ls960-ft"
        elif self.dataset in ["SIMS", "SIMSV2"]:
            self.text_extractor = "clue/roberta_chinese_large"
            self.audio_extractor = "TencentGameMate/chinese-hubert-large"
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        self.visual_extractor = "google/vivit-b-16x2-kinetics400"

    def get_hparams(self):
        dic = copy.deepcopy(self.__dict__)
        if self.ablation_mode == "wo_temporal_anchor":
            dic["enable_temporal_anchor"] = False
        dic["run_name"] = f"{self.ablation_mode}-{self.seed}"
        if not self.enable_temporal_anchor:
            dic["run_name"] = "wo_temporal_anchor"
        elif self.temporal_anchor_mode == "uniform" and self.ablation_mode in ("full", "wo_text_query", "wo_aggregation"):
            dic["run_name"] += "_uniform_anchor"
        if self.visual_input == "frame":
            dic["run_name"] += f"_{self.visual_input}"

        dic.pop("strategy")
        dic.pop("accelerator")
        dic.pop("num_workers")
        dic.pop("data_time")
        dic.pop("monitor")
        dic.pop("monitor_mode")
        dic.pop("fast_dev_run")
        dic.pop("enable_profiler")
        return dic


if __name__ == "__main__":
    config = Config("SIMS")
    print("data_time", config.data_time)
