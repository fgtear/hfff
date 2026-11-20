import os
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Timer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler
from utils.utils import seed_all
from config import Config
from lightning_data import LightningData
from lightning_model import LightningModel


def main(config):
    seed_all(config.seed)
    torch.set_float32_matmul_precision("high")

    mlf_logger = MLFlowLogger(
        tracking_uri="sqlite:///mlflow.db",
        experiment_name=config.dataset,
        run_name=config.get_hparams()["run_name"],
    )
    artifact_dir = f"./mlruns/{mlf_logger.experiment_id}/{mlf_logger.run_id}/artifacts"

    earlystop_callback = EarlyStopping(
        monitor=config.monitor,
        mode=config.monitor_mode,
        patience=config.early_stopping_patience,
        verbose=False,
        check_finite=True,
    )
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        dirpath=f"{artifact_dir}/checkpoints",
        monitor=config.monitor,
        mode=config.monitor_mode,
        save_top_k=1,
        every_n_epochs=1,
        filename="best_model",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    timer = Timer()

    trainer = L.Trainer(
        check_val_every_n_epoch=1,
        logger=mlf_logger,
        callbacks=[
            checkpoint_callback,
            earlystop_callback,
            lr_monitor,
            timer,
        ],
        log_every_n_steps=50,
        fast_dev_run=config.fast_dev_run,
        gradient_clip_val=config.gradient_clip_val,
        accelerator=config.accelerator,
        strategy=config.strategy,
        precision=config.precision,
        max_epochs=config.max_epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        num_sanity_val_steps=0,
    )

    model = LightningModel(config)
    dm = LightningData(config)

    trainer.fit(model=model, datamodule=dm)
    trainer.test(ckpt_path="best", datamodule=dm)

    print(f"训练总耗时: {timer.time_elapsed('train'):.2f} 秒")
    print(f"验证总耗时: {timer.time_elapsed('validate'):.2f} 秒")
    print(f"测试总耗时: {timer.time_elapsed('test'):.2f} 秒")

    if not config.save_model:
        os.remove(os.path.dirname(checkpoint_callback.best_model_path))

    del model
    del dm
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("MOSI", "MOSEI", "SIMS"), default="SIMS")
    parser.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ablation-mode", default="full", choices=(
        "full", "text_only", "audio_only", "visual_only", "cls_concat", "direct_fusion",
        "wo_text_query", "wo_av_query", "wo_aggregation", "wo_temporal_anchor",
    ))
    parser.add_argument("--visual-input", choices=("face", "frame"), default="face")
    args = parser.parse_args()
    config = Config(args.dataset)
    config.ablation_mode = args.ablation_mode
    config.visual_input = args.visual_input
    for seed in args.seed:
        config.seed = seed
        main(config)
