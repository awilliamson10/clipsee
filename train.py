import wandb
from sae.train import train_sae_group_on_vision_model
from typing import Any, cast
from sae_lens.training.sae_group import SparseAutoencoderDictionary
from transformers import CLIPProcessor
from clipsee.config import SAETrainConfig
from clipsee.dataloader import OpenCLIPActivationsStore, HFDataset
from vit_prisma.models.base_vit import HookedViT

def main():
    config = SAETrainConfig(
        model_name="laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
        expansion_factor=16,
        lr=0.0003,
        l1_coefficient=0.1,
        lr_warm_up_steps=500,
        l1_warm_up_steps=500,
        n_batches_in_buffer=24,
        store_batch_size_prompts=24,
        train_batch_size_tokens=24,
        wandb_project="fashion-sae",
        checkpoint_path="fashion-sae",
        seed=1337,
    )

    processor = CLIPProcessor.from_pretrained(config.model_name)
    dataset = HFDataset("ashraq/fashion-product-images-small", processor.image_processor, "image", "productDisplayName") # load_dataset("awilliamson/fashion-train", split="train")
    eval_dataset = HFDataset("ashraq/fashion-product-images-small", processor.image_processor, "image", "productDisplayName") # load_dataset("awilliamson/fashion-validation", split="train")
    # cfg.training_tokens = int(1_300_000*setup_args['num_epochs']) * cfg.context_size
    config.training_tokens = len(dataset) * config.num_epochs
    sae_group = SparseAutoencoderDictionary(config)
    model = HookedViT.from_pretrained(config.model_name, is_timm=False, is_clip=True)
    model.to(config.device)

    activation_store = OpenCLIPActivationsStore(
        config = config,
        model = model,
        dataset = dataset,
        eval_dataset = eval_dataset,
        num_workers = 0,
    )

    for i, (name, sae) in enumerate(sae_group):
        hyp = sae.cfg
        print(
            f"{i}: Name: {name} Layer {hyp.hook_point_layer}, p_norm {hyp.lp_norm}, alpha {hyp.l1_coefficient}"
        )

    if config.log_to_wandb:
        wandb.init(project=config.wandb_project, config=cast(Any, config), name=config.run_name)

    train_sae_group_on_vision_model(
        model,
        sae_group,
        activation_store,
        train_contexts=None, #TODO load checkpoints correctly to match saelens v2.1.3 lm_runner!
        training_run_state=None,  #TODO load checkpoints correctly to match saelens v2.1.3 lm_runner!
        n_checkpoints=config.n_checkpoints,
        batch_size=config.train_batch_size_tokens,
        feature_sampling_window=config.feature_sampling_window,
        use_wandb=config.log_to_wandb,
        wandb_log_frequency=config.wandb_log_frequency,
        eval_every_n_wandb_logs=config.eval_every_n_wandb_logs,
        autocast=config.autocast,
    )
    wandb.finish()


if __name__ == "__main__":
    main()


