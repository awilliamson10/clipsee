from dataclasses import dataclass, field
from typing import List
from sae_lens.training.config import LanguageModelSAERunnerConfig

@dataclass
class SAETrainConfig(LanguageModelSAERunnerConfig):
    dataset_path: str = 'imagenet_data'
    num_workers: int = 0
    num_epochs: int = 5

    expansion_factor: int = 8
    context_size: int = 257
    d_in: int = 1024
    model_name: str = "awilliamson/clipora-pubchem"
    hook_point: str = "blocks.{layer}.hook_mlp_out"
    hook_point_layer: List[int] = field(default_factory=lambda: [21])
    dead_feature_window: int = 5000
    use_ghost_grads: bool = True
    feature_sampling_window: int = 5000
    from_pretrained_path: str = None

    b_dec_init_method: str = "geometric_median"
    normalize_sae_decoder: bool = True

    lr: float = 0.0003
    l1_coefficient: int = 0.15
    lr_scheduler_name: str = "constant"
    train_batch_size_tokens: int = 24
    lr_warm_up_steps: int = 0

    n_batches_in_buffer: int = 24
    store_batch_size: int = 24

    log_to_wandb: bool = True
    wandb_project: str = "clipora-chemsae"
    wandb_entity: str = "willfulbytes"
    wandb_log_frequency: int = 25
    eval_every_n_wandb_logs: int = 10
    run_name: str = None

    device: str = "cuda"
    seed: int = 42
    n_checkpoints: int = 10
    checkpoint_path: str = "chem"
    dtype: str = "torch.float32"