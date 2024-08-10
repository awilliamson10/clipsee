import os
from typing import Any, Iterator, cast

import torch
from torch.utils.data import DataLoader
from vit_prisma.models.base_vit import HookedViT


def collate_fn(data):
    imgs = [d[0] for d in data]
    return torch.stack(imgs, dim=0)


def collate_fn_eval(data):
    imgs = [d[0] for d in data]
    return torch.stack(imgs, dim=0), torch.tensor([d[1] for d in data])

def transforms(image_processor, text_processor):
    def transform_fn(item):
        image, label = item
        image = image_processor(image)
        text = text_processor(label)
        return image, text
    return transform_fn

class OpenCLIPActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs.
    """

    def __init__(
        self,
        config: SAETrainConfig,
        model: HookedViT,
        processor: CLIPProcessor,
        dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset = None,
        num_workers: int = 0,
    ):
        self.config = config
        assert (
            not self.cfg.normalize_activations
        ), "Normalize activations is currently not implemented for vision, sorry!"
        self.normalize_activations = self.config.normalize_activations
        self.model = model
        self.dataset = dataset
        self.processor = processor

        self.dataset.set_transforms(
            transforms(
                self.processor.image_processor
            )
        )

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=True,
            num_workers=num_workers,
            batch_size=self.config.store_batch_size,
            collate_fn=collate_fn,
            drop_last=True,
        )
        self.dataloader_eval = torch.utils.data.DataLoader(
            eval_dataset,
            shuffle=True,
            num_workers=num_workers,
            batch_size=self.config.store_batch_size,
            collate_fn=collate_fn_eval,
            drop_last=True,
        )

        self.dataloader_iter = self.get_batch_tokens_internal()
        self.dataloader_eval_iter = self.get_val_batch_tokens_internal()


    def get_batch_tokens_internal(self):
        """
        Streams a batch of tokens from a dataset.
        """
        device = self.cfg.device
        while True:
            for data in self.image_dataloader:
                data.requires_grad_(False)
                yield data.to(device)  # 5

    def get_batch_tokens(self):
        return next(self.image_dataloader_iter)

    # returns the ground truth class as well.
    def get_val_batch_tokens_internal(self):
        """
        Streams a batch of tokens from a dataset.
        """
        device = self.cfg.device
        while True:
            for image_data, labels in self.image_dataloader_eval:
                image_data.requires_grad_(False)
                labels.requires_grad_(False)
                yield image_data.to(device), labels.to(device)

    def get_val_batch_tokens(self):
        return next(self.image_dataloader_eval_iter)

    def get_activations(self, batch_tokens: torch.Tensor, get_loss: bool = False):
        """
        Returns activations of shape (batches, context, num_layers, d_in)
        """
        layers = (
            self.cfg.hook_point_layer
            if isinstance(self.cfg.hook_point_layer, list)
            else [self.cfg.hook_point_layer]
        )
        act_names = [self.cfg.hook_point.format(layer=layer) for layer in layers]
        hook_point_max_layer = max(layers)

        if self.cfg.hook_point_head_index is not None:
            layerwise_activations = self.model.run_with_cache(
                batch_tokens,
                names_filter=act_names,
                stop_at_layer=hook_point_max_layer + 1,
            )[1]
            activations_list = [
                layerwise_activations[act_name][:, :, self.cfg.hook_point_head_index]
                for act_name in act_names
            ]
        else:
            layerwise_activations = self.model.run_with_cache(  ####
                batch_tokens,
                names_filter=act_names,
                stop_at_layer=hook_point_max_layer + 1,
            )[1]
            activations_list = [
                layerwise_activations[act_name] for act_name in act_names
            ]

        # Stack along a new dimension to keep separate layers distinct
        stacked_activations = torch.stack(activations_list, dim=2)

        return stacked_activations

    def get_buffer(self, n_batches_in_buffer: int):
        context_size = self.cfg.context_size
        batch_size = self.cfg.store_batch_size
        d_in = self.cfg.d_in
        total_size = batch_size * n_batches_in_buffer
        num_layers = (
            len(self.cfg.hook_point_layer)
            if isinstance(self.cfg.hook_point_layer, list)
            else 1
        )  # Number of hook points or layers

        refill_iterator = range(0, batch_size * n_batches_in_buffer, batch_size)
        # Initialize empty tensor buffer of the maximum required size with an additional dimension for layers
        new_buffer = torch.zeros(
            (total_size, context_size, num_layers, d_in),
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )

        for refill_batch_idx_start in refill_iterator:
            refill_batch_tokens = self.get_batch_tokens()  ######
            refill_activations = self.get_activations(refill_batch_tokens)

            new_buffer[
                refill_batch_idx_start : refill_batch_idx_start + batch_size, ...
            ] = refill_activations

            # pbar.update(1)

        new_buffer = new_buffer.reshape(-1, num_layers, d_in)
        new_buffer = new_buffer[torch.randperm(new_buffer.shape[0])]

        return new_buffer