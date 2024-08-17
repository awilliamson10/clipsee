import torch
from torch.utils.data import Dataset
from vit_prisma.models.base_vit import HookedViT
from open_clip import tokenize
import datasets
from typing import Any, Iterator, cast
from torch.utils.data import DataLoader
from clipsee.config import SAETrainConfig

class HFDataset(Dataset):
    def __init__(self, data_location, transforms, image_col, text_col):
        self.dataset = datasets.load_dataset(data_location, split="train")
        self.image_col = image_col
        self.text_col = text_col
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Remove the extra dimension by squeezing the tensor
        images = self.transforms(self.dataset[idx][self.image_col], return_tensors="pt")["pixel_values"].squeeze(0)
        texts = tokenize([self.dataset[idx][self.text_col]])[0]
        return images, texts, self.dataset[idx][self.image_col], self.dataset[idx][self.text_col]

# Update the collate functions accordingly
def collate_fn(data):
    imgs, _, _, _ = zip(*data)
    return torch.stack(imgs, dim=0)

def collate_fn_eval(data):
    imgs, texts, raw_imgs, raw_texts = zip(*data)
    return torch.stack(imgs, dim=0), torch.stack(texts, dim=0), raw_imgs, raw_texts


class OpenCLIPActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs.
    """

    def __init__(
        self,
        config: SAETrainConfig,
        model: HookedViT,
        dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset = None,
        num_workers: int = 0,
    ):
        self.config = config
        assert (
            not self.config.normalize_activations
        ), "Normalize activations is currently not implemented for vision, sorry!"
        self.normalize_activations = self.config.normalize_activations
        self.model = model
        self.dataset = dataset
        self.eval_dataset = eval_dataset

        self.image_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=True,
            num_workers=num_workers,
            batch_size=self.config.store_batch_size,
            collate_fn=collate_fn,
            drop_last=True,
        )
        self.image_dataloader_eval = torch.utils.data.DataLoader(
            self.eval_dataset,
            shuffle=True,
            num_workers=num_workers,
            batch_size=self.config.store_batch_size,
            collate_fn=collate_fn_eval,
            drop_last=True,
        )

        self.image_dataloader_iter = self.get_batch_tokens_internal()
        self.image_dataloader_eval_iter = self.get_val_batch_tokens_internal()

        self.storage_buffer = self.get_buffer(self.config.n_batches_in_buffer // 2)
        self.dataloader = self.get_data_loader()


    def get_batch_tokens_internal(self):
        """
        Streams a batch of tokens from a dataset.
        """
        device = self.config.device
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
        device = self.config.device
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
            self.config.hook_point_layer
            if isinstance(self.config.hook_point_layer, list)
            else [self.config.hook_point_layer]
        )
        act_names = [self.config.hook_point.format(layer=layer) for layer in layers]
        hook_point_max_layer = max(layers)

        if self.config.hook_point_head_index is not None:
            layerwise_activations = self.model.run_with_cache(
                batch_tokens,
                names_filter=act_names,
                stop_at_layer=hook_point_max_layer + 1,
            )[1]
            activations_list = [
                layerwise_activations[act_name][:, :, self.config.hook_point_head_index]
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
        context_size = self.config.context_size
        batch_size = self.config.store_batch_size
        d_in = self.config.d_in
        total_size = batch_size * n_batches_in_buffer
        num_layers = (
            len(self.config.hook_point_layer)
            if isinstance(self.config.hook_point_layer, list)
            else 1
        )  # Number of hook points or layers

        refill_iterator = range(0, batch_size * n_batches_in_buffer, batch_size)
        # Initialize empty tensor buffer of the maximum required size with an additional dimension for layers
        new_buffer = torch.zeros(
            (total_size, context_size, num_layers, d_in),
            dtype=self.config.dtype,
            device=self.config.device,
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

    def get_data_loader(
        self,
    ) -> Iterator[Any]:
        """
        Return a torch.utils.dataloader which you can get batches from.

        Should automatically refill the buffer when it gets to n % full.
        (better mixing if you refill and shuffle regularly).

        """

        batch_size = self.config.train_batch_size_tokens

        # 1. # create new buffer by mixing stored and new buffer
        mixing_buffer = torch.cat(
            [self.get_buffer(self.config.n_batches_in_buffer // 2), self.storage_buffer], ####
            dim=0,
        )

        mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]

        # 2.  put 50 % in storage
        self.storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

        # 3. put other 50 % in a dataloader
        dataloader = iter(
            DataLoader(
                # TODO: seems like a typing bug?
                cast(Any, mixing_buffer[mixing_buffer.shape[0] // 2 :]),
                batch_size=batch_size,
                shuffle=True,
            )
        )

        return dataloader

    def next_batch(self):
        """
        Get the next batch from the current DataLoader.
        If the DataLoader is exhausted, refill the buffer and create a new DataLoader.
        """
        try:
            # Try to get the next batch
            return next(self.dataloader)
        except StopIteration:
            # If the DataLoader is exhausted, create a new one
            self.dataloader = self.get_data_loader() #### 97
            return next(self.dataloader)
