import torch
from torch.utils import data
from typing import Tuple
from PIL import Image
import numpy as np

from image_manip import utils


class _BaseDataset(data.Dataset):
    def __init__(
        self,
        crop_size: Tuple[int, int],
        pixel_range: Tuple[float, float],
    ):
        super().__init__()

        self.crop_size = crop_size
        self.pixel_range = pixel_range

        # Need to define these in the child class.
        self.image_files = None
        self.mask_files = None

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:

        # Load the image file.
        image_file = self.image_files[idx]
        image = Image.open(image_file)

        # Force three color channels.
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Load the mask file.
        mask_file = self.mask_files[idx]
        pixel_min, pixel_max = self.pixel_range

        if mask_file is None:

            # The mask doesn't exist; assume it has no manipulated pixels.
            crop_size = self.crop_size if self.crop_size is not None else image.size
            mask = torch.zeros(crop_size).unsqueeze(dim=0)

            # Normalize the image.
            image = np.array(image) * (pixel_max - pixel_min) / 255.0 + pixel_min

            # Crop or pad the image.
            image = utils.crop_or_pad(image, crop_size, pad_value=pixel_max)

            # Convert the image to a tensor.
            image = torch.from_numpy(image).permute(2, 0, 1)

        else:

            # Load the mask.
            mask = Image.open(mask_file)

            # Force one color channel.
            if mask.mode != 'L':
                mask = mask.convert('L')

            # Resize the mask to match the image.
            mask = mask.resize(image.size[:2])

            # Normalize the image and mask.
            image = np.array(image) * (pixel_max - pixel_min) / 255.0 + pixel_min
            mask = np.array(mask) / 255.0

            # Convert partially mixed pixel labels to manipulated pixel labels.
            mask = (mask > 0.0).astype(float)

            # Crop or pad the image and mask.
            crop_size = (
                self.crop_size if self.crop_size is not None else image.shape[:2]
            )
            image, mask = utils.crop_or_pad(
                [image, mask], crop_size, pad_value=[pixel_max, 1.0]
            )

            # Convert the image and mask to tensors.
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask).permute(2, 0, 1)

        return image, mask

    def __len__(self) -> int:
        return len(self.image_files)
