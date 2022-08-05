import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
from typing import Tuple


class Coverage(Dataset):
    def __init__(
        self, data_dir: str, image_transform=None, mask_transform=None
    ) -> None:
        super().__init__()

        # Fetch the image filenames.
        self._image_dir = os.path.join(data_dir, 'image')
        self._input_files = [f for f in os.listdir(self._image_dir) if '.tif' in f]

        # Fetch the mask filenames.
        self._mask_dir = os.path.join(data_dir, 'mask')
        self._output_files = [f for f in os.listdir(self._mask_dir) if '.tif' in f]

        # Create transform callables for raw images and masks.
        if image_transform is None:
            self._image_transform = transforms.Compose(
                [
                    transforms.Resize([256, 256]),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                ]
            )

        else:
            self._image_transform = image_transform

        if mask_transform is None:
            self._mask_transform = transforms.Compose(
                [
                    transforms.Resize([256, 256]),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                ]
            )
        else:
            self._mask_transform = mask_transform

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load the image.
        image_file = self._input_files[idx]
        image = Image.open(os.path.join(self._image_dir, image_file))
        image = self._image_transform(image)

        # Load the mask.
        mask_file = self._output_files[idx]
        mask = Image.open(os.path.join(self._mask_dir, mask_file))
        mask = self._mask_transform(mask)

        return image, mask

    def __len__(self) -> int:
        return len(self._input_files)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='COVERAGE dataset loader')
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to the COVERAGE dataset directory.',
    )
    args = parser.parse_args()

    dataset = Coverage(data_dir=args.data_dir)
    image, mask = dataset[0]
    print('Sample:', image.size(), mask.size())
    print('Number of samples:', len(dataset))


if __name__ == '__main__':
    main()
