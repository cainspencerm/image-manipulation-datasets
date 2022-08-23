import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from typing import Tuple
import numpy as np

import utils


class Splicing(Dataset):
    '''Digital image forensic has gained a lot of attention as it is becoming easier
    for anyone to make forged images. Several areas are concerned by image
    manipulation: a doctored image can increase the credibility of fake news, impostors
    can use morphed images to pretend being someone else.

    It became of critical importance to be able to recognize the manipulations suffered
    by the images. To do this, the first need is to be able to rely on reliable and
    controlled data sets representing the most characteristic cases encountered. The
    purpose of this work is to lay the foundations of a body of tests allowing both the
    qualification of automatic methods of authentication and detection of manipulations
    and the training of these methods.

    This dataset contains about 105000 splicing forgeries are available under the
    splicing directory. Each splicing is accompanied by two binary masks. One under the
    probemask subdirectory indicates the location of the forgery and one under the
    donormask indicates the location of the source. The external image can be found in
    the JSON file under the graph subdirectory.

    To download the dataset, please visit the following link:
    https://defactodataset.github.io

    Note: The dataset has an issue between the image and probe mask sizes. We must be
    careful in how the images and masks are handled. Since the images have manipulation
    statistics embedded in the pixels, any sort of aggregation function could damage or
    destroy the statistics. Therefore, we need to resize the masks to match the images.
    Then we need to crop the images and masks to the provided crop size. This will
    preserve any manipulation statistics while removing issues in the dataset.

    Directory structure:
    Defacto Splicing
    ├── splicing_1_annotations
    │   ├── donor_mask
    │   │   ├── 0_000000195755.tif
    │   │   ├── 1000_000000348782.tif
    │   │   ├── ...
    │   │   └── 9997_000000206363.tif
    │   ├── graph
    │   │   ├── 0_000000195755.json
    │   │   ├── 1000_000000348782.json
    │   │   ├── ...
    │   │   └── 9997_000000206363.json
    │   └── probe_mask
    │       ├── 0_000000195755.jpg
    │       ├── 1000_000000348782.jpg
    │       ├── ...
    │       └── 9997_000000206363.jpg
    ├── splicing_1_img
    │   └── img
    │       ├── 0_000000195755.tif
    │       ├── 1000_000000348782.tif
    │       ├── ...
    │       └── 9997_000000206363.tif
    ├── ...
    ├── splicing_7_annotations
    │   ├── donor_mask
    │   │   ├── 0_000000529545.tif
    │   │   ├── 100_000000343187.tif
    │   │   ├── ...
    │   │   └── 9999_000000476500.tif
    │   ├── graph
    │   │   ├── 0_000000529545.json
    │   │   ├── 100_000000343187.json
    │   │   ├── ...
    │   │   └── 9999_000000476500.json
    │   └── probe_mask
    │       ├── 0_000000529545.jpg
    │       ├── 100_000000343187.jpg
    │       ├── ...
    │       └── 9999_000000476500.jpg
    └── splicing_7_img
        └── img
            ├── 0_000000529545.tif
            ├── 100_000000343187.tif
            ├── ...
            └── 9999_000000476500.tif

    Args:
        data_dir (str): The directory of the dataset.
        split (str): The split of the dataset. Can be one of 'train', 'valid', 'test',
            'benchmark', and 'full'.
        crop_size (tuple): The size of the crop to be applied on the image and mask.
        pixel_range (tuple): The range of the pixel values of the input images.
            Ex. (0, 1) scales the pixels from [0, 255] to [0, 1].
        download (bool): Whether to download the dataset.
    '''

    def __init__(
        self,
        data_dir: str,
        split: str = 'full',
        crop_size: Tuple[int, int] = (256, 256),
        pixel_range: Tuple[float, float] = (0.0, 1.0),
        download: bool = False,
    ) -> None:
        super().__init__()

        if download:
            raise NotImplementedError(
                'Downloading is not implemented yet due to the requirement of a '
                'browser to obtain the dataset. Please refer to the following link '
                'for more information: https://defactodataset.github.io.'
            )

        # Fetch the image filenames.
        self._image_dirs = [
            os.path.join(data_dir, f'splicing_{i}_img', 'img') for i in range(1, 8)
        ]
        image_files = [
            os.path.abspath(os.path.join(shard, f))
            for shard in self._image_dirs
            for f in os.listdir(shard)
            if '.tif' in f
        ]

        split_size = len(image_files) // 10

        # Note that the order of the output files is aligned with the input files.
        if split == 'train':
            self._input_files = image_files[: split_size * 8]

        elif split == 'valid':
            self._input_files = image_files[split_size * 8 : split_size * 9]

        elif split == 'test':
            self._input_files = image_files[split_size * 9 :]

        elif split == 'benchmark':
            self._input_files = image_files[:1000]

        elif split == 'full':
            self._input_files = image_files

        else:
            raise ValueError(f'Unknown split: {split}')

        # Fetch the mask files.
        self._mask_dirs = [
            os.path.join(data_dir, f'splicing_{i}_annotations', 'probe_mask')
            for i in range(1, 8)
        ]

        self._output_files = []
        for f in self._input_files:
            shard = f.split('/')[-3].split('_')[-2]
            f = f.replace('.tif', '.jpg').split('/')[-1]
            self._output_files.append(
                os.path.abspath(os.path.join(self._mask_dirs[int(shard) - 1], f))
            )

        self.crop_size = crop_size
        self.pixel_range = pixel_range

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image_file = self._input_files[idx]
        image = Image.open(image_file)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        mask_file = self._output_files[idx]
        mask = Image.open(mask_file)

        if mask.mode != 'L':
            mask = mask.convert('L')

        # Resize the mask to match the image.
        mask = mask.resize(image.size[:2])

        # Normalize the image and mask.
        minimum, maximum = self.pixel_range
        image, mask = (
            np.array(image) * (maximum - minimum) / 255.0 + minimum,
            np.array(mask) / 255.0,
        )

        # Convert partially mixed pixel labels to manipulated pixel labels.
        mask = (mask > 0.0).astype(float)

        # Crop or pad the image and mask.
        image, mask = utils.crop_or_pad(
            [image, mask], self.crop_size, pad_value=[maximum, 1.0]
        )

        image, mask = torch.from_numpy(image), torch.from_numpy(mask)
        image, mask = torch.permute(image, (2, 0, 1)), torch.permute(mask, (2, 0, 1))

        return image.float() / 255.0, mask.float() / 255.0

    def __len__(self) -> int:
        return len(self._input_files)


class CopyMove(Dataset):
    '''Digital image forensic has gained a lot of attention as it is becoming easier
    for anyone to make forged images. Several areas are concerned by image
    manipulation: a doctored image can increase the credibility of fake news, impostors
    can use morphed images to pretend being someone else.

    It became of critical importance to be able to recognize the manipulations suffered
    by the images. To do this, the first need is to be able to rely on reliable and
    controlled data sets representing the most characteristic cases encountered. The
    purpose of this work is to lay the foundations of a body of tests allowing both the
    qualification of automatic methods of authentication and detection of manipulations
    and the training of these methods.

    This dataset contains about 19000 copy-move forgeries are available under the
    copymoveimg directory. Each copy-move is accompanied by two binary masks. One under
    the probemask subdirectory indicates the location of the forgery and one under the
    donor_mask indicates the location of the source within the image.

    To download the dataset, please visit the following link:
    https://defactodataset.github.io

    Note: The dataset has an issue between the image and probe mask sizes. We must be
    careful in how the images and masks are handled. Since the images have manipulation
    statistics embedded in the pixels, any sort of aggregation function could damage or
    destroy the statistics. Therefore, we need to resize the masks to match the images.
    Then we need to crop the images and masks to the provided crop size. This will
    preserve any manipulation statistics while removing issues in the dataset.

    Directory structure:
    Defacto CopyMove
    ├── copymove_annotations
    │   ├── donor_mask
    │   │   ├── 0_000000000071.tif
    │   │   ├── 0_000000000109.tif
    │   │   ├── ...
    │   │   └── 9_000000581177.tif
    │   ├── graph
    │   │   ├── 0_000000000071.json
    │   │   ├── 0_000000000109.json
    │   │   ├── ...
    │   │   └── 9_000000581177.json
    │   └── probe_mask
    │       ├── 0_000000000071.jpg
    │       ├── 0_000000000109.jpg
    │       ├── ...
    │       └── 9_000000581177.jpg
    └── copymove_img
        └── img
            ├── 0_000000000071.tif
            ├── 0_000000000109.tif
            ├── ...
            └── 9_000000581177.tif

    Args:
        data_dir (str): The directory of the dataset.
        split (str): The split of the dataset. Can be one of 'train', 'valid', 'test',
            'benchmark', and 'full'.
        crop_size (tuple): The size of the crops.
        pixel_range (tuple): The range of the pixel values of the input images.
            Ex. (0, 1) scales the pixels from [0, 255] to [0, 1].
        download (bool): Whether to download the dataset.
    '''

    def __init__(
        self,
        data_dir: str,
        split: str = 'full',
        crop_size: Tuple[int, int] = (256, 256),
        pixel_range: Tuple[float, float] = (0.0, 1.0),
        download: bool = False,
    ) -> None:
        super().__init__()

        if download:
            raise NotImplementedError(
                'Downloading is not implemented yet due to the requirement of a '
                'browser to obtain the dataset. Please refer to the following link '
                'for more information: https://defactodataset.github.io.'
            )

        # Fetch the image filenames.
        self._image_dir = os.path.join(data_dir, 'copymove_img', 'img')
        image_files = [
            os.path.abspath(os.path.join(self._image_dir, f))
            for f in os.listdir(self._image_dir)
            if f.endswith('.tif')
        ]

        split_size = len(image_files) // 10

        # Note that the order of the output files is aligned with the input files.
        if split == 'train':
            self._input_files = image_files[: split_size * 8]

        elif split == 'valid':
            self._input_files = image_files[split_size * 8 : split_size * 9]

        elif split == 'test':
            self._input_files = image_files[split_size * 9 :]

        elif split == 'benchmark':
            self._input_files = image_files[:1000]

        elif split == 'full':
            self._input_files = image_files

        else:
            raise ValueError(f'Unknown split: {split}')

        # Fetch the mask files.
        self._mask_dir = os.path.join(data_dir, 'copymove_annotations', 'probe_mask')

        self._output_files = []
        for f in self._input_files:
            f = f.replace('.tif', '.jpg').split('/')[-1]
            self._output_files.append(os.path.join(self._mask_dir, f))

        self.crop_size = crop_size
        self.pixel_range = pixel_range

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image_file = self._input_files[idx]
        image = Image.open(image_file)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        mask_file = self._output_files[idx]
        mask = Image.open(mask_file)

        if mask.mode != 'L':
            mask = mask.convert('L')

        # Resize the mask to match the image.
        mask = mask.resize(image.size[:2])

        # Normalize the image and mask.
        minimum, maximum = self.pixel_range
        image, mask = (
            np.array(image) * (maximum - minimum) / 255.0 + minimum,
            np.array(mask) / 255.0,
        )

        # Convert partially mixed pixel labels to manipulated pixel labels.
        mask = (mask > 0.0).astype(float)

        # Crop or pad the image and mask.
        image, mask = utils.crop_or_pad(
            [image, mask], self.crop_size, pad_value=[maximum, 1.0]
        )

        image, mask = torch.from_numpy(image), torch.from_numpy(mask)
        image, mask = torch.permute(image, (2, 0, 1)), torch.permute(mask, (2, 0, 1))

        return image.float() / 255.0, mask.float() / 255.0

    def __len__(self) -> int:
        return len(self._input_files)


class Inpainting(Dataset):
    '''Digital image forensic has gained a lot of attention as it is becoming easier
    for anyone to make forged images. Several areas are concerned by image
    manipulation: a doctored image can increase the credibility of fake news, impostors
    can use morphed images to pretend being someone else.

    It became of critical importance to be able to recognize the manipulations suffered
    by the images. To do this, the first need is to be able to rely on reliable and
    controlled data sets representing the most characteristic cases encountered. The
    purpose of this work is to lay the foundations of a body of tests allowing both the
    qualification of automatic methods of authentication and detection of manipulations
    and the training of these methods.

    This dataset contains about 25000 object-removal forgeries are available under the
    inpainting directory. Each object-removal is accompanied by two binary masks. One
    under the probemask subdirectory indicates the location of the forgery and one
    under the inpaintmask which is the mask use for the inpainting algorithm.

    To download the dataset, please visit the following link:
    https://defactodataset.github.io

    Note: The dataset has an issue between the image and probe mask sizes. We must be
    careful in how the images and masks are handled. Since the images have manipulation
    statistics embedded in the pixels, any sort of aggregation function could damage or
    destroy the statistics. Therefore, we need to resize the masks to match the images.
    Then we need to crop the images and masks to the provided crop size. This will
    preserve any manipulation statistics while removing issues in the dataset.

    Directory structure:
    Defacto Inpainting
    ├── inpainting_annotations
    │   ├── graph
    │   │   ├── 0_000000000260.json
    │   │   ├── 0_000000000332.json
    │   │   ├── ...
    │   │   └── 9_000000581887.json
    │   ├── inpaint_mask
    │   │   ├── 0_000000000260.tif
    │   │   ├── 0_000000000332.tif
    │   │   ├── ...
    │   │   └── 9_000000581887.tif
    │   └── probe_mask
    │       ├── 0_000000000260.jpg
    │       ├── 0_000000000332.jpg
    │       ├── ...
    │       └── 9_000000581887.jpg
    └── inpainting_img
        └── img
            ├── 0_000000000260.tif
            ├── 0_000000000332.tif
            ├── ...
            └── 9_000000581887.tif

    Args:
        data_dir (str): The directory of the dataset.
        split (str): The split of the dataset. Must be 'train', 'valid', 'test',
            'benchmark', or 'full'.
        crop_size (tuple): The size of the crops.
        pixel_range (tuple): The range of the pixel values of the input images.
            Ex. (0, 1) scales the pixels from [0, 255] to [0, 1].
        download (bool): Whether to download the dataset.
    '''

    def __init__(
        self,
        data_dir: str,
        split: str = 'full',
        crop_size: Tuple[int, int] = (256, 256),
        pixel_range: Tuple[float, float] = (0.0, 1.0),
        download: bool = False,
    ) -> None:
        super().__init__()

        if download:
            raise NotImplementedError(
                'Downloading is not implemented yet due to the requirement of a '
                'browser to obtain the dataset. Please refer to the following link '
                'for more information: https://defactodataset.github.io.'
            )

        # Fetch the image filenames.
        self._image_dir = os.path.join(data_dir, 'inpainting_img', 'img')
        image_files = [
            os.path.abspath(os.path.join(self._image_dir, f))
            for f in os.listdir(self._image_dir)
            if f.endswith('.tif')
        ]

        split_size = len(image_files) // 10

        # Note that the order of the output files is aligned with the input files.
        if split == 'train':
            self._input_files = image_files[: split_size * 8]

        elif split == 'valid':
            self._input_files = image_files[split_size * 8 : split_size * 9]

        elif split == 'test':
            self._input_files = image_files[split_size * 9 :]

        elif split == 'benchmark':
            self._input_files = image_files[:1000]

        elif split == 'full':
            self._input_files = image_files

        else:
            raise ValueError(f'Unknown split: {split}')

        # Fetch the mask files.
        self._mask_dir = os.path.join(data_dir, 'inpainting_annotations', 'probe_mask')

        self._output_files = []
        for f in self._input_files:
            f = f.split('/')[-1]
            self._output_files.append(os.path.abspath(os.path.join(self._mask_dir, f)))

        self.crop_size = crop_size
        self.pixel_range = pixel_range

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image_file = self._input_files[idx]
        image = Image.open(image_file)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        mask_file = self._output_files[idx]
        mask = Image.open(mask_file)

        if mask.mode != 'L':
            mask = mask.convert('L')

        # Resize the mask to match the image.
        mask = mask.resize(image.size[:2])

        # Normalize the image and mask.
        minimum, maximum = self.pixel_range
        image, mask = (
            np.array(image) * (maximum - minimum) / 255.0 + minimum,
            np.array(mask) / 255.0,
        )

        # Convert partially mixed pixel labels to manipulated pixel labels.
        mask = (mask > 0.0).astype(float)

        # Crop or pad the image and mask.
        image, mask = utils.crop_or_pad(
            [image, mask], self.crop_size, pad_value=[maximum, 1.0]
        )

        image, mask = torch.from_numpy(image), torch.from_numpy(mask)
        image, mask = torch.permute(image, (2, 0, 1)), torch.permute(mask, (2, 0, 1))

        return image, mask

    def __len__(self) -> int:
        return len(self._input_files)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Defacto dataset loader')
    parser.add_argument(
        '--copy-move-data-dir',
        type=str,
        default=None,
        help='Path to the CopyMove dataset directory.',
    )
    parser.add_argument(
        '--inpainting-data-dir',
        type=str,
        default=None,
        help='Path to the Inpainting dataset directory.',
    )
    parser.add_argument(
        '--splicing-data-dir',
        type=str,
        default=None,
        help='Path to the Splicing dataset directory.',
    )
    args = parser.parse_args()

    if (
        args.copy_move_data_dir is None
        and args.inpainting_data_dir is None
        and args.splicing_data_dir is None
    ):
        parser.error('At least one dataset directory must be specified.')

    if args.copy_move_data_dir is not None:
        dataset = CopyMove(data_dir=args.copy_move_data_dir, split='benchmark')
        for image, mask in dataset:
            print('Sample:', image.size(), mask.size())
            break
        print('Number of samples:', len(dataset))

    if args.inpainting_data_dir is not None:
        dataset = Inpainting(data_dir=args.inpainting_data_dir, split='benchmark')
        for image, mask in dataset:
            print('Sample:', image.size(), mask.size())
            break
        print('Number of samples:', len(dataset))

    if args.splicing_data_dir is not None:
        dataset = Splicing(data_dir=args.splicing_data_dir, split='benchmark')
        for image, mask in dataset:
            print('Sample:', image.size(), mask.size())
            break
        print('Number of samples:', len(dataset))


if __name__ == '__main__':
    main()
