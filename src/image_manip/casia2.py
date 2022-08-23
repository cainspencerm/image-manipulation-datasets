import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from typing import Tuple
import numpy as np

from image_manip import utils


class Casia2(Dataset):
    '''CASIA V2 is a dataset for forgery classification. It contains 4795 images, 1701 authentic and 3274 forged.

    To download the dataset, please visit the following link:
    https://github.com/namtpham/casia2groundtruth

    Directory structure:
    CASIA 2.0
    ├── Au
    │   ├── Au_ani_00001.jpg
    │   ├── Au_ani_00002.jpg
    │   ├── ...
    │   └── Au_txt_30029.jpg
    ├── CASIA 2 Groundtruth
    │   ├── Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png
    │   ├── Tp_D_CND_M_N_art00076_art00077_10289_gt.png
    │   ├── ...
    │   └── Tp_S_NRN_S_O_sec00036_sec00036_00764_gt.png
    └── Tp
        ├── Tp_D_CND_M_N_ani00018_sec00096_00138.tif
        ├── Tp_D_CND_M_N_art00076_art00077_10289.tif
        ├── ...
        └── Tp_S_NRN_S_O_sec00036_sec00036_00764.tif

    Args:
        data_dir (str): The directory of the dataset.
        split (str): The split of the dataset. Must be 'train', 'valid', 'test',
            'benchmark', or 'full'.
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
                'for more information: https://github.com/namtpham/casia2groundtruth.'
            )

        # Fetch the image filenames.
        self._authentic_dir = os.path.join(data_dir, 'Au')
        auth_files = [
            os.path.abspath(os.path.join(self._authentic_dir, f))
            for f in os.listdir(self._authentic_dir)
            if '.tif' in f or '.jpg' in f
        ]
        auth_split_size = len(auth_files) // 10

        self._tampered_dir = os.path.join(data_dir, 'Tp')
        tamp_files = [
            os.path.abspath(os.path.join(self._tampered_dir, f))
            for f in os.listdir(self._tampered_dir)
            if '.tif' in f or '.jpg' in f
        ]
        tamp_split_size = len(tamp_files) // 10

        # Split the filenames into use cases.
        if split == 'train':
            self._input_files = auth_files[: auth_split_size * 8]
            self._input_files += tamp_files[: tamp_split_size * 8]

        elif split == 'valid':
            self._input_files = auth_files[auth_split_size * 8 : auth_split_size * 9]
            self._input_files += tamp_files[tamp_split_size * 8 : tamp_split_size * 9]

        elif split == 'test':
            self._input_files = auth_files[auth_split_size * 9 :]
            self._input_files += tamp_files[tamp_split_size * 9 :]

        elif split == 'benchmark':
            self._input_files = auth_files[:500]
            self._input_files += tamp_files[:500]

        elif split == 'full':
            self._input_files = auth_files + tamp_files

        else:
            raise ValueError('Unknown split: ' + split)

        # Ignore these files that have no ground truth masks.
        corrupted_files = [
            'Tp/Tp_D_NRD_S_N_cha10002_cha10001_20094.jpg',
            'Tp/Tp_S_NRD_S_N_arc20079_arc20079_01719.tif',
        ]

        remove_files = []
        for file in self._input_files:
            for f in corrupted_files:
                if f in file:
                    remove_files.append(file)

        for file in remove_files:
            self._input_files.remove(file)

        # Fetch the mask filenames.
        self._mask_dir = os.path.join(data_dir, 'CASIA 2 Groundtruth')
        mask_files = [
            f
            for f in os.listdir(self._mask_dir)
            if f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png')
        ]

        # Sort the output files based on the input files.
        self._output_files = []
        for file in self._input_files:
            tamp_id = file[-9:-4]
            mask = None
            for f in mask_files:
                if tamp_id + '_gt' == f[-12:-4]:
                    mask = f
                    break

            if mask is None and file.split('/')[-2] == 'Tp':
                raise ValueError('No ground truth file found for image: ' + file)

            mask_file = (
                os.path.abspath(os.path.join(self._mask_dir, mask))
                if mask is not None
                else None
            )
            self._output_files.append(mask_file)

        self.crop_size = crop_size
        self.pixel_range = pixel_range

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:

        minimum, maximum = self.pixel_range

        # Load the image.
        image_file = self._input_files[idx]
        image = Image.open(image_file)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Load the mask.
        mask_file = self._output_files[idx]

        if mask_file is None:
            image = np.array(image) * (maximum - minimum) / 255.0 + minimum
            image = utils.crop_or_pad(image, self.crop_size, pad_value=maximum)
            image = torch.from_numpy(image).permute(2, 0, 1)

            # An authentic image has no manipulation mask.
            mask = torch.zeros(size=[1] + list(image.size()[1:]))

        else:
            # Load the ground truth mask.
            mask = Image.open(mask_file)

            # Resize the mask to match the image.
            mask = mask.resize(image.size)

            if mask.mode != 'L':
                mask = mask.convert('L')

            image, mask = (
                np.array(image) * (maximum - minimum) / 255.0 + minimum,
                np.array(mask) / 255.0,
            )

            # Crop or pad the image and mask.
            image, mask = utils.crop_or_pad(
                [image, mask], self.crop_size, pad_value=[maximum, 1.0]
            )
            image, mask = torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(
                mask
            ).permute(2, 0, 1)

            image = image[:3]  # Remove the alpha channel if necessary.

        return image, mask

    def __len__(self):
        return len(self._input_files)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CASIA2 dataset loader')
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to the CASIA2 dataset directory.',
    )
    args = parser.parse_args()

    dataset = Casia2(data_dir=args.data_dir, split='benchmark')
    for image, mask in dataset:
        print('Sample:', image.size(), mask.size())
        break
    print('Number of samples:', len(dataset))


if __name__ == '__main__':
    main()
