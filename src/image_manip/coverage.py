import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from typing import Tuple
import numpy as np

import utils


class Coverage(Dataset):
    '''The Copy-Move Forgery Database with Similar but Genuine Objects (COVERAGE) accompanies the following publication: "COVERAGE--A NOVEL DATABASE FOR COPY-MOVE FORGERY DETECTION," IEEE International Conference on Image processing (ICIP), 2016.

    COVERAGE contains copymove forged (CMFD) images and their originals with similar but genuine objects (SGOs). COVERAGE is designed to highlight and address tamper detection ambiguity of popular methods, caused by self-similarity within natural images. In COVERAGE, forged-original pairs are annotated with (i) the duplicated and forged region masks, and (ii) the tampering factor/similarity metric. For benchmarking, forgery quality is evaluated using (i) computer vision-based methods, and (ii) human detection performance.

    To download the dataset, please visit the following link:
    https://github.com/wenbihan/coverage

    Directory structure:
    COVERAGE
    ├── image
    │   ├── 1.tif
    │   ├── 1t.tif
    │   ├── ...
    │   ├── 100.tif
    │   └── 100t.tif
    ├── label
    │   ├── ...  # Not implemented.
    ├── mask
    │   ├── 1copy.tif
    │   ├── 1forged.tif
    │   ├── 1paste.tif
    │   ├── ...
    │   ├── 100copy.tif
    │   ├── 100forged.tif
    │   └── 100paste.tif
    └── readme.txt

    Args:
        data_dir (str): The directory of the dataset.
        mask_type (str): The type of mask to use. Must be 'forged', 'copy', or 'paste'.
        crop_size (tuple): The size of the crop to be applied on the image and mask.
        pixel_range (tuple): The range of the pixel values of the input images.
            Ex. (0, 1) scales the pixels from [0, 255] to [0, 1].
        download (bool): Whether to download the dataset.
    '''

    def __init__(
        self,
        data_dir: str,
        mask_type: str = 'forged',
        crop_size: Tuple[int, int] = (256, 256),
        pixel_range: Tuple[float, float] = (0.0, 1.0),
        download: bool = False,
    ) -> None:
        super().__init__()

        assert mask_type in ['forged', 'copy', 'paste']

        if download:
            raise NotImplementedError(
                'Downloading is not implemented yet due to the requirement of a '
                'browser to obtain the dataset. Please refer to the following link '
                'for more information: https://github.com/wenbihan/coverage.'
            )

        # Fetch the image filenames.
        self._image_dir = os.path.join(data_dir, 'image')
        self._input_files = [
            os.path.abspath(os.path.join(self._image_dir, f))
            for f in os.listdir(self._image_dir)
            if '.tif' in f
        ]

        # Fetch the mask filenames in the correct order.
        self._mask_dir = os.path.abspath(os.path.join(data_dir, 'mask'))
        mask_files = [
            os.path.abspath(os.path.join(self._mask_dir, f))
            for f in os.listdir(self._mask_dir)
            if '.tif' in f
        ]
        self._output_files = []
        for f in self._input_files:
            f_name = f.split('.')[0]
            if f_name[-1] == 't':
                mask_file = f_name.split('/')[-1][:-1] + mask_type + '.tif'
                mask_file = os.path.abspath(os.path.join(self._mask_dir, mask_file))
                assert mask_file in mask_files
            else:
                mask_file = None

            self._output_files.append(mask_file)

        self.crop_size = crop_size
        self.pixel_range = pixel_range

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load the image.
        image_file = self._input_files[idx]
        image = Image.open(image_file)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        minimum, maximum = self.pixel_range

        # Load the mask.
        mask_file = self._output_files[idx]

        if mask_file is None:
            mask = torch.zeros(self.crop_size).unsqueeze(dim=0)

            # Crop or pad the image.
            image = np.array(image) * (maximum - minimum) / 255.0 + minimum
            image = utils.crop_or_pad(image, self.crop_size, pad_value=maximum)

            image = torch.from_numpy(image).permute(2, 0, 1)

        else:
            mask = Image.open(mask_file)

            if mask.mode != 'L':
                mask = mask.convert('L')

            # Resize the mask to match the image.
            mask = mask.resize(image.size)

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
    for image, mask in dataset:
        print('Sample:', image.size(), mask.size())
        break
    print('Number of samples:', len(dataset))


if __name__ == '__main__':
    main()
