import os
from typing import Tuple
import numpy as np

from image_manip import base


class IMD2020(base._BaseDataset):
    '''This dataset contains 2,010 real-life manipulated images downloaded from the
    Internet. Corresponding real versions of these images are also provided. Moreover,
    there is a manually created binary mask localizing the manipulated area of each
    manipulated image.

    To download the dataset, please visit the following link:
    http://staff.utia.cas.cz/novozada/db/

    Directory structure:
    IMD2020
    ├── 1a1ogs
    │   ├── 1a1ogs_orig.jpg
    │   ├── c8tf5mq_0.png
    │   └── c8tf5mq_0_mask.png
    ├── 1a3oag
    │   ├── 1a3oag_orig.jpg
    │   ├── c8tt7fg_0.jpg
    │   ├── ...
    │   └── c8u0wl4_0_mask.png
    ├── ...
    └── z41
        ├── 00109_fake.jpg
        ├── 00109_fake_mask.png
        └── 00109_orig.jpg

    Args:
        data_dir (str): The directory of the dataset.
        split (str): The split of the dataset. Must be 'train', 'valid', 'test',
            'benchmark', or 'full'.
        crop_size (tuple): The size of the crop to be applied on the image and mask.
        pixel_range (tuple): The range of the pixel values of the input images.
            Ex. (0, 1) scales the pixels from [0, 255] to [0, 1].
        shuffle (bool): Whether to shuffle the dataset before splitting.
        download (bool): Whether to download the dataset.
    '''

    def __init__(
        self,
        data_dir: str,
        split: str = 'full',
        crop_size: Tuple[int, int] = None,
        pixel_range: Tuple[float, float] = (0.0, 1.0),
        shuffle: bool = True,
        download: bool = False,
    ) -> None:
        super().__init__(crop_size, pixel_range)

        if download:
            raise NotImplementedError(
                'Downloading is not implemented yet due to the requirement of a '
                'browser to obtain the dataset. Please refer to the following link '
                'for more information: http://staff.utia.cas.cz/novozada/db/.'
            )

        subdirs = [
            os.path.join(data_dir, subdir)
            for subdir in os.listdir(data_dir)
            if '.' not in subdir
        ]

        # Fetch the authentic image filenames (they end in orig.jpg).
        image_files, mask_files = [], []
        for subdir in subdirs:
            for f in os.listdir(subdir):
                if 'orig' in f:
                    image_files.append(os.path.abspath(os.path.join(subdir, f)))
                    mask_files.append(None)
                elif 'mask' in f:
                    mask_file = os.path.abspath(os.path.join(subdir, f))
                    mask_files.append(mask_file)

                    # Locate the corresponding image file.
                    image_file = mask_file.replace('_mask', '')
                    if not os.path.exists(image_file):
                        image_file = image_file.replace('.png', '.jpg')
                        if not os.path.exists(image_file):
                            raise ValueError(
                                'Could not locate image for mask at {}'.format(
                                    mask_file
                                )
                            )
                    image_files.append(image_file)

        # Shuffle the image files for a random split.
        if shuffle:
            p = np.random.permutation(np.arange(len(image_files)))
            image_files = np.array(image_files)[p].tolist()
            mask_files = np.array(mask_files)[p].tolist()

        # Split the filenames into use cases.
        split_size = len(image_files) // 10
        if split == 'train':
            self.image_files = image_files[: split_size * 8]
            self.mask_files = mask_files[: split_size * 8]

        elif split == 'valid':
            self.image_files = image_files[split_size * 8 : split_size * 9]
            self.mask_files = mask_files[split_size * 8 : split_size * 9]

        elif split == 'test':
            self.image_files = image_files[split_size * 9 :]
            self.mask_files = mask_files[split_size * 9 :]

        elif split == 'benchmark':
            self.image_files = image_files[:500]
            self.mask_files = mask_files[:500]

        elif split == 'full':
            self.image_files = image_files
            self.mask_files = mask_files

        else:
            raise ValueError('Unknown split: ' + split)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='IMD2020 dataset loader')
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to the IMD2020 dataset directory.',
    )
    args = parser.parse_args()

    dataset = IMD2020(data_dir=args.data_dir)
    for image, mask in dataset:
        if mask is not None:
            print('Sample:', image.size(), mask.size())
        else:
            print('Sample:', image.size(), 'No mask')
        # break
    print('Number of samples:', len(dataset))


if __name__ == '__main__':
    main()
