import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
from typing import Tuple, Callable


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
        image_transform (callable): The transform to be applied on the image.
        mask_transform (callable): The transform to be applied on the mask.
        download (bool): Whether to download the dataset.
    '''

    def __init__(
        self,
        data_dir: str,
        split: str = 'full',
        image_transform: Callable = None,
        mask_transform: Callable = None,
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
            f for f in os.listdir(self._authentic_dir) if '.tif' in f or '.jpg' in f
        ]
        auth_split_size = len(auth_files) // 10

        self._tampered_dir = os.path.join(data_dir, 'Tp')
        tamp_files = [
            f for f in os.listdir(self._tampered_dir) if '.tif' in f or '.jpg' in f
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

        # Fetch the ground truth filenames.
        self._ground_truth_dir = os.path.join(data_dir, 'CASIA 2 Groundtruth')
        self._output_files = [
            f
            for f in os.listdir(self._ground_truth_dir)
            if f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png')
        ]

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

        file = self._input_files[idx]

        if file.startswith('Au'):
            # Load the image.
            image = Image.open(os.path.join(self._authentic_dir, file))

            if image.mode != 'RGB':
                image = image.convert('RGB')

            image = self._image_transform(image)

            # An authentic image has no manipulation mask.
            mask = torch.zeros(size=[1] + list(image.size()[1:]))

        elif file.startswith('Tp'):
            # Load the image.
            image = Image.open(os.path.join(self._tampered_dir, file))
            image = self._image_transform(image)

            # Find the corresponding ground truth file.
            tamp_id = file[-9:-4]
            gt_file = None
            for f in self._output_files:
                if tamp_id + '_gt' == f[-12:-4]:
                    gt_file = f
                    break

            if gt_file is None:
                raise ValueError('No ground truth file found for image: ' + file)

            # Load the ground truth mask.
            mask = Image.open(os.path.join(self._ground_truth_dir, gt_file))

            if mask.mode != 'L':
                mask = mask.convert('L')

            mask = self._mask_transform(mask)

        image = image[:3]

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
    image, mask = dataset[0]
    print('Sample:', image.size(), mask.size())
    print('Number of samples:', len(dataset))


if __name__ == '__main__':
    main()
