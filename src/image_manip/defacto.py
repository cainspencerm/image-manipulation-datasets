import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
from typing import Tuple, Callable


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
                'for more information: https://defactodataset.github.io.'
            )

        # Fetch the image filenames.
        self._image_dirs = [
            os.path.join(data_dir, f'splicing_{i}_img', 'img') for i in range(1, 8)
        ]
        image_files = [
            os.path.join(shard, f)
            for shard in self._image_dirs
            for f in os.listdir(shard)
            if '.tif' in f
        ]

        # Fetch the mask filenames.
        self._mask_dirs = [
            os.path.join(data_dir, f'splicing_{i}_annotations', 'probe_mask')
            for i in range(1, 8)
        ]
        mask_files = [
            os.path.join(shard, f)
            for shard in self._mask_dirs
            for f in os.listdir(shard)
            if '.jpg' in f
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

        # Fetch the ground truth filenames.
        self._output_files = []
        for f in self._input_files:
            if_id = f.split('.')[0].split('_')[-1]
            for mask_file in mask_files:
                if if_id + '.jpg' in mask_file:
                    self._output_files.append(mask_file)
                    break
            assert self._output_files[-1] == mask_file

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
        image_file = self._input_files[idx]
        image = Image.open(image_file)
        image = self._image_transform(image)

        mask_file = self._output_files[idx]
        mask = Image.open(mask_file)
        mask = self._mask_transform(mask)

        return image, mask

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
                'for more information: https://defactodataset.github.io.'
            )

        # Fetch the image filenames.
        self._image_dir = os.path.join(data_dir, 'copymove_img', 'img')
        image_files = [f for f in os.listdir(self._image_dir) if f.endswith('.tif')]

        # Fetch the mask filenames.
        self._mask_dir = os.path.join(data_dir, 'copymove_annotations', 'probe_mask')
        mask_files = [f for f in os.listdir(self._mask_dir) if f.endswith('.jpg')]

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

        # Fetch the ground truth filenames.
        self._output_files = []
        for f in self._input_files:
            if_id = f.split('.')[0]
            idx = mask_files.index(if_id + '.jpg')
            self._output_files.append(mask_files[idx])

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
        image_file = self._input_files[idx]
        image = Image.open(os.path.join(self._image_dir, image_file))
        image = self._image_transform(image)

        mask_file = self._output_files[idx]
        mask = Image.open(os.path.join(self._mask_dir, mask_file))
        mask = self._mask_transform(mask)

        return image, mask

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
                'for more information: https://defactodataset.github.io.'
            )

        # Fetch the image filenames.
        self._image_dir = os.path.join(data_dir, 'inpainting_img', 'img')
        image_files = [f for f in os.listdir(self._image_dir) if f.endswith('.tif')]

        # Fetch the ground truth filenames.
        self._mask_dir = os.path.join(data_dir, 'inpainting_annotations', 'probe_mask')
        mask_files = [f for f in os.listdir(self._mask_dir) if f.endswith('.tif')]

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

        # Fetch the ground truth filenames.
        self._output_files = []
        for f in self._input_files:
            if_id = f.split('.')[0]
            idx = mask_files.index(if_id + '.tif')
            self._output_files.append(mask_files[idx])

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
        image_file = self._input_files[idx]
        image = Image.open(os.path.join(self._image_dir, image_file))
        image = self._image_transform(image)

        mask_file = self._output_files[idx]
        mask = Image.open(os.path.join(self._mask_dir, mask_file))
        mask = self._mask_transform(mask)

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
        image, mask = dataset[0]
        print('Sample:', image.size(), mask.size())
        print('Number of samples:', len(dataset))

    if args.inpainting_data_dir is not None:
        dataset = Inpainting(data_dir=args.inpainting_data_dir, split='benchmark')
        image, mask = dataset[0]
        print('Samples:', image.size(), mask.size())
        print('Number of samples:', len(dataset))

    if args.splicing_data_dir is not None:
        dataset = Splicing(data_dir=args.splicing_data_dir, split='benchmark')
        image, mask = dataset[0]
        print('Samples:', image.size(), mask.size())
        print('Number of samples:', len(dataset))


if __name__ == '__main__':
    main()
