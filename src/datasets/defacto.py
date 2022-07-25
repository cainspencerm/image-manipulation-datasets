import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image


class Splicing(Dataset):
    def __init__(self, data_dir: str='data', split: str='full', image_transform=None, mask_transform=None):
        super().__init__()

        # Fetch the image filenames.
        self._image_dirs = [os.path.join(data_dir, f'splicing_{i}_img', 'img') for i in range(1, 8)]
        image_files = [os.path.join(shard, f) for shard in self._image_dirs for f in os.listdir(shard) if '.tif' in f]

        # Fetch the mask filenames.
        self._mask_dirs = [os.path.join(data_dir, f'splicing_{i}_annotations', 'probe_mask') for i in range(1, 8)]
        mask_files = [os.path.join(shard, f) for shard in self._mask_dirs for f in os.listdir(shard) if '.jpg' in f]

        split_size = len(image_files) // 10

        # Note that the order of the output files is aligned with the input files.
        if split == 'train':
            self._input_files = image_files[:split_size * 8]

        elif split == 'valid':
            self._input_files = image_files[split_size * 8 : split_size * 9]

        elif split == 'test':
            self._input_files = image_files[split_size * 9:]

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
            self._image_transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ])

        else:
            self._image_transform = image_transform

        if mask_transform is None:
            self._mask_transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ])
        else:
            self._mask_transform = mask_transform

    def __getitem__(self, idx):
        image_file = self._input_files[idx]
        image = Image.open(image_file)
        image = self._image_transform(image)

        mask_file = self._output_files[idx]
        mask = Image.open(mask_file)
        mask = self._mask_transform(mask)

        return image, mask

    def __len__(self):
        return len(self._input_files)


class CopyMove(Dataset):
    def __init__(self, data_dir: str='data', split: str='full', image_transform=None, mask_transform=None):
        super().__init__()

        # Fetch the image filenames.
        self._image_dir = os.path.join(data_dir, 'copymove_img', 'img')
        image_files = [f for f in os.listdir(self._image_dir) if f.endswith('.tif')]

        # Fetch the mask filenames.
        self._mask_dir = os.path.join(data_dir, 'copymove_annotations', 'probe_mask')
        mask_files = [f for f in os.listdir(self._mask_dir) if f.endswith('.jpg')]

        split_size = len(image_files) // 10

        # Note that the order of the output files is aligned with the input files.
        if split == 'train':
            self._input_files = image_files[:split_size * 8]

        elif split == 'valid':
            self._input_files = image_files[split_size * 8 : split_size * 9]

        elif split == 'test':
            self._input_files = image_files[split_size * 9:]

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
            self._image_transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ])

        else:
            self._image_transform = image_transform

        if mask_transform is None:
            self._mask_transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ])
        else:
            self._mask_transform = mask_transform

    def __getitem__(self, idx):
        image_file = self._input_files[idx]
        image = Image.open(os.path.join(self._image_dir, image_file))
        image = self._image_transform(image)

        mask_file = self._output_files[idx]
        mask = Image.open(os.path.join(self._mask_dir, mask_file))
        mask = self._mask_transform(mask)

        return image, mask

    def __len__(self):
        return len(self._input_files)


class Inpainting(Dataset):
    def __init__(self, data_dir: str='data', split: str='full', image_transform=None, mask_transform=None):
        super().__init__()

        # Fetch the image filenames.
        self._image_dir = os.path.join(data_dir, 'inpainting_img', 'img')
        image_files = [f for f in os.listdir(self._image_dir) if f.endswith('.tif')]

        # Fetch the ground truth filenames.
        self._mask_dir = os.path.join(data_dir, 'inpainting_annotations', 'probe_mask')
        mask_files = [f for f in os.listdir(self._mask_dir) if f.endswith('.tif')]

        split_size = len(image_files) // 10

        # Note that the order of the output files is aligned with the input files.
        if split == 'train':
            self._input_files = image_files[:split_size * 8]

        elif split == 'valid':
            self._input_files = image_files[split_size * 8 : split_size * 9]

        elif split == 'test':
            self._input_files = image_files[split_size * 9:]

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
            self._image_transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ])

        else:
            self._image_transform = image_transform

        if mask_transform is None:
            self._mask_transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ])
        else:
            self._mask_transform = mask_transform

    def __getitem__(self, idx):
        image_file = self._input_files[idx]
        image = Image.open(os.path.join(self._image_dir, image_file))
        image = self._image_transform(image)

        mask_file = self._output_files[idx]
        mask = Image.open(os.path.join(self._mask_dir, mask_file))
        mask = self._mask_transform(mask)

        return image, mask

    def __len__(self):
        return len(self._input_files)