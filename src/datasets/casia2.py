import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image


class Casia2(Dataset):
    def __init__(self, data_dir: str='data', split: str='full', image_transform=None, mask_transform=None) -> None:
        super().__init__()

        # Fetch the image filenames.
        self._authentic_dir = os.path.join(data_dir, 'Au')
        auth_files = [f for f in os.listdir(self._authentic_dir) if '.tif' in f or '.jpg' in f]
        auth_split_size = len(auth_files) // 10

        self._tampered_dir = os.path.join(data_dir, 'Tp')
        tamp_files = [f for f in os.listdir(self._tampered_dir) if '.tif' in f or '.jpg' in f]
        tamp_split_size = len(tamp_files) // 10

        # Split the filenames into use cases.
        if split == 'train':
            self._input_files = auth_files[:auth_split_size * 8]
            self._input_files += tamp_files[:tamp_split_size * 8]

        elif split == 'valid':
            self._input_files = auth_files[auth_split_size * 8 : auth_split_size * 9]
            self._input_files += tamp_files[tamp_split_size * 8 : tamp_split_size * 9]

        elif split == 'test':
            self._input_files = auth_files[auth_split_size * 9:]
            self._input_files += tamp_files[tamp_split_size * 9:]

        elif split == 'full':
            self._input_files = auth_files + tamp_files

        # Fetch the ground truth filenames.
        self._ground_truth_dir = os.path.join(data_dir, 'CASIA 2 Groundtruth')
        self._output_files = [f for f in os.listdir(self._ground_truth_dir) if f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png')]

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
