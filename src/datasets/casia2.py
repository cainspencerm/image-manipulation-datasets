import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image


class Casia2(Dataset):
    def __init__(self, data_dir=None, split=None):
        super().__init__()

        self._authentic_dir = os.path.join(data_dir, 'Au')
        self._tampered_dir = os.path.join(data_dir, 'Tp')

        auth_files = [f for f in os.listdir(self._authentic_dir) if '.tif' in f or '.jpg' in f]
        tamp_files = [f for f in os.listdir(self._tampered_dir) if '.tif' in f or '.jpg' in f]

        auth_split_size = len(auth_files) // 10
        tamp_split_size = len(tamp_files) // 10

        if split == 'train':
            self._input_files = auth_files[:auth_split_size * 8]
            self._input_files += tamp_files[:tamp_split_size * 8]

        elif split == 'val':
            self._input_files = auth_files[auth_split_size * 8 : auth_split_size * 9]
            self._input_files += tamp_files[tamp_split_size * 8 : tamp_split_size * 9]

        elif split == 'test':
            self._input_files = auth_files[auth_split_size * 9:]
            self._input_files += tamp_files[tamp_split_size * 9:]

        else:
            self._input_files = auth_files + tamp_files

        self._ground_truth_dir = os.path.join(data_dir, 'CASIA 2 Groundtruth')

        self._output_files = [f for f in os.listdir(self._ground_truth_dir) if f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png')]

        self._image_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ])

        self._mask_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.Grayscale(),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ])

    def __getitem__(self, idx):
        file = self._input_files[idx]
        if file.startswith('Au'):
            image = Image.open(os.path.join(self._authentic_dir, file))

            if image.mode != 'RGB':
                image = image.convert('RGB')

            image = self._image_transform(image)

            mask = torch.zeros(size=[1] + list(image.size()[1:]))

        else:
            image = Image.open(os.path.join(self._tampered_dir, file))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image = self._image_transform(image)

            tamp_id = file[-9:-4]
            for f in self._output_files:
                if tamp_id + '_gt' == f[-12:-4]:
                    gt_file = f
                    break

            mask = Image.open(os.path.join(self._ground_truth_dir, gt_file))
            mask = self._mask_transform(mask)

        return image, mask

    def __len__(self):
        return len(self._input_files)
