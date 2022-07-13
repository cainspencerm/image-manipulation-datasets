import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image


class Coverage(Dataset):
    def __init__(self, data_dir=None):
        super().__init__()

        self._image_dir = os.path.join(data_dir, 'image')
        self._input_files = [f for f in os.listdir(self._image_dir) if '.tif' in f]

        self._mask_dir = os.path.join(data_dir, 'mask')
        self._output_files = [f for f in os.listdir(self._mask_dir) if '.tif' in f]

        # self._label_dir = os.path.join(data_dir, 'label')

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
        image_file = self._input_files[idx]
        image = Image.open(os.path.join(self._image_dir, image_file))
        image = self._image_transform(image)

        mask_file = self._output_files[idx]
        mask = Image.open(os.path.join(self._mask_dir, mask_file))
        mask = self._mask_transform(mask)

        return image, mask

    def __len__(self):
        return len(self._input_files)


def main():
    dataset = Coverage(data_dir='coverage')
    for image, mask in dataset:
        pass
    print(image.shape, mask.shape)

if __name__ == '__main__':
    main()
