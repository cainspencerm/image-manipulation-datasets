import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
from typing import Tuple, Callable


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
        image_transform (callable): The transform to be applied on the image.
        mask_transform (callable): The transform to be applied on the mask.
        download (bool): Whether to download the dataset.
    '''

    def __init__(
        self,
        data_dir: str,
        mask_type: str = 'forged',
        image_transform: Callable = None,
        mask_transform: Callable = None,
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
        self._input_files = [f for f in os.listdir(self._image_dir) if '.tif' in f]

        # Fetch the mask filenames in the correct order.
        self._mask_dir = os.path.join(data_dir, 'mask')
        mask_files = [f for f in os.listdir(self._mask_dir) if '.tif' in f]
        self._output_files = []
        for f in self._input_files:
            f_name = f.split('.')[0]
            if f_name[-1] == 't':
                mask_file = f_name[:-1] + mask_type + '.tif'
                assert mask_file in mask_files
            else:
                mask_file = None

            self._output_files.append(mask_file)

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
        # Load the image.
        image_file = self._input_files[idx]
        image = Image.open(os.path.join(self._image_dir, image_file))
        image = self._image_transform(image)

        # Load the mask.
        mask_file = self._output_files[idx]
        if mask_file is None:
            mask = torch.zeros_like(image)
        else:
            mask = Image.open(os.path.join(self._mask_dir, mask_file))
            mask = self._mask_transform(mask)

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
    image, mask = dataset[0]
    print('Sample:', image.size(), mask.size())
    print('Number of samples:', len(dataset))


if __name__ == '__main__':
    main()
