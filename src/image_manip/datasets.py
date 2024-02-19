import torch
from torch.utils import data
from typing import Tuple, List, Union
from PIL import Image
import numpy as np
import os
import random


def crop_or_pad(
    arr: Union[List[np.ndarray], np.ndarray],
    shape: tuple,
    pad_value: Union[List[int], int] = 0,
) -> Union[List[np.ndarray], np.ndarray]:
    """Crop or pad an array (or arrays) to a given shape. Note that if multiple arrays
    are passed, they must all have the same height and width.
    Args:
        arr (list | np.ndarray): Array to crop or pad with format [B, H, W, C] or [H, W, C].
        shape (tuple): Shape of the cropped or padded array with format [B, H, W, C] or [H, W, C].
        pad_value (list | float): Value to use for padding.
    Returns:
        Cropped or padded array with format [B, H, W, C] or [H, W, C].
    """
    if isinstance(arr, list):
        arr_h, arr_w = arr[0].shape[:2]
        for i in range(len(arr)):
            if len(arr[i].shape) == 2:
                arr[i] = np.expand_dims(arr[i], axis=2)

            assert arr[i].shape[:2] == (
                arr_h,
                arr_w,
            ), f"All arrays must have the same height and width. {arr[i].shape[:2]} != {(arr_h, arr_w)}"

        assert len(arr) == len(
            pad_value
        ), "Number of arrays and number of pad values must match."

    elif isinstance(arr, np.ndarray):
        if len(arr.shape) == 2:
            arr = np.expand_dims(arr, axis=2)

        assert len(arr.shape) == 3, "Array must be of shape [H, W] or [H, W, C]."

        arr_h, arr_w = arr.shape[:2]

    else:
        raise ValueError("Invalid array type: {}".format(type(arr)))

    # This is used to determine the starting point of the crop.
    crop_height = (random.randint(0, max(arr_h - shape[0], 0)) // 8) * 8
    crop_width = (random.randint(0, max(arr_w - shape[1], 0)) // 8) * 8

    if isinstance(arr, list):
        return [
            _crop_or_pad(a, shape, (crop_height, crop_width), pv)
            for a, pv in zip(arr, pad_value)
        ]

    elif isinstance(arr, np.ndarray):
        return _crop_or_pad(arr, shape, (crop_height, crop_width), pad_value)


def _crop_or_pad(
    arr: np.ndarray, shape: tuple, crop_start: tuple, pad_value: int = 0
) -> np.ndarray:

    # Pad in the x-axis.
    if arr.shape[0] < shape[0]:
        arr = np.pad(
            arr,
            ((0, shape[0] - arr.shape[0]), (0, 0), (0, 0)),
            "constant",
            constant_values=(0, pad_value),
        )

    # Pad in the y-axis.
    if arr.shape[1] < shape[1]:
        arr = np.pad(
            arr,
            ((0, 0), (0, shape[1] - arr.shape[1]), (0, 0)),
            "constant",
            constant_values=(0, pad_value),
        )

    # Crop in both axes at the same time.
    if arr.shape[0] > shape[0] or arr.shape[1] > shape[1]:
        arr = arr[
            crop_start[0] : crop_start[0] + shape[0],
            crop_start[1] : crop_start[1] + shape[1],
            :,
        ]

    return arr


class _BaseDataset(data.Dataset):
    def __init__(
        self,
        crop_size: Tuple[int, int],
        pixel_range: Tuple[float, float],
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.crop_size = crop_size
        self.pixel_range = pixel_range
        self.data_type = dtype

        # Need to define these in the child class.
        self.image_files = None
        self.mask_files = None

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:

        # Load the image file.
        image_file = self.image_files[idx]
        image = Image.open(image_file)

        # Force three color channels.
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Load the mask file.
        mask_file = self.mask_files[idx]
        pixel_min, pixel_max = self.pixel_range

        if mask_file is None:

            # The mask doesn't exist; assume it has no manipulated pixels.
            crop_size = self.crop_size if self.crop_size is not None else image.size
            mask = torch.zeros(crop_size, dtype=self.data_type).unsqueeze(dim=0)

            # Normalize the image.
            image = np.array(image) * (pixel_max - pixel_min) / 255.0 + pixel_min

            # Crop or pad the image.
            image = crop_or_pad(image, crop_size, pad_value=pixel_max)

            # Convert the image to a tensor.
            image = torch.from_numpy(image).to(self.data_type).permute(2, 0, 1)

        else:

            # Load the mask.
            mask = Image.open(mask_file)

            # Force one color channel.
            if mask.mode != "L":
                mask = mask.convert("L")

            # Resize the mask to match the image.
            mask = mask.resize(image.size[:2])

            # Normalize the image and mask.
            image = np.array(image) * (pixel_max - pixel_min) / 255.0 + pixel_min
            mask = np.array(mask) / 255.0

            # Convert partially mixed pixel labels to manipulated pixel labels.
            mask = (mask > 0.0).astype(float)

            # Crop or pad the image and mask.
            crop_size = (
                self.crop_size if self.crop_size is not None else image.shape[:2]
            )
            image, mask = crop_or_pad(
                [image, mask], crop_size, pad_value=[pixel_max, 1.0]
            )

            # Convert the image and mask to tensors.
            image = torch.from_numpy(image).to(self.data_type).permute(2, 0, 1)
            mask = torch.from_numpy(mask).to(self.data_type).permute(2, 0, 1)

        return image, mask

    def __len__(self) -> int:
        return len(self.image_files)


class Splicing(_BaseDataset):
    """Digital image forensic has gained a lot of attention as it is becoming easier
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

    Note: The dataset has an issue between the image and probe mask sizes. We must be
    careful in how the images and masks are handled. Since the images have manipulation
    statistics embedded in the pixels, any sort of aggregation function could damage or
    destroy the statistics. Therefore, we need to resize the masks to match the images.
    Then we need to crop the images and masks to the provided crop size. This will
    preserve any manipulation statistics while removing issues in the dataset.

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
        crop_size (tuple): The size of the crop to be applied on the image and mask.
        pixel_range (tuple): The range of the pixel values of the input images.
            Ex. (0, 1) scales the pixels from [0, 255] to [0, 1].
        shuffle (bool): Whether to shuffle the dataset before splitting.
        download (bool): Whether to download the dataset.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "full",
        crop_size: Tuple[int, int] = (256, 256),
        pixel_range: Tuple[float, float] = (0.0, 1.0),
        shuffle: bool = True,
        download: bool = False,
    ) -> None:
        super().__init__(crop_size, pixel_range)

        if download:
            raise NotImplementedError(
                "Downloading is not implemented yet due to the requirement of a "
                "browser to obtain the dataset. Please refer to the following link "
                "for more information: https://defactodataset.github.io."
            )

        # Fetch the image filenames.
        image_dirs = [
            os.path.join(data_dir, f"splicing_{i}_img", "img") for i in range(1, 8)
        ]
        image_files = [
            os.path.abspath(os.path.join(shard, f))
            for shard in image_dirs
            for f in os.listdir(shard)
            if f.endswith("tif") or f.endswith("jpg")
        ]

        # Shuffle the image files for a random split.
        if shuffle:
            image_files = np.random.permutation(image_files).tolist()

        split_size = len(image_files) // 10

        if split == "train":
            self.image_files = image_files[: split_size * 8]

        elif split == "valid":
            self.image_files = image_files[split_size * 8 : split_size * 9]

        elif split == "test":
            self.image_files = image_files[split_size * 9 :]

        elif split == "benchmark":
            self.image_files = image_files[:1000]

        elif split == "full":
            self.image_files = image_files

        else:
            raise ValueError(f"Unknown split: {split}")

        # Fetch the mask files.
        mask_dirs = [
            os.path.join(data_dir, f"splicing_{i}_annotations", "probe_mask")
            for i in range(1, 8)
        ]

        self.mask_files = []
        for f in self.image_files:
            shard = f.split("/")[-3].split("_")[-2]
            f = f.split("/")[-1]
            mask_file = os.path.abspath(os.path.join(mask_dirs[int(shard) - 1], f))
            if not os.path.exists(mask_file) and mask_file[-3:] == "jpg":
                self.mask_files.append(mask_file.replace(".jpg", ".tif"))
            elif not os.path.exists(mask_file) and mask_file[-3:] == "tif":
                self.mask_files.append(mask_file.replace(".tif", ".jpg"))
            else:
                self.mask_files.append(mask_file)


class CopyMove(_BaseDataset):
    """Digital image forensic has gained a lot of attention as it is becoming easier
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

    Note: The dataset has an issue between the image and probe mask sizes. We must be
    careful in how the images and masks are handled. Since the images have manipulation
    statistics embedded in the pixels, any sort of aggregation function could damage or
    destroy the statistics. Therefore, we need to resize the masks to match the images.
    Then we need to crop the images and masks to the provided crop size. This will
    preserve any manipulation statistics while removing issues in the dataset.

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
        crop_size (tuple): The size of the crops.
        pixel_range (tuple): The range of the pixel values of the input images.
            Ex. (0, 1) scales the pixels from [0, 255] to [0, 1].
        shuffle (bool): Whether to shuffle the dataset before splitting.
        download (bool): Whether to download the dataset.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "full",
        crop_size: Tuple[int, int] = None,
        pixel_range: Tuple[float, float] = (0.0, 1.0),
        shuffle: bool = True,
        download: bool = False,
    ) -> None:
        super().__init__(crop_size, pixel_range)

        if download:
            raise NotImplementedError(
                "Downloading is not implemented yet due to the requirement of a "
                "browser to obtain the dataset. Please refer to the following link "
                "for more information: https://defactodataset.github.io."
            )

        # Fetch the image filenames.
        image_dir = os.path.join(data_dir, "copymove_img", "img")
        image_files = [
            os.path.abspath(os.path.join(image_dir, f))
            for f in os.listdir(image_dir)
            if f.endswith(".tif") or f.endswith(".jpg")
        ]

        # Shuffle the image files for a random split.
        if shuffle:
            image_files = np.random.permutation(image_files).tolist()

        split_size = len(image_files) // 10

        # Note that the order of the output files is aligned with the input files.
        if split == "train":
            self.image_files = image_files[: split_size * 8]

        elif split == "valid":
            self.image_files = image_files[split_size * 8 : split_size * 9]

        elif split == "test":
            self.image_files = image_files[split_size * 9 :]

        elif split == "benchmark":
            self.image_files = image_files[:1000]

        elif split == "full":
            self.image_files = image_files

        else:
            raise ValueError(f"Unknown split: {split}")

        # Fetch the mask files.
        mask_dir = os.path.join(data_dir, "copymove_annotations", "probe_mask")

        self.mask_files = []
        for f in self.image_files:
            f = f.split("/")[-1]
            mask_file = os.path.abspath(os.path.join(mask_dir, f))
            if not os.path.exists(mask_file) and mask_file[-3:] == "jpg":
                self.mask_files.append(mask_file.replace(".jpg", ".tif"))
            elif not os.path.exists(mask_file) and mask_file[-3:] == "tif":
                self.mask_files.append(mask_file.replace(".tif", ".jpg"))
            else:
                self.mask_files.append(mask_file)


class Inpainting(_BaseDataset):
    """Digital image forensic has gained a lot of attention as it is becoming easier
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

    Note: The dataset has an issue between the image and probe mask sizes. We must be
    careful in how the images and masks are handled. Since the images have manipulation
    statistics embedded in the pixels, any sort of aggregation function could damage or
    destroy the statistics. Therefore, we need to resize the masks to match the images.
    Then we need to crop the images and masks to the provided crop size. This will
    preserve any manipulation statistics while removing issues in the dataset.

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
        crop_size (tuple): The size of the crops.
        pixel_range (tuple): The range of the pixel values of the input images.
            Ex. (0, 1) scales the pixels from [0, 255] to [0, 1].
        shuffle (bool): Whether to shuffle the dataset before splitting.
        download (bool): Whether to download the dataset.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "full",
        crop_size: Tuple[int, int] = None,
        pixel_range: Tuple[float, float] = (0.0, 1.0),
        shuffle: bool = True,
        download: bool = False,
    ) -> None:
        super().__init__(crop_size, pixel_range)

        if download:
            raise NotImplementedError(
                "Downloading is not implemented yet due to the requirement of a "
                "browser to obtain the dataset. Please refer to the following link "
                "for more information: https://defactodataset.github.io."
            )

        # Fetch the image filenames.
        image_dir = os.path.join(data_dir, "inpainting_img", "img")
        image_files = [
            os.path.abspath(os.path.join(image_dir, f))
            for f in os.listdir(image_dir)
            if f.endswith(".tif") or f.endswith(".jpg")
        ]

        # Shuffle the image files for a random split.
        if shuffle:
            image_files = np.random.permutation(image_files).tolist()

        split_size = len(image_files) // 10

        # Note that the order of the output files is aligned with the input files.
        if split == "train":
            self.image_files = image_files[: split_size * 8]

        elif split == "valid":
            self.image_files = image_files[split_size * 8 : split_size * 9]

        elif split == "test":
            self.image_files = image_files[split_size * 9 :]

        elif split == "benchmark":
            self.image_files = image_files[:1000]

        elif split == "full":
            self.image_files = image_files

        else:
            raise ValueError(f"Unknown split: {split}")

        # Fetch the mask files.
        mask_dir = os.path.join(data_dir, "inpainting_annotations", "probe_mask")

        self.mask_files = []
        for f in self.image_files:
            f = f.split("/")[-1]
            mask_file = os.path.abspath(os.path.join(mask_dir, f))
            if not os.path.exists(mask_file) and mask_file[-3:] == "jpg":
                self.mask_files.append(mask_file.replace(".jpg", ".tif"))
            elif not os.path.exists(mask_file) and mask_file[-3:] == "tif":
                self.mask_files.append(mask_file.replace(".tif", ".jpg"))
            else:
                self.mask_files.append(mask_file)


class CASIA2(_BaseDataset):
    """CASIA V2 is a dataset for forgery classification. It contains 4795 images, 1701 authentic and 3274 forged.

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
        shuffle (bool): Whether to shuffle the dataset before splitting.
        download (bool): Whether to download the dataset.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "full",
        crop_size: Tuple[int, int] = None,
        pixel_range: Tuple[float, float] = (0.0, 1.0),
        shuffle: bool = True,
        download: bool = False,
    ) -> None:
        super().__init__(crop_size, pixel_range)

        if download:
            raise NotImplementedError(
                "Downloading is not implemented yet due to the requirement of a "
                "browser to obtain the dataset. Please refer to the following link "
                "for more information: https://github.com/namtpham/casia2groundtruth."
            )

        # Fetch the image filenames.
        authentic_dir = os.path.join(data_dir, "Au")
        auth_files = [
            os.path.abspath(os.path.join(authentic_dir, f))
            for f in os.listdir(authentic_dir)
            if f.endswith("tif") or f.endswith("jpg")
        ]

        tampered_dir = os.path.join(data_dir, "Tp")
        tamp_files = [
            os.path.abspath(os.path.join(tampered_dir, f))
            for f in os.listdir(tampered_dir)
            if f.endswith("tif") or f.endswith("jpg")
        ]

        # Ignore these files that have no ground truth masks.
        corrupted_files = [
            "Tp/Tp_D_NRD_S_N_cha10002_cha10001_20094.jpg",
            "Tp/Tp_S_NRD_S_N_arc20079_arc20079_01719.tif",
        ]

        remove_files = []
        for file in tamp_files:
            for f in corrupted_files:
                if f in file:
                    remove_files.append(file)

        for file in remove_files:
            tamp_files.remove(file)

        # Fetch the mask filenames.
        mask_dir = os.path.join(data_dir, "CASIA 2 Groundtruth")
        mask_files = [
            os.path.abspath(os.path.join(mask_dir, f))
            for f in os.listdir(mask_dir)
            if f.endswith(".tif") or f.endswith(".jpg") or f.endswith(".png")
        ]

        # Sort the mask files based on the tampered files.
        sorted_mask_files = []
        for file in tamp_files:
            tamp_id = file[-9:-4]
            mask = None
            for f in mask_files:
                if tamp_id + "_gt" == f[-12:-4]:
                    mask = f
                    break

            if mask is None and file.split("/")[-2] == "Tp":
                raise ValueError("No ground truth file found for image: " + file)

            mask_file = os.path.abspath(os.path.join(mask_dir, mask))
            sorted_mask_files.append(mask_file)

        mask_files = sorted_mask_files

        # Shuffle the image files for a random split.
        auth_files = np.random.permutation(auth_files).tolist()

        # Shuffle the tampered files in the same order as the masks.
        p = np.random.permutation(len(tamp_files))
        tamp_files = [tamp_files[i] for i in p]
        mask_files = [mask_files[i] for i in p]

        # Split the filenames into use cases.
        auth_split_size = len(auth_files) // 10
        tamp_split_size = len(tamp_files) // 10
        if split == "train":
            self.image_files = auth_files[: auth_split_size * 8]
            self.mask_files = [None for _ in range((auth_split_size * 8))]

            self.image_files += tamp_files[: tamp_split_size * 8]
            self.mask_files += mask_files[: tamp_split_size * 8]

        elif split == "valid":
            self.image_files = auth_files[auth_split_size * 8 : auth_split_size * 9]
            self.mask_files = [None for _ in range(len(self.image_files))]

            self.image_files += tamp_files[tamp_split_size * 8 : tamp_split_size * 9]
            self.mask_files += mask_files[tamp_split_size * 8 : tamp_split_size * 9]

        elif split == "test":
            self.image_files = auth_files[auth_split_size * 9 :]
            self.mask_files = [None for _ in range(len(self.image_files))]

            self.image_files += tamp_files[tamp_split_size * 9 :]
            self.mask_files += mask_files[tamp_split_size * 9 :]

        elif split == "benchmark":
            self.image_files = auth_files[:500]
            self.mask_files = [None for _ in range(500)]

            self.image_files += tamp_files[:500]
            self.mask_files += mask_files[:500]

        elif split == "full":
            self.image_files = auth_files + tamp_files

            self.mask_files = [None for _ in range(len(auth_files))]
            self.mask_files += mask_files

        else:
            raise ValueError("Unknown split: " + split)

        # Shuffle the image files to mix authentic and tampered images.
        if shuffle:
            p = np.random.permutation(len(self.image_files))
            self.image_files = [self.image_files[i] for i in p]
            self.mask_files = [self.mask_files[i] for i in p]


class Coverage(_BaseDataset):
    """The Copy-Move Forgery Database with Similar but Genuine Objects (COVERAGE)
    accompanies the following publication: "COVERAGE--A NOVEL DATABASE FOR COPY-MOVE
    FORGERY DETECTION," IEEE International Conference on Image processing (ICIP), 2016.

    COVERAGE contains copymove forged (CMFD) images and their originals with similar but
    genuine objects (SGOs). COVERAGE is designed to highlight and address tamper
    detection ambiguity of popular methods, caused by self-similarity within natural
    images. In COVERAGE, forged-original pairs are annotated with (i) the duplicated and
    forged region masks, and (ii) the tampering factor/similarity metric. For
    benchmarking, forgery quality is evaluated using (i) computer vision-based methods,
    and (ii) human detection performance.

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
        shuffle (bool): Whether to shuffle the dataset before splitting.
        download (bool): Whether to download the dataset.
    """

    def __init__(
        self,
        data_dir: str,
        mask_type: str = "forged",
        crop_size: Tuple[int, int] = None,
        pixel_range: Tuple[float, float] = (0.0, 1.0),
        shuffle: bool = True,
        download: bool = False,
    ) -> None:
        super().__init__(crop_size, pixel_range)

        assert mask_type in ["forged", "copy", "paste"]

        if download:
            raise NotImplementedError(
                "Downloading is not implemented yet due to the requirement of a "
                "browser to obtain the dataset. Please refer to the following link "
                "for more information: https://github.com/wenbihan/coverage."
            )

        # Fetch the image filenames.
        image_dir = os.path.join(data_dir, "image")
        self.image_files = [
            os.path.abspath(os.path.join(image_dir, f))
            for f in os.listdir(image_dir)
            if f.endswith("tif") or f.endswith("jpg")
        ]

        # Shuffle the image files for a random split.
        if shuffle:
            self.image_files = np.random.permutation(self.image_files).tolist()

        # Fetch the mask filenames in the correct order.
        mask_dir = os.path.abspath(os.path.join(data_dir, "mask"))
        mask_files = [
            os.path.abspath(os.path.join(mask_dir, f))
            for f in os.listdir(mask_dir)
            if ".tif" in f
        ]
        self.mask_files = []
        for f in self.image_files:
            f_name = f.split(".")[0]
            if f_name[-1] == "t":
                mask_file = f_name.split("/")[-1][:-1] + mask_type + ".tif"
                mask_file = os.path.abspath(os.path.join(mask_dir, mask_file))
                assert mask_file in mask_files
            else:
                mask_file = None

            self.mask_files.append(mask_file)


class IMD2020(_BaseDataset):
    """This dataset contains 2,010 real-life manipulated images downloaded from the
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
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "full",
        crop_size: Tuple[int, int] = None,
        pixel_range: Tuple[float, float] = (0.0, 1.0),
        shuffle: bool = True,
        download: bool = False,
    ) -> None:
        super().__init__(crop_size, pixel_range)

        if download:
            raise NotImplementedError(
                "Downloading is not implemented yet due to the requirement of a "
                "browser to obtain the dataset. Please refer to the following link "
                "for more information: http://staff.utia.cas.cz/novozada/db/."
            )

        subdirs = [
            os.path.join(data_dir, subdir)
            for subdir in os.listdir(data_dir)
            if "." not in subdir
        ]

        # Fetch the authentic image filenames (they end in orig.jpg).
        image_files, mask_files = [], []
        for subdir in subdirs:
            for f in os.listdir(subdir):
                if "orig" in f:
                    image_files.append(os.path.abspath(os.path.join(subdir, f)))
                    mask_files.append(None)
                elif "mask" in f:
                    mask_file = os.path.abspath(os.path.join(subdir, f))
                    mask_files.append(mask_file)

                    # Locate the corresponding image file.
                    image_file = mask_file.replace("_mask", "")
                    if not os.path.exists(image_file):
                        image_file = image_file.replace(".png", ".jpg")
                        if not os.path.exists(image_file):
                            raise ValueError(
                                "Could not locate image for mask at {}".format(
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
        if split == "train":
            self.image_files = image_files[: split_size * 8]
            self.mask_files = mask_files[: split_size * 8]

        elif split == "valid":
            self.image_files = image_files[split_size * 8 : split_size * 9]
            self.mask_files = mask_files[split_size * 8 : split_size * 9]

        elif split == "test":
            self.image_files = image_files[split_size * 9 :]
            self.mask_files = mask_files[split_size * 9 :]

        elif split == "benchmark":
            self.image_files = image_files[:500]
            self.mask_files = mask_files[:500]

        elif split == "full":
            self.image_files = image_files
            self.mask_files = mask_files

        else:
            raise ValueError("Unknown split: " + split)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Defacto dataset loader")
    parser.add_argument(
        "--copy-move-data-dir",
        type=str,
        default=None,
        help="Path to the CopyMove dataset directory.",
    )
    parser.add_argument(
        "--inpainting-data-dir",
        type=str,
        default=None,
        help="Path to the Inpainting dataset directory.",
    )
    parser.add_argument(
        "--splicing-data-dir",
        type=str,
        default=None,
        help="Path to the Splicing dataset directory.",
    )
    parser.add_argument(
        "--casia2-data-dir",
        type=str,
        default=None,
        help="Path to the CASIA2 dataset directory.",
    )
    parser.add_argument(
        "--coverage-data-dir",
        type=str,
        default=None,
        help="Path to the Coverage dataset directory.",
    )
    parser.add_argument(
        "--imd2020-data-dir",
        type=str,
        default=None,
        help="Path to the IMD2020 dataset directory.",
    )
    args = parser.parse_args()

    if (
        args.copy_move_data_dir is None
        and args.inpainting_data_dir is None
        and args.splicing_data_dir is None
        and args.casia2_data_dir is None
        and args.coverage_data_dir is None
        and args.imd2020_data_dir is None
    ):
        parser.error("At least one dataset directory must be specified.")

    if args.splicing_data_dir is not None:
        dataset = Splicing(data_dir=args.splicing_data_dir, split="valid")
        for image, mask in dataset:
            print("Sample:", image.size(), mask.size())
            break
        print("Number of samples:", len(dataset))

    if args.copy_move_data_dir is not None:
        dataset = CopyMove(data_dir=args.copy_move_data_dir, split="valid")
        for image, mask in dataset:
            print("Sample:", image.size(), mask.size())
            break
        print("Number of samples:", len(dataset))

    if args.inpainting_data_dir is not None:
        dataset = Inpainting(data_dir=args.inpainting_data_dir, split="valid")
        for image, mask in dataset:
            print("Sample:", image.size(), mask.size())
            break
        print("Number of samples:", len(dataset))

    if args.casia2_data_dir is not None:
        dataset = CASIA2(data_dir=args.casia2_data_dir, split="valid")
        for image, mask in dataset:
            print("Sample:", image.size(), mask.size())
            break
        print("Number of samples:", len(dataset))

    if args.coverage_data_dir is not None:
        dataset = Coverage(data_dir=args.coverage_data_dir)
        for image, mask in dataset:
            print("Sample:", image.size(), mask.size())
            break
        print("Number of samples:", len(dataset))

    if args.imd2020_data_dir is not None:
        dataset = IMD2020(data_dir=args.imd2020_data_dir, split="valid")
        for image, mask in dataset:
            print("Sample:", image.size(), mask.size())
            break
        print("Number of samples:", len(dataset))


if __name__ == "__main__":
    main()
