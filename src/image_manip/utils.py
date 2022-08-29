import numpy as np
import random
from typing import List, Union


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
            ), f'All arrays must have the same height and width. {arr[i].shape[:2]} != {(arr_h, arr_w)}'

        assert len(arr) == len(
            pad_value
        ), 'Number of arrays and number of pad values must match.'

    elif isinstance(arr, np.ndarray):
        if len(arr.shape) == 2:
            arr = np.expand_dims(arr, axis=2)

        assert len(arr.shape) == 3, 'Array must be of shape [H, W] or [H, W, C].'

        arr_h, arr_w = arr.shape[:2]

    else:
        raise ValueError('Invalid array type: {}'.format(type(arr)))

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
            'constant',
            constant_values=(0, pad_value),
        )

    # Pad in the y-axis.
    if arr.shape[1] < shape[1]:
        arr = np.pad(
            arr,
            ((0, 0), (0, shape[1] - arr.shape[1]), (0, 0)),
            'constant',
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
