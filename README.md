# Image Manipulation Datasets

This python package contains torch dataset classes for common image manipulation datasets.

Currently, the supported datasets are:
- CASIA 2.0

## Install
```bash
pip install git+https://github.com/cainspencerm/image-manipulation-datasets.git@0.1
```

## Examples

### CASIA 2.0

Ensure that the ground truth directory is in data_dir and named 'CASIA 2 Groundtruth'.

```python
from datasets import casia2
from torch.utils.data import DataLoader

# Create dataset object for dataloader.
dataset = casia2.Casia2(data_dir='data/CASIA2.0')  # optional split=['train', 'val', or 'test']

# Create dataloader from dataset.
full_loader = DataLoader(dataset, batch_size=8, shuffle=True)

for image, mask in full_loader:
    print(image.size(), mask.size())
    break
```
