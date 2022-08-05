# Image Manipulation Datasets

This python package contains torch dataset classes for common image manipulation datasets.

Currently, the supported datasets are:
- CASIA
    - CASIA 2.0
- Defacto
    - Copy/Move
    - Splicing
    - Inpainting
- Coverage

## Install
```bash
pip install git+https://github.com/cainspencerm/image-manipulation-datasets.git@0.5
```

## Examples

### CASIA 2.0

Ensure that the ground truth directory is in data_dir and named 'CASIA 2 Groundtruth'.

```python
from image_manip import casia2

# Create dataset object for dataloader.
dataset = casia2.Casia2(data_dir='data/CASIA2.0')  # optional split=['train', 'val', 'test', 'benchmark', 'full']
```

### Defacto Copy/Move

```python
from image_manip import defacto

# Create dataset object for dataloader.
dataset = defacto.CopyMove(data_dir='data/copy-move')  # optional split=['train', 'val', 'test', 'benchmark', 'full']
```

### Defacto Splicing

```python
from image_manip import defacto

dataset = defacto.Splicing(data_dir='data/splicing')  # optional split=['train', 'val', 'test', 'benchmark', 'full']
```

### Defacto Inpainting

```python
from image_manip import defacto

datatset = defacto.Inpainting(data_dir='data/inpainting') # optional split=['train', 'val', 'test', 'benchmark', 'full']
```

### Coverage

```python
from image_manip import coverage

dataset = coverage.Coverage(data_dir='data/coverage')
```

### Testing

To test the directory configurations:
```python
from torch.utils.data import DataLoader

# Create dataloader from dataset.
full_loader = DataLoader(dataset, batch_size=8, shuffle=True)

for image, mask in full_loader:
    print(image.size(), mask.size())
    break
```
