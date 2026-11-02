# PedVision hand X-ray test sets (RSNA + DHA)

These are the manually segmented hand radiographs used as test sets in
[PedVision](https://doi.org/10.1016/j.bspc.2025.108569) (RSNA and DHA).
The RSNA annotations were also used in
[Tay et al. (2025)](https://doi.org/10.1016/j.bspc.2024.107338).
Each image has a ground-truth mask of 19 hand bones in 4 bone classes,
manually annotated with [LabelMe](https://github.com/wkentaro/labelme).

| Set | Folder | Images | Ages |
|---|---|---|---|
| RSNA, Pediatric group | `RSNA/pediatric/` | 200 | 4–84 months |
| RSNA, Near-skeletally-mature (NSM) group | `RSNA/nsm/` | 200 | 169–228 months |
| Digital Hand Atlas (DHA) | `DHA/` | 152 | 0–18 years |

## Folder layout

```
RSNA/
  pediatric/
    images/<id>.jpg           grayscale radiograph, 1024 x 1024
    masks/<id>.png            RGB mask, 1024 x 1024
  nsm/
    images/  masks/
DHA/
  images/<id>.jpg             grayscale radiograph, resized to 1024 x 1024
  masks/<id>.png              RGB mask, 1024 x 1024
  original_size/
    images/<id>.jpg           original DHA JPEG, unmodified
    masks/<id>.png            RGB mask at the original resolution
    annotations/<id>.json     LabelMe polygons (the source of both DHA mask sets)
```

Every image has a mask with the same file name and the same height and width (`images/1378.jpg` ↔ `masks/1378.png`).
RSNA files are named by their RSNA Bone Age image ID, and DHA files by their Digital Hand Atlas image ID.

## Mask format

Masks are 3-channel RGB PNGs (uint8), with one colour per class:

| Class | Bones | RGB |
|---|---|---|
| Background | everything else (carpals, radius, ulna, soft tissue, ...) | `(0, 0, 0)` |
| Distal phalanges | 5 | `(128, 0, 0)` |
| Middle (intermediate) phalanges | 4 | `(0, 128, 0)` |
| Proximal phalanges | 5 | `(0, 0, 128)` |
| Metacarpals | 5 | `(128, 128, 0)` |

To convert a mask to a class-index map (0 = background, 1 = distal, 2 = middle, 3 = proximal, 4 = metacarpal):

```python
import numpy as np
from PIL import Image

PALETTE = {
    (0, 0, 0): 0,       # background
    (128, 0, 0): 1,     # distal phalanges
    (0, 128, 0): 2,     # middle phalanges
    (0, 0, 128): 3,     # proximal phalanges
    (128, 128, 0): 4,   # metacarpals
}

rgb = np.array(Image.open("RSNA/pediatric/masks/1378.png").convert("RGB"))
label = np.zeros(rgb.shape[:2], np.uint8)
for color, idx in PALETTE.items():
    label[(rgb == color).all(-1)] = idx
```

## RSNA

The 400 radiographs were randomly drawn from the
[RSNA Pediatric Bone Age Challenge](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017) dataset.
Half come from the Pediatric group (children up to 7 years old) and half from the near-skeletally-mature (NSM) group (14–19 years old).
The images were resized to 1024 × 1024 before annotation.

## DHA

The 152 radiographs come from the [Digital Hand Atlas](https://ipilab.usc.edu/research/baaweb/) (IPILab, USC).
There is one image for each combination of age (0–18 years, in one-year groups), sex (female/male) and race (Asian, African American, Caucasian, Hispanic):
19 × 2 × 4 = 152.
The Pediatric subgroup in the paper is ages 0–6 (56 images) and the NSM subgroup is ages 14–18 (40 images).
Age, sex and race for each image ID are available in the original Digital Hand Atlas.

The DHA images were annotated at their original resolution, and both mask sets are drawn from the same polygons:

- `DHA/images`, `DHA/masks`: resized to 1024 × 1024 to match the RSNA format.
  This is a plain resize, so the aspect ratio is not preserved.
  The polygons are drawn directly at 1024 × 1024 rather than by resizing a mask, which keeps edges sharp with no mixed colours.
- `DHA/original_size/`: the unmodified DHA JPEGs, with masks at their original resolution.


## Citation

If you use these data (RSNA or DHA annotations), please cite:

```bibtex
@article{homayounfar2026pedvision,
  title   = {PedVision: A manual-annotation-free and age scalable segmentation pipeline for bone analysis in hand X-ray images},
  author  = {Homayounfar, Morteza and Bierma-Zeinstra, Sita M. A. and Zadpoor, Amir A. and T{\"u}mer, Nazli},
  journal = {Biomedical Signal Processing and Control},
  volume  = {112},
  pages   = {108569},
  year    = {2026},
  doi     = {10.1016/j.bspc.2025.108569}
}
```

If you use the RSNA annotations, please also cite:

```bibtex
@article{tay2025growth,
  title   = {Towards growth-accommodating deep learning-based semantic segmentation of pediatric hand phalanges},
  author  = {Tay, Edwin and Zadpoor, Amir A. and T{\"u}mer, Nazli},
  journal = {Biomedical Signal Processing and Control},
  volume  = {102},
  pages   = {107338},
  year    = {2025},
  doi     = {10.1016/j.bspc.2024.107338}
}
```

Please also cite the original image datasets:

```bibtex
@article{halabi2019rsna,
  title   = {The {RSNA} Pediatric Bone Age Machine Learning Challenge},
  author  = {Halabi, Safwan S. and Prevedello, Luciano M. and Kalpathy-Cramer, Jayashree and Mamonov, Artem B. and Bilbily, Alexander and Cicero, Mark and Pan, Ian and Pereira, Lucas Ara{\'u}jo and Sousa, Rafael Teixeira and Abdala, Nitamar and Kitamura, Felipe Campos and Thodberg, Hans H. and Chen, Leon and Shih, George and Andriole, Katherine and Kohli, Marc D. and Erickson, Bradley J. and Flanders, Adam E.},
  journal = {Radiology},
  volume  = {290},
  number  = {2},
  pages   = {498--503},
  year    = {2019},
  doi     = {10.1148/radiol.2018180736}
}

@article{gertych2007dha,
  title   = {Bone age assessment of children using a digital hand atlas},
  author  = {Gertych, Arkadiusz and Zhang, Aifeng and Sayre, James and Pospiech-Kurkowska, Sylwia and Huang, H. K.},
  journal = {Computerized Medical Imaging and Graphics},
  volume  = {31},
  number  = {4-5},
  pages   = {322--331},
  year    = {2007},
  doi     = {10.1016/j.compmedimag.2007.02.012}
}
```

## Terms of use

The radiographs remain subject to the terms of their original sources: the RSNA Pediatric Bone Age Challenge dataset and the Digital Hand Atlas (IPILab, USC).
