from pathlib import Path

import numpy as np
import tifffile
from skimage.exposure import rescale_intensity

from abbott.fractal_tasks.conversions import to_itk, to_numpy
from abbott.registration.itk_elastix import apply_transform, register_transform_only

path = Path(__file__).parent.resolve()

files = [
    "./chik/stitched_p0000_w0000_t0000_R1.tif",
    "./chik/stitched_p0000_w0000_t0000_R2.tif",
]

# z, y, x
voxel_size = (2, 0.652, 0.652)

# registration parameters
parameter_files = [
    # str(path / "params_translation_level0_2.txt"),
    str(path / "params_rigid.txt"),
    str(path / "bspline_lvl2.txt"),
    # str(path / "params_similarity_level1.txt"),
    # str(path / "params_affine.txt"),
]


def main():
    imgs = []
    for f in files:
        img_path = path / f
        img = tifffile.imread(img_path)
        img = rescale_intensity(
            img, in_range=tuple(np.quantile(img, [0.01, 0.99])), out_range=(0, 1)
        )
        imgs.append(to_itk(img, scale=voxel_size))
    print("calculate transform")
    transform = register_transform_only(imgs[0], imgs[1], parameter_files)

    print("apply transform")
    registered = apply_transform(imgs[1], transform)

    print("save")
    result = to_numpy(registered)
    tifffile.imwrite("registered_img.tif", result)
    print("done")


if __name__ == "__main__":
    main()
