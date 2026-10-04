# Augmento

A desktop app that augments, renames and resizes image datasets for computer vision projects.

Website and examples: https://augmento.pythonanywhere.com

<!-- Add a screenshot of the app here, for example: ![Augmento](docs/screenshot.png) -->

## Features

**Augment**
- 25 augmentations: rotation, flipping, cropping, zoom, shear, perspective, affine, elastic, brightness, contrast, saturation, hue, blur, motion blur, three kinds of noise, cutout, random erasing, edge detection, emboss, sharpen, channel shuffle and channel drop
- Apply each method separately, or mix them in a combined pipeline
- Several copies per image, an intensity slider, a seed for repeatable results, and a border fill option
- Live preview before you generate thousands of images
- YOLO bounding box labels are transformed together with the images
- Runs in the background with a progress bar and a Cancel button

**Rename**
- Number all images in a folder (1.jpg, 2.jpg, 3.jpg ...) so you always know how many you have
- Natural sort, optional zero padding, subfolder support, label files follow their images, and an optional `rename_map.csv`
- Never overwrites files, and rolls back if something fails

**Resize**
- Resize a whole folder to one size, such as 640x640 or 720x720
- Stretch, letterbox (pad) or center crop modes
- YOLO labels are updated to match

## Install and run

You need Python 3.9 or newer. Tkinter is included with Python on Windows and macOS. On Ubuntu or Debian, install it with `sudo apt install python3-tk`.

```
git clone https://github.com/omarakl/Augmento.git
cd augmento
pip install -r requirements.txt
python augmento.py
```

Or download the ready-made Windows `.exe` from the [Releases](../../releases) page.

## YOLO labels

Augmento reads YOLO bounding box files: one `.txt` per image with the same name, and one object per line.

```
class  x_center  y_center  width  height
0      0.52      0.40      0.20   0.35
```

All values except the class are fractions of the image size (0 to 1). Turn on **Use YOLO bbox labels** (Augment), **Also process YOLO labels** (Resize) or **Also rename matching .txt label files** (Rename) when your dataset has them.

Not supported: segmentation polygons, Pascal VOC XML and COCO JSON. Files with polygons are skipped and logged.

Elastic Transformation is disabled when labels are on, because it cannot move boxes.

## Run the tests

```
python test_augmento.py
```

The GUI tests are skipped automatically when no display is available.

## Build a Windows exe

```
pip install pyinstaller
pyinstaller --clean --noconfirm --onefile --windowed --name Augmento augmento.py
```

The program is created at `dist/Augmento.exe`.

## Project layout

```
augmento.py        the application
test_augmento.py   automated tests
website/           the download page and the examples page
```

## Versions

- **2.0**: YOLO label support, combined pipelines, copies per image, intensity, seed, border fill, preview, rename and resize tools, background processing, and many bug fixes
- **1.0.0**: first release with 25 augmentation methods

## Contributing

Ideas, bug reports and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).

## Contact

Omar Akl: omar.aakl@hotmail.com
