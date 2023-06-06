# Zero Scratches
## Old Photo Restoration

This is a lightweight implementation of [Microsoft Bringing Old Photos Back to Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life)


### Install
```shell
pip install zeroscatches
```
### Basic usage
```python

import PIL.Image
from zeroscratches import EraseScratches


image_path = "/path/to/image-scratched.jpg"
eraser = EraseScratches()

image = PIL.Image.open(image_path)
new_img = eraser.erase(image)

new_img = PIL.Image.fromarray(new_img)
new_img.show()
```