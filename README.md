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

Get the pretrained models at [Hugging Face Zero Scratches](https://huggingface.co/leonelhs/zeroscratches)

## Some Apps using the library:

### [Face Shine](https://github.com/leonelhs/face-shine) 
Face Shine Is a backend server for photo enhancement and restoration.

### [Super Face](https://github.com/leonelhs/SuperFace/)
Super Face is a Python QT frontend for Face Shine server.

<img src="https://drive.google.com/uc?export=view&id=1D7hpjQSlUkzfTba-E5Ul4Rb1c8lYkFj5"/>
<img src="https://drive.google.com/uc?export=view&id=1oKpJe-Ff3SeEekhGVRP1Ap3eIFqt0c8u"/>
