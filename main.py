import logging
import PIL.Image

from zeroscratches import EraseScratches

logging.basicConfig(level=logging.INFO)

image_path = "/home/leonel/old-couple.jpg"
eraser = EraseScratches()

image = PIL.Image.open(image_path)
new_img = eraser.erase(image)

new_img = PIL.Image.fromarray(new_img)
new_img.show()
