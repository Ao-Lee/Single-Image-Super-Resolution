import os
from PIL import Image

for name in os.listdir('data'):
    path_data = os.path.join('data', name)
    path_image = os.path.join('Images', name)
    path_label = os.path.join('Labels', name)
    image = Image.open(path_data)
    w, h = image.size
    w_new = w - w % 4
    h_new = h - h % 4
    image = image.crop((0, 0, w_new, h_new)) # left, upper, right lower
    image.save(path_label)
    resized_image = image.resize((w_new//4, h_new//4))
    resized_image.save(path_image)