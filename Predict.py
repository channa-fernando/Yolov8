from ultralytics import YOLO

import numpy as np


model = YOLO('C:\\Users\\Channa CC\\PycharmProjects\\YoloV8\\runs\\classify\\train6\\weights\\best.pt')  # load a custom model

results = model('C:\\Users\\Channa CC\\Desktop\\ToPrint\\test1_pera.jpg')  # predict on an image

names_dict = results[0].names

probs = results[0].probs.data.tolist()

print(names_dict)
print(probs)

print(names_dict[np.argmax(probs)])