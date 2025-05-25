from ultralytics import YOLO
import numpy as np

model = YOLO('./ver7.pt')


def classify_model(image):
    results = model.predict(image)

    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    classification_result = names_dict[np.argmax(probs)]
    return classification_result
