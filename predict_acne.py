from ultralytics import YOLO
import cv2
import numpy as np
from GLT import GLT


def predict_acne(image):
    acne_detection_model = YOLO('./acne_detection.pt')
    acne_classification_model = YOLO('./ver7.pt')
    orig_img = cv2.imread(image)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    result_list = []
    results_detetion = acne_detection_model.predict(source=image, conf=0.4)
    bounding_boxe = results_detetion[0].boxes.xyxy
    if len(bounding_boxe) > 0:
        for i, box in enumerate(bounding_boxe):
            x1, y1, x2, y2 = box.int().cpu().numpy()
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_img.shape[1], x2)
            y2 = min(orig_img.shape[0], y2)

            # Crop the image
            cropped_image = orig_img[y1:y2, x1:x2]

            # Super resolution enhancement
            # 1. Scale up the image using cv2
            scale_factor = 3  # Increased scale factor for better visualization
            enhanced_image = cv2.resize(
                cropped_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

            # 2. Apply sharpening
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            sharpened_image = cv2.filter2D(enhanced_image, -1, kernel)

            lab = cv2.cvtColor(sharpened_image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            contrast_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

            # 4. Apply GLT (log transform) to enhance details
            log_enhanced_image = GLT(contrast_enhanced, 'log', coeff=15.0)

            median_blur = cv2.medianBlur(log_enhanced_image, 5)

            results_classify = acne_classification_model.predict(median_blur)

            names_dict = results_classify[0].names
            probs = results_classify[0].probs.data.tolist()
            result_list.append(names_dict[np.argmax(probs)])

    return result_list
