import cv2
import json
import numpy as np
import os
import time
import glob

from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes

from tensorflow.keras import utils

import sys
sys.path.append('/content/drive/MyDrive/face_detector')
from preprocessing import ImageFaceDetector

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    image_face_detector = ImageFaceDetector('/content/drive/MyDrive/face_detector/preprocessing/shape_predictor_68_face_landmarks.dat')

    phi = 0
    weighted_bifpn = False
    model_path = '/content/drive/MyDrive/EfficientDet/snapshots/pascal_60_0.0199_0.0291.h5'
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    # coco classes
    classes = ['fake', 'real']
    num_classes = 2
    score_threshold = 0.5
    colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
    _, model = efficientdet(phi=phi,
                            weighted_bifpn=weighted_bifpn,
                            num_classes=num_classes,
                            score_threshold=score_threshold)
    model.load_weights(model_path, by_name=True)
    
    # utils.plot_model(model, 'my_first_model.png')

    for image_path in ['/content/drive/MyDrive/test_photos/frame0-00-10.93.jpg']:
        image = cv2.imread(image_path)
        faces = image_face_detector.preprocess_image(image)
        (orig, image) = faces[0]
        src_image = image.copy()
        image, scale = preprocess_image(image, image_size=image_size)
        # run network
        start = time.time()
        boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        print(time.time() - start)
        indices = np.where(scores[:] > score_threshold)[0]

        # select those detections
        boxes = boxes[indices]
        labels = labels[indices]

        draw_boxes(src_image, boxes, scores, labels, colors, classes)
        cv2.imwrite("result.jpg", src_image)


if __name__ == '__main__':
    main()
