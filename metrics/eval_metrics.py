import argparse

import cv2
import json
import numpy as np
import os
import time
import glob

import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)

from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes

from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score, roc_curve, roc_auc_score

import pickle

import matplotlib.pyplot as plt

def parse_args(args):
    parser = argparse.ArgumentParser(description='Evaluate metrics for Deepfake Detection')
    parser.add_argument('--model-path', help='Path to model')
    parser.add_argument('--save-metrics', help='Filename for saving metrics', default=None)
    parser.add_argument('--dataset-path', help='Path to dataset images')
    parser.add_argument('--list-path', help='Path to image set')
    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)

def save_metrics(path, list_of_results):
    with open(path, 'wb') as fp:
        pickle.dump(list_of_results, fp)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model_path = args.model_path
    image_size = 512

    classes = ['fake', 'real']
    _, model = efficientdet(phi=0,
                            weighted_bifpn=False,
                            num_classes=2,
                            score_threshold=0.5)
    model.load_weights(model_path, by_name=True)


    path_to_files = args.dataset_path

    with open(args.list_path) as f:
      validate_set = f.read().splitlines()

    validate_set = list(map(lambda x: f"{path_to_files}{x}.jpg", validate_set)) 
    list_of_results = []

    for index, val_image in enumerate(validate_set):
        image = cv2.imread(val_image)
        src_image = image.copy()
        # BGR -> RGB
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale = preprocess_image(image, image_size=image_size)
        # run network
        boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        res = zip(boxes[0], scores[0], labels[0])
        res = list(res)
        (box_boundaries, score, predicted_class) = max(res,key=lambda item:item[1])
        if predicted_class == 1:
          score = 1 - score
          predicted_class = 0
        else :
          predicted_class = 1
        real_class = 0
        if 'fake' in val_image :
          real_class = 1

        print(f"Progress: {index + 1}/{len(validate_set)}")
        list_of_results.append((score, predicted_class, real_class))
      
    if args.save_metrics is not None:
      save_metrics(args.save_metrics, list_of_results)
    
    unzipped_object = zip(*list_of_results)
    unzipped_list = list(unzipped_object)
    predicted = unzipped_list[1]
    actual = unzipped_list[2]
    scores = unzipped_list[0]
    results = confusion_matrix(actual, predicted)
    print ('Confusion Matrix :')
    print(results)

    recall = recall_score(actual, predicted)
    print ('Recall:')
    print(recall)

    accuracy = accuracy_score(actual, predicted)
    print ('Accuracy:')
    print(accuracy)

    precision = precision_score(actual, predicted)
    print ('Precision:')
    print(precision)

    fscore = f1_score(actual, predicted)
    print ('F1:')
    print(fscore)

    fpr, tpr, _ = roc_curve(actual, scores)
    auc = roc_auc_score(actual, scores)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.savefig("/content/drive/MyDrive/EfficientDet/metrics/roc.jpg")


if __name__ == '__main__':
    main()
