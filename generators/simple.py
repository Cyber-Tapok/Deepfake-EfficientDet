from generators.common import Generator
import os
import numpy as np
from six import raise_from
import cv2
import xml.etree.ElementTree as ET

accepted_classes = {
    'fake': 0,
    'real': 1,
}


class SimpleGenerator(Generator):

    def __init__(
            self,
            data_dir,
            set_name,
            classes=accepted_classes,
            image_extension='.jpg',
            skip_truncated=False,
            skip_difficult=False,
            **kwargs
    ):
        """
        Args:
            data_dir: the path of directory which contains ImageSets directory
            set_name: test|trainval|train|val
            classes: class names tos id mapping
            image_extension: image filename ext
            **kwargs:
        """
        self.data_dir = data_dir
        self.set_name = set_name
        self.classes = classes
        self.image_names = [l.strip().split(None, 1)[0] for l in
                            open(os.path.join(data_dir, 'ImageSets', 'Main', set_name + '.txt')).readlines()]
        self.image_extension = image_extension
        # class ids to names mapping
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(SimpleGenerator, self).__init__(**kwargs)

    def size(self):
        return len(self.image_names)

    def num_classes(self):
        return len(self.classes)

    def has_label(self, label):
        return label in self.labels

    def has_name(self, name):
        return name in self.classes

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        path = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        image = cv2.imread(path)
        h, w = image.shape[:2]
        return float(w) / float(h)

    def load_image(self, image_index):
        path = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __parse_annotation(self, filename: str):

        file_class = 'real'

        if "fake" in filename:
            file_class = 'fake'

        box = np.zeros((4,))
        label = self.name_to_label(file_class)

        box[0] = 110.0 - 1
        box[1] = 110.0 - 1
        box[2] = 410.0 - 1
        box[3] = 410.0 - 1

        return box, label

    def _parse_annotations(self, filename: str):
        annotations = {'labels': np.empty((0,), dtype=np.int32), 'bboxes': np.empty((0, 4))}
        box, label = self.__parse_annotation(filename)
        annotations['bboxes'] = np.concatenate([annotations['bboxes'], [box]])
        annotations['labels'] = np.concatenate([annotations['labels'], [label]])

        return annotations

    def load_annotations(self, image_index):
        filename = self.image_names[image_index]
        return self._parse_annotations(filename)
