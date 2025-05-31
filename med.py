import re
from collections import Counter
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api.layers import Conv2D, MaxPool2D
from keras.api.layers import Flatten, Dense, Dropout
from keras.api.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DefaultConv2D = partial(Conv2D, kernel_size=3, padding="same",
                        activation="relu", kernel_initializer="he_normal")
images_120 = "data/300"
image_index = pd.read_csv(r"data/table.csv")


def get_label(image_path: Path, image_labels: pd.DataFrame):
    image = cv2.imread(image_path)
    image_array = np.array(image)
    name = image_path.name
    for index, row in image_labels.iterrows():
        file_name = row['RXBASE 300'].split("/")[-1]
        if file_name == name:
            label = row['name']
            matcher = re.findall(r"\[.+]", str(label))
            if matcher:
                with_label = ((image_array / 255.0).astype(np.float32), matcher[0])

                return with_label


c = 0
with_labels = []
for img in Path(images_120).iterdir():
    label = get_label(img, image_index)
    if c > 100:
        break
    if label:
        c += 1
        with_labels.append(label)
for i in with_labels:
    print(i[1])
imgs = list(map(lambda x: x[0], with_labels))
labels = list(map(lambda x: x[1], with_labels))
label_counts = Counter(labels)
imgs_filtered = [x for x, label in zip(imgs, labels) if label_counts[label] > 1]
y_filtered = [label for label in labels if label_counts[label] > 1]
le = LabelEncoder()
integer_labels = le.fit_transform(y_filtered)
x_train, x_test, y_train, y_test = train_test_split(imgs_filtered, integer_labels, stratify=integer_labels,
                                                    test_size=0.5)
model = Sequential([
    DefaultConv2D(filters=32, kernel_size=3, input_shape=[225, 300, 3]),
    MaxPool2D(),
    DefaultConv2D(filters=64),
    DefaultConv2D(filters=64),
    MaxPool2D(),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    MaxPool2D(),
    Flatten(),
    Dense(units=64, activation="relu",
          kernel_initializer="he_normal"),
    Dropout(0.25),
    Dense(units=64, activation="relu",
          kernel_initializer="he_normal"),
    Dropout(0.25),
    Dense(units=le.classes_.shape[0], activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam")
x_train = np.array(x_train, dtype=np.float32)
y_train = tf.convert_to_tensor(y_train)

history = model.fit(x_train, y_train, epochs=50)

y_pred = model.predict(np.array(x_test))
print(y_pred)

predicted_classes = tf.argmax(y_pred, axis=1)

print(predicted_classes)
correct = 0
for pred, label in zip(predicted_classes, y_test):
    print(le.classes_[pred], le.classes_[label])
    if pred == label:
        correct += 1

print(f"correct: {correct}, total: {len(predicted_classes)}, accuracy: {correct / len(predicted_classes)}")
