from __future__ import division, print_function, absolute_import

import os
import numpy as np
import tensorflow as tf
import tflearn

from settings import delta, start, end, batch, training_epoch, training_size
from model import create_model
from tflearn.data_utils import load_csv

data, labels = load_csv(
    "./samples/sample-%s-%s-%s-%s.csv" % (
        "delta" if delta else "value", start, end, batch
    ),
    target_column=0,
    has_header=False,
    categorical_labels=True,
    n_classes=3
)

def preprocess(data):
    return np.array(data, dtype=np.float32)

data = preprocess(data)

model = create_model()
model.fit(
    data, labels,
    n_epoch=training_epoch,
    batch_size=training_size,
    show_metric=True
)

print("Saving model...")
if not os.path.exists("./models"):
    os.makedirs("./models")
model.save("./models/model")
