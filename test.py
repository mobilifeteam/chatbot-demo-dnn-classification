from __future__ import division, print_function, absolute_import

import os
import tflearn
import re
import time

number_pattern = re.compile("^\\d+$")

from settings import delta
from sample import generate_sample
from model import create_model

model = create_model()

print("Loading model...")
if not os.path.exists("./models/model.index"):
    print("No model found")
    exit(1)
model.load("./models/model")

valid = 0
total = 0
auto = 0
next_valid = False
next_invalid = False
stuck_timer = None
while True:
    sample = generate_sample()
    label, values, deltas = [sample[k] for k in ("label", "values", "deltas")]

    data = deltas if delta else values
    print("  [%s] %s" % (label, ", ".join([str(c) for c in data])))
    if delta:
        print("   -> %s" % (", ".join([str(c) for c in values])))
    prediction = model.predict([data])[0]
    
    index = 0
    max_index = -1
    max_value = -1
    rows = []
    for value in prediction:
        if max_value < value:
            max_value = value
            max_index = index
        rows.append("  [%s] %.6f" % (index, value))
        index += 1
    if max_index == label:
        next_valid = False
        valid += 1
    else:
        next_invalid = False
    total += 1
    index = 0
    print("Predicted (%s/%s : %.2f%%):" % (valid, total, valid * 100 / total))
    for row in rows:
        print("%s%s" % (row, " <" if index == max_index else ""))
        index += 1

    if stuck_timer is not None and time.time() - stuck_timer > 10:
        next_valid = False
        next_invalid = False
        print("==== Search Time Exceeded ====")
    
    if auto > 0 or next_valid or next_invalid:
        auto -= 1
        print("=" * 16)
        continue
    command = input("> ")
    if command == "q":
        break
    elif command == "valid":
        next_valid = True
        stuck_timer = time.time()
    elif command == "invalid":
        next_invalid = True
        stuck_timer = time.time()
    elif number_pattern.match(command) is not None:
        auto = int(command) - 1
        if auto < 0:
            auto = 0
