import os
import random
from settings import start, end, size, batch


def generate_sample():
    if random.choice([True, False]):
        # Ascending
        value = random.randint(start, end - size)
        values = [value]
        deltas = []

        while len(values) < size:
            old_value = value
            value = random.randint(value, end - size + len(values) - 1)
            values.append(value)
            deltas.append(value - old_value)
        
        return {
            "label": 2,
            "values": values,
            "deltas": deltas
        }
    elif random.choice([True, False]):
        # Deascending
        value = random.randint(size, end)
        values = [value]
        deltas = []

        while len(values) < size:
            old_value = value
            value = random.randint(start + size - len(values), value)
            values.append(value)
            deltas.append(value - old_value)
        
        return {
            "label": 0,
            "values": values,
            "deltas": deltas
        }
    else:
        # Same
        value = random.randint(start + 1, end - 1)
        values = [value]
        deltas = []

        while len(values) < size:
            old_value = value
            value = random.randint(value - 1, value + 1)
            values.append(value)
            deltas.append(value - old_value)
        
        return {
            "label": 1,
            "values": values,
            "deltas": deltas
        }


if not os.path.exists("./samples"):
    os.makedirs("./samples")

values_file = open("./samples/sample-value-%s-%s-%s.csv" % (
    start, end, batch
), "w")
deltas_file = open("./samples/sample-delta-%s-%s-%s.csv" % (
    start, end, batch
), "w")

while batch > 0:
    sample = generate_sample()
    label, values, deltas = [sample[k] for k in ("label", "values", "deltas")]
    values_file.write(",".join([str(c) for c in ([label] + values)]) + "\n")
    deltas_file.write(",".join([str(c) for c in ([label] + deltas)]) + "\n")
    batch -= 1

values_file.close()
deltas_file.close()
