import tflearn

from settings import delta, size


def create_model():
    net = tflearn.input_data([None, (size - 1) if delta else size])

    # net = tflearn.embedding(net, input_dim=10000, output_dim=256)
    # net = tflearn.lstm(net, 256, dropout=0.9)
    # net = tflearn.fully_connected(net, 3, activation='softmax')
    # net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

    net = tflearn.fully_connected(net, 128, activation="tanh", regularizer="L2")
    net = tflearn.dropout(net, 0.8)
    net = tflearn.fully_connected(net, 256, activation="tanh", regularizer="L2")
    net = tflearn.dropout(net, 0.8)
    net = tflearn.fully_connected(net, 3, activation="softmax")
    net = tflearn.regression(net, optimizer="adam", learning_rate=0.001, loss="categorical_crossentropy")

    return tflearn.DNN(net)
