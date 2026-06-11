import numpy as np

INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
EXPECTED = np.array([0, 1, 1, 0], dtype=float)
LEARNING_RATE = 0.5


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class XORNet:
    def __init__(self):
        self.w1 = np.zeros((2, 2))
        self.b1 = np.zeros(2)
        self.w2 = np.zeros(2)
        self.b2 = 0.0
        self.hidden = np.zeros(2)
        self.prediction = 0.0
        self.delta_hidden = np.zeros(2)

    def initialize(self):
        self.w1 = np.random.uniform(-1, 1, (2, 2))
        self.b1 = np.random.uniform(-1, 1, 2)
        self.w2 = np.random.uniform(-1, 1, 2)
        self.b2 = float(np.random.uniform(-1, 1))

    def forward_prop(self, x, y):
        inputs = np.array([x, y])
        self.hidden = sigmoid(inputs @ self.w1 + self.b1)
        self.prediction = float(sigmoid(self.hidden @ self.w2 + self.b2))

    def compute_loss(self):
        total = 0.0
        for i in range(4):
            self.forward_prop(INPUTS[i][0], INPUTS[i][1])
            diff = EXPECTED[i] - self.prediction
            total += diff * diff
        return total / 4.0

    def back_prop(self, i):
        d_loss = 2.0 * (self.prediction - EXPECTED[i])
        d_pred = self.prediction * (1.0 - self.prediction)
        delta_out = d_loss * d_pred

        self.delta_hidden = delta_out * self.w2 * self.hidden * (1.0 - self.hidden)

        self.w2 -= LEARNING_RATE * (delta_out * self.hidden)
        self.b2 -= LEARNING_RATE * delta_out

        for k in range(2):
            for j in range(2):
                self.w1[k][j] -= LEARNING_RATE * self.delta_hidden[j] * INPUTS[i][k]

        self.b1 -= LEARNING_RATE * self.delta_hidden


if __name__ == "__main__":
    net = XORNet()
    net.initialize()

    epochs = 50000
    for epoch in range(epochs):
        for i in range(4):
            net.forward_prop(INPUTS[i][0], INPUTS[i][1])
            net.back_prop(i)
        if epoch % 1000 == 0:
            print(f"epoch {epoch} - loss: {net.compute_loss():.4f}")

    print("\ntraining done\npredictions:")
    for i in range(4):
        net.forward_prop(INPUTS[i][0], INPUTS[i][1])
        print(f"[{INPUTS[i][0]:.0f}, {INPUTS[i][1]:.0f}] -> {net.prediction:.3f} (rounds to {round(net.prediction)})")
