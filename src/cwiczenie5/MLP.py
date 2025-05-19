import numpy as np

class MLP:
    def __init__(self, layer_sizes, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.L = len(layer_sizes) - 1
        self.sizes = layer_sizes

        # Inicjalizacja wag i biasów (He)
        self.W = {}
        self.b = {}
        for l in range(1, self.L + 1):
            self.W[l] = np.random.randn(self.sizes[l], self.sizes[l-1]) * np.sqrt(2. / self.sizes[l-1])
            self.b[l] = np.random.randn(self.sizes[l], 1) * np.sqrt(2. / self.sizes[l-1])

    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def sigmoid_derivative(A):
        return A * (1 - A)

    @staticmethod
    def softmax(Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def forward(self, X):
        cache = {'A0': X}
        A = X
        # warstwy ukryte: sigmoid
        for l in range(1, self.L):
            Z = self.W[l] @ A + self.b[l]
            A = self.sigmoid(Z)
            cache[f'Z{l}'], cache[f'A{l}'] = Z, A
        # warstwa wyjściowa: softmax
        ZL = self.W[self.L] @ A + self.b[self.L]
        AL = self.softmax(ZL)
        cache[f'Z{self.L}'], cache[f'A{self.L}'] = ZL, AL
        return AL, cache

    def compute_cost(self, AL, Y):

        m = Y.shape[1]
        # dodaj epsilon dla stabilności
        eps = 1e-8
        cost = -np.sum(Y * np.log(AL + eps)) / m
        return cost

    def backward(self, AL, Y, cache):

        m = Y.shape[1]
        grads = {}

        # dZ dla ostatniej warstwy
        dZ = AL - Y
        A_prev = cache[f'A{self.L-1}']
        grads[f'dW{self.L}'] = (dZ @ A_prev.T) / m
        grads[f'db{self.L}'] = np.sum(dZ, axis=1, keepdims=True) / m

        dA_prev = self.W[self.L].T @ dZ
        # wstecz przez warstwy ukryte
        for l in reversed(range(1, self.L)):
            A = cache[f'A{l}']
            A_prev = cache[f'A{l-1}']
            dZ = dA_prev * self.sigmoid_derivative(A)
            grads[f'dW{l}'] = (dZ @ A_prev.T) / m
            grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m
            dA_prev = self.W[l].T @ dZ

        return grads

    def update_parameters(self, grads, alpha):
        for l in range(1, self.L + 1):
            self.W[l] -= alpha * grads[f'dW{l}']
            self.b[l] -= alpha * grads[f'db{l}']

    def fit(self,
            X_train, Y_train,
            X_val, Y_val,
            epochs=1000,
            alpha=0.01,
            batch_size=32,
            print_every=100):
        m_train = X_train.shape[1]
        min_cost = float('inf')
        best_W, best_b = {}, {}

        for e in range(1, epochs + 1):
            perm = np.random.permutation(m_train)
            X_shuf = X_train[:, perm]
            Y_shuf = Y_train[:, perm]
            epoch_cost = 0
            for i in range(0, m_train, batch_size):
                X_batch = X_shuf[:, i:i+batch_size]
                Y_batch = Y_shuf[:, i:i+batch_size]
                AL, cache = self.forward(X_batch)
                cost = self.compute_cost(AL, Y_batch)
                epoch_cost += cost * X_batch.shape[1]
                grads = self.backward(AL, Y_batch, cache)
                self.update_parameters(grads, alpha)
            epoch_cost /= m_train
            AL_val, _ = self.forward(X_val)
            val_cost = self.compute_cost(AL_val, Y_val)
            if print_every and e % print_every == 0:
                print(f"Epoch {e}/{epochs} - train cost: {epoch_cost:.6f}, val cost: {val_cost:.6f}")
            # zapis najlepszych wag
            if val_cost < min_cost:
                min_cost = val_cost
                best_W = {l: self.W[l].copy() for l in self.W}
                best_b = {l: self.b[l].copy() for l in self.b}
            if val_cost > min_cost * 1.4:
                self.W, self.b = best_W, best_b
                return
        # przywrócenie najlepszych
        self.W, self.b = best_W, best_b

    def predict(self, X):
        AL, _ = self.forward(X)
        return np.argmax(AL, axis=0)
