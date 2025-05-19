import numpy as np


class MLP_Better:
    def __init__(self, layer_sizes, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.L = len(layer_sizes) - 1
        self.sizes = layer_sizes

        # Inicjalizacja wag i biasów (He)
        self.W = {}
        self.b = {}
        for l in range(1, self.L + 1):
            self.W[l] = np.random.randn(self.sizes[l], self.sizes[l - 1]) * np.sqrt(2. / self.sizes[l - 1])
            self.b[l] = np.random.randn(self.sizes[l], 1) * np.sqrt(2. / self.sizes[l - 1])

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

    def forward(self, X, is_training=True, keep_prob=0.8):

        cache = {'A0': X}
        A = X
        # Warstwy ukryte: sigmoid + dropout
        for l in range(1, self.L):
            Z = self.W[l] @ A + self.b[l]
            A = self.sigmoid(Z)
            if is_training:
                D = np.random.rand(*A.shape) < keep_prob
                A *= D / keep_prob  # Skalowanie dla zachowania oczekiwanej wartości
                cache[f'D{l}'] = D
            cache[f'Z{l}'], cache[f'A{l}'] = Z, A
        # Warstwa wyjściowa: softmax (bez dropoutu)
        ZL = self.W[self.L] @ A + self.b[self.L]
        AL = self.softmax(ZL)
        cache[f'Z{self.L}'], cache[f'A{self.L}'] = ZL, AL
        return AL, cache

    def compute_cost(self, AL, Y, lambd=0.01):
        m = Y.shape[1]
        eps = 1e-8
        cross_entropy = -np.sum(Y * np.log(AL + eps)) / m
        # Regularyzacja L2 dla wszystkich wag
        L2_cost = 0
        for l in range(1, self.L + 1):
            L2_cost += np.sum(self.W[l] ** 2)
        L2_cost = (lambd / (2 * m)) * L2_cost
        return cross_entropy + L2_cost

    def backward(self, AL, Y, cache, lambd=0.01):
        m = Y.shape[1]
        grads = {}

        dZ = AL - Y
        A_prev = cache[f'A{self.L - 1}']
        grads[f'dW{self.L}'] = (dZ @ A_prev.T) / m + (lambd / m) * self.W[self.L]  # L2
        grads[f'db{self.L}'] = np.sum(dZ, axis=1, keepdims=True) / m

        dA_prev = self.W[self.L].T @ dZ
        # Wstecz przez warstwy ukryte (sigmoid + dropout)
        for l in reversed(range(1, self.L)):
            A = cache[f'A{l}']
            A_prev = cache[f'A{l - 1}']
            dZ = dA_prev * self.sigmoid_derivative(A)
            if f'D{l}' in cache:  # Uwzględnij dropout
                dZ *= cache[f'D{l}'] / cache.get('keep_prob', 0.8)
            grads[f'dW{l}'] = (dZ @ A_prev.T) / m + (lambd / m) * self.W[l]  # L2
            grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m
            dA_prev = self.W[l].T @ dZ

        return grads

    def update_parameters(self, grads, alpha):
        for l in range(1, self.L + 1):
            self.W[l] -= alpha * grads[f'dW{l}']
            self.b[l] -= alpha * grads[f'db{l}']

    def fit(self, X_train, Y_train, X_val, Y_val,
            epochs=1000, alpha=0.01, batch_size=32,
            print_every=100, lambd=0.01, keep_prob=0.8):

        m_train = X_train.shape[1]
        min_cost = float('inf')
        best_W, best_b = {}, {}

        for e in range(1, epochs + 1):
            perm = np.random.permutation(m_train)
            X_shuf = X_train[:, perm]
            Y_shuf = Y_train[:, perm]
            epoch_cost = 0
            for i in range(0, m_train, batch_size):
                X_batch = X_shuf[:, i:i + batch_size]
                Y_batch = Y_shuf[:, i:i + batch_size]
                AL, cache = self.forward(X_batch, is_training=True, keep_prob=keep_prob)
                cost = self.compute_cost(AL, Y_batch, lambd=lambd)
                epoch_cost += cost * X_batch.shape[1]
                grads = self.backward(AL, Y_batch, cache, lambd=lambd)
                self.update_parameters(grads, alpha)
            epoch_cost /= m_train
            AL_val, _ = self.forward(X_val, is_training=False)
            val_cost = self.compute_cost(AL_val, Y_val, lambd=0)  # Bez L2 dla walidacji
            if print_every and e % print_every == 0:
                print(f"Epoch {e}/{epochs} - train cost: {epoch_cost:.6f}, val cost: {val_cost:.6f}")
            if val_cost < min_cost:
                min_cost = val_cost
                best_W = {l: self.W[l].copy() for l in self.W}
                best_b = {l: self.b[l].copy() for l in self.b}
            if val_cost > min_cost * 1.4:
                print("Early stopping...")
                self.W, self.b = best_W, best_b
                return
        self.W, self.b = best_W, best_b

    def predict(self, X):
        AL, _ = self.forward(X, is_training=False)
        return np.argmax(AL, axis=0)

