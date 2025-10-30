# handwritten_digit_recognizer_numpy.py
# NumPy-based MLP for MNIST (one hidden layer). Includes training, evaluation,
# plotting, saving/loading model, and custom image prediction helper.

import os
import sys
import time
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# ---------------------------
# Data loading utilities
# ---------------------------
def load_mnist_try_keras():
    try:
        # Attempt to load via tensorflow.keras (works if TF is installed)
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return x_train, y_train, x_test, y_test
    except Exception:
        return None

def load_mnist_try_sklearn():
    try:
        from sklearn.datasets import fetch_openml
        mn = fetch_openml('mnist_784', version=1, as_frame=False, cache=True)
        data = mn['data']
        targets = mn['target'].astype(np.int64)
        x = data.reshape(-1, 28, 28)
        y = targets
        # split: first 60k train, last 10k test (same split as classic MNIST)
        x_train, y_train = x[:60000], y[:60000]
        x_test, y_test = x[60000:], y[60000:]
        return x_train, y_train, x_test, y_test
    except Exception:
        return None

def load_mnist_from_npz(npz_path='mnist.npz'):
    if os.path.exists(npz_path):
        with np.load(npz_path) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
        return x_train, y_train, x_test, y_test
    return None

def load_mnist():
    # Try multiple loaders
    print("Loading MNIST dataset...")
    res = load_mnist_try_keras()
    if res:
        print("Loaded MNIST via tensorflow.keras.datasets")
        return res
    res = load_mnist_try_sklearn()
    if res:
        print("Loaded MNIST via sklearn.fetch_openml")
        return res
    res = load_mnist_from_npz()
    if res:
        print("Loaded MNIST from local mnist.npz")
        return res
    raise RuntimeError("Could not load MNIST. Install tensorflow or sklearn, or place 'mnist.npz' in working directory.")

# ---------------------------
# Model (NumPy MLP)
# ---------------------------
def one_hot(y, n_classes=10):
    out = np.zeros((y.size, n_classes))
    out[np.arange(y.size), y] = 1
    return out

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(z):
    # numerically stable softmax
    z = z - np.max(z, axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / np.sum(expz, axis=1, keepdims=True)

def cross_entropy_loss(probs, y_true_onehot):
    # probs: (N, C), y_true_onehot: (N, C)
    N = probs.shape[0]
    # avoid log(0)
    clipped = np.clip(probs, 1e-12, 1.0)
    loss = -np.sum(y_true_onehot * np.log(clipped)) / N
    return loss

class SimpleMLP:
    def __init__(self, input_dim=28*28, hidden_dim=128, output_dim=10, seed=42):
        rng = np.random.RandomState(seed)
        # Xavier init for weights
        self.W1 = rng.randn(input_dim, hidden_dim) * np.sqrt(2.0/(input_dim + hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.randn(hidden_dim, output_dim) * np.sqrt(2.0/(hidden_dim + output_dim))
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        # X: (N, D)
        Z1 = X.dot(self.W1) + self.b1    # (N, H)
        A1 = relu(Z1)                    # (N, H)
        Z2 = A1.dot(self.W2) + self.b2   # (N, C)
        probs = softmax(Z2)              # (N, C)
        cache = (X, Z1, A1, Z2, probs)
        return probs, cache

    def backward(self, cache, y_onehot):
        # cache from forward
        X, Z1, A1, Z2, probs = cache
        N = X.shape[0]
        # derivative w.r.t. Z2
        dZ2 = (probs - y_onehot) / N     # (N, C)
        dW2 = A1.T.dot(dZ2)              # (H, C)
        db2 = np.sum(dZ2, axis=0, keepdims=True)  # (1, C)

        dA1 = dZ2.dot(self.W2.T)         # (N, H)
        dZ1 = dA1 * relu_derivative(Z1)  # (N, H)
        dW1 = X.T.dot(dZ1)               # (D, H)
        db1 = np.sum(dZ1, axis=0, keepdims=True)  # (1, H)

        grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        return grads

    def update_params(self, grads, lr=0.01):
        self.W1 -= lr * grads['dW1']
        self.b1 -= lr * grads['db1']
        self.W2 -= lr * grads['dW2']
        self.b2 -= lr * grads['db2']

    def predict(self, X):
        probs, _ = self.forward(X)
        preds = np.argmax(probs, axis=1)
        return preds, probs

    def save(self, path='mlp_weights.npz'):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"Saved model to {path}")

    def load(self, path='mlp_weights.npz'):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        data = np.load(path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        print(f"Loaded model from {path}")

# ---------------------------
# Training & evaluation
# ---------------------------
def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def train(model, X_train, y_train, X_val, y_val,
          epochs=10, batch_size=128, lr=0.1, verbose=True):
    n = X_train.shape[0]
    steps = int(np.ceil(n / batch_size))
    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        # shuffle
        perm = np.random.permutation(n)
        X_train = X_train[perm]
        y_train = y_train[perm]

        epoch_loss = 0.0
        t0 = time.time()
        for b in range(steps):
            start = b * batch_size
            end = min(start + batch_size, n)
            Xb = X_train[start:end]
            yb = y_train[start:end]
            yb_onehot = one_hot(yb, 10)

            probs, cache = model.forward(Xb)
            loss = cross_entropy_loss(probs, yb_onehot)
            epoch_loss += loss * (end - start)

            grads = model.backward(cache, yb_onehot)
            model.update_params(grads, lr=lr)

        epoch_loss /= n
        # evaluate
        y_pred_train, _ = model.predict(X_train[:10000])  # quick estimate
        train_acc = accuracy(y_pred_train, y_train[:10000])
        y_pred_val, _ = model.predict(X_val)
        val_acc = accuracy(y_pred_val, y_val)
        # compute val loss
        val_probs, _ = model.forward(X_val)
        val_loss = cross_entropy_loss(val_probs, one_hot(y_val, 10))

        history['loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if verbose:
            print(f"Epoch {epoch}/{epochs} - time: {time.time()-t0:.1f}s - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f} - acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")

    return history

def plot_history(history):
    epochs = len(history['loss'])
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(range(1, epochs+1), history['loss'], label='train loss')
    plt.plot(range(1, epochs+1), history['val_loss'], label='val loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(range(1, epochs+1), history['acc'], label='train acc')
    plt.plot(range(1, epochs+1), history['val_acc'], label='val acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy')

    plt.tight_layout()
    plt.show()

# ---------------------------
# Preprocessing helpers
# ---------------------------
def preprocess_images(x):
    # x: (N, 28, 28) uint8 -> flatten, normalize to [0,1]
    x = x.astype(np.float32) / 255.0
    N = x.shape[0]
    return x.reshape(N, -1)

def prepare_data():
    x_train, y_train, x_test, y_test = load_mnist()
    # use a validation set from train
    # shuffle & split
    perm = np.random.permutation(x_train.shape[0])
    x_train = x_train[perm]
    y_train = y_train[perm]
    val_size = 10000
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train2 = x_train[val_size:]
    y_train2 = y_train[val_size:]

    X_train = preprocess_images(x_train2)
    X_val = preprocess_images(x_val)
    X_test = preprocess_images(x_test)
    return X_train, y_train2, X_val, y_val, X_test, y_test

def preprocess_single_image(path):
    # loads image, converts to grayscale 28x28, invert & normalize like MNIST
    img = Image.open(path).convert('L')  # grayscale
    # auto-crop? make square & resize
    img = ImageOps.invert(img)  # invert so that digit is white on black as in MNIST
    img = img.resize((28, 28), Image.ANTIALIAS)
    arr = np.array(img).astype(np.float32) / 255.0
    flat = arr.reshape(1, -1)
    return flat

# ---------------------------
# CLI-like main
# ---------------------------
def main_train_and_demo():
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    print("Data shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    # Instantiate model
    model = SimpleMLP(input_dim=28*28, hidden_dim=128, output_dim=10, seed=123)

    # Train
    history = train(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=128, lr=0.1, verbose=True)

    # Plot history
    plot_history(history)

    # Evaluate on test set
    y_pred_test, probs = model.predict(X_test)
    test_acc = accuracy(y_pred_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save model
    model.save('mlp_mnist_weights.npz')

    # show some predictions
    n_show = 8
    idxs = np.random.choice(X_test.shape[0], size=n_show, replace=False)
    plt.figure(figsize=(12,3))
    for i, idx in enumerate(idxs):
        ax = plt.subplot(1, n_show, i+1)
        img = X_test[idx].reshape(28,28)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f"Pred: {y_pred_test[idx]} / True: {y_test[idx]}")
    plt.show()

    # Return trained model
    return model

def demo_predict_custom_image(model_path='mlp_mnist_weights.npz', image_path=None):
    model = SimpleMLP(input_dim=28*28, hidden_dim=128, output_dim=10)
    model.load(model_path)
    if image_path is None:
        print("No image provided. Provide path to an image of a digit (image_path).")
        return
    x = preprocess_single_image(image_path)
    pred, probs = model.predict(x)
    print("Predicted digit:", int(pred[0]))
    print("Probabilities:", probs[0])
    # show image
    plt.imshow(x.reshape(28,28), cmap='gray')
    plt.title(f"Predicted: {int(pred[0])}")
    plt.axis('off')
    plt.show()

# Only run training/demo if executed as main script
if __name__ == '__main__':
    # Quick menu
    print("Handwritten Digit Recognizer (NumPy MLP)\nOptions:\n1. Train model\n2. Predict a custom image using saved model\n")
    choice = input("Enter 1 or 2: ").strip()
    if choice == '1':
        model = main_train_and_demo()
    elif choice == '2':
        model_path = input("Enter model path (default mlp_mnist_weights.npz): ").strip() or 'mlp_mnist_weights.npz'
        img_path = input("Enter path to digit image: ").strip()
        demo_predict_custom_image(model_path=model_path, image_path=img_path)
    else:
        print("Invalid choice. Exiting.")
