# ANN-Digit-Recognizer
A simple handwritten digit recognition system built completely from scratch using NumPy.
It trains a small neural network (MLP) on the MNIST dataset to classify digits (0–9).

⚙️ Requirements
pip install numpy matplotlib pillow
# Optional (for MNIST download)
pip install tensorflow  # or scikit-learn


If offline, download mnist.npz

and place it in the same folder as the script.

🚀 Usage

Train the model

python handwritten_digit_recognizer_numpy.py
# → Enter 1 when prompted


Predict a custom image

python handwritten_digit_recognizer_numpy.py
# → Enter 2 and provide image path (e.g., digit.png)

🧩 Model Summary

Input: 784 (28×28)

Hidden: 128 (ReLU)

Output: 10 (Softmax)

Loss: Cross-Entropy

Optimizer: SGD

📈 Results

~95% test accuracy on MNIST after 10 epochs.
Supports prediction from user-uploaded digit images (PNG/JPG).

📜 Author

Varun Kumar Madala
