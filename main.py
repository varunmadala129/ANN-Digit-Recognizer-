def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def feedforward(input_layer, weights, biases):
    activations = [input_layer]
    for i in range(len(weights)):
        net_input = np.dot(activations[-1], weights[i]) + biases[i]
        activation = sigmoid(net_input)
        activations.append(activation)
    return activations

def backpropagate(input_layer, weights, biases, y_true, layer_sizes, epochs=1000, learning_rate=0.1):
    for epoch in range(epochs):
        activations = feedforward(input_layer, weights, biases)
        error = y_true - activations[-1]
        deltas = [error * sigmoid_derivative(activations[-1])]

        for i in range(len(layer_sizes) - 2, -1, -1):
            delta = deltas[-1].dot(weights[i + 1].T) * sigmoid_derivative(activations[i + 1])
            deltas.append(delta)
        deltas.reverse()

        for i in range(len(weights)):
            weights[i] += learning_rate * np.outer(activations[i], deltas[i])
            biases[i] += learning_rate * deltas[i]
    return weights, biases
rom IPython.display import display, HTML

# Prompt user
display(HTML("<h3 style='color:green;'>📤 Please upload a clear handwritten digit image (28x28 or larger)...</h3>"))
uploaded = files.upload()

for filename in uploaded.keys():
    img = Image.open(filename).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))               # Resize to match MNIST format
    img_array = np.array(img)

    # Optional: Invert colors if background is white (common for paper)
    img_array = 255 - img_array

    # Normalize and flatten
    img_array = img_array.reshape(784) / 255.0

    # Predict using trained network
    prediction = feedforward(img_array, weights, biases)[-1]
    predicted_digit = np.argmax(prediction)

    # Show the image and prediction
    plt.imshow(np.array(img).reshape(28, 28), cmap='gray')
    plt.title(f"🧠 Predicted Digit: {predicted_digit}")
    plt.axis('off')
    plt.show()
