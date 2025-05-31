import numpy as np
from tensorflow.keras.datasets import mnist

def softmax(logits):
    """
    Computing softmax probabilities
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def cross_entropy_loss(probs, y_true):
    """
    Compute the cross-entropy loss
    y_true: vector of integer labels
    """
    n = y_true.shape[0]
    # add a small constant for numerical stability
    correct_logprobs = -np.log(probs[np.arange(n), y_true] + 1e-10)
    return np.sum(correct_logprobs) / n

def compute_gradients(X, y, W):
    """
    Computing the gradients for the linear classifier
    X: Input data matrix of shape
    y: True labels
    W: Weight matrix of shape
    Returns:
        grad_W: Gradient with respect to W
        probs: Predicted probabilities
    """
    n = X.shape[0]
    logits = X.dot(W)           # shape: (n_samples, n_classes)
    probs = softmax(logits)     # shape: (n_samples, n_classes)
    
    # create one-hot encoding of y
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), y] = 1
    
    # gradient of the loss with respect to logits
    dlogits = (probs - one_hot) / n
    # gradient with respect to W
    grad_W = X.T.dot(dlogits)
    
    return grad_W, probs

# loading the MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# preprocess the data by flattening and normalize images
x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0

x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
x_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))])

n_features = x_train.shape[1]
n_classes = 10

# initialize weights
np.random.seed(0)
W = np.random.randn(n_features, n_classes) * 0.01

# hyperparameters
learning_rate = 0.1
n_epochs = 100  

for epoch in range(n_epochs):
    grad_W, probs = compute_gradients(x_train, y_train, W)
    W -= learning_rate * grad_W

    # computing training loss and accuracy
    loss = cross_entropy_loss(probs, y_train)
    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y_train)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Training Accuracy = {acc*100:.2f}%")

# evaluating the model on the test set.
test_logits = x_test.dot(W)
test_probs = softmax(test_logits)
test_preds = np.argmax(test_probs, axis=1)
test_acc = np.mean(test_preds == y_test)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
