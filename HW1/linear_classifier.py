import numpy as np
import time
from download_mnist import load

# Load MNIST dataset
x_train, y_train, x_test, y_test = load()

# Normalize the images to have pixel values between 0 and 1
x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32) / 255.0

# Add a bias term by appending a column of ones to each image
x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
x_test  = np.concatenate([x_test,  np.ones((x_test.shape[0], 1))], axis=1)

input_dim = x_train.shape[1]  # 785 features (784 pixels + 1 bias)
num_classes = 10              # Digits 0 through 9

def softmax(scores):
    # Compute the softmax probabilities
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def cross_entropy_loss(W, x, y):
    # Calculate the cross-entropy loss for weight matrix W
    scores = x.dot(W)              
    probs = softmax(scores)        
    # Get the log probability for the correct class for each sample
    correct_logprobs = -np.log(probs[np.arange(len(y)), y] + 1e-8)
    return np.mean(correct_logprobs)

def compute_accuracy(W, x, y):
    scores = x.dot(W)
    predictions = np.argmax(scores, axis=1)
    return np.mean(predictions == y)

# Use random search to find a good weight matrix W
best_loss = float('inf')
best_W = None
num_iterations = 100  # Try 100 random candidates

start_time = time.time()
for i in range(num_iterations):
    # Generate a candidate weight matrix with small random values
    W_candidate = np.random.randn(input_dim, num_classes) * 0.01
    loss = cross_entropy_loss(W_candidate, x_train, y_train)
    if loss < best_loss:
        best_loss = loss
        best_W = W_candidate
        print(f"Iteration {i}: Loss = {best_loss:.4f}")

elapsed_time = time.time() - start_time

# Test the best weight matrix on the test set
test_acc = compute_accuracy(best_W, x_test, y_test)
print(f"\nTime taken: {elapsed_time:.2f} seconds")
print(f"Training loss: {best_loss:.4f}")
print(f"Test accuracy: {test_acc * 100:.2f}%")
