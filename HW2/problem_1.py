import numpy as np

def relu(z):
    return np.maximum(0, z)

def f(W, x):
    """
    Compute f(W, x) = ||ReLU(Wx)||^2
    """
    z = W @ x         
    a = relu(z)       # apply ReLU element-wise
    return np.sum(a**2)

def grad_f(W, x):
    """
    Compute the gradients of f(W, x) with respect to W and x.
    Returns:
        grad_W: Gradient with respect to W (same shape as W)
        grad_x: Gradient with respect to x (same shape as x)
    """
    z = W @ x
    a = relu(z)
    
    # derivative of f with respect to z: 2a multiplied by indicator function (z > 0)
    dz = 2 * a * (z > 0).astype(float)
    
    # gradient with respect to x (using chain rule)
    grad_x = W.T @ dz
    
    # gradient with respect to W is the outer product of dz and x.
    grad_W = np.outer(dz, x)
    
    return grad_W, grad_x

if __name__ == "__main__":
    np.random.seed(0)
    W = np.random.randn(3, 3)  # 3x3 matrix
    x = np.random.randn(3, 1)  # 3x1 vector

    print("f(W, x) =", f(W, x))
    gradW, gradx = grad_f(W, x)
    print("Gradient with respect to W:\n", gradW)
    print("Gradient with respect to x:\n", gradx)
