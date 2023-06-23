import numpy as np

class GaussianProcessRegression:
    def __init__(self, kernel, sigma_noise=1e-3):
        self.kernel = kernel
        self.sigma_noise = sigma_noise
        self.X = None
        self.y = None
        self.K = None
        self.K_inv = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.K = self.kernel(X, X) + self.sigma_noise * np.eye(len(X))
        self.K_inv = np.linalg.inv(self.K)

    def predict(self, X_test):
        K_star = self.kernel(self.X, X_test)
        y_pred_mean = K_star.T @ self.K_inv @ self.y
        y_pred_cov = self.kernel(X_test, X_test) - K_star.T @ self.K_inv @ K_star
        return y_pred_mean, y_pred_cov

# Define the kernel function (RBF)
def rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    pairwise_sq_dists = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return variance * np.exp(-0.5 * pairwise_sq_dists / lengthscale**2)

# Generate toy dataset
np.random.seed(0)
X_train = np.linspace(-5, 5, 10).reshape(-1, 1)
y_train = np.sin(X_train) + 0.2 * np.random.randn(len(X_train), 1)

X_test = np.linspace(-6, 6, 100).reshape(-1, 1)

# Create an instance of GaussianProcessRegression
gpr = GaussianProcessRegression(kernel=rbf_kernel)

# Fit the model to the training data
gpr.fit(X_train, y_train)

# Predict on the test data
y_pred_mean, y_pred_cov = gpr.predict(X_test)

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color='b', label='Training Data')
plt.plot(X_test, y_pred_mean, color='r', label='Mean Prediction')
plt.fill_between(X_test.ravel(), y_pred_mean.ravel() - 2*np.sqrt(np.diag(y_pred_cov)), y_pred_mean.ravel() + 2*np.sqrt(np.diag(y_pred_cov)), color='gray', alpha=0.3, label='Uncertainty')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Gaussian Process Regression')
plt.legend()
plt.show()
