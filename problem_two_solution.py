import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('D3.csv')
X = data[['X1', 'X2', 'X3']].values
y = data['Y'].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def gradient_descent(X, y, learning_rate, max_iterations, tolerance=1e-6):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.zeros((n + 1, 1))
    y = y.reshape(-1, 1)
    losses = []
    
    for iteration in range(max_iterations):
        predictions = X_b.dot(theta)
        error = predictions - y
        gradients = 2/m * X_b.T.dot(error)
        theta_new = theta - learning_rate * gradients
        
        loss = (1/m) * np.sum(np.square(error))
        losses.append(loss)
        
        if np.all(np.abs(theta_new - theta) < tolerance):
            break
        
        theta = theta_new
        
        if np.isnan(loss) or np.isinf(loss) or np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            print(f"Encountered NaN or Inf at iteration {iteration}. Stopping early.")
            break
    
    return theta, losses, iteration + 1

learning_rates = [0.1, 0.01, 0.001]
max_iterations = 10000
results = {}

plt.figure(figsize=(15, 5))
for i, lr in enumerate(learning_rates):
    print(f"\nTraining with Learning Rate: {lr}")
    theta, losses, iterations = gradient_descent(X_scaled, y, learning_rate=lr, max_iterations=max_iterations)
    results[lr] = (theta, losses, iterations)
    
    print(f"Final Loss: {losses[-1]:.6f}")
    print(f"Iterations: {iterations}")
    print(f"Model: Y = {theta[0][0]:.4f} + {theta[1][0]:.4f}*X1 + {theta[2][0]:.4f}*X2 + {theta[3][0]:.4f}*X3")
    
    plt.subplot(1, 3, i+1)
    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"Loss Over Iterations (LR={lr})")
    plt.yscale('log')

plt.tight_layout()
plt.savefig('multivariate_loss_plot.png')
plt.show()

# Find the best model (lowest final loss)
best_lr = min(results, key=lambda x: results[x][1][-1])
best_theta, best_losses, best_iterations = results[best_lr]

print("\n1. Best Model:")
print(f"Learning Rate: {best_lr}")
print(f"Final Loss: {best_losses[-1]:.6f}")
print(f"Iterations: {best_iterations}")
print(f"Model: Y = {best_theta[0][0]:.4f} + {best_theta[1][0]:.4f}*X1 + {best_theta[2][0]:.4f}*X2 + {best_theta[3][0]:.4f}*X3")

print("\n2. Loss over iteration plot has been saved as 'multivariate_loss_plot.png'")

print("\n3. Impact of Learning Rates:")
for lr, (_, losses, iterations) in results.items():
    print(f"\nLearning Rate {lr}:")
    print(f"  Final Loss: {losses[-1]:.6f}")
    print(f"  Iterations: {iterations}")

# Predict for new values
new_points = np.array([[1, 1, 1], [2, 0, 4], [3, 2, 1]])
new_points_scaled = scaler.transform(new_points)
predictions = np.c_[np.ones((3, 1)), new_points_scaled].dot(best_theta)

print("\n4. Predictions:")
for point, pred in zip(new_points, predictions):
    print(f"For (X1, X2, X3) = {tuple(point)}: Predicted Y = {pred[0]:.4f}")