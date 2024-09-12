import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('D3.csv')
X = data[['X1', 'X2', 'X3']].values
Y = data['Y'].values

def gradient_descent(X, y, learning_rate, max_iterations, tolerance=1e-6):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.zeros((n + 1, 1))
    y = y.reshape(-1, 1)
    losses = []
    
    for iteration in range(max_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta_new = theta - learning_rate * gradients
        
        loss = (1/m) * np.sum(np.square(X_b.dot(theta) - y))
        losses.append(loss)
        
        if np.all(np.abs(theta_new - theta) < tolerance):
            break
        
        theta = theta_new
    
    return theta, losses, iteration + 1

learning_rates = [0.1, 0.01, 0.001]
max_iterations = 10000
results = {}

plt.figure(figsize=(10, 6))
for lr in learning_rates:
    print(f"\nTraining with Learning Rate: {lr}")
    theta, losses, iterations = gradient_descent(X, Y, learning_rate=lr, max_iterations=max_iterations)
    results[lr] = (theta, losses, iterations)
    
    print(f"Final Loss: {losses[-1]:.6f}")
    print(f"Iterations: {iterations}")
    print(f"Model: Y = {theta[0][0]:.4f} + {theta[1][0]:.4f}*X1 + {theta[2][0]:.4f}*X2 + {theta[3][0]:.4f}*X3")
    
    plt.plot(losses, label=f"LR={lr}")

plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss Over Iterations for Different Learning Rates")
plt.legend()
plt.yscale('log')
plt.savefig('multivariate_loss_plot.png')
plt.show()

# Find the best model (lowest final loss)
best_lr = min(results, key=lambda x: results[x][1][-1])
best_theta, best_losses, best_iterations = results[best_lr]

print("\nBest Model:")
print(f"Learning Rate: {best_lr}")
print(f"Final Loss: {best_losses[-1]:.6f}")
print(f"Iterations: {best_iterations}")
print(f"Model: Y = {best_theta[0][0]:.4f} + {best_theta[1][0]:.4f}*X1 + {best_theta[2][0]:.4f}*X2 + {best_theta[3][0]:.4f}*X3")

# Predict for new values
new_points = np.array([[1, 1, 1], [2, 0, 4], [3, 2, 1]])
predictions = np.c_[np.ones((3, 1)), new_points].dot(best_theta)

print("\nPredictions:")
for point, pred in zip(new_points, predictions):
    print(f"For (X1, X2, X3) = {tuple(point)}: Predicted Y = {pred[0]:.4f}")

print("\nImpact of Learning Rates:")
for lr, (_, losses, iterations) in results.items():
    print(f"\nLearning Rate {lr}:")
    print(f"  Final Loss: {losses[-1]:.6f}")
    print(f"  Iterations: {iterations}")