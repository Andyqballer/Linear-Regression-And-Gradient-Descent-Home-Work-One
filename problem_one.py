import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('D3.csv')
X1, X2, X3, Y = data['X1'].values, data['X2'].values, data['X3'].values, data['Y'].values

def gradient_descent(X, y, learning_rate, max_iterations, tolerance=1e-6):
    m = len(X)
    X_b = np.c_[np.ones((m, 1)), X.reshape(-1, 1)]
    theta = np.zeros((X_b.shape[1], 1))
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

learning_rates = [0.1, 0.05, 0.01]
max_iterations = 1000
variables = [('X1', X1), ('X2', X2), ('X3', X3)]
results = {}

plt.figure(figsize=(15, 5))
for i, (label, X) in enumerate(variables):
    plt.subplot(1, 3, i+1)
    print(f"\nTraining with {label}")
    for lr in learning_rates:
        print(f"Learning Rate: {lr}")
        theta, losses, iterations = gradient_descent(X, Y, learning_rate=lr, max_iterations=max_iterations)
        results[(label, lr)] = (theta, losses, iterations)
        
        print(f"Model: Y = {theta[0][0]:.4f} + {theta[1][0]:.4f} * {label}")
        print(f"Final Loss: {losses[-1]:.6f}")
        print(f"Iterations: {iterations}")
        
        plt.plot(losses, label=f"LR={lr}")
    
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"Loss Over Iterations for {label}")
    plt.legend()
plt.tight_layout()
plt.savefig('loss_plots.png')
plt.show()

# Analyze which explanatory variable has the lowest final loss
final_losses = {key: value[1][-1] for key, value in results.items()}
best_variable = min(final_losses, key=final_losses.get)
print(f"\nBest variable with lowest final loss: {best_variable[0]} (Learning Rate: {best_variable[1]})")

# Plot final regression models
plt.figure(figsize=(15, 5))
for i, (label, X) in enumerate(variables):
    plt.subplot(1, 3, i+1)
    X_b = np.c_[np.ones((len(X), 1)), X.reshape(-1, 1)]
    for lr in learning_rates:
        theta, _, _ = results[(label, lr)]
        plt.scatter(X, Y, color='blue', alpha=0.5, label='Data')
        plt.plot(X, X_b.dot(theta), label=f'LR={lr}')
    plt.xlabel(label)
    plt.ylabel('Y')
    plt.title(f'Linear Regression for {label}')
    plt.legend()
plt.tight_layout()
plt.savefig('regression_plots.png')
plt.show()

print("\nImpact of Learning Rates:")
for label, _ in variables:
    print(f"\nFor {label}:")
    for lr in learning_rates:
        _, losses, iterations = results[(label, lr)]
        print(f"  Learning Rate {lr}:")
        print(f"    Final Loss: {losses[-1]:.6f}")
        print(f"    Iterations: {iterations}")