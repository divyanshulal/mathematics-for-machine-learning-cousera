#!/usr/bin/env python3
"""
Neural Network Mathematics Demonstration
Combining concepts from all three courses: Linear Algebra, Calculus, and PCA
"""

import numpy as np
import matplotlib.pyplot as plt


class SimpleNeuralNetwork:
    """
    A simple 2-layer neural network implemented from scratch
    Demonstrates mathematical concepts:
    - Linear algebra: matrix multiplications, transformations
    - Calculus: gradients, chain rule, backpropagation
    - Optimization: gradient descent
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """Initialize network with random weights"""
        np.random.seed(42)

        # Weights and biases (Linear Algebra!)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

        self.learning_rate = learning_rate

        # For visualization
        self.loss_history = []

    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, z):
        """Derivative of sigmoid (Calculus!)"""
        s = self.sigmoid(z)
        return s * (1 - s)

    def forward(self, X):
        """
        Forward propagation (Linear Algebra!)
        Computes: a2 = σ(σ(X·W1 + b1)·W2 + b2)
        """
        # Hidden layer
        self.z1 = X @ self.W1 + self.b1  # Linear transformation
        self.a1 = self.sigmoid(self.z1)  # Activation

        # Output layer
        self.z2 = self.a1 @ self.W2 + self.b2  # Linear transformation
        self.a2 = self.sigmoid(self.z2)  # Activation

        return self.a2

    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, X, y):
        """
        Backpropagation (Calculus + Chain Rule!)
        Computes gradients of loss with respect to all parameters
        """
        m = X.shape[0]

        # Output layer gradients
        dz2 = self.a2 - y  # Derivative of loss w.r.t. z2
        dW2 = (self.a1.T @ dz2) / m  # Gradient for W2
        db2 = np.sum(dz2, axis=0, keepdims=True) / m  # Gradient for b2

        # Hidden layer gradients (Chain rule!)
        dz1 = (dz2 @ self.W2.T) * self.sigmoid_derivative(self.z1)
        dW1 = (X.T @ dz1) / m  # Gradient for W1
        db1 = np.sum(dz1, axis=0, keepdims=True) / m  # Gradient for b1

        return dW1, db1, dW2, db2

    def update_parameters(self, dW1, db1, dW2, db2):
        """
        Gradient descent update (Optimization!)
        θ_new = θ_old - α·∇L(θ)
        """
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, y, epochs=1000):
        """Train the network"""
        for epoch in range(epochs):
            # Forward propagation
            y_pred = self.forward(X)

            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # Backward propagation
            dW1, db1, dW2, db2 = self.backward(X, y)

            # Update parameters
            self.update_parameters(dW1, db1, dW2, db2)

            if epoch % 100 == 0:
                accuracy = np.mean((y_pred > 0.5) == y)
                print(f"Epoch {epoch:4d}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    def predict(self, X):
        """Make predictions"""
        return self.forward(X) > 0.5


def demonstrate_xor_problem():
    """
    Solve the XOR problem - a classic non-linearly separable problem
    This demonstrates why we need hidden layers and non-linear activations
    """
    print("=" * 60)
    print("NEURAL NETWORK: XOR PROBLEM")
    print("=" * 60)

    # XOR dataset
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    print("\nXOR Truth Table:")
    print("  Input  | Output")
    print("  -------|-------")
    for i in range(len(X)):
        print(f"  {X[i]}  |   {y[i][0]}")

    # Create and train network
    print("\nTraining neural network...")
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
    nn.train(X, y, epochs=1000)

    # Test predictions
    print("\n" + "=" * 60)
    print("PREDICTIONS:")
    print("=" * 60)
    predictions = nn.forward(X)
    for i in range(len(X)):
        print(f"Input: {X[i]}, True: {y[i][0]}, Predicted: {predictions[i][0]:.4f}, "
              f"Class: {int(predictions[i][0] > 0.5)}")

    accuracy = np.mean((predictions > 0.5) == y)
    print(f"\nFinal Accuracy: {accuracy:.2%}")

    # Visualize
    fig = plt.figure(figsize=(14, 5))

    # Decision boundary
    ax1 = plt.subplot(1, 3, 1)
    h = 0.01
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    Z = nn.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
    plt.colorbar(label='Predicted Probability')

    # Plot data points
    for i in range(len(X)):
        color = 'red' if y[i][0] == 1 else 'blue'
        plt.scatter(X[i, 0], X[i, 1], c=color, s=200, edgecolors='black', linewidths=2)

    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Decision Boundary for XOR Problem')
    plt.grid(True, alpha=0.3)

    # Loss curve
    ax2 = plt.subplot(1, 3, 2)
    plt.plot(nn.loss_history, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)

    # Weight visualization
    ax3 = plt.subplot(1, 3, 3)
    weights = [nn.W1.flatten(), nn.W2.flatten()]
    labels = ['W1 (Input→Hidden)', 'W2 (Hidden→Output)']
    colors = ['blue', 'red']

    for w, label, color in zip(weights, labels, colors):
        plt.hist(w, bins=20, alpha=0.6, label=label, color=color)

    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Learned Weights')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('claude-demos/neural_network_xor.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'neural_network_xor.png'")


def demonstrate_gradient_flow():
    """Demonstrate gradient flow and backpropagation"""
    print("\n" + "=" * 60)
    print("GRADIENT FLOW IN BACKPROPAGATION")
    print("=" * 60)

    # Simple example with one sample
    X = np.array([[1.0, 0.5]])
    y = np.array([[1.0]])

    nn = SimpleNeuralNetwork(input_size=2, hidden_size=3, output_size=1, learning_rate=0.1)

    # Forward pass
    print("\nForward Pass:")
    print(f"Input X: {X}")
    y_pred = nn.forward(X)
    print(f"Predicted output: {y_pred[0, 0]:.4f}")
    print(f"True output: {y[0, 0]:.4f}")

    loss = nn.compute_loss(y, y_pred)
    print(f"Loss: {loss:.4f}")

    # Backward pass
    print("\nBackward Pass (Computing Gradients):")
    dW1, db1, dW2, db2 = nn.backward(X, y)

    print(f"\nGradient shapes:")
    print(f"  dW1: {dW1.shape} (Input→Hidden weights)")
    print(f"  db1: {db1.shape} (Hidden biases)")
    print(f"  dW2: {dW2.shape} (Hidden→Output weights)")
    print(f"  db2: {db2.shape} (Output biases)")

    print(f"\nGradient magnitudes:")
    print(f"  ||dW1||: {np.linalg.norm(dW1):.6f}")
    print(f"  ||dW2||: {np.linalg.norm(dW2):.6f}")

    # Show weight update
    W1_old = nn.W1.copy()
    W2_old = nn.W2.copy()

    nn.update_parameters(dW1, db1, dW2, db2)

    print(f"\nWeight changes after one update:")
    print(f"  ||ΔW1||: {np.linalg.norm(nn.W1 - W1_old):.6f}")
    print(f"  ||ΔW2||: {np.linalg.norm(nn.W2 - W2_old):.6f}")


def demonstrate_mathematical_concepts():
    """Highlight the mathematical concepts used"""
    print("\n" + "=" * 60)
    print("MATHEMATICAL CONCEPTS IN NEURAL NETWORKS")
    print("=" * 60)

    print("\n1. LINEAR ALGEBRA:")
    print("   • Matrix multiplication: z = X·W + b")
    print("   • Vector transformations")
    print("   • Dot products and norms")

    print("\n2. MULTIVARIATE CALCULUS:")
    print("   • Partial derivatives: ∂L/∂W, ∂L/∂b")
    print("   • Chain rule: ∂L/∂W₁ = ∂L/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁")
    print("   • Gradient descent: W ← W - α·∇L(W)")

    print("\n3. OPTIMIZATION:")
    print("   • Loss minimization")
    print("   • Gradient descent algorithm")
    print("   • Learning rate selection")

    print("\n4. ACTIVATION FUNCTIONS:")
    print("   • Non-linear transformations")
    print("   • Sigmoid: σ(z) = 1/(1 + e⁻ᶻ)")
    print("   • Derivative: σ'(z) = σ(z)·(1 - σ(z))")

    # Visualize activation function
    z = np.linspace(-6, 6, 200)
    sigmoid = 1 / (1 + np.exp(-z))
    sigmoid_deriv = sigmoid * (1 - sigmoid)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(z, sigmoid, linewidth=2, label='σ(z)')
    plt.xlabel('z')
    plt.ylabel('σ(z)')
    plt.title('Sigmoid Activation Function')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(z, sigmoid_deriv, linewidth=2, color='red', label="σ'(z)")
    plt.xlabel('z')
    plt.ylabel("σ'(z)")
    plt.title('Sigmoid Derivative')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('claude-demos/activation_functions.png', dpi=150, bbox_inches='tight')
    print("\nActivation function visualization saved as 'activation_functions.png'")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 60)
    print("NEURAL NETWORK MATHEMATICS DEMONSTRATION")
    print("Combining Linear Algebra, Calculus, and Optimization")
    print("=" * 60)

    demonstrate_xor_problem()
    demonstrate_gradient_flow()
    demonstrate_mathematical_concepts()

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
