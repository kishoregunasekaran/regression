import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data from the image
X = np.array([
    [1, 4],
    [2, 5],
    [3, 8],
    [4, 2]
])
y = np.array([1, 6, 8, 12])

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Print the coefficients and intercept
print("Coefficients:")
print(f"a0 (Intercept): {model.intercept_:.3f}")
print(f"a1 (Product 1): {model.coef_[0]:.3f}")
print(f"a2 (Product 2): {model.coef_[1]:.3f}")

# Construct the equation
equation = f"y = {model.intercept_:.3f} + {model.coef_[0]:.3f}x1 + {model.coef_[1]:.3f}x2"
print(f"\nMultiple Linear Regression Equation:\n{equation}")

# Predict 5th week sales
week_5_data = np.array([[5, 6]])  # 5th week data for Product 1 and Product 2
predicted_sales = model.predict(week_5_data)
print(f"\nPredicted 5th week sales: {predicted_sales[0]:.3f} lakhs")

# Visualize the data and predictions
fig = plt.figure(figsize=(12, 6))

# 3D scatter plot
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o')
ax.set_xlabel('Product 1 Sales')
ax.set_ylabel('Product 2 Sales')
ax.set_zlabel('Weekly Sales (in lakhs)')
ax.set_title('3D Scatter Plot of Sales Data')

# Create a meshgrid for the prediction surface
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 10)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)

# Plot the prediction surface
ax.plot_surface(X1, X2, Z, alpha=0.5)

# 2D plot for actual vs predicted values
ax2 = fig.add_subplot(122)
y_pred = model.predict(X)
ax2.scatter(y, y_pred, c='b', marker='o')
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Sales')
ax2.set_ylabel('Predicted Sales')
ax2.set_title('Actual vs Predicted Sales')

plt.tight_layout()
plt.show()