import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Define the dataset
X = np.array([1, 2, 3, 4]).reshape(-1, 1)
y = np.array([1, 4, 9, 15])

# Perform polynomial regression
def polynomial_regression(X, y, degree=2):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    return model, poly_features

# Create a function to plot the results
def plot_polynomial_regression(X, y, model, poly_features):
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_plot_poly = poly_features.transform(X_plot)

    y_plot = model.predict(X_plot_poly)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X_plot, y_plot, color='red', label='Polynomial regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Polynomial Regression (Degree {poly_features.degree})')

    # Create equation string
    coef = model.coef_.flatten()  # Flatten the coefficient array
    eq = f'y = {model.intercept_:.2f}'
    for i, c in enumerate(coef):
        if i == 0:
            eq += f' + {c:.2f}x'
        else:
            eq += f' + {c:.2f}x^{i+1}'




    plt.legend()
    plt.grid(True)
    plt.show()

# Perform regression and plot
model, poly_features = polynomial_regression(X, y)
plot_polynomial_regression(X, y, model, poly_features)

# Print the coefficients and R-squared score
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Print the equation
coef = model.coef_.flatten()  # Flatten the coefficient array
eq = f'y = {model.intercept_:.2f}'
for i, c in enumerate(coef):
    if i == 0:
        eq += f' + {c:.2f}x'
    else:
        eq += f' + {c:.2f}x^{i+1}'
print("\nPolynomial Regression Equation:")
print(eq)