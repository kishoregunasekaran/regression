import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Data
weeks = [1, 2, 3, 4,5]
sales = [1.2, 1.8, 2.6, 3.2,3.8]

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(weeks, sales, color='red', marker='o')

# Fit the linear regression model
X = np.array(weeks).reshape(-1, 1)
y = np.array(sales)
reg = LinearRegression().fit(X, y)

# Get intercept and slope
intercept = reg.intercept_
slope = reg.coef_[0]

# Generate points for the regression line
line_x = np.array([0, 5])
line_y = intercept + slope * line_x

# Plot the regression line
plt.plot(line_x, line_y, color='blue')

# Customize the plot
plt.title('Sales Regression Analysis')
plt.xlabel('Week')
plt.ylabel('Sales')
plt.xlim(0, 5)
plt.ylim(0, 5)

# Add text for intercept and regression equation
plt.text(0.1, 0.2, f'Intercept: {intercept:.2f}', transform=plt.gca().transAxes)
plt.text(0.1, 0.1, f'y = {intercept:.2f} + {slope:.2f}x', transform=plt.gca().transAxes)

# Add point labels
for i, (x, y) in enumerate(zip(weeks, sales)):
    plt.annotate(f'({x}, {y:.1f})', (x, y), xytext=(5, 5), textcoords='offset points')

# Calculate and print predictions for 7th and 9th month
month_7 = intercept + slope * 7
month_9 = intercept + slope * 9
print(f"7th month sales: y = {intercept:.2f} + ({slope:.2f} * 7) = {month_7:.2f}")
print(f"9th month sales: y = {intercept:.2f} + ({slope:.2f} * 9) = {month_9:.2f}")

# Show the plot
plt.show()

# Print regression details
print(f"\nRegression Equation: y = {intercept:.2f} + {slope:.2f}x")
print(f"Intercept: {intercept:.2f}")
print(f"Slope: {slope:.2f}")