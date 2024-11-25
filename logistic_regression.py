import numpy as np
import matplotlib.pyplot as plt

def logistic_function(x, a0, a1):
    z = a0 + a1 * x
    return 1 / (1 + np.exp(-z))

# Parameters
a0 = 1
a1 = 8
threshold = 0.5

# Student's score
x = 60

# Calculate probability
p = a0 + a1 * x

# Calculate logistic regression result
y = logistic_function(x, a0, a1)

# Determine if student is selected
selected = y > threshold

# Print results
print(f"The equation for Logistic regression is:")
print(f"y = 1 / (1 + e^(-x))")
print(f"\nThe probability for x is:")
print(f"p(x) = z = a0 + a1*x")
print(f"\nGiven a0 = {a0}, a1 = {a1}, x = {x} marks, threshold > {threshold}")
print(f"\np(x) = z = {a0} + {a1} * {x} = {p}")
print(f"\nThe logistic regression equation is:")
print(f"y = 1 / (1 + e^(-{p:.2f})) = {y:.10f}")
print(f"\nSince {y:.10f} > {threshold}, the student with marks = {x}, is {'selected' if selected else 'not selected'}")

# Plotting
x_range = np.linspace(0, 100, 1000)
y_range = logistic_function(x_range, a0, a1)

plt.figure(figsize=(10, 6))
plt.plot(x_range, y_range, label='Logistic Curve')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.axvline(x=x, color='g', linestyle='--', label='Student Score')
plt.scatter(x, y, color='b', s=100, zorder=5, label='Student')
plt.xlabel('Score')
plt.ylabel('Probability of Selection')
plt.title('Logistic Regression for Student Selection')
plt.legend()
plt.grid(True)
plt.show()