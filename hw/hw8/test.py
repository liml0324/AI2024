import numpy as np
import matplotlib.pyplot as plt

# Define the hyperplane equation and the margin
def hyperplane(x1):
    return (1/2) * (x1 + 2)

def margin_positive(x1):
    return (1/2) * (x1 + 2) + 1 / np.sqrt(5)

def margin_negative(x1):
    return (1/2) * (x1 + 2) - 1 / np.sqrt(5)

# Support vectors
support_vectors = np.array([[1, 2], [3, 2], [3, 3]])
un_support_vectors = np.array([[2, 1], [2, 3]])

# Create the plot
x1_range = np.linspace(0, 4, 100)
plt.figure(figsize=(8, 6))

# Plot hyperplane
plt.plot(x1_range, hyperplane(x1_range), 'b-', label='Hyperplane: -x1 + 2x2 - 2 = 0')

# Plot margins
plt.plot(x1_range, margin_positive(x1_range), 'r--', label='Positive Margin')
plt.plot(x1_range, margin_negative(x1_range), 'g--', label='Negative Margin')

# Plot support vectors
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], color='red', zorder=5)
for sv in support_vectors:
    plt.annotate(f"({sv[0]}, {sv[1]})", (sv[0], sv[1]), textcoords="offset points", xytext=(0,10), ha='center')
    
# Plot unsupport vectors
plt.scatter(un_support_vectors[:, 0], un_support_vectors[:, 1], color='black', zorder=5)
for sv in un_support_vectors:
    plt.annotate(f"({sv[0]}, {sv[1]})", (sv[0], sv[1]), textcoords="offset points", xytext=(0,10), ha='center')

# Set labels and title
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.xlim(0, 4)
plt.ylim(0, 4)
plt.legend()
plt.title('Maximal Margin Classifier with Hyperplane and Margins')

plt.grid(True)
plt.show()