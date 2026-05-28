
import matplotlib.pyplot as plt
import numpy as np

# Data
num_demos = np.array([10, 20, 200])
accuracy = np.array([98.0, 100.0, 100.0])
variance = np.array([0.14, 0.0, 0.0])

# Lower and upper bounds for variance shading
lower_bound = accuracy - variance
upper_bound = accuracy + variance

# Create plot
plt.figure(figsize=(8, 5))

# Plot accuracy line
plt.plot(
    num_demos,
    accuracy,
    marker='o',
    markersize=8,
    linewidth=2,
    label='Accuracy'
)

# Plot variance shading
plt.fill_between(
    num_demos,
    lower_bound,
    upper_bound,
    alpha=0.4,
    label='± Variance'
)

# Labels and title
plt.xlabel('Number of Demonstrations')
plt.ylabel('Accuracy (%)')
plt.title('Block Insertion Task Accuracy vs Number of Demonstrations')

# Zoom in further so the variance region is visible
plt.ylim(70, 105)
plt.xlim(0, 210)

# Grid and legend
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

plt.savefig('block_insertion_accuracy.png')