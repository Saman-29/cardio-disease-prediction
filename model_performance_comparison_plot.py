import matplotlib.pyplot as plt
import numpy as np

# Model names and performance metrics
models = ['Logistic Regression', 'Random Forest', 'Hybrid (Proposed Model)']
metrics = {
    'Accuracy': [0.62, 0.67, 0.82],
    'Precision': [0.65, 0.68, 0.80],
    'Recall': [0.55, 0.60, 0.82],
    'F1-Score': [0.59, 0.63, 0.81],
    'ROC-AUC': [0.58, 0.63, 0.79]
}

# X-axis positions
x = np.arange(len(models))
width = 0.5  # Width for bars

# Create subplots for all metrics
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Model Performance Comparisons', fontsize=16)

# Flatten axes for easier iteration
axes = axes.ravel()

# Iterate and plot each metric
for idx, (metric_name, values) in enumerate(metrics.items()):
    axes[idx].bar(x, values, color='skyblue', width=width)
    axes[idx].set_title(f'{metric_name} Comparison')
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(models, rotation=15)
    axes[idx].set_ylim(0, 1)
    axes[idx].set_ylabel('Score')

# Remove the extra subplot (6th plot)
fig.delaxes(axes[-1])

# Adjust layout for readability
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
