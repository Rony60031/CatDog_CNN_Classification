import matplotlib.pyplot as plt

# Actual results from Adam training
epochs = [1, 2, 3, 4, 5]

accuracy = [53.67, 68.31, 75.79, 81.56, 85.30]

loss = [0.6899, 0.6042, 0.4959, 0.4094, 0.3429]

# Accuracy Graph
plt.figure(figsize=(8, 5))
plt.plot(epochs, accuracy, marker='o', label='Training Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("CNN Training Accuracy")
plt.xticks(epochs)
plt.grid(True)
plt.legend()

plt.savefig("results/training_accuracy.png", dpi=300)
plt.show()

# Loss Graph
plt.figure(figsize=(8, 5))
plt.plot(epochs, loss, marker='o', color='red', label='Training Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training Loss")
plt.xticks(epochs)
plt.grid(True)
plt.legend()

plt.savefig("results/training_loss.png", dpi=300)
plt.show()

print("Graphs saved successfully!")