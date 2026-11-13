import csv
import matplotlib.pyplot as plt

# Read training history
epochs = []
train_loss = []
val_loss = []
train_acc = []
val_acc = []

with open("results/validation_history.csv", "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        val_loss.append(float(row["val_loss"]))
        train_acc.append(float(row["train_accuracy"]))
        val_acc.append(float(row["val_accuracy"]))


# Accuracy Graph
plt.figure(figsize=(9, 5))

plt.plot(
    epochs, train_acc,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs, val_acc,
    marker="s",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")

plt.xticks(epochs)
plt.grid(True)
plt.legend()

plt.savefig(
    "results/validation_accuracy.png",
    dpi=300
)

plt.show()


# Loss Graph
plt.figure(figsize=(9, 5))

plt.plot(
    epochs, train_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs, val_loss,
    marker="s",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.xticks(epochs)
plt.grid(True)
plt.legend()

plt.savefig(
    "results/validation_loss.png",
    dpi=300
)

plt.show()

print("Validation graphs saved successfully!")