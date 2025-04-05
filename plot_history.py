# plot_history.py
import json
import matplotlib.pyplot as plt

# Load the saved training history
history_file = "training_history.json"
try:
    with open(history_file, "r") as f:
        history = json.load(f)
except FileNotFoundError:
    print(f"History file '{history_file}' not found.")
    exit(1)

epochs = range(1, len(history["accuracy"]) + 1)

plt.figure(figsize=(12, 5))
# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, history["accuracy"], "bo-", label="Training Accuracy")
plt.plot(epochs, history["val_accuracy"], "ro-", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(epochs, history["loss"], "bo-", label="Training Loss")
plt.plot(epochs, history["val_loss"], "ro-", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("static/training_plots.png")
plt.show()
