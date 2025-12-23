# 1. Loss Curve
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)
plt.show()

# 2. Train vs Test Accuracy
train_acc = [t for t in train_correct]
test_acc = [t for t in test_correct]

plt.plot(train_acc, label="Train Accuracy")
plt.plot(test_acc, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.show()

# 3. Evaluate
# Load weights
model.load_state_dict(
    torch.load("model/final_satellite_cnn_5channel.pth", map_location=device)
)
# Move model to device
model = model.to(device)

model.eval()
test_correct = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_val = model(X_batch)
        predicted = torch.max(y_val, 1)[1]
        test_correct += (predicted == y_batch).sum().item()

print(f"Final Test Accuracy: {test_correct / len(test_loader.dataset) * 100 :.2f}%")
# Final Test Accuracy: 98.14%
