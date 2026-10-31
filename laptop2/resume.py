
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import psutil

from torch.utils.data import DataLoader
from model import CNN


# Training settings
TOTAL_EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001

CHECKPOINT_INTERVAL = 5

device = torch.device("cpu")


# Memory utilization
def get_memory_usage():

    try:
        with open("/sys/fs/cgroup/memory.current", "r") as f:
            used = int(f.read())

        with open("/sys/fs/cgroup/memory.max", "r") as f:
            limit = f.read().strip()

        if limit != "max" and int(limit) > 0:
            return (used / int(limit)) * 100

    except (OSError, ValueError):
        pass

    return psutil.virtual_memory().percent


# Test accuracy
def evaluate_accuracy(model, test_loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


# Load checkpoint
def load_checkpoint(checkpoint_path):

    print("\nLoading received checkpoint...", flush=True)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Checkpoint file not found")

    # Measure complete checkpoint restoration time
    restore_start = time.perf_counter()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    required = [
        "epoch",
        "model_state_dict",
        "optimizer_state_dict"
    ]

    for key in required:

        if key not in checkpoint:
            raise ValueError(f"Missing checkpoint field: {key}")

    saved_epoch = checkpoint["epoch"]

    if not isinstance(saved_epoch, int) or saved_epoch < 0:
        raise ValueError("Invalid checkpoint epoch")

    model = CNN().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = saved_epoch + 1

    restore_time = time.perf_counter() - restore_start

    checkpoint_size = (
        os.path.getsize(checkpoint_path) / (1024 * 1024)
    )

    print("Checkpoint restored successfully!", flush=True)
    print("Last completed epoch:", saved_epoch, flush=True)
    print("Resuming from epoch:", start_epoch, flush=True)

    print(
        f"Checkpoint size: {checkpoint_size:.2f} MB",
        flush=True
    )

    print(
        f"Checkpoint restoration time: {restore_time:.4f} seconds",
        flush=True
    )

    return model, optimizer, start_epoch


# Save checkpoint
def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    accuracy,
    checkpoint_path
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "accuracy": accuracy
    }

    temp_path = checkpoint_path + ".tmp"

    torch.save(checkpoint, temp_path)

    os.replace(temp_path, checkpoint_path)

    print(
        f"Edge checkpoint saved at epoch {epoch}",
        flush=True
    )


# Resume training
def resume_training(checkpoint_path):

    model, optimizer, start_epoch = load_checkpoint(
        checkpoint_path
    )

    if start_epoch > TOTAL_EPOCHS:

        print("Training was already completed.", flush=True)
        return

    print("\nLoading MNIST dataset...", flush=True)

    # Must match Laptop 1 preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = torchvision.datasets.MNIST(
        root="/app/data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_data = torchvision.datasets.MNIST(
        root="/app/data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_data,
        batch_size=256,
        shuffle=False
    )

    criterion = nn.CrossEntropyLoss()

    print("\nEdge System training started!", flush=True)

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):

        model.train()

        total_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Calculate test accuracy
        accuracy = evaluate_accuracy(
            model,
            test_loader
        )

        # Resource utilization
        cpu_usage = psutil.cpu_percent(interval=1)

        memory_usage = get_memory_usage()

        # Display all metrics on one line
        print(
            f"Edge Epoch [{epoch}/{TOTAL_EPOCHS}] "
            f"Loss: {avg_loss:.4f} "
            f"Accuracy: {accuracy:.2f}% "
            f"CPU Utilization: {cpu_usage:.2f}% "
            f"Memory Utilization: {memory_usage:.2f}%",
            flush=True
        )

        # Save checkpoint every 5 epochs
        if epoch % CHECKPOINT_INTERVAL == 0:

            save_checkpoint(
                model,
                optimizer,
                epoch,
                avg_loss,
                accuracy,
                checkpoint_path
            )

    print("\nEdge System training completed!", flush=True)


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage: python resume.py checkpoint.pth")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    try:

        resume_training(checkpoint_path)

    except Exception as error:

        print("Training failed:", error, flush=True)
        sys.exit(1)
