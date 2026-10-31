import os
import threading
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import psutil

from torch.utils.data import DataLoader
from model import CNN


# Configuration
TOTAL_EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001

CHECKPOINT_INTERVAL = 5

CPU_THRESHOLD = 75.0
CPU_MONITOR_INTERVAL = 1

CHECKPOINT_DIR = "/app/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")

EDGE_URL = "http://10.53.165.152:5000/upload"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("cpu")

# Shared monitoring variables
migration_event = threading.Event()
monitor_stop = threading.Event()
cpu_lock = threading.Lock()

latest_cpu_usage = 0.0


# Monitor CPU continuously in the background
def monitor_cpu():
    global latest_cpu_usage

    psutil.cpu_percent(interval=None)

    while not monitor_stop.wait(CPU_MONITOR_INTERVAL):

        cpu_usage = psutil.cpu_percent(interval=None)

        with cpu_lock:
            latest_cpu_usage = cpu_usage

        if cpu_usage >= CPU_THRESHOLD:

            print(
                f"\nHIGH CPU UTILIZATION DETECTED: "
                f"{cpu_usage:.2f}%",
                flush=True
            )

            migration_event.set()
            break


# Get latest CPU reading
def get_cpu_usage():

    with cpu_lock:
        return latest_cpu_usage


# Get memory utilization
def get_memory_usage():

    # Prefer Docker container memory utilization (cgroup v2)
    try:
        with open("/sys/fs/cgroup/memory.current", "r") as f:
            used = int(f.read())

        with open("/sys/fs/cgroup/memory.max", "r") as f:
            limit = f.read().strip()

        if limit != "max" and int(limit) > 0:
            return (used / int(limit)) * 100

    except (OSError, ValueError):
        pass

    # Fallback to system memory utilization
    return psutil.virtual_memory().percent


# Save checkpoint
def save_checkpoint(epoch, model, optimizer, loss, accuracy):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "accuracy": accuracy
    }

    temp_path = CHECKPOINT_PATH + ".tmp"

    torch.save(checkpoint, temp_path)
    os.replace(temp_path, CHECKPOINT_PATH)

    print(
        f"Checkpoint saved at epoch {epoch}",
        flush=True
    )


# Evaluate test accuracy
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


# Transfer latest checkpoint
def send_checkpoint():

    print("\nSending latest checkpoint to Edge System...", flush=True)

    try:

        with open(CHECKPOINT_PATH, "rb") as file:

            response = requests.post(
                EDGE_URL,
                files={
                    "checkpoint": ("checkpoint.pth", file)
                },
                timeout=120
            )

        print("Edge response:", response.text, flush=True)

        if response.status_code == 200:

            print(
                "Migration request accepted by Edge System.",
                flush=True
            )

            return True

    except requests.RequestException as e:

        print("Migration failed:", e, flush=True)

    return False


def main():

    print("Main System training started!", flush=True)

    print(
        f"CPU migration threshold: {CPU_THRESHOLD}%",
        flush=True
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Training dataset
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

    # Testing dataset
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

    model = CNN().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    # Save initial checkpoint for recovery before epoch 5
    save_checkpoint(0, model, optimizer, 0.0, 0.0)

    # Start continuous CPU monitoring
    monitor_thread = threading.Thread(
        target=monitor_cpu,
        daemon=True
    )

    monitor_thread.start()

    try:

        for epoch in range(1, TOTAL_EPOCHS + 1):

            model.train()

            total_loss = 0.0
            interrupted = False

            for images, labels in train_loader:

                # Stop at the next batch boundary if migration triggers
                if migration_event.is_set():

                    interrupted = True
                    break

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Do not evaluate or save an incomplete epoch
            if interrupted:

                print(
                    f"Training interrupted during epoch {epoch}.",
                    flush=True
                )

                break

            avg_loss = total_loss / len(train_loader)

            # Calculate test accuracy
            accuracy = evaluate_accuracy(
                model,
                test_loader
            )

            # Collect resource utilization
            cpu_usage = get_cpu_usage()
            memory_usage = get_memory_usage()

            # Display all metrics on one line
            print(
                f"Main Epoch [{epoch}/{TOTAL_EPOCHS}] "
                f"Loss: {avg_loss:.4f} "
                f"Accuracy: {accuracy:.2f}% "
                f"CPU Utilization: {cpu_usage:.2f}% "
                f"Memory Utilization: {memory_usage:.2f}%",
                flush=True
            )

            # Save checkpoint after every 5 completed epochs
            if epoch % CHECKPOINT_INTERVAL == 0:

                save_checkpoint(
                    epoch,
                    model,
                    optimizer,
                    avg_loss,
                    accuracy
                )

            # Migrate if CPU trigger has occurred
            if migration_event.is_set():

                print(
                    "\nMigration condition reached!",
                    flush=True
                )

                print(
                    "Transferring the latest saved checkpoint.",
                    flush=True
                )

                break

    finally:

        monitor_stop.set()
        monitor_thread.join(timeout=2)

    # Transfer latest checkpoint if migration was triggered
    if migration_event.is_set():

        if send_checkpoint():

            print(
                "Main System stopping after checkpoint transfer.",
                flush=True
            )

        else:

            print(
                "Checkpoint transfer failed. "
                "Latest checkpoint remains on Main System.",
                flush=True
            )

    else:

        print(
            "All training epochs completed on Main System.",
            flush=True
        )


if __name__ == "__main__":
    main()
