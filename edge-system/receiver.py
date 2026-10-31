
from flask import Flask, request, jsonify
import os
import subprocess
import sys
import hashlib
import torch
from model import CNN
import torch.optim as optim

app = Flask(__name__)

CHECKPOINT_DIR = "/app/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
HASH_PATH = os.path.join(CHECKPOINT_DIR, "last_received.sha256")

MAX_FILE_SIZE = 100 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

training_process = None


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Edge System is ready",
        "status": "waiting for checkpoint"
    })


@app.route("/upload", methods=["POST"])
def receive_checkpoint():
    global training_process

    if "checkpoint" not in request.files:
        return jsonify({"error": "No checkpoint received"}), 400

    if training_process is not None and training_process.poll() is None:
        return jsonify({"error": "Training is already running"}), 409

    file = request.files["checkpoint"]

    if not file.filename.endswith(".pth"):
        return jsonify({"error": "Only .pth files are accepted"}), 400

    temp_path = CHECKPOINT_PATH + ".tmp"

    try:
        file.save(temp_path)

        # Calculate checkpoint hash
        sha256 = hashlib.sha256()

        with open(temp_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        checkpoint_hash = sha256.hexdigest()

        # Prevent duplicate checkpoint uploads
        if os.path.exists(HASH_PATH):
            with open(HASH_PATH, "r") as f:
                old_hash = f.read().strip()

            if checkpoint_hash == old_hash:
                os.remove(temp_path)
                return jsonify({
                    "error": "This checkpoint was already received"
                }), 409

        # Validate checkpoint before accepting it
        checkpoint = torch.load(
            temp_path,
            map_location="cpu",
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

        model = CNN()
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if not isinstance(checkpoint["epoch"], int):
            raise ValueError("Invalid epoch value")

        # Save validated checkpoint
        os.replace(temp_path, CHECKPOINT_PATH)

        with open(HASH_PATH, "w") as f:
            f.write(checkpoint_hash)

        # Start resumed training
        training_process = subprocess.Popen([
            sys.executable,
            "resume.py",
            CHECKPOINT_PATH
        ])

        return jsonify({
            "status": "success",
            "message": "Checkpoint validated and training started",
            "last_completed_epoch": checkpoint["epoch"],
            "resume_epoch": checkpoint["epoch"] + 1
        }), 200

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 400


@app.route("/status", methods=["GET"])
def status():
    if training_process is None:
        return jsonify({"status": "waiting"})

    exit_code = training_process.poll()

    if exit_code is None:
        return jsonify({"status": "training"})

    if exit_code == 0:
        return jsonify({
            "status": "training process finished",
            "exit_code": 0
        })

    return jsonify({
        "status": "failed",
        "exit_code": exit_code
    })


if __name__ == "__main__":
    print("Starting Edge System HTTP Receiver...")
    print("Waiting for checkpoint from Main System...")

    app.run(host="0.0.0.0", port=5000, debug=False)
