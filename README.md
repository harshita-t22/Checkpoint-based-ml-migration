\# Checkpoint-Based Migration of ML Training Tasks Across Local and Edge Systems



\## Overview



This project implements checkpoint-based migration of machine learning training tasks from a main system to an edge system.



The objective is to demonstrate how a running ML training task can be interrupted, its training state saved as a checkpoint, transferred to another system, and resumed from the saved state.



The migration mechanism uses CPU utilization as a trigger to identify high resource usage on the main system.



\## Objectives



\- Train a machine learning model on a main system.

\- Monitor CPU utilization during training.

\- Save model and optimizer states periodically using checkpoints.

\- Trigger migration when CPU utilization crosses a predefined threshold.

\- Transfer the checkpoint to an edge system over HTTP.

\- Restore the model and optimizer states on the edge system.

\- Resume training from the saved epoch.



\## System Architecture



The project consists of two systems:



\### 1. Main System



\- Executes the ML training task.

\- Monitors CPU utilization.

\- Creates periodic training checkpoints.

\- Transfers the latest checkpoint to the edge system when migration is triggered.



\### 2. Edge System



\- Runs an HTTP receiver using Flask.

\- Receives checkpoint files from the main system.

\- Restores the saved model and optimizer states.

\- Resumes training from the checkpoint.



\## Technologies Used



\- Python

\- PyTorch

\- Torchvision

\- Flask

\- Docker

\- Docker Volumes

\- HTTP

\- psutil



\## Project Structure



```text

Checkpoint-Based-ML-Migration/

│

├── laptop1/

│   ├── train.py

│   ├── model.py

│   ├── requirements.txt

│   └── Dockerfile

│

└── laptop2/

&#x20;   ├── receiver.py

&#x20;   ├── resume.py

&#x20;   ├── model.py

&#x20;   ├── requirements.txt

&#x20;   └── Dockerfile

```



\## Workflow



1\. The main system starts ML model training.

2\. CPU utilization is monitored during execution.

3\. Checkpoints are periodically saved.

4\. When CPU utilization exceeds the configured threshold, training is interrupted at a batch boundary.

5\. The latest available checkpoint is transferred to the edge system.

6\. The edge system restores the checkpoint.

7\. Training resumes from the saved epoch.



\## Experimental Evaluation



The project evaluates:



\- Checkpoint creation time.

\- Checkpoint file size.

\- Checkpoint transfer time.

\- Checkpoint restoration time.

\- CPU and memory utilization.

\- Training progress preserved during migration.



Experimental results will be documented based on actual measurements.



\## Running the Project



Both systems use Docker containers to provide isolated execution environments.



Refer to the Dockerfiles and requirements files in the respective `laptop1` and `laptop2` directories for setup details.







