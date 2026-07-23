import os
import torch
from torch.utils.data import DataLoader

# Pipeline imports
from src.data_pipeline.prepare_data import build_data_lists
from src.data_pipeline.dataset import WakeWordDataset
from src.models.stage1_cnn import WakeWordTriggerCNN
from src.training.train_stage1 import FocalLoss, train_one_epoch

def main():
    # 1. Device Diagnostics
    # Optimization for macOS hardware: leverage Apple Silicon GPU acceleration (MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training executing on hardware device: {device}")

    # 2. Extract Phase: Discover paths and build balanced data arrays
    print("Scanning directories and splitting data...")
    train_paths, test_paths, train_labels, test_labels = build_data_lists(
        data_dir="data", 
        target_word="marvin"
    )

    # 3. Transform Phase: Instantiate the PyTorch Dataset pipes
    train_dataset = WakeWordDataset(train_paths, train_labels)
    test_dataset = WakeWordDataset(test_paths, test_labels)

    # 4. Load Phase: Create performance-optimized DataLoaders
    # num_workers=2 utilizes multi-threaded CPU loading to prevent GPU starvation
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, 
        num_workers=2, pin_memory=True if device != torch.device("cpu") else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, 
        num_workers=2, pin_memory=True if device != torch.device("cpu") else False
    )

    # 5. Model Initialization
    model = WakeWordTriggerCNN(num_classes=1).to(device)
    print(f"Total Trainable Parameter Footprint: {model.count_parameters()} params")

    # 6. Optimization Stack Setup
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # 7. Core Training Loop Execution
    epochs = 15
    print(f"Commencing training loop ({epochs} epochs)...")
    
    for epoch in range(epochs):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{epochs}] -> Loss: {loss:.4f} | Accuracy: {acc * 100:.2f}%")

    # 8. Storage Persistency
    os.makedirs("models", exist_ok=True)
    model_path = "models/stage1_trigger.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Stage 1 model state weights safely serialized to: {model_path}")

if __name__ == "__main__":
    main()