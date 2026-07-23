import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Pipeline imports
from src.data_pipeline.prepare_data import build_data_lists
from src.data_pipeline.dataset import WakeWordDataset
from src.models.stage1_cnn import WakeWordTriggerCNN

@torch.no_grad()
def evaluate_model():
    # 1. Hardware allocation
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Running evaluation on: {device}")

    # 2. Data Preparation (Grabbing the unseen test split)
    print("Fetching unseen test data...")
    _, test_paths, _, test_labels = build_data_lists(
        data_dir="data", target_word="marvin"
    )

    test_dataset = WakeWordDataset(test_paths, test_labels)
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, 
        num_workers=2, pin_memory=True if device != torch.device("cpu") else False
    )

    # 3. Model Instantiation & Weight Loading
    model = WakeWordTriggerCNN(num_classes=1).to(device)
    weights_path = "models/stage1_trigger.pth"
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval() # CRITICAL: Disables dropout and batch norm tracking
        print(f"Successfully loaded weights from {weights_path}")
    except FileNotFoundError:
        print(f"Error: Could not find {weights_path}. Ensure training completed.")
        return

    # 4. Evaluation Loop
    all_predictions = []
    all_targets = []
    
    print("Scanning test dataset...")
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        
        logits = model(inputs)
        probs = torch.sigmoid(logits)
        
        # Apply the Stage 1 threshold (0.70) defined in our cascade architecture
        preds = (probs >= 0.70).int().cpu().numpy()
        
        all_predictions.extend(preds)
        all_targets.extend(labels.numpy())

    # 5. Metric Calculation
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions).squeeze()
    
    print("\n" + "="*50)
    print(" STAGE 1 CNN: UNSEEN DATA PERFORMANCE")
    print("="*50 + "\n")
    
    print(classification_report(
        all_targets, 
        all_predictions, 
        target_names=["Negative (Noise/Other)", "Positive (Marvin)"],
        digits=4
    ))
    
    print("CONFUSION MATRIX:")
    cm = confusion_matrix(all_targets, all_predictions)
    print(f"True Negatives (Correct Silence/Noise) : {cm[0][0]}")
    print(f"False Positives (Accidental Triggers)  : {cm[0][1]}")
    print(f"False Negatives (Missed Wake Words)    : {cm[1][0]}")
    print(f"True Positives (Caught Wake Words)     : {cm[1][1]}\n")

if __name__ == "__main__":
    evaluate_model()