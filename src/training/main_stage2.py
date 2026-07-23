import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Pipeline imports
from src.data_pipeline.prepare_data import build_data_lists
from src.data_pipeline.dataset import WakeWordDataset
from src.models.stage1_cnn import WakeWordTriggerCNN
from src.models.stage2_rnn import WakeWordVerifierRNN

@torch.no_grad()
def mine_hard_examples_topk(stage1_model, raw_dataloader, device):
    """
    Passes raw dataset through Stage 1, retains all True Positives, 
    and ranks all True Negatives by Stage 1 probability to extract the 
    Top-K hardest negatives matching the positive sample support.
    """
    stage1_model.eval()
    
    pos_inputs, pos_labels = [], []
    neg_inputs, neg_labels, neg_probs = [], [], []
    
    print("Evaluating corpus through Stage 1 CNN for Top-K Hard Mining...")
    
    for inputs, labels in raw_dataloader:
        inputs = inputs.to(device)
        
        # Forward pass on device
        logits = stage1_model(inputs)
        probs = torch.sigmoid(logits).squeeze()
        
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
            
        # Immediately push tensors back to CPU host RAM to prevent GPU/MPS OOM
        probs_cpu = probs.cpu()
        inputs_cpu = inputs.cpu()
        
        # Masking performed safely on CPU
        pos_mask = (labels == 1.0)
        neg_mask = (labels == 0.0)
        
        if pos_mask.any():
            pos_inputs.append(inputs_cpu[pos_mask])
            pos_labels.append(labels[pos_mask])
            
        if neg_mask.any():
            neg_inputs.append(inputs_cpu[neg_mask])
            neg_labels.append(labels[neg_mask])
            neg_probs.append(probs_cpu[neg_mask])
            
    # Concatenate complete feature collections
    all_pos_inputs = torch.cat(pos_inputs, dim=0)
    all_pos_labels = torch.cat(pos_labels, dim=0)
    
    all_neg_inputs = torch.cat(neg_inputs, dim=0)
    all_neg_labels = torch.cat(neg_labels, dim=0)
    all_neg_probs = torch.cat(neg_probs, dim=0)
    
    # Establish K based on available positive class support
    num_pos = len(all_pos_labels)
    
    # Sort negative samples by Stage 1 confidence descending
    sorted_indices = torch.argsort(all_neg_probs, descending=True)
    topk_indices = sorted_indices[:num_pos]
    
    # Extract maximum-entropy negative instances
    topk_neg_inputs = all_neg_inputs[topk_indices]
    topk_neg_labels = all_neg_labels[topk_indices]
    
    # Combine into a 1:1 balanced corpus
    final_inputs = torch.cat([all_pos_inputs, topk_neg_inputs], dim=0)
    final_labels = torch.cat([all_pos_labels, topk_neg_labels], dim=0)
    
    print(f"    -> Mining Complete.")
    print(f"       Positives retained       : {num_pos}")
    print(f"       Top-K Negatives selected : {len(topk_neg_labels)}")
    print(f"       Max Stage 1 Prob on Neg  : {all_neg_probs[sorted_indices[0]]:.4f}")
    print(f"       Min Stage 1 Prob in Top-K: {all_neg_probs[topk_indices[-1]]:.4f}")
    
    return TensorDataset(final_inputs, final_labels)


def train_stage2_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        logits = model(inputs)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        predictions = (torch.sigmoid(logits) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += inputs.size(0)
        
    return running_loss / total, correct / total


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Executing Stage 2 Mining & Training on: {device}")

    # 1. Dataset Discovery
    print("Scanning audio files...")
    train_paths, _, train_labels, _ = build_data_lists(
        data_dir="data", target_word="marvin"
    )

    raw_train_dataset = WakeWordDataset(train_paths, train_labels)
    raw_train_loader = DataLoader(
        raw_train_dataset, batch_size=128, shuffle=False, num_workers=2
    )

    # 2. Load Pre-trained Stage 1 Weights
    print("Loading Stage 1 CNN Spatial Gate...")
    stage1_model = WakeWordTriggerCNN(num_classes=1).to(device)
    stage1_weights = "models/stage1_trigger.pth"
    
    if not os.path.exists(stage1_weights):
        raise FileNotFoundError(f"Missing {stage1_weights}. Train Stage 1 first!")
        
    stage1_model.load_state_dict(torch.load(stage1_weights, map_location=device))
    
    # 3. Deterministic Top-K Mining
    hard_train_dataset = mine_hard_examples_topk(stage1_model, raw_train_loader, device)
    
    stage2_train_loader = DataLoader(
        hard_train_dataset, batch_size=64, shuffle=True, 
        pin_memory=True if device != torch.device("cpu") else False
    )

    # 4. Model & Optimizer Instantiation
    model = WakeWordVerifierRNN(num_classes=1).to(device)
    print(f"Stage 2 GRU Total Parameters: {model.count_parameters()}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)

    # 5. Training Loop
    epochs = 15
    print("Commencing Stage 2 Sequence Training on Mined Dataset...")
    
    for epoch in range(epochs):
        loss, acc = train_stage2_epoch(model, stage2_train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{epochs}] -> Loss: {loss:.4f} | Accuracy: {acc * 100:.2f}%")

    # 6. Weight Persistence
    os.makedirs("models", exist_ok=True)
    model_path = "models/stage2_verifier.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Stage 2 weights successfully serialized to: {model_path}")

if __name__ == "__main__":
    main()