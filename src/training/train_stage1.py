import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Computes Focal Loss to handle extreme class imbalance by dynamically 
    scaling down the loss of well-classified examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Calculate standard binary cross-entropy with raw logits for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Compute probability of the true class (p_t)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Calculate modulating factor: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply weighting factor alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute final loss
        loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Executes a single training epoch across the dataset.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        # Reshape labels to match out_features dimension (Batch, 1)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Forward Pass
        logits = model(inputs)
        loss = criterion(logits, labels)
        
        # Backward Pass & Gradient Descent step
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item() * inputs.size(0)
        predictions = (torch.sigmoid(logits) > 0.5).float()
        correct_predictions += (predictions == labels).sum().item()
        total_samples += inputs.size(0)
        
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc