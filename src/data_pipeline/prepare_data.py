import os
import glob
from sklearn.model_selection import train_test_split

def build_data_lists(data_dir="data", target_word="marvin"):
    """
    Crawls the raw downloaded folders and builds flat lists of file paths 
    and binary labels to feed into the PyTorch Dataset.
    """
    file_paths = []
    labels = []
    
    # ---------------------------------------------------------
    # 1. Parse Google Speech Commands (The Target and Speech Negatives)
    # ---------------------------------------------------------
    gsc_dir = os.path.join(data_dir, "speech_commands")
    
    # Positive samples (Target wake word -> Label 1.0)
    target_dir = os.path.join(gsc_dir, target_word)
    if os.path.exists(target_dir):
        target_files = glob.glob(os.path.join(target_dir, "*.wav"))
        file_paths.extend(target_files)
        labels.extend([1.0] * len(target_files))
        
    # Negative samples (Other random words -> Label 0.0)
    # We select a subset of words so the dataset doesn't become impossibly large
    negative_words = ["stop", "go", "yes", "no", "up", "down", "left", "right"] 
    for word in negative_words:
        word_dir = os.path.join(gsc_dir, word)
        if os.path.exists(word_dir):
            word_files = glob.glob(os.path.join(word_dir, "*.wav"))
            file_paths.extend(word_files)
            labels.extend([0.0] * len(word_files))
            
    # ---------------------------------------------------------
    # 2. Parse ESC-50 (Hard Negative Background Noise -> Label 0.0)
    # ---------------------------------------------------------
    esc50_dir = os.path.join(data_dir, "ESC-50", "audio")
    if os.path.exists(esc50_dir):
        noise_files = glob.glob(os.path.join(esc50_dir, "*.wav"))
        file_paths.extend(noise_files)
        labels.extend([0.0] * len(noise_files))
        
    # ---------------------------------------------------------
    # 3. Stratified Train/Test Split
    # ---------------------------------------------------------
    # stratify=labels ensures our 1:10 imbalance ratio remains identical in both sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Total files found: {len(file_paths)}")
    print(f"Training on: {len(train_paths)} files")
    print(f"Testing on: {len(test_paths)} files")
    print(f"Target word '{target_word}' examples: {labels.count(1.0)}")
    
    return train_paths, test_paths, train_labels, test_labels