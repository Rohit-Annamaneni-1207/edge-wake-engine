import os
import time
import torch
import torchaudio
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

# Pipeline imports
from src.models.stage1_cnn import WakeWordTriggerCNN
from src.models.stage2_rnn import WakeWordVerifierRNN

# ------------------------------------------------------------------
# 1. System Initialization & Parameter Counting
# ------------------------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"⚙️ Initializing Edge Pipeline UI on {device}...")

stage1 = WakeWordTriggerCNN(num_classes=1)
stage2 = WakeWordVerifierRNN(num_classes=1)

# Dynamically count parameters
s0_params = 0
s1_params = sum(p.numel() for p in stage1.parameters() if p.requires_grad)
s2_params = sum(p.numel() for p in stage2.parameters() if p.requires_grad)

try:
    stage1.load_state_dict(torch.load("models/stage1_trigger.pth", map_location="cpu"))
    stage2.load_state_dict(torch.load("models/stage2_verifier.pth", map_location="cpu"))
    
    stage1.to(device).eval()
    stage2.to(device).eval()
    print("✅ Cascade weights loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError("Missing weight files! Ensure Stage 1 and Stage 2 are trained.")

STAGE0_ENERGY_THRESHOLD = 0.005  
STAGE1_CNN_THRESHOLD = 0.70      
STAGE2_GRU_THRESHOLD = 0.50      

# ------------------------------------------------------------------
# 2. Continuous Rolling-Buffer Inference Engine
# ------------------------------------------------------------------
@torch.no_grad()
def process_audio(audio_filepath):
    if audio_filepath is None:
        return None, "No audio provided", "", "", "🔴 PIPELINE IDLE", "0.00 ms"

    # Start the hardware latency clock
    start_time = time.perf_counter()

    # 1. Base ETL
    waveform, sr = torchaudio.load(audio_filepath)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=512, n_mels=40
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    # 2. Rolling Buffer Execution
    window_size = 16000 # 1.0 seconds
    stride = 8000       # 0.5 seconds hop
    
    furthest_stage = -1
    best_logs = ()
    
    # Slide the 1-second window across the entire recording
    for i in range(0, max(1, waveform.shape[1] - window_size + 1), stride):
        chunk = waveform[:, i : i + window_size]
        
        # Pad the very last chunk if it's too short
        if chunk.shape[1] < window_size:
            chunk = torch.nn.functional.pad(chunk, (0, window_size - chunk.shape[1]))

        # ==========================================
        # STAGE 0: RMS Energy Gate
        # ==========================================
        rms_energy = torch.sqrt(torch.mean(chunk ** 2)).item()
        s0_pass = rms_energy > STAGE0_ENERGY_THRESHOLD
        s0_status = f"{'🟢 PASS' if s0_pass else '🔴 FAIL'} | RMS: {rms_energy:.4f}"
        
        # Generate Spec for UI Visualization
        mel_spec = amp_to_db(mel_transform(chunk))
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(mel_spec[0].numpy(), origin='lower', aspect='auto', cmap='viridis')
        ax.set_title(f"Evaluated Window: {i/16000:.1f}s to {(i+window_size)/16000:.1f}s")
        ax.set_xlabel("Time (Frames)")
        ax.set_ylabel("Frequency (Mels)")
        plt.tight_layout()

        if not s0_pass:
            if 0 >= furthest_stage:
                furthest_stage = 0
                best_logs = (fig, s0_status, "⚪ BYPASSED", "⚪ BYPASSED", "🔴 REJECTED AT STAGE 0")
            plt.close(fig) # Free memory
            continue

        # ==========================================
        # STAGE 1: Spatial CNN Gate
        # ==========================================
        input_tensor = mel_spec.unsqueeze(0).to(device)
        s1_prob = torch.sigmoid(stage1(input_tensor)).item()
        s1_pass = s1_prob >= STAGE1_CNN_THRESHOLD
        s1_status = f"{'🟢 PASS' if s1_pass else '🔴 FAIL'} | Conf: {s1_prob:.4f}"

        if not s1_pass:
            if 1 >= furthest_stage:
                furthest_stage = 1
                best_logs = (fig, s0_status, s1_status, "⚪ BYPASSED", "🔴 REJECTED AT STAGE 1")
            plt.close(fig)
            continue

        # ==========================================
        # STAGE 2: Temporal GRU Verifier
        # ==========================================
        s2_prob = torch.sigmoid(stage2(input_tensor)).item()
        s2_pass = s2_prob >= STAGE2_GRU_THRESHOLD
        s2_status = f"{'🟢 PASS' if s2_pass else '🔴 FAIL'} | Conf: {s2_prob:.4f}"

        if not s2_pass:
            if 2 >= furthest_stage:
                furthest_stage = 2
                best_logs = (fig, s0_status, s1_status, s2_status, "🔴 REJECTED AT STAGE 2")
            plt.close(fig)
            continue

        # 🎯 WAKE WORD FOUND! Halt the rolling buffer immediately.
        best_logs = (fig, s0_status, s1_status, s2_status, "🚀 🟢 WAKE WORD CONFIRMED!")
        break 

    # Stop the clock
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    latency_str = f"{latency_ms:.2f} ms"

    return (*best_logs, latency_str)

# ------------------------------------------------------------------
# 3. Gradio Interface Construction
# ------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    gr.Markdown("# 🎙️ Edge wake-word Cascade: Rolling Buffer Diagnostics")
    gr.Markdown("Record as long as you want. The engine will scan chronologically, just like a real edge device.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Input Audio")
            submit_btn = gr.Button("Process Buffer", variant="primary")
            
        with gr.Column():
            spec_plot = gr.Plot(label="DSP Output (Most Relevant Window)")
            
    with gr.Row():
        s0_out = gr.Textbox(label=f"Stage 0: RMS Energy ({s0_params} Parameters)")
        s1_out = gr.Textbox(label=f"Stage 1: CNN Gate ({s1_params:,} Parameters)")
        s2_out = gr.Textbox(label=f"Stage 2: GRU Verifier ({s2_params:,} Parameters)")
        
    with gr.Row():
        final_out = gr.Textbox(label="End-to-End Pipeline Decision", scale=2)
        latency_out = gr.Textbox(label="Total Compute Latency", scale=1)

    submit_btn.click(
        fn=process_audio,
        inputs=audio_input,
        outputs=[spec_plot, s0_out, s1_out, s2_out, final_out, latency_out]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)