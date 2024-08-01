import os
import numpy as np
import librosa
import soundfile as sf
import pickle
import sys
import torch
sys.path.append('speaker_encoder/')
from speaker_encoder.encoder import inference as spk_encoder
from speaker_encoder.encoder.inference import load_model, embed_utterance
from pathlib import Path

# loading speaker encoder
enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt')  # speaker encoder path
use_gpu = torch.cuda.is_available()  # Check if GPU is available
if use_gpu:
    spk_encoder.load_model(enc_model_fpath, device="cuda")
else:
    spk_encoder.load_model(enc_model_fpath, device="cpu")

# Define mel basis
mel_basis = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)

# Define functions
def get_mel(wav_path):
    wav, _ = librosa.load(wav_path, sr=22050)
    wav = wav[:(wav.shape[0] // 256)*256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram

def get_embed(wav_path):
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed

# Define paths
data_dir = 'Data'
wavs_dir = os.path.join(data_dir, 'wavs')
mels_dir = os.path.join(data_dir, 'mels')
embeds_dir = os.path.join(data_dir, 'embeds')

# Process each speaker directory
for speaker in os.listdir(wavs_dir):
    speaker_wavs_dir = os.path.join(wavs_dir, speaker)
    speaker_mels_dir = os.path.join(mels_dir, speaker)
    speaker_embeds_dir = os.path.join(embeds_dir, speaker)
    
    # Create directories for each speaker
    os.makedirs(speaker_mels_dir, exist_ok=True)
    os.makedirs(speaker_embeds_dir, exist_ok=True)
    
    # Process each wav file for the speaker
    for wav_file in os.listdir(speaker_wavs_dir):
        wav_path = os.path.join(speaker_wavs_dir, wav_file)
        
        # Get mel-spectrogram and embedding
        mel_spectrogram = get_mel(wav_path)
        embed = get_embed(wav_path)
        
        # Save mel-spectrogram
        mel_filename = wav_file.replace('.wav', '.npy')
        mel_path = os.path.join(speaker_mels_dir, mel_filename)
        np.save(mel_path, mel_spectrogram)
        
        # Save embedding
        embed_filename = wav_file.replace('.wav', '.npy')
        embed_path = os.path.join(speaker_embeds_dir, embed_filename)
        np.save(embed_path, embed)

print("Processing complete.")
