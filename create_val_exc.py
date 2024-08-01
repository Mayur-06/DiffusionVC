import os
import random

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)

data_dir = 'Data'
mel_dir = os.path.join(data_dir, 'mels')
#filelists_dir = 'filelists'
# List all speakers
speakers = [spk for spk in os.listdir(mel_dir)]

# Split speakers into training and test speakers
test_speakers = random.sample(speakers, k=5)  # Adjust the number of test speakers
train_speakers = [spk for spk in speakers if spk not in test_speakers]

# Collect all audio IDs for training, validation, and exceptions
train_ids = []
val_ids = []
exc_ids = []

for spk in train_speakers:
    mel_files = os.listdir(os.path.join(mel_dir, spk))
    mel_ids = [m[:-4] for m in mel_files if m.endswith('.npy')]  # Remove the _mel.npy extension
    random.shuffle(mel_ids)
    split_val = int(0.1 * len(mel_ids))  # Use 10% for validation
    split_exc = int(0.05 * len(mel_ids))  # Use 5% for exceptions
    val_ids.extend(mel_ids[:split_val])
    exc_ids.extend(mel_ids[split_val:split_val + split_exc])
    train_ids.extend(mel_ids[split_val + split_exc:])


with open('val_file.txt', 'w') as f:
    for val_id in val_ids:
        f.write(val_id + '\n')

# Write exception IDs to exc_file.txt
with open('exc_file.txt', 'w') as f:
    for exc_id in exc_ids:
        f.write(exc_id + '\n')
print("Validation and exception files created.")
