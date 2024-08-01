
---

# Diffusion-based Voice Conversion Model

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
The Diffusion-based Voice Conversion Model is a state-of-the-art voice conversion framework that leverages diffusion processes to achieve high-quality voice synthesis. This project aims to convert the voice of one speaker to another while preserving the linguistic content and naturalness of the speech. [Note that this is an on-going research project]

## Features
- High-quality voice conversion using diffusion processes.
- Support for various datasets such as CREMA, LRW_full_size, avspeech, and vox.
- Customizable model architecture and training settings.
- Preprocessing scripts for audio and video data.
- Efficient inference with pre-trained models.

## Installation
To get started with the project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/diffusion-voice-conversion.git
   cd diffusion-voice-conversion
   ```

2. **Install dependencies:**
   Ensure you have Python 3.8 or higher and then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download datasets:**
   Place your datasets in the `Data` directory following the structure mentioned in the [Data Preparation](#data-preparation) section. Create `wavs` folder to store speaker folders in it.

## Usage

### Data Preparation
Run extract_mel_emb.py file to get mel-spectrograms and embeddings.
Organize your datasets as follows:
```
Data/
├── wavs/
│   ├── s1/
│   │   ├── 1001_DFA_DIS_XX.wav
│   ├── s2/
│   │   ├── 1002_DFA_DIS_XX.wav
│   └── ...
├── mels/
|   ├──s1/
|   |  ├── 1001_DFA_DIS_XX.npy
|   |__ ...
├── embeds/
|   ├──s1/
|   |  ├──1001_DFA_DIS_XX.pkl
|   |__ ...
```

### Training
There are two parts two be trained in this model. One for train_enc file (forward diffusion) and another is train_dec.py (reverse diffusion)
Now, you can even use a pretrained model instead of training the forward diffusion part. Model will be saved in `logs_dec` which will be used for inference.
otherwise for reverse diffusion part, run the following command:
```bash
python train_dec.py 
```

### Inference
For inference, save some seen audio files in `example` folder and run `inference` jupyter notebook.

Ensure that the model checkpoint is saved in `checkpts\vc`.

## Model Architecture
The model architecture includes:
- **UNet**: Used for encoding and decoding audio features.
- **Diffusion Process**: Applied to enhance the quality of voice conversion.
- **Speaker Encoder**: Extracts speaker embeddings from input audio.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
We would like to thank the contributors and the open-source community for their valuable work and support.

---
