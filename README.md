# NoMoColor: Unified Noise Modulation for Enhanced Diffusion-based Image Colorization (LANA)

<div align="center">

[![AAAI 2026/2027](https://img.shields.io/badge/AAAI-2026%20%2F%202027-blue.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/42207)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1609%2Faaai.v40i48.42207-orange.svg)](https://doi.org/10.1609/aaai.v40i48.42207)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](https://ojs.aaai.org/index.php/AAAI/article/download/42207/46168)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)](#citation)

**Official implementation of the paper accepted at AAAI Conference on Artificial Intelligence (AAAI)**  
*(Student Abstract and Poster Program)*

**Ankan Deria**<sup>1,2</sup> &nbsp;|&nbsp; 
**Dwarikanath Mahapatra**<sup>3</sup> &nbsp;|&nbsp; 
**Murari Mondal**<sup>4</sup> &nbsp;|&nbsp; 
**Sudipta Roy**<sup>2</sup>

<sup>1</sup> Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)  
<sup>2</sup> Jio University &nbsp;&nbsp;|&nbsp;&nbsp; <sup>3</sup> Khalifa University &nbsp;&nbsp;|&nbsp;&nbsp; <sup>4</sup> Kalinga Institute of Industrial Technology (KIIT)

</div>

---

## 📌 News & Updates
- **[March 2026]**: **NoMoColor (LANA)** paper accepted at **AAAI Conference on Artificial Intelligence**! Available online at [AAAI OJS Library](https://ojs.aaai.org/index.php/AAAI/article/view/42207).
- **[Code Release]**: Full inference, training, noise modulation pipeline, and Grounding-DINO + SAM integration released.

---

## 📖 Abstract

> *We present a language-based noise modulation module for diffusion models that improves image color generation under textual guidance. Unlike standard approaches that inject noise uniformly, our method leverages semantic cues from text to selectively control the noise injection process, preserving local details and enhancing color accuracy even when descriptions are ambiguous or incomplete. Applied to language-guided image colorization, this targeted modulation leads to more faithful and visually consistent results. The proposed module is lightweight, generalizable, and can be integrated into existing diffusion pipelines, offering a simple yet effective step toward more controllable text-to-image generation.*

---

## 💡 Key Contributions

Standard diffusion-based colorization frameworks suffer from **color bleeding**, **background degradation**, and **hallucination of inaccurate hues** because they inject isotropic Gaussian noise uniformly across all spatial positions. 

**NoMoColor (LANA)** addresses this with:
1. **Semantic-Guided Noise Modulation (NoMo)**: 
   - Uses zero-shot grounded segmentation (**Grounding DINO + Segment Anything / SAM**) to localize user prompt tokens in the input grayscale image.
   - Modulates the initial noise distribution non-uniformly ($`\epsilon_{mod} = \epsilon \odot \mathcal{M}_{latent}`$), dampening perturbations on regions that need preservation while concentrating stochastic exploration on target objects.
2. **Grayscale Structural Guidance & Enhanced Decoder**:
   - A dedicated grayscale feature encoder (`g_encoder`) extracts multi-scale luminance embeddings from the original image.
   - The enhanced VAE decoder injects high-frequency spatial features directly into the latent decoding pass, guaranteeing **zero background drift** and razor-sharp edge reconstruction.
3. **Cross-Attention & Token Guidance**:
   - Leverages localized cross-attention maps to ensure specific color descriptors bind precisely to target objects (e.g. multi-object colorization like *"the green cup, the blue cup, the cyan cup"*).
4. **Lightweight & Plug-and-Play**:
   - Compatible with DDIM and PLMS samplers (`DDIMSampler_withsam`, `PLMSSampler_withsam`) without retraining the base generative model from scratch.

---


## 📂 Repository Structure

```text
LANA-main/
├── cldm/                          # Control Latent Diffusion Model modules
│   ├── cldm.py                    # ControlLDM_cat architecture & model definition
│   ├── hack.py                    # Attention slicing & memory optimization hooks
│   ├── logger.py                  # PyTorch Lightning image logger
│   ├── model.py                   # Model instantiation & checkpoint loading
│   └── struct_tool.py             # Structural tools & feature processors
├── configs/                       # Model configuration YAMLs
│   ├── cldm_sample.yaml           # Inference configuration
│   └── cldm_v15_ehdecoder.yaml    # Training & enhanced decoder configuration
├── example/                       # Demo grayscale input images
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
├── ldm/                           # Latent Diffusion Models core library
│   ├── models/
│   │   ├── autoencoder.py         # AutoencoderKL_enhanceD with grayscale encoder
│   │   └── diffusion/
│   │       ├── ddim.py            # DDIMSampler & DDIMSampler_withsam (Noise Modulation)
│   │       ├── ddpm.py            # LatentDiffusion base implementation
│   │       └── plms.py            # PLMSSampler & PLMSSampler_withsam
│   ├── modules/                   # Attention, OpenAIMODEL (CatUNet), CLIP encoders
│   └── ptp/                       # Prompt-to-Prompt & cross-attention controllers
├── sample_text/                   # Sample test prompts & pairs
│   ├── pairs.json                 # Image-to-prompt test mapping
│   └── test.json                  # Sample test prompts
├── colorization_dataset.py        # Dataset loader for training & evaluation
├── colorization_dataset_test.py   # Dataset loader for inference with SAM masks
├── colorization_main.py           # Training & validation entrypoint
├── config.py                      # Global flags (e.g. save_memory)
├── inference.py                   # Inference script with PLMS/DDIM noise modulation
├── sam_mask.py                    # Zero-shot grounding & segmentation (Grounding-DINO + SAM)
├── share.py                       # Pre-import environment & hardware configuration
└── requirements.txt               # Environment dependencies
```

---

## ⚙️ Environment Setup

### 1. Prerequisites
- Linux or macOS (CUDA-enabled GPU with >= 8GB VRAM recommended for training/inference)
- Python 3.9 or 3.10
- PyTorch 1.13.1+ with torchvision matching your CUDA version

### 2. Conda Environment
```bash
# Create and activate conda environment
conda create -n nomocolor python=3.10 -y
conda activate nomocolor

# Install PyTorch (example for CUDA 11.8; adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### 3. Automatic Pretrained Weights
- **Grounding DINO** (`IDEA-Research/grounding-dino-tiny`) and **Segment Anything** (`facebook/sam-vit-base`) will automatically download upon first run via Hugging Face `transformers`.
- Ensure your environment has internet access or set `HF_HOME` to your local cache.

---

## 🚀 Quickstart & Inference

### 1. Model Checkpoint
Place your trained checkpoint (e.g., `coco_weight.ckpt`) in a directory such as `./checkpoints/`:
```bash
mkdir -p checkpoints
# Copy or symlink your checkpoint to checkpoints/coco_weight.ckpt
```

In `inference.py`, verify or update `resume_path`:
```python
resume_path = "checkpoints/coco_weight.ckpt"
```

### 2. Run Inference on Example Images
The repository includes sample images in `example/` and test prompt pairings in `sample_text/pairs.json`:

```bash
python inference.py
```

The script will:
1. Load test samples from `example/` and prompts from `sample_text/pairs.json`.
2. Extract semantic object masks using **Grounding DINO + SAM** (`sam_mask.py`).
3. Apply **Noise Modulation** (`latten_mask` * `noise`) in the latent space.
4. Execute diffusion sampling with `PLMSSampler_withsam` (or `DDIMSampler_withsam`).
5. Save colorized outputs to `./image_log/test_<timestamp>/`.

### 3. Custom Prompts & Input Images
To colorize your own images:
1. Add your `.jpg` or `.png` images into `example/`.
2. Edit `sample_text/pairs.json` with the filename, original reference caption, and desired colorization prompt:
   ```json
   [
       ["my_photo.jpg", "A cat sitting on a chair", "A fluffy golden cat sitting on a blue velvet chair"]
   ]
   ```
3. Run `python inference.py`.

---

## 🏋️ Training & Fine-Tuning

### 1. Dataset Preparation
Prepare your dataset (e.g. COCO 2017) formatted as follows:
```text
dataset_root/
├── train2017/                     # Training images
├── val2017/                       # Validation images
└── metadata/
    ├── caption_train.json         # Dict: { "img_name.jpg": ["caption 1", "caption 2"] }
    └── caption_val.json           # Dict: { "img_name.jpg": ["caption 1"] }
```

### 2. Launch Training
Update dataset paths and base checkpoint inside `colorization_main.py`, then run:
```bash
python colorization_main.py --train
```

Key arguments:
- `-t`, `--train`: Enable training mode.
- `-r`, `--resume`: Resume training from an existing checkpoint.
- `-m`, `--multicolor`: Multi-color test mode.
- `-s`, `--usesam`: Enable SAM-guided segmentation masks during evaluation.

### 3. Multi-Color Evaluation with SAM
To evaluate multi-color control using Grounding-DINO + SAM masks:
```bash
python colorization_main.py --multicolor --usesam
```

---

## 📊 Key Inference Parameters

In `inference.py`, you can easily tune:

| Parameter | Default | Description |
|---|---|---|
| `ddim_steps` | `50` | Number of denoising sampling steps (PLMS / DDIM). |
| `unconditional_guidance_scale` | `7.0` | Classifier-Free Guidance (CFG) scale for text fidelity. |
| `ddim_eta` | `0.0` | Stochasticity factor (`0.0` for deterministic DDIM/PLMS). |
| `use_attn_guidance` | `True` | Enables cross-attention token guidance. |
| `threshold` (in `sam_mask.py`) | `0.3` | Detection threshold for Grounding DINO. |

---

## 📝 Citation

If you find **NoMoColor (LANA)** useful in your research or applications, please cite our paper:

```bibtex
@inproceedings{deria2026nomocolor,
  title={NoMoColor: Unified Noise Modulation for Enhanced Diffusion-based Image Colorization (Student Abstract)},
  author={Deria, Ankan and Mahapatra, Dwarikanath and Mondal, Murari and Roy, Sudipta},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={48},
  pages={41182--41184},
  year={2026}
}
```

- **Publication Link**: [https://ojs.aaai.org/index.php/AAAI/article/view/42207](https://ojs.aaai.org/index.php/AAAI/article/view/42207)
- **DOI**: [10.1609/aaai.v40i48.42207](https://doi.org/10.1609/aaai.v40i48.42207)

---

## 🙏 Acknowledgements

This work builds upon foundational open-source models and frameworks:
- [Latent Diffusion Models (LDM)](https://github.com/CompVis/latent-diffusion) & [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt)
