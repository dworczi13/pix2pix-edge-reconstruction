Pix2Pix Edge Reconstruction after Lossy Compression

Implementation of a Pix2Pix conditional GAN model for reconstruction of edge maps from images degraded by lossy compression.

The project was developed as part of an engineering thesis in Electronics and Telecommunications at Poznan University of Technology.

The goal of the project is to investigate whether deep learning models can reconstruct structural information (edges) lost during compression better than classical edge detection algorithms.

Project Motivation

Lossy compression algorithms remove information from images in order to reduce data size.
As a result, structural details such as edges become degraded, which significantly reduces the effectiveness of classical edge detection methods.

Traditional methods such as Canny edge detection rely on local gradient analysis and perform well on high-quality images but struggle when compression artifacts are present.

This project explores whether a conditional generative adversarial network (Pix2Pix) can learn the relationship between compressed images and their corresponding edge maps and reconstruct edge information more effectively.

Method Overview

The project uses the Pix2Pix architecture, a conditional GAN designed for image-to-image translation.

The network learns to transform:

compressed image  →  reconstructed edge map
Architecture

The model consists of two neural networks:

Generator

U-Net architecture

encoder–decoder structure

skip connections for preserving spatial information

Discriminator

PatchGAN discriminator

evaluates local image patches instead of the whole image

encourages structural consistency in generated edge maps

Dataset Preparation

The training pipeline consists of several preprocessing steps.

1. Image compression

Original images are compressed using the VVC (Versatile Video Coding) codec.

Different compression levels are used by varying the quantization parameter (QP):

QP = 22, 27, 32, 37, 42, 47, 52, 57

This allows evaluation of the model under different levels of degradation.

2. Edge map generation

Reference edge maps are generated using the Canny edge detector.

These edge maps serve as ground truth targets during training.

3. Dataset pairing

Each training sample consists of:

Input  → compressed image
Target → reference edge map

These pairs are used to train the Pix2Pix model.

Training

The model is trained using a combination of two loss functions:

Adversarial Loss

Encourages the generator to produce edge maps that look realistic to the discriminator.

Reconstruction Loss (L1)

Ensures structural similarity between generated and reference edge maps.

The training process involves approximately:

~800,000 training iterations

Training is performed on image patches of size:

128 × 128 pixels
Evaluation

Model performance is evaluated using both visual and quantitative analysis.

Quantitative metrics

PSNR (Peak Signal-to-Noise Ratio)

SSIM (Structural Similarity Index)

Precision

Recall

F1 Score

Baseline comparison

The results produced by the Pix2Pix model are compared with:

Canny edge detection

applied directly to compressed images.

Project Structure
pix2pix-edge-reconstruction
│
├── prepare_dataset.py
│   Dataset preprocessing and preparation
│
├── generate_canny_edge.py
│   Generation of reference edge maps using Canny algorithm
│
├── conversions.py
│   Image preprocessing utilities
│
├── s2a_x57_005_11_tgd.py
│   Pix2Pix generator implementation (U-Net)
│
├── s2a_x57_005_12_vgd.py
│   Pix2Pix discriminator implementation (PatchGAN)
│
├── requirements.txt
│   Python dependencies
│
└── README.md
Technologies

Python

Deep Learning

Computer Vision

Conditional GAN (Pix2Pix)

U-Net architecture

PatchGAN discriminator

Results

The experiments demonstrate that deep learning models can reconstruct edge structures in compressed images more effectively than classical gradient-based detectors in scenarios with strong compression artifacts.

Author

Kacper Dworczak

Engineering Thesis
Electronics and Telecommunications
Poznan University of Technology
