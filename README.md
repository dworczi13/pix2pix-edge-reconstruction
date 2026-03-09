# Edge Recovery in Compressed Images using Pix2Pix GAN
### (Rekonstrukcja krawędzi w obrazach kompresowanych stratnie przy użyciu sieci cGAN)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Type](https://img.shields.io/badge/Project-Engineering%20Thesis-green)

##  Abstract / Streszczenie
English: This project focuses on the detection and reconstruction of edges in images with reduced quality due to lossy compression. Compression artifacts significantly degrade the performance of traditional edge detection methods, such as the Canny algorithm. To address this, a Deep Learning approach using a Pix2Pix GAN (Conditional Generative Adversarial Network) was implemented.

The model consists of a U-Net generator with skip connections to preserve spatial information and a PatchGAN discriminator to ensure sharp details. The network was trained to map compressed images (with artifacts) to clean edge maps, using the COCO dataset. Quantitative analysis (PSNR, SSIM, Boundary F1 Score, ERC) demonstrates that the proposed model effectively learns edge representation robust to compression artifacts, outperforming classical methods in high-compression scenarios.

Polski: Projekt koncentruje się na problemie detekcji i rekonstrukcji krawędzi w obrazach o obniżonej jakości, wynikającej z kompresji stratnej. Artefakty powstające podczas kompresji w znacznym stopniu pogarszają działanie tradycyjnych metod, takich jak algorytm Canny'ego. W ramach projektu wykorzystano model Pix2Pix, będący odmianą warunkowej sieci przeciwstawnej (cGAN).

Architektura modelu składa się z generatora typu U-Net oraz dyskryminatora PatchGAN. Model uczono transformacji obrazów po kompresji stratnej do map krawędzi. Skuteczność oceniono przy użyciu metryk PSNR, SSIM, Boundary F1 Score oraz współczynnika ERC. Wyniki wskazują, że model Pix2Pix potrafi skutecznie odtworzyć krawędzie nawet przy silnej degradacji obrazu.
Pix2Pix Edge Reconstruction after Lossy Compression

# Model Architecture

The implemented architecture consists of two neural networks:

### Generator
- **U-Net architecture**
- skip connections to preserve spatial information
- generates edge maps from compressed images

### Discriminator
- **PatchGAN discriminator**
- evaluates realism of local image patches
- encourages generation of sharper edges

This adversarial setup allows the model to reconstruct high-quality edge structures even under strong compression artifacts.

---

# Dataset

Training data was prepared using the **COCO dataset**.

Dataset pipeline:

1. Original images
2. Apply **lossy compression**
3. Generate **ground truth edge maps** using the Canny detector
4. Train Pix2Pix to learn the mapping between compressed images and clean edges

---

# Evaluation Metrics

The model was evaluated using several quantitative metrics:

- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index)**
- **Boundary F1 Score**
- **Edge Reconstruction Coefficient (ERC)**

Results show that the Pix2Pix model can reconstruct meaningful edge structures even when compression artifacts significantly degrade image quality.

---

# Technologies Used

- Python
- TensorFlow
- OpenCV
- NumPy
- COCO Dataset

---

# Key Concepts

- Conditional GAN (cGAN)
- Pix2Pix architecture
- Image-to-image translation
- Computer Vision
- Edge detection


