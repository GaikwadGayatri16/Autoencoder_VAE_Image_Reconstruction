# Autoencoder and Variational Autoencoder for Image Reconstruction

A Generative AI practical implementing an **Autoencoder (AE)** and a **Variational Autoencoder (VAE)** for image reconstruction and generation using the **MNIST handwritten digit dataset**.

The project demonstrates the difference between deterministic and probabilistic latent representations and compares the reconstruction performance of both models using **Mean Squared Error (MSE)** and **Structural Similarity Index (SSIM)**.

---

## 📌 Project Overview

An **Autoencoder** learns to compress an input image into a lower-dimensional latent representation and reconstruct the original image from it.

A **Variational Autoencoder (VAE)** extends this concept by learning a probability distribution in the latent space. This allows the model not only to reconstruct images but also to generate new digit-like images by sampling random points from the learned latent distribution.

This practical implements both approaches on the **MNIST handwritten digit dataset**.

---

## 🎯 Objectives

- Understand the working of Autoencoders.
- Understand the working of Variational Autoencoders.
- Implement an Autoencoder using TensorFlow/Keras.
- Implement a VAE using the reparameterization trick.
- Reconstruct MNIST images using both models.
- Generate new digit-like images using the VAE.
- Compare AE and VAE reconstruction performance.
- Evaluate reconstruction using MSE and SSIM.
- Visualize training and validation losses.
- Understand the difference between deterministic and probabilistic latent spaces.

---

## 🧠 Concepts Covered

### Autoencoder

An Autoencoder consists of three major components:

```text
Input Image
    ↓
Encoder
    ↓
Latent Representation
    ↓
Decoder
    ↓
Reconstructed Image
```

The encoder compresses the input into a smaller latent representation, while the decoder reconstructs the original input.

For this project:

```text
784 → 256 → 128 → 32 → 128 → 256 → 784
```

The **32-dimensional vector** acts as the bottleneck/latent representation.

---

### Variational Autoencoder

A VAE learns a probability distribution instead of a single deterministic latent vector.

The encoder produces:

- `mu` — mean of the latent distribution
- `log_var` — logarithm of the variance

The latent vector is sampled using the **reparameterization trick**:

```text
z = mu + exp(0.5 × log_var) × epsilon
```

where:

```text
epsilon ~ N(0, I)
```

The VAE loss consists of:

```text
VAE Loss = Reconstruction Loss + KL Divergence
```

The KL divergence term regularizes the latent space and makes random image generation possible.

---

## 📊 Dataset

The project uses the **MNIST handwritten digit dataset**.

MNIST contains grayscale images of handwritten digits from **0 to 9**.

Each image has dimensions:

```text
28 × 28 pixels
```

Before being provided to the fully connected networks, the images are:

1. Converted to `float32`.
2. Normalized to the range `[0, 1]`.
3. Flattened from `28 × 28` into `784` values.

```text
28 × 28 → 784
```

The notebook uses:

```python
keras.datasets.mnist.load_data()
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | Neural network implementation |
| NumPy | Numerical computation |
| Matplotlib | Visualization |
| MNIST | Image dataset |
| Google Colab | Development and execution environment |

---

## ⚙️ Project Configuration

The notebook uses the following configuration:

| Parameter | Value |
|---|---:|
| Image Size | 28 × 28 |
| Input Dimension | 784 |
| Latent Dimension | 32 |
| Batch Size | 128 |
| Autoencoder Epochs | 5 |
| VAE Epochs | 5 |
| Random Seed | 42 |

---

## 🏗️ Autoencoder Architecture

The Autoencoder consists of an encoder and decoder.

### Encoder

```text
784
 ↓
256 neurons
 ↓
128 neurons
 ↓
32-dimensional latent vector
```

### Decoder

```text
32-dimensional latent vector
 ↓
128 neurons
 ↓
256 neurons
 ↓
784
```

### Complete Architecture

```text
784 → 256 → 128 → 32 → 128 → 256 → 784
```

The model uses **ReLU** activations in the hidden layers and a **sigmoid** activation in the output layer.

The Autoencoder is trained using:

```text
Binary Cross-Entropy
```

with the **Adam optimizer**.

---

## 🔄 Autoencoder Workflow

```text
MNIST Image
     │
     ▼
Normalize Pixel Values
     │
     ▼
Flatten 28×28 → 784
     │
     ▼
┌───────────────┐
│    Encoder    │
│ 784 → 256     │
│ 256 → 128     │
│ 128 → 32      │
└───────────────┘
     │
     ▼
Latent Vector (32)
     │
     ▼
┌───────────────┐
│    Decoder    │
│ 32 → 128      │
│ 128 → 256     │
│ 256 → 784     │
└───────────────┘
     │
     ▼
Reconstructed Image
```

---

## 🧬 VAE Architecture

The VAE uses a similar encoder-decoder structure, but instead of directly producing one latent vector, the encoder produces:

```text
mu
log_var
```

These values are used to sample the latent representation:

```text
(mu, log_var)
       │
       ▼
Reparameterization
       │
       ▼
     z (32)
       │
       ▼
    Decoder
       │
       ▼
Reconstructed Image
```

### VAE Architecture

```text
784 → 256 → 128 → (mu, log_var)
                         ↓
                    z (32)
                         ↓
                  128 → 256 → 784
```

---

## 🔬 VAE Loss

The VAE uses two major loss components.

### 1. Reconstruction Loss

Measures how close the reconstructed image is to the original image.

### 2. KL Divergence

Regularizes the learned latent distribution so that it remains close to a standard normal distribution.

The total loss is:

```text
Total Loss =
Reconstruction Loss + KL Divergence
```

---

## 🖼️ Image Reconstruction

After training, both models reconstruct images from the MNIST test dataset.

The notebook displays:

```text
Original Image
      ↓
Autoencoder Reconstruction
      ↓
VAE Reconstruction
```

This allows visual comparison of the reconstruction quality of both models.

---

## ✨ VAE Image Generation

One of the major advantages of the VAE is its ability to generate new images.

The notebook samples random latent vectors from a standard normal distribution:

```python
random_latent_vectors = tf.random.normal(
    shape=(num_generated, LATENT_DIM),
    seed=42
)
```

These latent vectors are passed through the VAE decoder to generate new digit-like images.

The project generates:

```text
20 new images
```

The generated images demonstrate how the VAE can learn a structured latent space suitable for generative modeling.

---

## 📏 Evaluation Metrics

The project compares the Autoencoder and VAE using two image reconstruction metrics.

### Mean Squared Error (MSE)

MSE measures the average squared difference between the original and reconstructed images.

```text
Lower MSE → Better numerical reconstruction
```

The notebook calculates:

```python
ae_mse
vae_mse
```

---

### Structural Similarity Index (SSIM)

SSIM measures structural similarity between the original and reconstructed images.

```text
Higher SSIM → Greater structural similarity
```

The notebook calculates:

```python
ae_ssim
vae_ssim
```

> **Note:** The actual numerical values depend on the runtime and training execution. The notebook should be run to obtain the current results rather than inserting guessed values.

---

## 📈 Visualizations

The project includes several visualizations:

### 1. MNIST Samples

Displays sample handwritten digits from the dataset.

### 2. Autoencoder Reconstruction

Compares original MNIST images with AE reconstructions.

### 3. VAE Reconstruction

Compares original MNIST images with VAE reconstructions.

### 4. Autoencoder Training Loss

Shows:

```text
Training Loss
Validation Loss
```

across epochs.

### 5. VAE Training Loss

Tracks:

```text
Total Loss
Reconstruction Loss
KL Loss
```

### 6. Generated Images

Displays new digit-like images generated from random latent vectors.

### 7. Final Comparison

The notebook provides a final visual comparison:

```text
Original
   ↓
AE Reconstruction
   ↓
VAE Reconstruction
```

---

## 🔍 Autoencoder vs VAE

| Feature | Autoencoder | Variational Autoencoder |
|---|---|---|
| Latent Representation | Deterministic vector | Probability distribution |
| Encoder Output | Latent vector | `mu` and `log_var` |
| Latent Space | Not explicitly regularized | Regularized using KL divergence |
| Loss | Reconstruction loss | Reconstruction + KL divergence |
| Reconstruction | Usually sharper | Often smoother |
| Image Generation | Limited | Natural capability |
| Main Purpose | Compression / representation / denoising | Representation + generative modeling |

---

## 📁 Repository Structure

A simple repository structure for this project is:

```text
Autoencoder-VAE-Image-Reconstruction/
│
├── README.md
│
└── Practical_03_Autoencoder_VAE_Image_Reconstruction.ipynb
```

If additional result images are added later:

```text
Autoencoder-VAE-Image-Reconstruction/
│
├── README.md
├── Practical_03_Autoencoder_VAE_Image_Reconstruction.ipynb
│
└── results/
    ├── mnist_samples.png
    ├── ae_reconstruction.png
    ├── vae_reconstruction.png
    ├── generated_images.png
    └── loss_comparison.png
```

---

## ▶️ How to Run

### Option 1 — Google Colab

1. Open the notebook in Google Colab.
2. Run the cells sequentially.
3. The MNIST dataset will be downloaded automatically through Keras.
4. Train the Autoencoder.
5. Train the VAE.
6. View the reconstruction results.
7. View the generated images.
8. Check the MSE and SSIM comparison.

### Option 2 — Local Environment

Install the required packages:

```bash
pip install tensorflow numpy matplotlib
```

Then launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook:

```text
Practical_03_Autoencoder_VAE_Image_Reconstruction.ipynb
```

and execute the cells sequentially.

---

## 🧪 Experimental Workflow

The complete experiment follows these steps:

```text
1. Import Libraries
        ↓
2. Set Configuration
        ↓
3. Load MNIST Dataset
        ↓
4. Normalize & Flatten Images
        ↓
5. Build Autoencoder
        ↓
6. Train Autoencoder
        ↓
7. Reconstruct Images
        ↓
8. Plot AE Loss
        ↓
9. Build VAE
        ↓
10. Train VAE
        ↓
11. Reconstruct Images
        ↓
12. Generate New Images
        ↓
13. Calculate MSE & SSIM
        ↓
14. Compare Training Loss
        ↓
15. Visual Comparison
        ↓
16. Conclusion
```

---

## 📌 Results

The experiment demonstrates that:

- The **Autoencoder** successfully learns a compressed representation of MNIST images and reconstructs them.
- The **VAE** also reconstructs MNIST images while learning a probabilistic and regularized latent space.
- The VAE can generate new digit-like images by sampling from its latent distribution.
- MSE provides a numerical measure of reconstruction error.
- SSIM provides a measure of structural similarity.
- Reconstruction metrics alone do not fully capture the additional generative capability of the VAE.

The exact MSE and SSIM values should be taken from the actual execution of the notebook.

---

## 🎓 Learning Outcomes

After completing this practical, the following concepts are demonstrated:

- Autoencoder architecture
- Encoder-decoder networks
- Bottleneck representations
- Latent spaces
- Variational Autoencoders
- Mean and log variance
- Reparameterization trick
- Reconstruction loss
- KL divergence
- Probabilistic latent representations
- Image reconstruction
- Generative image synthesis
- MSE evaluation
- SSIM evaluation
- Training-loss visualization

---

## 📝 Conclusion

This practical demonstrates the difference between an **Autoencoder** and a **Variational Autoencoder** for image reconstruction.

The Autoencoder focuses primarily on learning a compressed representation that can be decoded back into the original image. In contrast, the VAE combines image reconstruction with probabilistic latent-space learning through KL-divergence regularization.

As a result, the VAE provides an additional capability: **generating new digit-like images by sampling from the learned latent distribution**.

Therefore, the experiment illustrates how VAEs extend the traditional autoencoder architecture from reconstruction and representation learning toward **generative modeling**.

---

## 👤 Author

**Gayatri Pandharinath Gaikwad**

**Course:** Generative AI  
**Batch:** TY-A3  
**Practical:** No. 03

---

## 📚 Project Type

**Generative AI — Deep Learning Practical**

**Topic:** Autoencoder and Variational Autoencoder for Image Reconstruction
