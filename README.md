# <b>📍 Project Roadmap</b></br>
This project aims to develop a lightweight AI model that sharpens blurry video images, especially useful for video calls on low internet. We use a Teacher-Student model approach to train a fast, real-time solution.

## 🧩 Phase 1: Data Preparation & Simulation
### Goal: Simulate video call blur and build a training dataset.
- Collect high-resolution images (e.g., DIV2K, BSD500, etc.)
- Simulate blur using downscale → upscale (bicubic/bilinear)
- Create paired dataset:
  - High-Resolution (HR) = Original sharp image
  - Low-Quality (LQ) = Simulated blurry version
- Organize folders: train/, val/, test/

## 🧠 Phase 2: Model Design & Knowledge Distillation
### Goal: Build two models — one smart (Teacher), one fast (Student).
- Use a pre-trained model (e.g., ESRGAN, EDSR) as the Teacher
- Design a lightweight CNN for the Student
- Use Knowledge Distillation to train the Student by mimicking the Teacher
- Combine MSE Loss + SSIM Loss for better visual results

## 🏋️ Phase 3: Training, Evaluation & Testing
### Goal: Train the Student and check how well it performs.
- Train the Student on LQ → HR image pairs
- Evaluate with:
  - SSIM (Structural Similarity Index) – Goal: >90%
  - MOS (Mean Opinion Score) – Human opinion on quality
- Test on over 100 diverse images (text, nature, people, animals, games)
- Visualize side-by-side comparisons (Input vs Output vs Ground Truth)

## ⚡ Phase 4: Optimization & Deployment
### Goal: Make the model fast enough for real-time use.
- Optimize model for 30–60 FPS on 1920x1080 images
- Convert to TorchScript or ONNX for deployment
- Benchmark performance (CPU & GPU)
- Package the final model and create a user-friendly interface (optional)
