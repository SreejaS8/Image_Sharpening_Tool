# 🎓 Student Video Restoration Model (Restormer-inspired)

Welcome to our lightweight student model for video restoration!  
Built in just **one month** by a team of 3 students, this project explores **Knowledge Distillation** using the powerful [Restormer](https://arxiv.org/abs/2111.09881) architecture as a reference.

While the results aren't perfect yet, this was an incredible learning journey into transformer-based video enhancement. 🚀

---
## 📎 Quick Links

🔗 **📽️ Final Output Video:**  
[Watch Output on Google Drive](https://drive.google.com/file/d/1zisauunRZvulfONcYg9C9EJtpjXfMZ3s/view?usp=sharing)

📄 **📚 Report Document:**  
[Read Report PDF](https://drive.google.com/file/d/1EOEkHlK25ipUehwKS4FFm9zRc5Y7-bjs/view?usp=sharing)

---

## 📁 Project Overview

🎯 **Goal:**  
To design a compact student model for **video sharpening and restoration**, learning from a teacher model based on Restormer.

🧠 **Technique Used:**  
- Knowledge Distillation (KD)
- Inspired by Restormer
- Custom student architectures for:
  - Defocus deblurring: | SSIM: 0.9216 | PSNR: 28.33 dB
  - Motion deblurring: | SSIM: 0.9595 | PSNR: 33.16 dB
  - Deraining: | SSIM: 0.9078 | PSNR: 30.52 dB

📆 **Duration:**  
1 Month (Intense 🧪🔥)

---

📂 **💻 Code Files:**  
Check the code above in this repository.

---

## 🛠️ Features

- Tiny student model trained via knowledge distillation
- Inference pipeline for full video processing
- Batch-wise processing for faster testing
- Multiple sub-models stacked: Defocus → Derain → Deblur
