import sys
import os
import torch

# Add Restormer root folder to sys.path (folder containing this script)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from basicsr.archs.restormer_arch import Restormer

def load_teacher_model():
    teacher_model = Restormer()

    checkpoint_path = os.path.join(os.path.dirname(__file__), 'pretrained_models', 'motion_deblurring.pth')
    checkpoint = torch.load(checkpoint_path)

    teacher_model.load_state_dict(checkpoint['params'])
    print("✅ Pretrained weights loaded!")

    teacher_model.eval()

    for param in teacher_model.parameters():
        param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)

    print("✅ Teacher model ready and frozen for distillation.")
    return teacher_model

if __name__ == "__main__":
    model = load_teacher_model()
