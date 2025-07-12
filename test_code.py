import os, cv2, sys, torch, numpy as np
from torchvision import transforms
sys.path.append(os.path.abspath('./Restormer/basicsr/models/archs'))
from restormer_arch import Restormer
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# ────────────── Configuration ──────────────
IN_VIDEO   = r"test_video1.mp4"
OUT_VIDEO  = r"out_combined.mp4"
WORK_SIZE  = 128
BATCH_SIZE = 4
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Device:", device)

# ────────────── Models ──────────────
class ResidualBlock(torch.nn.Module):
    def _init_(self, c):
        super()._init_()
        self.conv1 = torch.nn.Conv2d(c, c, 3, padding=1)
        self.relu  = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(c, c, 3, padding=1)
    def forward(self,x): return x + self.conv2(self.relu(self.conv1(x)))

class StudentDeblurNet(torch.nn.Module):
    def _init_(self):
        super()._init_()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3,32,3,padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(32,64,3,stride=2,padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(64,64,3,padding=1), torch.nn.ReLU()
        )
        self.middle  = torch.nn.Sequential(*[ResidualBlock(64) for _ in range(4)])
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64,32,2,stride=2), torch.nn.ReLU(),
            torch.nn.Conv2d(32,3,3,padding=1)
        )
    def forward(self,x): return self.decoder(self.middle(self.encoder(x)))

class NanoRestormer(Restormer):
    def _init_(self):
        super()._init_(
            inp_channels=3, out_channels=3,
            dim=24,
            num_blocks=[2,2,2,2],
            num_refinement_blocks=2,
            heads=[1,2,4,8],
            ffn_expansion_factor=2.0,
            bias=False,
            LayerNorm_type='WithBias',
            dual_pixel_task=False
        )

class StudentSRCNN(torch.nn.Module):
    def _init_(self):
        super()._init_()
        self.conv1 = torch.nn.Conv2d(3,64,9,padding=4)
        self.conv2 = torch.nn.Conv2d(64,32,5,padding=2)
        self.conv3 = torch.nn.Conv2d(32,3,5,padding=2)
    def forward(self,x):
        x=torch.relu(self.conv1(x))
        x=torch.relu(self.conv2(x))
        return self.conv3(x)

def load_model(pth, cls):
    net=cls().to(device)
    ckpt=torch.load(pth,map_location=device)
    net.load_state_dict(ckpt.get("params",ckpt), strict=True)
    net.eval()
    return net

print("📦 Loading models …")
model_kd  = load_model("student_kd.pth",              StudentDeblurNet)
model_me  = load_model("student_motion_enhanced.pth", NanoRestormer)
model_amp = load_model("student_model_amp.pth",       StudentSRCNN)
print("✅ Models ready\n")

# ────────────── Transforms ──────────────
tfm_in  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((WORK_SIZE, WORK_SIZE))
])
tfm_out = transforms.ToPILImage()

# ────────────── Video IO ──────────────
cap = cv2.VideoCapture(IN_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc= cv2.VideoWriter_fourcc(*'mp4v')
writer= cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w*2, h))  # side-by-side

print(f"🎞  Processing {count} frames at {fps:.1f} FPS …")

frames, tensors = [], []
ssim_scores = []

with torch.no_grad():
    for idx in tqdm(range(count)):
        ret, frame = cap.read()
        if not ret: break

        frames.append(frame)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensors.append(tfm_in(img))

        if len(tensors)==BATCH_SIZE or idx==count-1:
            batch = torch.stack(tensors).to(device)  # [B,3,H,W]
            out = model_kd(batch)
            out = model_me(out)
            out = model_amp(out)
            out = out.clamp(0,1).cpu()

            for i in range(out.size(0)):
                pred_img = tfm_out(out[i])
                pred_np  = cv2.cvtColor(np.array(pred_img), cv2.COLOR_RGB2BGR)
                pred_resized = cv2.resize(pred_np, (w, h))

                orig = frames[i]
                concat = np.concatenate((orig, pred_resized), axis=1)
                writer.write(concat)

                # ─── SSIM ───
                gray_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
                gray_pred = cv2.cvtColor(pred_resized, cv2.COLOR_BGR2GRAY)
                score = ssim(gray_orig, gray_pred, data_range=255)
                ssim_scores.append(score)

            frames, tensors = [], []

cap.release(); writer.release()
mean_ssim = np.mean(ssim_scores)
print(f"\n✅ Done! Video saved to:\n{OUT_VIDEO}")
print(f"📊 Average SSIM: {mean_ssim:.4f}")