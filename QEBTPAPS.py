import os
import glob
import random
import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import pennylane as qml
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import openai
from skimage import exposure

ROOT_DIR = "/path/to/BraTS2021_TrainingData"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
openai.api_key = OPENAI_API_KEY
CLASS_NAMES = ["meningioma", "pituitary", "glioma", "no_tumor"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ITERATIONS = 4
DATA_LIMIT = None

def normalize_clahe(img):
    a = (img - img.min()) / (img.max() - img.min() + 1e-8)
    u = (a * 255).astype("uint8")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    o = clahe.apply(u).astype("float32") / 255.0
    mn, mx = o.min(), o.max()
    return (o - mn) / (mx - mn + 1e-8)

def random_augment(img, mask):
    if random.random() < 0.5:
        img = np.fliplr(img).copy()
        mask = np.fliplr(mask).copy()
    if random.random() < 0.5:
        img = np.flipud(img).copy()
        mask = np.flipud(mask).copy()
    ang = random.uniform(-12, 12)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
    img = cv2.warpAffine((img*255).astype("uint8"), M, (w, h), flags=cv2.INTER_LINEAR)/255.0
    mask = cv2.warpAffine((mask*255).astype("uint8"), M, (w, h), flags=cv2.INTER_NEAREST)/255.0
    return img, mask

class BraTSDataset(Dataset):
    def __init__(self, root_dir, size=128, sequences=("flair",), augment=False, limit=None):
        self.size = size
        self.sequences = sequences
        self.augment = augment
        cases = sorted(glob.glob(os.path.join(root_dir, "BraTS*")))
        if limit:
            cases = cases[:limit]
        self.slices = []
        for case in cases:
            segg = glob.glob(os.path.join(case, "*_seg.nii*"))
            if not segg:
                continue
            seg = nib.load(segg[0]).get_fdata().astype("float32")
            seg = (seg > 0).astype("float32")
            img_paths = []
            for s in self.sequences:
                found = glob.glob(os.path.join(case, f"*_{s}.nii*"))
                if not found:
                    break
                img_paths.append(found[0])
            if len(img_paths) != len(self.sequences):
                continue
            imgs = [nib.load(p).get_fdata().astype("float32") for p in img_paths]
            img_mean = np.mean(imgs, axis=0)
            zdim = img_mean.shape[2]
            for z in range(zdim):
                sl = img_mean[:, :, z]
                ms = seg[:, :, z]
                if ms.sum() < 50:
                    continue
                slr = cv2.resize(sl, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
                msr = cv2.resize(ms, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
                sln = normalize_clahe(slr)
                self.slices.append((sln.astype("float32"), msr.astype("float32")))
        if not self.slices:
            raise RuntimeError("No slices prepared. Check dataset path.")
    def __len__(self):
        return len(self.slices)
    def __getitem__(self, idx):
        img, mask = self.slices[idx]
        if self.augment:
            img, mask = random_augment(img, mask)
        img = np.expand_dims(img, 0).astype("float32")
        mask = np.expand_dims(mask, 0).astype("float32")
        return torch.tensor(img), torch.tensor(mask)

class TumorClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        r = models.resnet18(pretrained=False)
        r.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        r.fc = nn.Linear(r.fc.in_features, num_classes)
        self.model = r
    def forward(self, x):
        return self.model(x)

n_qubits = 8
qlayer_shape = (3, n_qubits)
dev = qml.device("default.qubit", wires=n_qubits)

def entangling():
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[n_qubits - 1, 0])

@qml.qnode(dev, interface="torch")
def qnode(inputs, params):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for l in range(params.shape[0]):
        for q in range(n_qubits):
            qml.RX(params[l, q], wires=q)
        entangling()
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QCNNModule(nn.Module):
    def __init__(self, in_features=64, out_features=64):
        super().__init__()
        self.fc = nn.Linear(in_features, n_qubits)
        init = np.random.randn(*qlayer_shape).astype("float32") * 0.1
        self.params = nn.Parameter(torch.tensor(init))
        self.post = nn.Linear(n_qubits, out_features)
    def forward(self, x):
        xproj = self.fc(x)
        outs = []
        for i in range(xproj.shape[0]):
            out = qnode(xproj[i], self.params)
            outs.append(torch.tensor(out))
        stacked = torch.stack(outs).to(x.device)
        return self.post(stacked)

class ProgressionGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.qc = QCNNModule(32, 32)
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        b = x.shape[0]
        f = self.encoder(x).view(b, -1)
        q = self.qc(f)
        return self.fc(q).squeeze(-1)

class QuantumRNNCell(nn.Module):
    def __init__(self, in_size=16, hidden_size=16):
        super().__init__()
        self.fc_in = nn.Linear(in_size, n_qubits)
        self.fc_h = nn.Linear(hidden_size, n_qubits)
        init = np.random.randn(*qlayer_shape).astype("float32") * 0.05
        self.params = nn.Parameter(torch.tensor(init))
        self.fc_out = nn.Linear(n_qubits, hidden_size)
    @qml.qnode(dev, interface="torch")
    def _rqnode(self, x_in, h_in, params):
        for i in range(n_qubits):
            val = 0.0
            if i < x_in.shape[0]:
                val += x_in[i]
            if i < h_in.shape[0]:
                val += h_in[i]
            qml.RY(val, wires=i)
        for layer in range(params.shape[0]):
            for q in range(n_qubits):
                qml.RZ(params[layer, q], wires=q)
            entangling()
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    def forward(self, x, h):
        xq = self.fc_in(x)
        hq = self.fc_h(h)
        res = self._rqnode(xq, hq, self.params)
        stacked = torch.stack([torch.tensor(r) for r in res]).to(x.device)
        return torch.tanh(self.fc_out(stacked))

class QETAFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.qcnn = QCNNModule(64, 64)
        self.rqcell = QuantumRNNCell(16, 16)
        self.rqout = nn.Linear(16, 1)
    def forward(self, seq):
        batch, tim, c, h, w = seq.shape
        segs = []
        feats = []
        for t in range(tim):
            x = seq[:, t]
            z = self.encoder(x)
            seg = self.decoder(z)
            segs.append(seg)
            pooled = self.global_pool(z).view(batch, -1)
            qf = self.qcnn(pooled)
            feats.append(qf)
        segs = torch.stack(segs, dim=1)
        feats = torch.stack(feats, dim=1)
        h0 = torch.zeros(batch, 16).to(seq.device)
        hid = h0
        for t in range(tim):
            inp = feats[:, t, :16]
            hid = self.rqcell(inp, hid)
        prog = self.rqout(hid).squeeze(-1)
        return segs[:, -1], prog

def classify_image(model, img_np):
    t = torch.tensor(np.expand_dims(img_np, 0)).to(DEVICE)
    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        return CLASS_NAMES[idx], float(probs[idx]), probs.tolist()

def generate_future_sequence(base_img, base_mask, num_steps, progressor):
    seq = []
    current_img = base_img.copy()
    current_mask = base_mask.copy()
    for i in range(num_steps):
        inp = torch.tensor(np.expand_dims(current_img, 0)).unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            growth = float(progressor(inp).detach().cpu().numpy()[0])
        k = 1 + growth * 6
        kw = max(1, int(k))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kw))
        new_mask = cv2.dilate((current_mask * 255).astype("uint8"), kernel).astype("float32") / 255.0
        new_mask = np.clip(new_mask, 0, 1)
        new_img = current_img.copy()
        new_img[new_mask > 0.5] = np.clip(new_img[new_mask > 0.5] + 0.15 * growth, 0, 1)
        seq.append((new_img.copy(), new_mask.copy(), growth))
        current_img = new_img
        current_mask = new_mask
    return seq

def build_prompt(stages):
    parts = []
    for i, s in enumerate(stages):
        parts.append(f"Stage {i+1}: class={s['class']} prob={s['prob']:.3f} growth={s['growth']:.3f}")
    return "You are a clinical assistant. Summarize the following tumor progression stages and provide likely clinical implications and suggested next steps: " + " ; ".join(parts)

def gpt3_summary(prompt):
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY not set"
    resp = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=300, temperature=0.7)
    return resp.choices[0].text.strip()

def visualize_and_save(base_img, base_mask, seq, outdir):
    os.makedirs(outdir, exist_ok=True)
    for i, (img, mask, growth) in enumerate(seq):
        fig, axs = plt.subplots(1,3, figsize=(9,3))
        axs[0].imshow(base_img, cmap="gray")
        axs[0].set_title("original")
        axs[1].imshow(mask, cmap="gray")
        axs[1].set_title(f"mask t{i+1}")
        axs[2].imshow(img, cmap="gray")
        axs[2].set_title(f"image t{i+1} g={growth:.3f}")
        for a in axs:
            a.axis("off")
        path = os.path.join(outdir, f"stage_{i+1}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)

def main():
    ds = BraTSDataset(ROOT_DIR, size=128, sequences=("flair",), augment=False, limit=DATA_LIMIT)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    classifier = TumorClassifier(num_classes=4).to(DEVICE)
    progressor = ProgressionGenerator().to(DEVICE)
    qetaf = QETAFModel().to(DEVICE)
    optim_c = optim.Adam(classifier.parameters(), lr=1e-3)
    optim_p = optim.Adam(progressor.parameters(), lr=1e-3)
    for epoch in range(1):
        classifier.train()
        progressor.train()
        for i, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            labels = torch.randint(0, 4, (imgs.size(0),)).to(DEVICE)
            logits = classifier(imgs)
            lossc = nn.CrossEntropyLoss()(logits, labels)
            optim_c.zero_grad()
            lossc.backward()
            optim_c.step()
            predg = progressor(imgs)
            lossg = ((predg.mean() - 0.2) ** 2)
            optim_p.zero_grad()
            lossg.backward()
            optim_p.step()
    classifier.eval()
    progressor.eval()
    out_root = "qetaf_results"
    os.makedirs(out_root, exist_ok=True)
    report = []
    for idx in range(min(len(ds), 20)):
        img, mask = ds[idx]
        img_np = img.squeeze(0).numpy()
        mask_np = mask.squeeze(0).numpy()
        cls, p, probs = classify_image(classifier, img_np)
        entry = {"case": idx, "initial_class": cls, "initial_prob": p, "stages": [], "explanation": ""}
        if cls != "no_tumor":
            seq = generate_future_sequence(img_np, mask_np, NUM_ITERATIONS, progressor)
            stages = []
            for s_img, s_mask, growth in seq:
                c2, p2, probs2 = classify_image(classifier, s_img)
                stages.append({"class": c2, "prob": p2, "probs": probs2, "growth": growth})
            visualize_and_save(img_np, mask_np, seq, os.path.join(out_root, f"case_{idx}"))
            prompt = build_prompt(stages)
            explanation = gpt3_summary(prompt)
            entry["stages"] = stages
            entry["explanation"] = explanation
        else:
            prompt = f"No tumor detected for case {idx}. classifier={cls} prob={p:.3f}"
            explanation = gpt3_summary(prompt)
            entry["explanation"] = explanation
        report.append(entry)
    with open(os.path.join(out_root, "report.txt"), "w") as f:
        for r in report:
            f.write(str(r) + "\n\n")

if __name__ == "__main__":
    main()
