import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

try:
    import torchvision.io as io
except RuntimeError:
    io = None


class GLipsFullClipDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, num_frames=25):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.num_frames = num_frames
        self.samples = []

        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            split_folder = os.path.join(root_dir, cls_name, split)
            if not os.path.exists(split_folder):
                for alt in ['train', 'validation', 'val', 'test']:
                    alt_folder = os.path.join(root_dir, cls_name, alt)
                    if os.path.exists(alt_folder):
                        split_folder = alt_folder
                        break

            if os.path.exists(split_folder):
                for file in os.listdir(split_folder):
                    if file.endswith('.mp4'):
                        self.samples.append((os.path.join(split_folder, file), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video = None

        if io is not None and hasattr(io, 'read_video'):
            try:
                video, _, _ = io.read_video(video_path, pts_unit='sec', output_format='TCHW')
            except Exception:
                video = None

        if video is None:
            try:
                import imageio.v2 as imageio
            except Exception:
                try:
                    import imageio
                except Exception:
                    imageio = None

            if imageio is not None:
                frames = []
                try:
                    reader = imageio.get_reader(video_path, 'ffmpeg')
                    for frame in reader:
                        frames.append(frame)
                    reader.close()
                    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
                except Exception:
                    video = None

        if video is None:
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                if not frames:
                    raise RuntimeError(f"No frames read from {video_path}")
                video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
            except Exception as e:
                raise RuntimeError(f"Could not read video. Install imageio or opencv. Error: {e}")

        video = video.float() / 255.0
        T = video.size(0)

        if T > self.num_frames:
            indices = np.linspace(0, T - 1, num=self.num_frames).astype(int)
            video = video[indices]
        elif T < self.num_frames:
            pad_count = self.num_frames - T
            video = torch.cat([video, video[-1:].repeat(pad_count, 1, 1, 1)], dim=0)

        if self.transform:
            video = self.transform(video)

        return video.permute(1, 0, 2, 3), label


class Frontend3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

    def forward(self, x):
        return self.layers(x)


class ResNet2DBackend(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.layers = nn.Sequential(*list(resnet.children())[4:-2])

    def forward(self, x):
        return self.layers(x)


class CNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = Frontend3D()
        self.resnet = ResNet2DBackend()

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = self.frontend(x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x = self.resnet(x)
        return x.view(B, T, 512, x.size(2), x.size(3))


class LipreadingGRU(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.proj = nn.Linear(hidden_size * 2, 512)

    def forward(self, x):
        x, _ = self.gru(x)
        return self.proj(x)


class GLipsModel(nn.Module):
    def __init__(self, num_classes=500):
        super().__init__()
        self.cnn = CNN3D()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gru = LipreadingGRU(input_size=512, hidden_size=512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        B, T, C, H, W = x.size()
        x = self.avgpool(x.view(B * T, C, H, W)).view(B, T, 512)
        x = self.gru(x)
        x = torch.mean(x, dim=1)
        return self.classifier(x)


if __name__ == '__main__':
    from tqdm import tqdm

    torch.backends.cudnn.benchmark = True

    num_workers = min(4, os.cpu_count() or 0)
    train_dataset = GLipsFullClipDataset(root_dir='./GLips/lipread_files/', split='train')
    val_dataset = GLipsFullClipDataset(root_dir='./GLips/lipread_files/', split='validation')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=num_workers,
                              pin_memory=True, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers,
                            pin_memory=True, persistent_workers=num_workers > 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == 'cuda', f"CUDA not available — got device '{device}'"
    print(f"Using device: {torch.cuda.get_device_name(device)}")
    model = GLipsModel()
    model.to(device)
    if hasattr(torch, 'compile'):
        model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    num_epochs = 100
    best_val_acc = 0.0
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    epoch_bar = tqdm(range(num_epochs), desc='Epochs', unit='epoch')
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        num_samples = 0

        train_bar = tqdm(train_loader, desc=f'Train {epoch+1}/{num_epochs}', leave=False, unit='batch')
        for data, target in train_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                loss = criterion(model(data), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * data.size(0)
            num_samples += data.size(0)
            train_bar.set_postfix(loss=f'{loss.item():.4f}')

        epoch_loss = running_loss / num_samples if num_samples > 0 else 0.0

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Val', leave=False, unit='batch'):
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    preds = torch.argmax(model(data), dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)

        val_acc = correct / total if total > 0 else 0.0
        epoch_bar.set_postfix(loss=f'{epoch_loss:.4f}', val_acc=f'{val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            tqdm.write(f"Saved best model (val_acc={best_val_acc:.4f})")

    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    print(f"Training finished. Best val acc: {best_val_acc:.4f}. Models saved in {save_dir}")
