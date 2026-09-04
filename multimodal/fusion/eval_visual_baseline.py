"""Evaluate the standalone visual checkpoint on val for both the mouth-crop and full-face trees."""
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

LIPREAD = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lipreading'))
sys.path.insert(0, os.path.join(LIPREAD, 'Transformer_based'))
sys.path.insert(0, LIPREAD)
from model import GLipsNet, _strip_orig_mod        # noqa: E402
from dataset import GLipsFullClipDataset, VideoAugment  # noqa: E402

CKPT = os.path.join(LIPREAD, 'Transformer_based', 'checkpoints_500', 'best_model.pth')
ROOTS = {
    'mouth_crop': os.path.join(LIPREAD, 'GLips_mouth', 'lipread_files'),
    'full_face':  os.path.join(LIPREAD, 'GLips', 'lipread_files'),
}
EXCLUDED = ()  # hier/soll mouth-ROI bug fixed upstream; full 500-class vocabulary now


def evaluate(root, device):
    if not os.path.isdir(root):
        return None
    classes = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)) and d not in EXCLUDED)
    tf = VideoAugment(crop_size=88, resize_size=96, is_train=False)
    ds = GLipsFullClipDataset(root, split='val', num_frames=25, transform=tf, classes=classes)
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4,
                        pin_memory=True, persistent_workers=True)

    model = GLipsNet(num_classes=len(classes), pool='attn').to(device)
    model.load_state_dict(_strip_orig_mod(torch.load(CKPT, map_location=device, weights_only=True)))
    model.eval()

    c1 = c5 = n = 0
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.no_grad():
        for video, target in tqdm(loader, desc=os.path.basename(os.path.dirname(root)), unit='batch'):
            video, target = video.to(device), target.to(device)
            with torch.amp.autocast('cuda', dtype=amp):
                logits = model(video)
            c1 += (logits.argmax(1) == target).sum().item()
            c5 += (logits.topk(5, 1).indices == target.unsqueeze(1)).any(1).sum().item()
            n += target.size(0)
    return len(classes), n, c1 / n, c5 / n


def main():
    device = torch.device('cuda')
    print(f'checkpoint: {CKPT}\n')
    for name, root in ROOTS.items():
        res = evaluate(root, device)
        if res is None:
            print(f'{name:12s}: no val clips at {root}')
            continue
        ncls, n, top1, top5 = res
        print(f'{name:12s}: {ncls} classes, {n} clips -> top1={top1:.4f} top5={top5:.4f}')


if __name__ == '__main__':
    main()
