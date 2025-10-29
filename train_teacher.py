import argparse, torch, torch.optim as optim
import torch.nn.functional as F
from kd.models import resnet18_cifar, resnet34_cifar, resnet50_cifar
from kd.utils import make_cifar10_loaders, evaluate, save_ckpt
from tqdm import tqdm

MODEL_MAP = {
    "resnet18": resnet18_cifar,
    "resnet34": resnet34_cifar,
    "resnet50": resnet50_cifar,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet34", choices=list(MODEL_MAP.keys()))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--milestones", type=int, nargs="*", default=[60, 120, 160])
    ap.add_argument("--save", default="checkpoints/teacher_{model}_e{epoch}.pt")
    ap.add_argument("--early_snapshots", type=int, nargs="*", default=[35, 50, 65, 80])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    train_loader, test_loader = make_cifar10_loaders(batch_size=args.batch_size)

    model = MODEL_MAP[args.model]().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Teacher][{args.model}] epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        scheduler.step()
        acc = evaluate(model, test_loader, device)
        print(f"Test Acc @epoch {epoch}: {acc:.4f}")

        # snimi “rane” snapshotove + finalni
        if epoch in set(args.early_snapshots + [args.epochs]):
            path = args.save.format(model=args.model, epoch=epoch)
            save_ckpt(model, path)

if __name__ == "__main__":
    main()
