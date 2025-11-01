# train_student_kd.py
import argparse, torch, torch.optim as optim
from tqdm import tqdm

from kd.models import resnet18_cifar, resnet34_cifar, resnet50_cifar
from kd.utils  import evaluate, make_cifar10_loaders
from kd.losses import kd_loss

from kd.utils import load_ckpt  # koristimo isti util kao za teachera

MODEL_MAP = {
    "resnet18": resnet18_cifar,
    "resnet34": resnet34_cifar,
    "resnet50": resnet50_cifar,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="resnet18", choices=list(MODEL_MAP.keys()))
    ap.add_argument("--teacher", default="resnet18", choices=list(MODEL_MAP.keys()))
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--kd_epochs", type=int, default=20)   # ESKD: posle ovoga KD off
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--milestones", type=int, nargs="*", default=[30, 40])
    args = ap.parse_args()

    device = torch.device(args.device)
    train_loader, test_loader = make_cifar10_loaders(batch_size=args.batch_size)

    # student i teacher
    student = MODEL_MAP[args.student]().to(device)
    teacher = MODEL_MAP[args.teacher]().to(device)
    teacher = load_ckpt(teacher, args.teacher_ckpt, map_location=device)
    teacher.eval()

    opt   = optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones, gamma=0.2)

    for epoch in range(1, args.epochs + 1):
        student.train()
        use_kd = (epoch <= args.kd_epochs)
        pbar = tqdm(train_loader, desc=f"[Student KD] epoch {epoch} ({'KD' if use_kd else 'CE-only'})")
        for x,y in pbar:
            x,y = x.to(device), y.to(device)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            loss = kd_loss(s_logits, t_logits, y, alpha=args.alpha, temperature=args.temperature, use_kd=use_kd)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        sched.step()
        acc = evaluate(student, test_loader, device)
        print(f"Student Test Acc @epoch {epoch}: {acc:.4f}")

if __name__ == "__main__":
    main()
