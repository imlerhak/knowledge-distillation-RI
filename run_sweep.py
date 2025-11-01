import os, re, csv, sys, subprocess, datetime, argparse

CKPT_RE = re.compile(r"teacher_(?P<teacher>resnet18|resnet34|resnet50)_e(?P<epoch>\d+)\.pt$", re.I)
RESULTS_DIR = "results"
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
CSV_PATH = os.path.join(RESULTS_DIR, "sweep_results.csv")

def find_teacher_ckpts(folder="checkpoints"):
    if not os.path.isdir(folder):
        return []
    out = []
    for name in os.listdir(folder):
        m = CKPT_RE.match(name)
        if m:
            out.append({
                "path": os.path.join(folder, name),
                "teacher": m.group("teacher"),
                "epoch": int(m.group("epoch")),
            })
    out.sort(key=lambda x: (x["teacher"], x["epoch"]))
    return out

def run_cmd(cmd, log_file):
    with open(log_file, "w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last_acc = None
        for line in p.stdout:
            f.write(line)
            f.flush()
            if "Student Test Acc @epoch" in line:
                last_acc = line.strip().split(":")[-1].strip()
        ret = p.wait()
    return ret, last_acc

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="resnet18", choices=["resnet18","resnet34","resnet50"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--kd_epochs", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=4.0)
    args = ap.parse_args()

    ensure_dirs()
    ckpts = find_teacher_ckpts()
    if not ckpts:
        print("Nema teacher checkpointova. Prvo ih treniraj!")
        sys.exit(1)

    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        if write_header:
            writer.writerow([
                "timestamp","student","teacher","teacher_epoch",
                "epochs","kd_epochs","alpha","temperature","final_acc",
                "teacher_ckpt","log_file"
            ])

        for ck in ckpts:
            teacher = ck["teacher"]
            t_epoch = ck["epoch"]
            ckpt = ck["path"]

            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(LOGS_DIR, f"{stamp}__{teacher}_e{t_epoch}.log")

            cmd = [
                sys.executable, "train_student_kd.py",
                "--student", args.student,
                "--teacher", teacher,
                "--teacher_ckpt", ckpt,
                "--epochs", str(args.epochs),
                "--kd_epochs", str(args.kd_epochs),
                "--alpha", str(args.alpha),
                "--temperature", str(args.temperature),
            ]

            print(">>", " ".join(cmd))
            ret, final_acc = run_cmd(cmd, log_path)

            writer.writerow([
                stamp, args.student, teacher, t_epoch,
                args.epochs, args.kd_epochs, args.alpha, args.temperature,
                final_acc or "", ckpt, log_path
            ])
            print("   => final_acc:", final_acc)

    print("\nGotovo")
    print("Rezultati:   ", CSV_PATH)
    print("Log fajlovi: ", LOGS_DIR)

if __name__ == "__main__":
    main()
