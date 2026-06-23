import hashlib
import json
import random
import shutil
from pathlib import Path

from .paths import norm_path, now_stamp


def parse_ratio(text, default=1.0):
    raw = str(text).strip()
    if not raw:
        return default
    value = float(raw[:-1]) / 100.0 if raw.endswith("%") else float(raw)
    if value > 1:
        value = value / 100.0
    if value < 0:
        raise ValueError("比例不能小于 0")
    return min(value, 1.0)


def split_label_line(line):
    line = line.rstrip("\r\n")
    if "\t" in line:
        image, label = line.split("\t", 1)
        return image, "\t", label
    parts = line.split(maxsplit=1)
    if len(parts) == 2:
        return parts[0], " ", parts[1]
    return line, "", ""


def discover_label_files(root):
    root = Path(root)
    if (root / "train.txt").exists() and (root / "val.txt").exists():
        return {"train": root / "train.txt", "val": root / "val.txt"}
    if (root / "label.txt").exists():
        return {"all": root / "label.txt"}
    train = list(root.glob("**/train.txt"))
    val = list(root.glob("**/val.txt"))
    if train and val:
        return {"train": train[0], "val": val[0]}
    label = list(root.glob("**/label.txt"))
    if label:
        return {"all": label[0]}
    raise FileNotFoundError(f"未找到 train.txt/val.txt 或 label.txt：{root}")


def read_lines(root, val_ratio, rng):
    files = discover_label_files(root)
    if "all" in files:
        lines = [
            x
            for x in files["all"].read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        rng.shuffle(lines)
        val_count = int(len(lines) * val_ratio)
        return lines[val_count:], lines[:val_count], files["all"].parent
    train = [
        x
        for x in files["train"].read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    val = [
        x for x in files["val"].read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    return train, val, files["train"].parent


def sample(lines, ratio, rng):
    lines = list(lines)
    rng.shuffle(lines)
    if ratio >= 1:
        return lines
    count = int(len(lines) * ratio)
    if lines and ratio > 0 and count == 0:
        count = 1
    return lines[:count]


def resolve_image(label_root, image_ref):
    raw = image_ref.strip()
    candidate = Path(raw)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for item in (
        Path(label_root) / raw,
        Path(label_root).parent / raw,
        Path(label_root) / "images" / raw,
    ):
        if item.exists():
            return item
    return Path(label_root) / raw


def copy_line(line, label_root, out_dir, source_name, index):
    image_ref, sep, label = split_label_line(line)
    src = resolve_image(label_root, image_ref)
    if not src.exists():
        raise FileNotFoundError(str(src))
    digest = hashlib.md5(f"{source_name}-{index}-{src}".encode("utf-8")).hexdigest()[
        :10
    ]
    dest_rel = Path("images") / source_name / f"{src.stem}_{digest}{src.suffix}"
    dest = Path(out_dir) / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return f"{dest_rel.as_posix()}{sep}{label}".rstrip()


def merge_datasets(sources, out_dir, val_ratio, seed, log=None):
    rng = random.Random(seed)
    out = Path(norm_path(out_dir))
    out.mkdir(parents=True, exist_ok=True)
    train_out, val_out, missing = [], [], []
    manifest = {"created_at": now_stamp(), "sources": []}
    for idx, (path, ratio, tag) in enumerate(sources, start=1):
        root = Path(norm_path(path))
        source_name = f"{tag}_{idx}_{root.name}"
        if log:
            log(f"读取数据集：{root}，抽样比例：{ratio:.2f}")
        train, val, label_root = read_lines(root, val_ratio, rng)
        before = (len(train_out), len(val_out))
        for line_idx, line in enumerate(sample(train, ratio, rng)):
            try:
                train_out.append(copy_line(line, label_root, out, source_name, line_idx))
            except FileNotFoundError as exc:
                missing.append(str(exc))
        for line_idx, line in enumerate(sample(val, ratio, rng)):
            try:
                val_out.append(copy_line(line, label_root, out, source_name, line_idx))
            except FileNotFoundError as exc:
                missing.append(str(exc))
        manifest["sources"].append(
            {
                "path": str(root),
                "ratio": ratio,
                "train_added": len(train_out) - before[0],
                "val_added": len(val_out) - before[1],
            }
        )
    rng.shuffle(train_out)
    rng.shuffle(val_out)
    (out / "train.txt").write_text("\n".join(train_out) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val_out) + "\n", encoding="utf-8")
    manifest.update(
        {
            "train_samples": len(train_out),
            "val_samples": len(val_out),
            "missing_count": len(missing),
            "missing_images": missing[:200],
        }
    )
    (out / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out, manifest
