# -*- coding: utf-8 -*-
"""下载 VQA-RAD 与 IU-Xray 数据集到本地（数据本身不进仓库）

用法:
  python download_data.py --vqa-rad          # 走 HuggingFace datasets（默认镜像）
  python download_data.py --iu-xray          # 下载 IU-Xray jsonl + 图片包并解压
  python download_data.py --all
可选环境变量: HF_ENDPOINT=https://hf-mirror.com（国内加速）
下载后目录:
  data/vqa-rad/   （由 datasets 缓存管理）
  data/iu-xray/   image.zip + train/val/test.jsonl + images/（解压后）
"""
import argparse
import os
import zipfile

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def download_vqa_rad():
    from datasets import load_dataset
    ds = load_dataset("flaviagiammarino/vqa-rad")
    print(ds)
    print("VQA-RAD ready (datasets cache).")


def download_iu_xray():
    import urllib.request

    out = os.path.join(DATA_DIR, "iu-xray")
    os.makedirs(out, exist_ok=True)
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url_base = f"{base}/datasets/dz-osamu/IU-Xray/resolve/main"
    for f in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        dst = os.path.join(out, f)
        if not os.path.exists(dst):
            print("downloading", f)
            urllib.request.urlretrieve(f"{url_base}/{f}", dst)
    zpath = os.path.join(out, "image.zip")
    if not os.path.exists(zpath):
        print("downloading image.zip (~1.1GB, please wait)")
        urllib.request.urlretrieve(f"{url_base}/image.zip", zpath)
    img_root = os.path.join(out, "images")
    if not os.path.exists(os.path.join(img_root, "iu_xray")):
        print("unzipping images ...")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(img_root)
    print("IU-Xray ready at", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa-rad", action="store_true")
    ap.add_argument("--iu-xray", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all or args.vqa_rad:
        download_vqa_rad()
    if args.all or args.iu_xray:
        download_iu_xray()
