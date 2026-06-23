"""
模型测试
对比 PaddleOCR（完整管线） vs TextDetection（纯检测）

UI 调用: validation_code() 返回脚本字符串, latest_image() 找结果图片
直接运行: python validation.py <img_path>
"""
import os
import sys
from pathlib import Path


def main():
    """命令行入口: python validation.py <图片路径>"""
    if len(sys.argv) < 2:
        print("用法: python validation.py <图片路径>")
        sys.exit(1)

    img_path = sys.argv[1]
    det_model_dir = os.environ.get(
        "DET_MODEL_DIR",
        fr'D:\dl\projects\Git\PaddleX\mytools\exports\det_5',
    )
    rec_model_dir = os.environ.get(
        "REC_MODEL_DIR",
        fr'D:\app\app1\models\ocr\paddle_ocr\rec',
    )
    textline_ori_dir = os.environ.get(
        "TEXTLINE_ORI_DIR",
        fr'D:\app\app1\models\ocr\paddle_ocr\textline_orientation',
    )
    save_dir = os.environ.get("SAVE_DIR", "./output/")
    os.makedirs(save_dir, exist_ok=True)

    from paddleocr import PaddleOCR, TextDetection

    # ========== 1. TextDetection（纯检测） ==========
    print("=" * 60)
    print("TextDetection（纯检测）")
    print("=" * 60)
    model_det = TextDetection(
        model_dir=det_model_dir,
        limit_side_len=1216,
        thresh=0.3,
        box_thresh=0.5,
        unclip_ratio=1.5,
    )
    output_det = model_det.predict(img_path, batch_size=1)
    for res in output_det:
        res.print()
        res.save_to_img(save_path=os.path.join(save_dir, "td_det"))

    # ========== 2. PaddleOCR（完整管线） ==========
    print("\n" + "=" * 60)
    print("PaddleOCR（检测+识别+方向）")
    print("=" * 60)
    try:
        ocr = PaddleOCR(
            text_detection_model_dir=det_model_dir,
            text_det_limit_side_len=1216,
            text_det_thresh=0.3,
            text_det_box_thresh=0.5,
            text_det_unclip_ratio=1.5,
            text_recognition_model_dir=rec_model_dir,
            use_doc_unwarping=False,
            use_doc_orientation_classify=False,
            textline_orientation_model_dir=textline_ori_dir,
            use_textline_orientation=True,
            device='gpu',
        )
        output_ocr = ocr.predict(img_path)
        for res in output_ocr:
            res.print()
            res.save_to_img(save_path=os.path.join(save_dir, "paddleocr"))
    except AssertionError as e:
        print(f"⚠️  PaddleOCR 管线执行失败（textline_orientation 返回了意外角度）: {e}")
        print("   建议设置 use_textline_orientation=False 重试")

    print(f"\n✅ 对比完成，结果已保存到 {save_dir}")
    print(f"  - TextDetection:  {save_dir}td_det/")
    print(f"  - PaddleOCR:      {save_dir}paddleocr/")


def validation_code():
    """返回被 UI 用 python -c 执行的验证脚本字符串"""
    return r'''
import inspect
import sys
from pathlib import Path

image_path = sys.argv[1]
output_dir = Path(sys.argv[2])
det_model_dir = sys.argv[3]
use_rec = sys.argv[4].lower() == "true"
rec_model_dir = sys.argv[5]
device = sys.argv[6]
det_model_name = sys.argv[7]
rec_model_name = sys.argv[8]
repo_root = Path(sys.argv[9])
output_dir.mkdir(parents=True, exist_ok=True)

try:
    from paddleocr import PaddleOCR
except Exception as exc:
    paddleocr_repo = repo_root / "paddlex" / "repo_manager" / "repos" / "PaddleOCR"
    if paddleocr_repo.exists():
        sys.path.insert(0, str(paddleocr_repo))
    try:
        from paddleocr import PaddleOCR
    except Exception as fallback_exc:
        raise SystemExit(
            "无法导入 paddleocr，请在 GUI 的 Python 命令中选择已安装 paddleocr 的环境，"
            "或确认 PaddleX 内置 PaddleOCR repo 存在。"
            f" 原始错误：{exc}; fallback 错误：{fallback_exc}"
        )

signature = inspect.signature(PaddleOCR)
params = signature.parameters
kwargs = {}

def put(name, value):
    if name in params and value not in (None, ""):
        kwargs[name] = value

put("use_doc_orientation_classify", False)
put("use_doc_unwarping", False)
put("use_textline_orientation", False)
put("text_detection_model_name", det_model_name)
put("text_recognition_model_name", rec_model_name if use_rec else None)
put("text_detection_model_dir", det_model_dir)
put("text_recognition_model_dir", rec_model_dir if use_rec else None)
put("det_model_dir", det_model_dir)
put("rec_model_dir", rec_model_dir if use_rec else None)
put("device", device)
if "use_gpu" in params:
    kwargs["use_gpu"] = device.lower().startswith("gpu")
if not use_rec:
    if "use_text_recognition" in params:
        kwargs["use_text_recognition"] = False
    elif "use_recognition" in params:
        kwargs["use_recognition"] = False

print("PaddleOCR kwargs:", kwargs)
ocr = PaddleOCR(**kwargs)
result = ocr.predict(image_path)
for res in result:
    if hasattr(res, "print"):
        res.print()
    else:
        print(res)
    if hasattr(res, "save_to_img"):
        res.save_to_img(str(output_dir))
    if hasattr(res, "save_to_json"):
        res.save_to_json(str(output_dir))
print("VALIDATION_OUTPUT_DIR=" + str(output_dir))
'''


def latest_image(output_dir):
    """在 output_dir 中找最新的图片，供 UI 预览"""
    candidates = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        candidates.extend(Path(output_dir).glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    main()
