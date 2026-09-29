"""核对 PDF 与提取图片，仅整理符合既定规则的肝脏超声及诊断标签。

sha256_bytes 核原字节，pixel_hash 核解码像素，main 复算纳入规则并输出溯源。
这是既定 PDF 批次的准备脚本，不在推理中运行；需要额外的 pypdf 依赖。
"""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from project_paths import display_path, load_paths  # noqa: E402


def sha256_bytes(data):
    """计算文件字节哈希，用于核对原始提取内容。"""
    return hashlib.sha256(data).hexdigest()


def pixel_hash(path):
    """按图像尺寸与解码后的 RGB 像素计算哈希，不修改输入文件。"""
    with Image.open(path) as im:
        im.load()
        rgb = im.convert("RGB")
        return sha256_bytes(str(rgb.size).encode() + rgb.tobytes())


def main():
    """校验提取记录与 PDF 内容，重算筛选规则并输出图片、标签及溯源信息。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    project, out = args.project.resolve(), args.output.resolve()
    paths = load_paths(project=project)
    raw = paths["bioengineering_source"]
    extracted = paths["bioengineering_extracted"]
    if out.exists():
        raise FileExistsError(out)
    pdfpath = raw / "Dataset2_US.pdf"
    old_summary = json.loads((extracted / "summary.json").read_text())
    assert sha256_bytes(pdfpath.read_bytes()) == old_summary["source_sha256"]
    with (extracted / "labels.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len({r["image_id"] for r in rows}) == len(rows)
    reader = PdfReader(pdfpath)
    page_images = {}
    groups = defaultdict(list)
    for r in rows:
        f = extracted / r["native_path"]
        assert sha256_bytes(f.read_bytes()) == r["file_sha256"], r["image_id"]
        assert pixel_hash(f) == r["pixel_sha256"], r["image_id"]
        page = int(r["source_page"]) - 1
        if page not in page_images:
            page_images[page] = {
                im.indirect_reference.idnum: im for im in reader.pages[page].images
            }
        im = page_images[page][int(r["pdf_object_id"])]
        obj = im.indirect_reference.get_object()
        if r["extraction_method"] == "original_DCT_stream_no_reencoding":
            assert sha256_bytes(obj.get_data()) == r["file_sha256"], r["image_id"]
        else:
            assert sha256_bytes(im.data) == r["file_sha256"], r["image_id"]
        if r["imagefolder_ce5_path"]:
            assert pixel_hash(extracted / r["imagefolder_ce5_path"]) == r["pixel_sha256"]
        groups[r["pixel_sha256"]].append(r)
    # 重新按模态、器官、诊断及重复关系决定纳入，不能直接相信提取表的标记。
    prepared = []
    for r in rows:
        reasons = []
        if r["modality"] != "US" or r["organ"] != "liver":
            reasons.append("not_liver_ultrasound")
        if r["label_original"] not in {"CE1", "CE2", "CE3", "CE3a", "CE3b", "CE4", "CE5"}:
            reasons.append("no_explicit_CE_label")
        peers = groups[r["pixel_sha256"]]
        if len(peers) > 1:
            reasons.append(
                "duplicate_image_conflicting_labels"
                if len({p["label_original"] for p in peers}) > 1
                else "duplicate_image_requires_review"
            )
        included = not reasons
        assert included == (r["include_in_filtered"] == "True"), r["image_id"]
        # CE3a/b 可合并为五类分期 CE3；活动性不决定二分类，CE4/5 仍是患病。
        ce5 = "CE3" if r["label_original"].startswith("CE3") else r["label_original"]
        assert ce5 == r["label_ce5"]
        stage = {
            "CE1": "active",
            "CE2": "active",
            "CE3": "transitional",
            "CE4": "inactive",
            "CE5": "inactive",
        }.get(ce5, "")
        assert stage == r["stage3_paper_convention"]
        rec = dict(r)
        rec.update(
            image=("images/" if included else "review/images/") + Path(r["native_path"]).name,
            label=1 if included else None,
            included=included,
            review_reasons=reasons,
            source_pdf=display_path(pdfpath, project),
            source_extracted_image=display_path(extracted / r["native_path"], project),
            ce5_index={"CE1": 0, "CE2": 1, "CE3": 2, "CE4": 3, "CE5": 4}.get(ce5),
            patient_key_scope="PDF province + local patient number, not verified globally unique patient identity",
        )
        prepared.append(rec)
    clean = [r for r in prepared if r["included"]]
    review = [r for r in prepared if not r["included"]]
    # 与已有 HE 图片比较解码像素，寻找跨来源重复；此处只记录，不合并数据。
    he_index = defaultdict(list)
    he_dir = paths["processed_he"] / "images"
    if he_dir.exists():
        for f in he_dir.iterdir():
            if f.is_file():
                he_index[pixel_hash(f)].append(display_path(f, project))
    overlaps = [
        {"image_id": r["image_id"], "matches": he_index[r["pixel_sha256"]]}
        for r in prepared
        if r["pixel_sha256"] in he_index
    ]
    sources = []
    for p in sorted(raw.glob("*.pdf")):
        sources.append(
            {
                "path": display_path(p, project),
                "sha256": sha256_bytes(p.read_bytes()),
                "pages": len(PdfReader(p).pages),
                "action": "validated_US_extraction"
                if p.name == "Dataset2_US.pdf"
                else "excluded_from_ultrasound_task",
                "reason": ""
                if p.name == "Dataset2_US.pdf"
                else "MRI supplementary material; retained in external",
            }
        )
    out.mkdir(parents=True)
    for r in prepared:
        dest = out / r["image"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / r["source_extracted_image"], dest)
        assert sha256_bytes(dest.read_bytes()) == r["file_sha256"]
    (out / "labels.txt").write_text("".join(f"{r['image']} 1\n" for r in clean))
    (out / "labels_ce5.txt").write_text("".join(f"{r['image']} {r['ce5_index']}\n" for r in clean))
    (out / "provenance.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in prepared)
    )
    (out / "review/records.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review)
    )
    mapping = json.loads((extracted / "class_mapping.json").read_text())
    mapping["binary_class_to_idx"] = {"non_echinococcosis": 0, "echinococcosis": 1}
    mapping["binary_note"] = "No negative images in this export; non-active CE remains positive."
    (out / "class_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n"
    )
    summary = {
        "sources": sources,
        "extracted_source_directory": display_path(extracted, project),
        "pdf_and_extracted_are_same_dataset": True,
        "source_records": len(rows),
        "source_modalities": dict(Counter(r["modality"] for r in rows)),
        "source_unique_pixel_images": len(groups),
        "included_images": len(clean),
        "binary_counts": {"0": 0, "1": len(clean)},
        "ce5_counts": dict(Counter(r["label_ce5"] for r in clean)),
        "stage3_counts": dict(Counter(r["stage3_paper_convention"] for r in clean)),
        "review_records": len(review),
        "review_unique_pixel_images": len({r["pixel_sha256"] for r in review}),
        "review_reason_counts": dict(
            Counter(reason for r in review for reason in r["review_reasons"])
        ),
        "cross_HE_DCNN_exact_pixel_matches": overlaps,
        "cross_HE_DCNN_check_performed": he_dir.exists(),
        "near_duplicate_review_completed": False,
        "deidentification_completed": False,
        "visual_review_this_run_completed": False,
        "clinical_label_review_completed": False,
        "training_split_created": False,
        "verification": "PDF hash, all embedded image bytes, decoded pixels, ImageFolder copies, recomputed inclusion rules and output copies checked",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (out / "README.md").write_text("""# Bioengineering 肝包虫病超声整理版

## 使用入口

- images/：80 张有明确 CE 图注的肝脏超声，保留原生 JPG/PNG 字节与尺寸。
- labels.txt：每行 `images/文件名 1`。二分类定义为 0=非包虫病，1=包虫病；本数据没有阴性。
- labels_ce5.txt：可选分期标签，CE1=0、CE2=1、CE3=2、CE4=3、CE5=4。不要将此文件作为二分类标签。
- provenance.jsonl：全部 88 条原始图像记录、图注、页码、原始 CE 标签、文档患者键、原路径和哈希。
- review/images/ 与 review/records.jsonl：8 条暂不纳入的图像记录及原因。
- class_mapping.json：二分类、五分类及论文三组定义。
- summary.json：校验结果、来源指纹、数量与跨 HE_DCNN 精确像素重复检查。

## 数据来源与筛选

两个输入目录是同一数据的 PDF 和提取版，不能重复计数。本次从当前 PDF 重新提取并核对全部 88 条 CSV 记录，原文件字节与图注全部对应。处理脚本还核查 PDF 内嵌图像对象、文件哈希及 RGB 像素哈希。
Dataset2_US.pdf 共 45 页、87 个 Figure、88 次图片出现：US 85 条、肺部 CT 2 条、胸片 1 条。Dataset3_MRI.pdf 共 5 页，不纳入超声任务，保留在 external。

采用保守筛选：必须是肝脏超声、有明确 CE 图注且无精确重复/分期冲突。80 张入选图片中 CE1=6、CE2=20、CE3=24、CE4=18、CE5=12，全部映射为包虫病阳性。CE3a/CE3b 在五分类中归 CE3，原分期保留。论文三组为 active=26、transitional=24、inactive=30，仅沿用论文分组，不重新判断生物活性。

## 待复核条目

- fig_018、fig_019：肝脏超声，但图注没有明确 CE 分期。本轮沿用既有保守筛选，不擅自补充标签。缺失分期不等于阴性，也不表示已证实不是包虫病。
- fig_008、fig_042、fig_076：同一 PDF 图像对象、相同像素，却标为 CE2/CE3a/CE2，并对应不同文档患者键。保留三条来源用于审核。分期冲突并非二分类阴阳性冲突，但患者关联和分期不清，因此全部暂缓纳入。
- fig_025_part_01、fig_025_part_02：肺部 CT；fig_085：胸片。不得将这些标成超声阴性。

## 实验限制

这是完成来源一致性检查的整理版，不是临床复核完成的数据集。已有源 README 记录 fig_013 的色彩异常、fig_005/058 和 fig_036/041 的疑似近重复，以及图内 CE 文字和身份信息；本轮保留原图并未改像素，未完成新增视觉核查、近重复排查或去标识。后续应优先审核这些问题。
patient_key 仅是 PDF 省份与局部编号组合，不保证真实患者独立。不要随机按图划分后宣称独立验证。没有创建 train/val/test，也没有扩增数据。图注只用于标签与溯源，不能作为模型诊断输入；pdf_bbox_json 是页面坐标，不是病灶框。
本目录没有与 HE_DCNN 合并；二者可按 labels.txt 接口读取，但跨来源训练评估需检查设备/来源差异。本次只核查精确像素重复，不能排除不同裁剪或缩放版本。

## 复现

在项目根目录使用含 Pillow、pypdf 的 Python 环境：

```sh
python scripts/data/prepare_bioengineering_us.py --project . --output data/processed/bioengineering_US_binary_new
```

脚本拒绝覆盖已有输出。原始数据与原 HE_DCNN 整理结果均保持不变。许可和使用范围继承原论文及数据来源，不因本次整理新增授权。
""")
    assert len({r["pixel_sha256"] for r in clean}) == len(clean)
    assert len(clean) + len(review) == len(rows)
    assert all((out / r["image"]).exists() for r in prepared)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
