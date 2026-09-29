"""复现指定网络来源的筛选；只有 --apply 才写入项目并删除指定来源目录。

main 先在固定临时目录准备筛选结果，再按 --apply 决定是否应用。
write_json/write_jsonl 保存记录；此为已完成的一次性历史操作，不应重复执行。
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from project_paths import display_path, load_paths  # noqa: E402


def write_json(path, obj):
    """保存来源清单或整理统计。"""
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rs):
    """按行保存逐图来源与筛选记录。"""
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))


def main():
    """先在临时目录准备筛选结果；--apply 时才应用到项目目录。"""
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    root = a.project
    paths = load_paths(project=root)
    src = paths["web_source"]
    dst = paths["processed_web"]
    stage = paths["web_staging"]
    stage.mkdir(exist_ok=False)
    ext = stage / "external_metadata"
    out = stage / "processed"
    ext.mkdir()
    out.mkdir()
    removed_ids = {"PMC3319939", "PMC8792127", "PMC11409352"}
    rows = [json.loads(x) for x in (src / "manifest.jsonl").read_text().splitlines()]
    removed = [r for r in rows if r["source_id"] in removed_ids]
    kept = [r for r in rows if r["source_id"] not in removed_ids]
    assert len(removed) == 6 and len(kept) == 103
    assert {r["source_id"] for r in kept} == {"PMC10047450", "zenodo_7272660"}
    assert not dst.exists(), "Refusing to overwrite processed output"

    for sid in removed_ids:
        folder = src / "positive" / sid
        expected = {src / r["path"] for r in removed if r["source_id"] == sid}
        actual = {f for f in folder.rglob("*") if f.is_file() and f.name != ".DS_Store"}
        assert actual == expected, f"Unexpected directory contents: {sid}"

    for r in rows:
        assert hashlib.sha256((src / r["path"]).read_bytes()).hexdigest() == r["sha256"]

    records = []
    pixelhashes = set()

    for original in kept:
        r = dict(original)
        normal = r["source_id"] == "zenodo_7272660"
        rel = Path("normal_source/images" if normal else "images") / (r["sha256"] + ".jpg")
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / r["path"], target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == r["sha256"]
        with Image.open(target) as im:
            im.load()
            assert list(im.size) == [r["width"], r["height"]]
            rgb = im.convert("RGB")
            ph = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            assert ph == r["pixel_sha256"] and ph not in pixelhashes
            pixelhashes.add(ph)
        r["external_path"] = display_path(src / original["path"], root)
        r["path"] = str(rel)
        r["selection_basis"] = (
            "User retained PMC10047450; three other positive sources excluded for arrow annotations"
            if not normal
            else "Source Normal category; HE-specific exclusion remains unverified"
        )
        r["model_input"] = "image only; do not pass caption or diagnosis metadata"
        records.append(r)

    positive = [r for r in records if r["strict_he_label"] == 1]
    normal = [r for r in records if r["source_class"] == "Normal"]
    assert len(positive) == 3 and len(normal) == 100
    (out / "labels.txt").write_text("".join(f"{r['path']} 1\n" for r in positive))
    (out / "labels_normal_source.txt").write_text("".join(f"{r['path']} 0\n" for r in normal))
    write_jsonl(out / "provenance.jsonl", records)
    write_json(
        out / "class_mapping.json",
        {
            "binary_definition": {"0": "non-HE", "1": "HE"},
            "labels.txt": "Only 3 confirmed source-diagnosis HE-positive figures",
            "labels_normal_source.txt": "100 source Normal proxy labels, NOT individually verified HE-negative",
            "normal_strict_he_label": None,
        },
    )
    summary = {
        "image_files": 103,
        "positive_publication_figures": 3,
        "normal_source_images": 100,
        "strictly_verified_he_negative_images": 0,
        "excluded_arrow_marked_figures": 6,
        "within_batch_exact_pixel_duplicates": 0,
        "all_images_decoded": True,
        "all_copies_sha256_verified": True,
        "image_transformations": [],
        "train_test_split_created": False,
    }
    write_json(out / "summary.json", summary)
    active_sources = [
        s
        for s in json.loads((src / "sources.json").read_text())
        if s["source_id"] not in removed_ids
    ]
    write_json(out / "sources.json", active_sources)
    (out / "README.md").write_text("""# 网络肝脏超声数据：箭头筛选后整理版

本次仅整理 external/web_liver_ultrasound_20260918，本目录不覆盖其他已处理的数据集。

- `images/`：3 个 PMC10047450 肝包虫病阳性图文件，沿用用户指定的保留范围。包含 CE1、CE2、CE3b；多面板图不拆分，不按面板虚增样本数。
- `labels.txt`：仅以上 3 个文件，每行 `images/文件名 1`，0=非包虫病，1=包虫病。
- `normal_source/images/`：100 张 Zenodo Normal 类原图。
- `labels_normal_source.txt`：这 100 张图片的来源正常标签 0。公开资料未提供逐人包虫病专项排除记录，因此不自动加入主 labels.txt；provenance 中 strict_he_label 为 null。
- `provenance.jsonl`：103 条逐图记录，包含原路径、图注、标签依据、尺寸、SHA256、像素哈希与来源分组。
- `sources.json`、`class_mapping.json`、`summary.json`：来源及许可、标签规则和校验统计。

## 筛选与质量

按用户要求从 external/positive 删除 PMC3319939、PMC8792127、PMC11409352 三个目录，共 6 个带箭头提示的图文件；未用修图、遮挡、生成式填补来抹除箭头。删除记录保存在 external/excluded_records.jsonl，原始来源网页证据继续保留，但被删除图片不进入处理目录。

103 个输出文件均与保留的输入逐字节一致，未缩放、裁剪、重新压缩、增强或拆分；全部可解码，内部无完全像素重复。沿用此前与 161 张既有处理图像的精确重复核查结果（无重复），未重新进行近重复检测。

去掉这三个来源只解决用户指出的箭头问题，不代表整个数据集已经通过盲法或临床质量审查。保留阳性图仍是出版图，含拼图和文字等来源线索；正常组是另一来源的原 JPG，两组风格不同，不应据此直接宣称模型学到了疾病特征。图注、疾病分期、标签文件和来源元数据不得作为模型诊断输入。

未创建 train/val/test。按来源/患者分组再设计实验；正常图患者标识未知，100 张图不等于 100 人。出版图质量不等于设备原始 DICOM。此处“入选”是数据筛选结论，不是新的法律授权。

PMC10047450 许可 CC BY 4.0。Zenodo 元数据标 CC BY 4.0，说明另要求联系作者并引用论文和网址；详见 sources.json。原正常 ZIP 和分割标注仍存于 external，本次只整理分类所需图像。
""")
    write_jsonl(ext / "manifest.jsonl", kept)

    for r in removed:
        r["exclusion_reason"] = "User requested removal of source folder due to arrow cues"
        r["image_deleted"] = True

    write_jsonl(ext / "excluded_records.jsonl", removed)
    write_json(ext / "sources.json", active_sources)
    write_json(
        ext / "excluded_sources.json",
        [
            s
            for s in json.loads((src / "sources.json").read_text())
            if s["source_id"] in removed_ids
        ],
    )
    old = json.loads((src / "summary.json").read_text())
    old.update(
        image_files=103,
        positive_publication_figures=3,
        excluded_arrow_marked_figures=6,
        active_positive_sources=["PMC10047450"],
    )
    write_json(ext / "summary.json", old)
    (ext / "labels_he_positive.txt").write_text(
        "# path\tlabel\n" + "".join(f"{r['path']}\t1\n" for r in kept if r["strict_he_label"] == 1)
    )
    (ext / "README.md").write_text("""# 网络补充肝脏超声数据（2026-09-18，箭头筛选后）

当前保留 3 个 PMC10047450 肝包虫病超声出版图文件，以及 100 张 Zenodo Normal 类原始 JPG。

用户指出其他三个阳性来源有箭头标记，已按要求删除 positive/PMC3319939、positive/PMC8792127、positive/PMC11409352，共 6 个图文件。排除记录见 excluded_records.jsonl 和 excluded_sources.json；网页证据保留用于追溯，但不属于模型输入。

positive/PMC10047450 对应包虫病标签 1。normal/zenodo_7272660 是来源 Normal 类，source_binary_label 为 0，但未提供逐人包虫病排除证据，strict_he_label 和 label 为 null。两组标签单独保存，不自动组成已经临床验证的二分类金标准。

manifest.jsonl 是当前 103 个图文件的有效清单；sources.json 是当前有效来源。summary.json 保存数量与校验统计。evidence/ 保留下载时的来源页面/元数据，可能包含已排除图片的链接，不作为图像清单。

保留图像未裁剪、缩放、重压缩或修图。3 个阳性出版图可能包含多个面板，不是原始设备 DICOM，文件数不是患者数。Normal.zip 的原始 MD5 已与 Zenodo 核对，100 张 JPG 和 200 份分割 JSON 保持原样；分割标注不是疾病诊断标签。

有效来源：
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10047450/ — CC BY 4.0。
- https://zenodo.org/records/7272660 — 元数据为 CC BY 4.0，说明另要求联系作者并引用论文及数据网址；未代为联系。

处理后图像及标签位于 ../../processed/web_liver_ultrasound_20260918。主 labels.txt 仅含 3 个阳性文件，100 个正常来源标签单独保存。去掉箭头来源不等于消除了拼图、文字、设备和来源差异等全部捷径线索。未划分训练与测试集。
""")

    # 此开关会写入项目并删除三个指定来源目录；默认仅准备临时结果。
    if a.apply:
        shutil.copytree(out, dst)
        for f in ext.iterdir():
            shutil.copy2(f, src / f.name)
        for sid in sorted(removed_ids):
            shutil.rmtree(src / "positive" / sid)
        scripts = root / "scripts/data"
        scripts.mkdir(parents=True, exist_ok=True)
        if Path(__file__).resolve() != (scripts / "prepare_web_us_filtered.py").resolve():
            shutil.copy2(Path(__file__), scripts / "prepare_web_us_filtered.py")
        assert {d.name for d in (src / "positive").iterdir() if d.is_dir()} == {"PMC10047450"}
        for r in records:
            assert hashlib.sha256((dst / r["path"]).read_bytes()).hexdigest() == r["sha256"]

    print(
        json.dumps(
            {"applied": a.apply, "output": str(dst), "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
