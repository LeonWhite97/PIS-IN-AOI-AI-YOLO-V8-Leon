# 公开外部候选审查工作台

本目录是 Baining FC-BGA YOLO PoC 公开外部语料（CC BY 4.0）的人工审查工作台。

## 文件导航

| 文件 | 作用 |
|------|------|
| `candidates.jsonl` | 110 张候选主清单（每条含 `review_status` 与 `accepted_classes`） |
| `candidates.enriched.json` | 同一清单的富化版（附带 `_w`/`_h` 尺寸，供联系表复用） |
| `images/` | 110 张缩略原图（`public-{hash16}.jpg`，按内容去重命名） |
| `contact_sheet.html` | 可视化联系表：110 张缩略图 + 七类图例，审查时对照用 |
| `ANNOTATION_SPEC.md` | **七类缺陷标注规范**：定义、视觉判据、决策流程、落盘字段 |
| `predefined_classes.txt` | LabelImg 类下拉顺序文件（7 类合同顺序 0-6，防索引错位） |
| `labels/` | 已接受样本的 YOLO 标签（由 `apply` 落盘，git 跟踪） |
| `../sources.json` `../ATTRIBUTION.md` | 来源登记与署名（CC BY 4.0 合规） |

## 审查三步

1. 打开 `contact_sheet.html` 浏览全部候选；
2. 按 `ANNOTATION_SPEC.md` 判定每张图的 `review_status` 与 `accepted_classes`；
   - 可判定的图：在 `data/to_annotate/` 里用 LabelImg 画框（YOLO 格式）→
     `make apply LABEL_DIR=data/to_annotate` 落盘；
   - 拿不准的图：`make quarantine IDS="public-xxxxxxxx"` 隔离（默认 `DEFECT_UNCLEAR`）；
3. 跑进度追踪器确认 B0 门控状态：

```bash
python tools/vision/fc_bga_yolo/review_progress.py --json
```

## 目标（B0 门控）

- 已接受图像 **≥ 20** 张；
- 覆盖缺陷类 **≥ 2** 个；
- 解锁后可发布 `public-external-v0.1`。

详见父目录 `../README.md` 的 gate 状态说明。
