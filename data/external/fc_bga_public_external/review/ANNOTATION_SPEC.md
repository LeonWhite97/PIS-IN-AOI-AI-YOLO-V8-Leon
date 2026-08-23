# 七类缺陷标注规范（公开外部候选审查）

> 用途：指导人工审查员对 `review/candidates.jsonl` 中的 110 张公开外部候选图，
> 判定其**可接受性**以及它**示范了哪几类缺陷**，从而解锁 B0 门控
> （≥20 张已接受图、≥2 个被覆盖的缺陷类）。
>
> 配套工具：
> - `review/contact_sheet.html` — 110 张缩略图联系表（懒加载，含七类图例）
> - `tools/vision/fc_bga_yolo/review_progress.py` — 进度 / B0 门控追踪器
> - `review/candidates.jsonl` — 候选清单（每条含 `review_status` 与 `accepted_classes`）

---

## 0. 这份标注在做什么（先对齐认知）

公开外部候选来自 Roboflow 的 CC BY 4.0 数据集（BGA RAM Chips 56 张、BGA-Balls 54 张），
它们**不是**本项目正式的四光源（R/G/B/RING）FC-BGA 实拍。
因此本阶段标注的目的不是"给实拍打边界框"，而是：

1. 判断这张公开图**能否**作为某一类缺陷的**视觉示范样本**进入公开外部语料；
2. 若可以，标记它清晰示范了七类缺陷中的**哪几类**（整图多标签）；
3. 给出 `accepted` / `quarantined` 决定，供 B0 门控统计。

**边界框（bounding box）标注是后续 B1/B2 阶段的事**，不属于 B0。
B0 只要求"已接受图像 + 类覆盖度"，不要求框。

---

## 1. 输入契约提醒（为什么公开图不能直接当训练输入）

正式训练输入契约为 `rgb_grayscale_stack_v1`：将 R、G、B 三个灰度通道沿通道维堆叠，
**RING 通道仅作为证据/参考，不参与模型输入**。

公开候选多为 300×300 或 640×640 的普通 RGB/BGR 图，**不满足该契约**。
所以：

- 标注阶段只评估"缺陷形态是否可见、是否典型"，不评估通道格式；
- 进入正式训练前，必须通过 `prepare_public_external_candidates.py` 做
  格式归一化与（如需要）通道重组；
- 若某图肉眼就无法分辨焊球阵列（过曝、失焦、裁切过紧），直接 `quarantined`，
  不要强行归类。

---

## 2. 七类缺陷定义与视觉判据

判定优先级：先看"有没有球/球阵是否完整"，再看"单球几何/位置异常"，
最后看"异物/桥连"。拿不准一律 `quarantined`。

### 2.1 `MISSING_BALL`（缺球）
- **定义**：阵列中应有焊球的位置为空缺、露铜或仅有焊盘而无凸起球体。
- **视觉判据**：规则网格上出现明显"洞"——周围球均匀，唯独某格缺失；
  或整行/整列缺球。
- **典型公开图**：BGA-Balls 中故意挖掉若干球的合成图。
- **注意**：不要和 `BALL_OFFSET`（球在但偏了）混淆——缺球是"没有"，偏移是"有但歪"。

### 2.2 `EXTRA_BALL`（多球 / 多余球）
- **定义**：网格之外或两格之间出现多余球体、散落焊料球。
- **视觉判据**：网格规则边界外多出孤立球；或相邻球之间夹着一颗不该有的小球。
- **典型公开图**：BGA-Balls 中在空隙处人为加上额外球的图。
- **注意**：区分"阵列边缘正常末排球"与"溢出网格的散球"。

### 2.3 `BALL_BRIDGE`（桥连）
- **定义**：相邻两球被焊料连成一体，失去各自独立轮廓。
- **视觉判据**：两球之间出现连续的金属连接带，轮廓 merged；
  在灰度图上呈现"两座山峰被同一底座连起"。
- **典型公开图**：BGA-Balls 中相邻球接触/融合的合成图。
- **注意**：和 `BALL_SIZE_ABNORMAL`（单球过大但独立）区分——桥连的关键是**跨球连通**。

### 2.4 `BALL_SIZE_ABNORMAL`（球体尺寸异常）
- **定义**：单球直径显著大于或小于同阵列其他球（非桥连、非缺球）。
- **视觉判据**：在均匀阵列中某颗球明显偏小（缩球）或偏大（鼓包），但轮廓独立。
- **注意**：需与邻球做相对对比，单看一张图无法判定"尺寸异常"时标 `quarantined`。

### 2.5 `BALL_OFFSET`（球偏移）
- **定义**：焊球相对于其理论焊盘中心发生平移，未落在格点正中。
- **视觉判据**：球体中心偏离所在网格格点；阵列整体看呈"错位棋盘"感。
- **注意**：和 `MISSING_BALL` 区分——偏移是"球在但歪"，缺球是"没有"。

### 2.6 `BALL_SHAPE_ABNORMAL`（球形状异常）
- **定义**：球体轮廓非圆形/椭球，出现塌陷、泪滴、卫星球、不规则凸起等。
- **视觉判据**：轮廓明显偏离光滑圆弧；出现"尾巴""双头""凹陷"。
- **注意**：与 `BALL_SIZE_ABNORMAL` 区分——这里强调**形状**而非**大小**；
  与 `BALL_BRIDGE` 区分——形状异常仍是单球内部形变，不跨球连通。

### 2.7 `FOREIGN_MATERIAL`（异物）
- **定义**：球阵中出现非焊球、非基板的异物（纤维、粉尘、焊渣、指纹、划痕污染物）。
- **视觉判据**：与焊球材质/灰度明显不同的斑点、线条或团块，位置随机。
- **注意**：与 `EXTRA_BALL` 区分——异物**不是球形焊料**，形态杂乱、灰度异质。

---

## 3. 审查决策流程（每条候选必走）

```
读取候选 (sample_id, image_path, source_id)
   │
   ├─ 图像本身不可用？（过曝/失焦/裁切致无法辨阵）
   │     └─→ review_status = "quarantined", quarantine_reason = "UNREADABLE"
   │
   ├─ 能辨阵，但无法确定示范了哪类缺陷？
   │     └─→ "quarantined", quarantine_reason = "DEFECT_UNCLEAR"
   │
   └─ 能明确判定的缺陷类（可多选）→ accepted_classes = (...)

判定可接受的图：
   review_status = "accepted", accepted_classes = 命中的类（≥1，至多 7）
   若 accepted_classes 为空但图质量好、只是"无缺陷良品"：
       仍记 accepted，accepted_classes = ()  （良品也可入库作负样本）
```

**三不原则**：
- 不强行给模糊图归类；
- 不把"良品"误标成缺陷类；
- 不一图多标到失去意义（命中几类写几类，宁少勿错）。

---

## 4. 如何落盘（写入 candidates.jsonl）

每条记录需维护以下字段：

| 字段 | 取值 | 说明 |
|------|------|------|
| `review_status` | `review_required` \| `accepted` \| `quarantined` | 初审后必须改写 |
| `accepted_classes` | 七类名的元组，如 `("MISSING_BALL",)` 或 `()` | 仅 `accepted` 时有意义 |
| `label_path` | 如 `"labels/public-xxxxxxxxxxxxxxxx.txt"` | **accepted 且要训练的样本必须填**；文件放在 `review/labels/` |
| `quarantine_reason` | 字符串，如 `UNREADABLE` / `DEFECT_UNCLEAR` / `LICENSE_AMBIGUOUS` | 仅 `quarantined` 时填写 |

> 字段名与类型以 `tools/vision/fc_bga_yolo/public_external_manifest.py`
> 的 `CandidateRecord` 为准。**不要手改 JSONL**——用
> `tools/vision/fc_bga_yolo/apply_review_labels.py` 把标注工具导出的
> `{sample_id}.txt` 一键落盘（自动校验框、推导 `accepted_classes`、写
> `label_path` 并把标签复制到 `review/labels/`）。手改只用于 `quarantined`
> 这类工具无法表达的状态。

### 4.1 边界框标签文件（B0 训练必需）

B0 门控不仅数图像数，还会校验每个 accepted 样本的 YOLO 标签框
（`evaluate_revision_gate` 逐条 `_validate_label`）。因此：

- **位置**：`review/labels/{sample_id}.txt`（与图像同名，`sample_id` 即 `public-xxxxxxxxxxxxxxxx`）；
- **格式**：每框一行 `class_id cx cy w h`，坐标归一化到 [0,1]；
- **类别索引必须按合同固定顺序**：

| id | 类名 |
|----|------|
| 0 | BALL_BRIDGE |
| 1 | MISSING_BALL |
| 2 | EXTRA_BALL |
| 3 | BALL_SIZE_ABNORMAL |
| 4 | BALL_OFFSET |
| 5 | BALL_SHAPE_ABNORMAL |
| 6 | FOREIGN_MATERIAL |

- 良品（`accepted_classes = ()`）不需要标签文件，`label_path` 留 `null` 即可。

### 4.2 推荐标注工具与就绪工作区

**就绪工作区（推荐）：`data/to_annotate/`**
- 已放入全部 110 张候选图（文件名 = `sample_id`，与 `candidates.jsonl` 逐一对应）；
- 已放置 `predefined_classes.txt`（7 类按 §4.1 合同顺序）。LabelImg 启动时会从
  **当前工作目录**读取该文件，因此请**在 `data/to_annotate/` 目录内启动 LabelImg**
  （或在启动处放一份同文件），下拉框即按序出现 7 类，`class_id` 0-6 与 §4.1 表严格一致。

- **LabelImg**（本地轻量）：`pip install labelImg`；打开 `data/to_annotate/`，
  工具栏把 **PascalVOC 切换为 YOLO**（关键，否则导出 XML），逐张画框 → 选类 → Ctrl+S；
  `.txt` 与图像同目录生成在 `data/to_annotate/` 下。
- **makesense.ai**（零安装网页）：上传 `data/to_annotate/` 里的图，按 §4.1 顺序添加
  7 个标签，导出 YOLO txt 到任意目录，交给下方 apply 落盘。
- **Label Studio**（功能最全）：`pip install label-studio`，建 Object Detection 项目，
  导出选 YOLO。

任选其一；关键是**类别顺序严格一致**、文件名与 `sample_id` 对应、
保存格式是 **YOLO 而非 PascalVOC XML**。

**落盘**（人工标完后由脚本执行；自动校验每个框、推导 `accepted_classes`、写
`review_status=accepted` + `label_path`，并把标签复制到 git 跟踪的 `review/labels/`）：

```bash
make apply LABEL_DIR=data/to_annotate
# 等价于：
# python tools/vision/fc_bga_yolo/apply_review_labels.py --label-dir data/to_annotate
```

先预览零副作用的结果：`make apply LABEL_DIR=data/to_annotate DRY=1`

- **良品（无缺陷）**：保存空 `.txt`（0 字节）即可——校验通过、`accepted_classes=()`，
  可作负样本入库。
- **无法判定的图**：不要保存标签文件，把 `sample_id` 交给 AI 执行隔离落盘：

```bash
# 预览零副作用
make quarantine IDS="public-xxxxxxxxxxxxxxxx public-yyyyyyyyyyyyyyyy" DRY=1
# 正式隔离（默认 DEFECT_UNCLEAR，可换 UNREADABLE / LICENSE_AMBIGUOUS）
make quarantine IDS="public-xxxxxxxxxxxxxxxx"
# 等价于：
# python tools/vision/fc_bga_yolo/quarantine_candidates.py --ids public-xxxxxxxxxxxxxxxx
```

---

## 5. B0 门控回顾（验收标准）

`review_progress.py` 会调用 `public_external_revision.evaluate_revision_gate` 计算 B0：

- 已接受图像数 **≥ 20**；
- 已接受图像覆盖的缺陷类数 **≥ 2**；
- train/val/test 三个子集均非空（B0 阶段由脚本从 accepted 图自动切分）。

当前基线（2026-08-17）：110 张全部 `review_required`，B0 = `blocked_data`，
距解锁还差 **20 张已接受图 + 2 个类别**。

**建议的最低可行路径**：
1. 先扫 contact sheet，挑出最有把握的 ~25 张（覆盖至少 2 类，例如 MISSING_BALL + EXTRA_BALL）；
2. 标为 `accepted` 并填 `accepted_classes`；
3. 跑 `review_progress.py --json` 确认 B0 翻转为解锁；
4. 再逐步扩充其余类。

---

## 6. 公开数据合规提醒

- 全部来源为 **CC BY 4.0**，使用/再分发须保留署名（见 `../ATTRIBUTION.md`）。
- 仅接受**已确认许可**的源；若某图来源存疑，标 `quarantined` 且
  `quarantine_reason = "LICENSE_AMBIGUOUS"`，不要入库。
- 不对公开图做任何声称"真实产线缺陷"的表述——它们是**视觉形态示范**。

---

## 7. 快速命令

```bash
# 文本进度
python tools/vision/fc_bga_yolo/review_progress.py

# 机器可读（CI / 后续脚本消费）
python tools/vision/fc_bga_yolo/review_progress.py --json

# 标注落盘（标完一批后执行；--dry-run 先预览）
python tools/vision/fc_bga_yolo/apply_review_labels.py --label-dir data/to_annotate --dry-run
python tools/vision/fc_bga_yolo/apply_review_labels.py --label-dir data/to_annotate

# 隔离无法判定的图（拿不准的 sample_id）
python tools/vision/fc_bga_yolo/quarantine_candidates.py --ids public-xxxxxxxx --dry-run
python tools/vision/fc_bga_yolo/quarantine_candidates.py --ids public-xxxxxxxx --reason DEFECT_UNCLEAR

# B0 门控检查 / 实体化版本
python tools/vision/fc_bga_yolo/build_b0_version.py            # 检查清单（blocked 时退出码 1）
python tools/vision/fc_bga_yolo/build_b0_version.py --publish  # 门控通过后生成 versions/public-external-v0.1/

# 或全部走单命令入口
bash tools/vision/fc_bga_yolo/review-loop.sh all

# 打开联系表（审查时对照）
# 直接用浏览器打开 data/external/fc_bga_public_external/review/contact_sheet.html
```

---

_本规范服务于 Baining FC-BGA YOLO PoC 的公开外部语料 B0 阶段，
随审查实践迭代更新。_
