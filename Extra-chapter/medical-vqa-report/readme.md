# 医疗影像 VQA 与报告生成：用消费级显卡跑通垂直领域 VLM 最小闭环

> **⚠️ 免责声明：本项目仅用于教学与技术研究，输出内容不构成医学建议，禁止用于任何临床用途。**
> 模型可能生成不准确甚至错误的描述，所有示例结果请仅作为技术演示看待。

> 阅读建议：适合读完主线 [chapter20（BLIP-2 与 LLaVA）](../../docs/chapter20/01_blip2_llava.md) 与 [chapter11（LoRA 实战）](../../docs/chapter11/03_peft_lora.md) 后阅读。配套代码见 `code/`，结果文件见 `results/`。

## 背景与动机

主线教程里我们知道了 VLM（视觉语言模型）怎么读图、LoRA 怎么省显存微调。但在真实的垂直领域（医疗是典型代表），通用 VLM 直接用效果如何？领域微调能带来多少增益？这两个问题教程没有回答。

本专题用 **公开的医疗影像问答数据集 VQA-RAD** 和 **公开胸片报告数据集 IU-Xray**，在一块 6GB 显存的消费级显卡（RTX 4050 Laptop）上，完整跑通「通用 VLM zero-shot → 量化评估 → QLoRA 领域微调 → 增益对比」的最小闭环。全程可复现，单卡可跑。

## 场景与问题描述

- **VQA（视觉问答）**：给一张影像（X 光/CT 切片）和一个自然语言问题，模型输出简短答案。VQA-RAD 里的问题分两类：closed（is/are/do 开头，答案基本是 yes/no）与 open（如 "what organ is abnormal?"）。
- **报告生成**：给一张/一对胸片，模型输出一段规范的放射学报告（findings 风格），与真实报告做 BLEU-4 / ROUGE-L 对比。

## 数据来源与合规说明

| 数据集 | 来源 | 授权/条款 | 本专题用法 |
|---|---|---|---|
| VQA-RAD | Lau et al., 2018，经 [HuggingFace `flaviagiammarino/vqa-rad`](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) 分发 | 原始数据来自公开 HeadXNet 项目（Stanford AIMI），仅用于研究 | VQA 训练/评估 |
| IU-Xray | Demner-Fushman et al., 2016（Open-i），经 [HuggingFace `dz-osamu/IU-Xray`](https://huggingface.co/datasets/dz-osamu/IU-Xray) 分发 | Open-i 公开数据，使用时需遵守其条款并引用原始论文 | 报告生成评估 |

**合规约定**：
1. 本专题**不随仓库分发任何医疗影像或报告原文**（`data/` 只提供下载脚本）；请读者自行从上述来源获取，并遵守其原始条款；
2. 所有示例图片均已在文档中脱敏说明，报告样例仅用于展示模型输出格式；
3. 再次强调：**输出不构成医学建议，非临床用途**。

## 技术方案

```
                    ┌────────────────────────────┐
 VQA-RAD 图+问题 ──▶ │  Qwen2.5-VL-3B-Instruct    │ ──▶ 答案（yes/no 或短语）
                    │  4bit NF4 量化（bitsandbytes）│
 IU-Xray 胸片 ────▶ │                            │ ──▶ findings 风格报告
                    └────────────────────────────┘
                            │
                    QLoRA（r=16, NF4, 仅 LLM 投影层）
                            │
                    领域微调后 VQA/报告链路复评，对比增益
```

- 量化：NF4 + double quant，计算精度 bf16；
- 微调：仅调 LLM 部分 q/k/v/o + gate/up/down 投影，视觉塔冻结；梯度检查点 + 梯度累积；
- 环境：Python 3.11，torch (CUDA 12.6)，transformers，peft，trl 未用（自写训练循环），bitsandbytes（Windows 原生支持）。

## 实践步骤

完整代码在 [`code/`](./code/)，建议按顺序：

1. **准备数据**：`python code/download_data.py --all`（下载 VQA-RAD 与 IU-Xray 到本地，不进仓库）
2. **zero-shot VQA 评估**：`python code/vqa_inference.py --max-samples 100`
3. **zero-shot 报告生成评估**：`python code/report_inference.py --max-samples 30`
4. **QLoRA 微调**：`python code/finetune_qlora.py --epochs 1 --max-samples 800`
5. **微调后复评**：加 `--adapter adapters/vqa-rad-lora` 重跑 2/3

## 结果

### VQA（VQA-RAD test 全量 451 条，越界匹配评测）

| 指标 | zero-shot | QLoRA 微调后 | 增益 |
|---|---|---|---|
| Closed-ended 准确率（269 条） | 67.66% | 69.52% | +1.86pp |
| Open-ended 准确率（182 条） | 25.82% | **36.26%** | **+10.44pp（相对 +40%）** |
| 总体准确率 | 50.78% | 56.10% | +5.32pp |

- 微调配置：Qwen2.5-VL-3B-Instruct 4bit NF4，LoRA r=16 / alpha=32，仅调 LLM 侧 q/k/v/o + gate/up/down 投影（视觉塔冻结），800 条训练样本 × 1 epoch，batch 1 × 梯度累积 16，lr 1e-4；训练 loss 3.91 → 0.68。
- **主要增益来自 open-ended**：yes/no 类问题 zero-shot 已接近模型能力上限，而「医疗术语 + 回答风格」这类开放式能力正是领域微调最见效的部分。

### 报告生成（IU-Xray test 前 30 例，2 图输入）

| 指标 | zero-shot |
|---|---|
| BLEU-4 | 0.0120 |
| ROUGE-L (F1) | 0.1699 |

zero-shot 3B 模型与放射科参考报告的措辞重叠很低，这与文献中通用 VLM zero-shot 报告生成的典型水平一致；受 6GB 显存限制，本专题未做报告生成的微调对比（VQA 链路已能说明微调增益），这留给显存更充裕的读者作为练习。

> ⚠️ 再次声明：以上所有数字仅说明模型行为，**不构成任何医学有效性证据，禁止临床使用**。

## 经验总结与坑点

1. **6GB 显存能跑多模态 QLoRA**：3B VLM + NF4 + 梯度检查点 + batch 1 累积 16，峰值显存 5.8GB，刚好塞下；再大一点（如 7B）就需要换方案了。
2. **transformers 5.x 的类名没变**：`Qwen2_5_VLForConditionalGeneration` 在 5.x 依然可用，但 `min_pixels/max_pixels` 传参已废弃，新写法是 `size={'longest_edge': ..., 'shortest_edge': ...}`。
3. **Windows 的 bitsandbytes 原生可用**（0.50+），不需要 WSL；但下载要自备镜像（`HF_ENDPOINT=https://hf-mirror.com`）。
4. **labels 掩码两个易漏点**：除了 `pad_token_id`，**视觉占位符 token（`image_token_id`）也必须置 -100**，否则 loss 会被图像 token 拉偏。
5. **VQA-RAD 的 open-ended 评测口径**：论文常用含别名的匹配，我们用「规范化后互为子串」的宽松匹配，数字会略高于严格口径，横向对比时注意对齐。
6. **hf-mirror 单线程下载慢**（~0.7MB/s），大模型文件用 `aria2c -x 16 -s 16` 能到 ~30MB/s。

## 参考资料

- Lau et al., "A Dataset of Clinically Realistic Visual Question Answering for Medical Imaging", EMNLP 2018（VQA-RAD）
- Demner-Fushman et al., "Creating a Public Annotated Dataset of Chest X-rays...", JAMIA 2016（IU-Xray）
- Qwen2.5-VL Technical Report
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023
