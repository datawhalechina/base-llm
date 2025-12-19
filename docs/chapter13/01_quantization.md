# 第一节 模型量化实战

在前面的实战章节中，为了在消费级显卡上加载大模型，我们已经初步体验了量化技术的魔力——只需几行代码配置 `BitsAndBytesConfig`，庞大的模型就能“塞”进显存。但你是否好奇，这背后的 `int8` 或 `nf4` 到底发生了什么？除了微调时用到的 BitsAndBytes，还有哪些量化技术更适合推理部署？

## 一、小资源干大活

### 1.1 冗余与压缩

量化，听起来是一个复杂的数学概念，但实际非常简单，就是**用较少的信息来表示数据，在尽量不损失模型性能的前提下，降低资源开销**。深度学习模型（无论是 CV 还是 NLP 领域）普遍表现出显著的**参数冗余性**。早在 1989 年，Yann LeCun 等人就在论文《Optimal Brain Damage》 [^1]中指出神经网络中存在大量参数可以被删除而不影响准确率；而后续著名的“彩票假设”（The Lottery Ticket Hypothesis） [^2]更是进一步证明，密集网络中包含一个极小的子网络（“中奖彩票”），它的性能可与原始网络媲美。量化技术正是利用这一特性，通过降低非关键参数的数值精度（例如从 FP16 降至 INT4），在大幅减少显存占用和计算量的同时，尽可能保持模型的原始性能。

打个比方，一张原本几十 MB 的高清无损照片（如 RAW 格式），在压缩为几百 KB 的 JPG 格式后，虽然丢失了大量人眼难以察觉的色彩细节（精度降低），但我们依然能清晰地识别出照片中的人物和风景。说明原始数据中包含大量对于“视觉理解”来说非必须的冗余信息。量化的过程也是类似，我们试图找出模型参数中那些对最终输出影响不大的微小精度，将其削减，在大幅降低显存占用的同时，保留模型的核心能力，实现**“瘦身不降智”**。

### 1.2 量化的核心价值

量化技术主要带来两方面的巨大收益：

（1）**降低显存开销**：通常模型以 FP16（16位浮点数）格式存储，若量化为 INT8（8位整数），显存占用直接减半；若进一步量化为 INT4（4位整数），显存占用仅为原来的 1/4。原本需要多张 A100 才能加载的千亿模型，量化后可能只需一张消费级显卡即可运行。

（2）**提升推理速度**：数据量的减少意味着内存带宽（Memory Bandwidth）压力的降低。在 LLM 推理这种典型的“内存受限（Memory-bound）”场景下，更快的权重加载速度直接转化为更快的 Token 生成速度。

## 二、从“装不下”到“跑得动”

在动手实践之前，我们需要学会如何估算一个模型到底需要多少显存。这不是玄学，而是一道简单的算术题。

### 2.1 精度与显存的关系

模型权重通常以浮点数形式存储，不同的精度决定了每个参数占用的字节数：

- **FP32 (Full Precision)**：单精度浮点数，占用 **4 Bytes**。这是深度学习训练的默认精度，但在推理时通常不需要这么高。
- **FP16 / BF16 (Half Precision)**：半精度浮点数，占用 **2 Bytes**。
    - **FP16**：传统的半精度，数值范围较小，容易溢出。
    - **BF16 (BFloat16)**：Google 提出的格式，牺牲了小数位精度以换取与 FP32 相同的数值范围（指数位），训练更稳定，是目前大模型训练的主流选择。
- **INT8**：8 位整数，占用 **1 Byte**。
- **INT4**：4 位整数，占用 **0.5 Byte**（即 4 bit）。

### 2.2 显存估算公式

在计算机存储单位中，1 GB = 1024 MB，1 MB = 1024 KB。但在估算模型参数量（如 7B = 7 Billion）和显存（GB）时，为了方便口算，我们通常近似认为 $1 \text{ GB} \approx 10^9 \text{ Bytes}$。如果追求精确计算，记得除以 $1024^3$。模型所需显存大小的**通用估算公式**如下：

$$ \text{权重显存占用} \approx \text{模型参数量} \times \text{每参数占用字节数} \tag{13.1} $$

以我们之前学习过的 Qwen2.5 为例，这里选择 **Qwen2.5-7B**（约 70 亿参数，即 $7 \times 10^9$）：

（1）**FP16 / BF16 精度（2 Bytes/参数）**：

$$ 7 \times 10^9 \times 2 \text{ Bytes} \approx 14 \text{ GB} \tag{13.2} $$


（2）**INT8 量化（1 Byte/参数）**：

$$ 7 \times 10^9 \times 1 \text{ Byte} \approx 7 \text{ GB} \tag{13.3} $$


（3）**INT4 量化（0.5 Byte/参数）**：

$$ 7 \times 10^9 \times 0.5 \text{ Byte} \approx 3.5 \text{ GB} \tag{13.4} $$

> 这只是模型权重的静态占用。实际运行时，还需要预留显存给：
> - **KV Cache**：上下文缓存，与序列长度（Context Length）成正比，上下文越长，占用越大。
> - **激活值**：中间层计算结果，与 Batch Size 和序列长度相关。
> - **框架开销**：PyTorch / CUDA context 本身会占用一定开销。
>
> 所以，实际显存需求通常比估算值高 **20%~30%**。例如加载 7B 的 INT4 模型（3.5GB 权重），推荐显存至少 6GB 起步。

## 三、Transformers 中的主流集成方案

虽然量化方法层出不穷，但在 Hugging Face `Transformers` 的官方文档与实践中，最常用的三类集成方式是加载 **GPTQ**、**AWQ** 以及 **bitsandbytes（bnb）**。在代码层面，它们通常通过 `AutoModel*.from_pretrained(..., quantization_config=...)` 搭配相应的配置类（如 `GPTQConfig`、`AwqConfig`、`BitsAndBytesConfig`）实现相对统一的调用体验。

如果从使用场景来区分，**GPTQ 和 AWQ** 主要面向**推理部署与加速**，它们属于 Post-Training Quantization (PTQ) 算法，生成的模型通常以量化后的检查点形式保存，加载后显存占用低且推理速度快。**bitsandbytes** 则既常用于 8bit/4bit **推理**，也是诸如 QLoRA 在内的一系列**低显存微调方案**的核心依赖，尤其擅长让大模型在单卡上完成 4-bit 训练。

### 3.1 面向生成式模型的高效量化

**GPTQ (Generative Pre-trained Transformer Quantization)** [^3]是一种面向大规模生成式 Transformer 的**训练后量化（Post-Training Quantization, PTQ）**技术。它是经典的 **OBQ (Optimal Brain Quantization)** 算法在超大模型上的高效进化版，基于**近似二阶信息**实现了一次性权重量化（one-shot weight quantization）。GPTQ 解决了以往简单的“四舍五入”（Round-to-Nearest, RTN）量化在模型参数超过百亿级时会导致严重精度崩塌的问题，成功将 1750 亿参数的超大模型压缩至 3-bit 或 4-bit，且几乎不损失精度。

GPTQ 的量化目标是最小化量化前后激活值的平方误差：

$$ \min_{\widehat{\mathbf{W}}} \| \mathbf{WX} - \widehat{\mathbf{W}}\mathbf{X} \|_2^2 \tag{13.5} $$

其中，$\mathbf{W}$ 为原始权重矩阵，$\mathbf{X}$ 为输入激活值矩阵，$\widehat{\mathbf{W}}$ 为量化后的权重矩阵。

GPTQ 的成功依赖于三个**关键机制**：

（1）**二阶信息补偿**：它利用海森矩阵（Hessian Matrix，$\mathbf{H} = 2\mathbf{X}\mathbf{X}^\top$）的二阶信息来判断权重的重要性。**这就是识别“冗余参数”的重要数学工具**。海森矩阵描述了损失函数曲面的曲率，如果某个权重方向上的曲率很小（平坦），说明该权重的微小变化对总误差影响不大，它是相对“冗余”的；反之则是“关键”参数。GPTQ 利用其逆矩阵 $\mathbf{H}^{-1}$ 来更新剩余权重，以补偿当前权重量化带来的误差 $\delta$：

$$ \boldsymbol{\delta}_F = - \frac{w_q - Q(w_q)}{[\mathbf{H}_F^{-1}]_{qq}} \cdot (\mathbf{H}_F^{-1})_{:, q} \tag{13.6} $$

其中，$w_q$ 是当前被量化的权重，$Q(w_q)$ 是其量化值，$\boldsymbol{\delta}_F$ 是对剩余未量化权重集合 $F$ 的更新向量，$\mathbf{H}_F^{-1}$ 是对应当前未量化权重的海森矩阵逆矩阵。

> 这里给出的只是 GPTQ 在分块/局部参数子集上的近似更新形式，省略了部分实现细节，主要是理解它利用二阶信息做误差补偿的思想。

（2）**任意顺序与延迟批量更新**：GPTQ 发现大模型不需要像 OBQ 那样进行昂贵的“贪心排序”，只需按顺序量化即可。同时，它引入了 **Lazy Batch-Updates（延迟批量更新）** 策略，将计算密集型的更新操作分块执行（如 128 列为一组），极大地提升了 GPU 利用率。

（3）**Cholesky 分解**：为了解决大模型下海森矩阵逆计算的数值不稳定性问题，GPTQ 引入了 Cholesky 分解，确保了算法在千亿参数规模下的稳健运行。

GPTQ 的量化过程是分块进行的。如图 13-1-1，加粗的列块（Block）表示当前正在处理的列。左侧灰色部分是利用 Cholesky 分解预先计算好的逆 Hessian 信息。在处理当前块（橙色部分）时，算法会递归地逐列量化（中间白色列），并将量化误差利用预计算的 Hessian 信息“推”给后续未量化的权重（右侧蓝色部分）进行更新补偿，从而最大程度保留模型精度。

<p align="center">
  <img src="./images/13_1_1.png" width="80%" alt="GPTQ 量化过程示意图" />
  <br />
  <em>图 13-1-1 GPTQ 量化过程示意图：分块更新与 Cholesky 分解的应用</em>
</p>

得益于上述优化，GPTQ 能以极快的速度（如 1750 亿参数仅需 4 小时）完成量化。实验表明，模型规模越大，GPTQ 带来的相对精度损失反而越小。如图 13-1-2，在 OPT 模型家族中，随着参数量增加（横轴向右），传统 RTN 方法（蓝线）的困惑度（PPL）急剧上升，意味着模型“崩了”；而 GPTQ（红线）则紧贴全精度基线（黑虚线），展现了极强的鲁棒性。生成的 INT4 模型配合 ExLlama 等专用内核，推理速度可达 FP16 的 3~4 倍。

<p align="center">
  <img src="./images/13_1_2.png" width="90%" alt="GPTQ 与 RTN 在不同规模模型上的 PPL 对比" />
  <br />
  <em>图 13-1-2 GPTQ 与 RTN 在不同规模模型上的 PPL 对比：模型越大，GPTQ 优势越明显</em>
</p>

### 3.2 激活感知权重量化

**AWQ (Activation-aware Weight Quantization)** [^4] 提出了一种更符合直觉且高效的量化思路，特别适合**端侧**部署。与 GPTQ 依赖复杂的二阶信息进行误差补偿不同，AWQ 另辟蹊径，发现**权重的“重要性”并不取决于权重本身的大小，而取决于它所处理的激活值的大小**。实验表明，仅保留 1% 的“显著权重”（即对应激活值较大的通道）为 FP16 精度，就能极大恢复模型性能。有趣的是，如果按权重本身的 L2 范数来选这 1%，效果和随机选差不多；但如果按**激活值幅值**来选，效果立竿见影。

为了工程落地，AWQ 并没有真正把这 1% 的权重存成 FP16（混合精度会拖累推理速度），而是采用了一种精妙的**数学等价变换**。如图 13-1-3 所示，(a) 中简单的 RTN 量化导致 PPL 高达 43.2，模型基本“报废”；(b) 展示了如果保留 1% 显著权重为 FP16，PPL 能降回 13.0，但混合精度效率低。AWQ 的做法是 (c)：找出那些对应较大激活值的权重通道，给它们乘上一个放大系数 $s$（Scale up），同时在输入 $x$ 上除以 $s$。
- **原理**：$y = \mathbf{w}\mathbf{x} = (\mathbf{w} \cdot s) \cdot (\mathbf{x} / s)$
- **效果**：当权重被放大后，其数值范围变大，相对量化误差（Relative Quantization Error）就会变小。AWQ 的优化目标是找到一组最优的缩放因子 $\mathbf{s}$，使得量化误差最小：

  $$ \mathbf{s}^* = \arg \min_{\mathbf{s}} \mathcal{L}(\mathbf{s}) \tag{13.7} $$

  $$ \mathcal{L}(\mathbf{s}) = \| Q(\mathbf{W} \cdot \text{diag}(\mathbf{s})) (\text{diag}(\mathbf{s})^{-1} \cdot \mathbf{X}) - \mathbf{WX} \| \tag{13.8} $$

  其中，$Q(\cdot)$ 表示量化函数，$\mathbf{W}$ 为原始权重，$\mathbf{X}$ 为输入特征，$\mathbf{s}$ 为我们需要寻找的最佳缩放因子向量，$\text{diag}(\mathbf{s})$ 是由 $\mathbf{s}$ 构成的对角矩阵。这就好比用一把尺子去量物体，把物体放大后再量，读数的相对精度自然就高了。最终 AWQ 在全 INT 量化下也能达到与混合精度相当的性能（PPL 13.0）。

<p align="center">
  <img src="./images/13_1_3.png" width="95%" alt="AWQ 量化原理示意图" />
  <br />
  <em>图 13-1-3 AWQ 量化原理示意图：(a) RTN 导致精度崩塌；(b) 混合精度效果好但效率低；(c) AWQ 通过等价缩放实现全 INT 量化下的高性能</em>
</p>

通过表 13-1-1 的实验结果可以看到，在 Llama-2-7B/13B/70B 等不同规模的模型上，AWQ（W4-g128）的困惑度始终低于 RTN 和 GPTQ。特别是在 70B 模型上，AWQ 的 INT4 量化效果（PPL 3.41）几乎与 FP16 全精度基线（PPL 3.32）持平，证明了其在保护模型性能方面的优越性。

<div align="center">

<table border="1" style="margin: 0 auto;">
<tr>
  <td rowspan="2" style="text-align: center;"><strong>PPL &darr;</strong></td>
  <td style="text-align: center;"></td>
  <td colspan="3" style="text-align: center;"><strong>Llama-2</strong></td>
  <td colspan="4" style="text-align: center;"><strong>LLaMA</strong></td>
</tr>
<tr>
  <td style="text-align: center;"></td>
  <td style="text-align: center;"><strong>7B</strong></td>
  <td style="text-align: center;"><strong>13B</strong></td>
  <td style="text-align: center;"><strong>70B</strong></td>
  <td style="text-align: center;"><strong>7B</strong></td>
  <td style="text-align: center;"><strong>13B</strong></td>
  <td style="text-align: center;"><strong>30B</strong></td>
  <td style="text-align: center;"><strong>65B</strong></td>
</tr>
<tr>
  <td style="text-align: center;"><strong>FP16</strong></td>
  <td style="text-align: center;">-</td>
  <td style="text-align: center;">5.47</td>
  <td style="text-align: center;">4.88</td>
  <td style="text-align: center;">3.32</td>
  <td style="text-align: center;">5.68</td>
  <td style="text-align: center;">5.09</td>
  <td style="text-align: center;">4.10</td>
  <td style="text-align: center;">3.53</td>
</tr>
<tr>
  <td rowspan="2" style="text-align: center;"><strong>INT3</strong></td>
  <td style="text-align: center;">RTN</td>
  <td style="text-align: center;">6.66</td>
  <td style="text-align: center;">5.52</td>
  <td style="text-align: center;">3.98</td>
  <td style="text-align: center;">7.01</td>
  <td style="text-align: center;">5.88</td>
  <td style="text-align: center;">4.88</td>
  <td style="text-align: center;">4.24</td>
</tr>
<tr>
  <td style="text-align: center;">GPTQ</td>
  <td style="text-align: center;">6.43</td>
  <td style="text-align: center;">5.48</td>
  <td style="text-align: center;">3.88</td>
  <td style="text-align: center;">8.81</td>
  <td style="text-align: center;">5.66</td>
  <td style="text-align: center;">4.88</td>
  <td style="text-align: center;">4.17</td>
</tr>
<tr>
  <td rowspan="2" style="text-align: center;"><strong>g128</strong></td>
  <td style="text-align: center;">GPTQ-R</td>
  <td style="text-align: center;">6.42</td>
  <td style="text-align: center;">5.41</td>
  <td style="text-align: center;">3.86</td>
  <td style="text-align: center;">6.53</td>
  <td style="text-align: center;">5.64</td>
  <td style="text-align: center;">4.74</td>
  <td style="text-align: center;">4.21</td>
</tr>
<tr>
  <td style="text-align: center;">AWQ</td>
  <td style="text-align: center;">6.24</td>
  <td style="text-align: center;">5.32</td>
  <td style="text-align: center;">3.74</td>
  <td style="text-align: center;">6.35</td>
  <td style="text-align: center;">5.52</td>
  <td style="text-align: center;">4.61</td>
  <td style="text-align: center;">3.95</td>
</tr>
<tr>
  <td rowspan="2" style="text-align: center;"><strong>INT4</strong></td>
  <td style="text-align: center;">RTN</td>
  <td style="text-align: center;">5.73</td>
  <td style="text-align: center;">4.98</td>
  <td style="text-align: center;">3.46</td>
  <td style="text-align: center;">5.96</td>
  <td style="text-align: center;">5.25</td>
  <td style="text-align: center;">4.23</td>
  <td style="text-align: center;">3.67</td>
</tr>
<tr>
  <td style="text-align: center;">GPTQ</td>
  <td style="text-align: center;">5.69</td>
  <td style="text-align: center;">4.98</td>
  <td style="text-align: center;">3.42</td>
  <td style="text-align: center;">6.22</td>
  <td style="text-align: center;">5.23</td>
  <td style="text-align: center;">4.24</td>
  <td style="text-align: center;">3.66</td>
</tr>
<tr>
  <td rowspan="2" style="text-align: center;"><strong>g128</strong></td>
  <td style="text-align: center;">GPTQ-R</td>
  <td style="text-align: center;">5.63</td>
  <td style="text-align: center;">4.99</td>
  <td style="text-align: center;">3.43</td>
  <td style="text-align: center;">5.83</td>
  <td style="text-align: center;">5.20</td>
  <td style="text-align: center;">4.22</td>
  <td style="text-align: center;">3.66</td>
</tr>
<tr>
  <td style="text-align: center;">AWQ</td>
  <td style="text-align: center;">5.60</td>
  <td style="text-align: center;">4.97</td>
  <td style="text-align: center;">3.41</td>
  <td style="text-align: center;">5.78</td>
  <td style="text-align: center;">5.19</td>
  <td style="text-align: center;">4.21</td>
  <td style="text-align: center;">3.62</td>
</tr>
</table>

<p><em>表 13-1-1 不同量化方法在 LLaMA/Llama-2 上的 PPL 对比</em></p>

</div>

在 LLaMA、Mistral 等模型上，AWQ 的 INT4 量化几乎能达到 FP16 的无损性能水平。而且它具有**端侧友好**性，配合论文提出的 **TinyChat** 推理框架，作者展示了在高配 Jetson Orin 上以小 batch 形式运行 Llama-2-70B 的可能性；在树莓派等资源更受限的设备上，理论上也可以以 INT4 形式跑 7B 模型，但整体更偏 Demo/实验性质，速度和体验都会受到一定限制。

### 3.3 BitsAndBytes (BNB)

在前面的实战中，我们其实已经尝试使用了 BNB，通过配置 `BitsAndBytesConfig` 轻松实现了 4-bit 量化加载。如果说 GPTQ 和 AWQ 是侧重于“精打细算”的量化算法，那么 BNB 则是承载了 **LLM.int8()** [^6]和 **QLoRA** 等前沿研究的工程基石。它不仅是一个底层的 CUDA 库，更包含了一整套处理大模型量化难题的解决方案。BNB 的主要贡献是解决了大模型量化中一个棘手的**“离群值”**问题。研究发现，当模型参数规模超过 67 亿（6.7B）时，Transformer 层中会系统性地涌现出少量数值巨大的**离群特征（Emergent Outliers）**。虽然这些特征只占所有参数的约 0.1%，但它们对模型性能非常重要。传统的 8-bit 量化会将这些巨大的数值强制截断或粗糙量化，导致模型精度瞬间崩塌（如困惑度暴增）。如图 13-1-4 所示，在 6.7B 参数规模处，普通 8-bit 量化（橙线）的准确率急剧下降，而 LLM.int8()（蓝线）则保持了与 16-bit 基线一致的性能。

<p align="center">
  <img src="./images/13_1_4.png" width="80%" alt="模型规模与量化性能的关系" />
  <br />
  <em>图 13-1-4 模型规模与量化性能的关系：6.7B 参数后，传统量化（W8A16）失效，LLM.int8() 保持无损</em>
</p>

**LLM.int8()** 解决这一问题的主要原理是**混合精度分解（Mixed-precision Decomposition）**。它就像一个智能筛子，在推理过程中动态探测特征值的大小。对于 99.9% 的常规数值，使用**向量级量化（Vector-wise Quantization）**把它们压缩为 8-bit 进行矩阵乘法，以节省显存：

$$ \mathbf{C}_{f16} \approx \frac{1}{\mathbf{c}_{x} \otimes \mathbf{c}_{w}} \mathbf{C}_{i32} \tag{13.9} $$

其中，$\mathbf{C}_{f16}$ 是近似的 FP16 输出结果，$\mathbf{C}_{i32}$ 是 INT8 矩阵乘法得到的 INT32 结果，$\mathbf{c}_{x}$ 和 $\mathbf{c}_{w}$ 分别是输入 $\mathbf{X}$ 和权重 $\mathbf{W}$ 的量化缩放因子（Scaling Factors）。

而对于那 0.1% 超过阈值（如 6.0）的“离群”维度 $O$，则自动拆分出来，保持 **FP16** 高精度计算。最后将两部分结果合并（见图 13-1-5）：

$$ \mathbf{C}_{f16} \approx \underbrace{\sum_{h \in O} \mathbf{X}_{f16}^h \mathbf{W}_{f16}^h}_{\text{离群值 (FP16)}} + \underbrace{S \cdot \sum_{h \notin O} \mathbf{X}_{i8}^h \mathbf{W}_{i8}^h}_{\text{常规值 (INT8)}} \tag{13.10} $$

其中，$h$ 代表特征维度，$O$ 是离群特征维度的集合，$S$ 是去归一化项（对应上面的缩放因子乘积）。为了便于理解，这里对缩放因子 $\mathbf{c}_{x}$、$\mathbf{c}_{w}$ 和 $S$ 的张量维度及广播方式做了简化，实际工程实现中会针对 batch/head/channel 等维度分别维护缩放因子。

这种“抓大放小”的策略，让我们可以在几乎不损失任何精度的情况下，用 INT8 的显存开销运行超大模型。

<p align="center">
  <img src="./images/13_1_5.png" width="95%" alt="LLM.int8() 混合精度分解示意图" />
  <br />
  <em>图 13-1-5 LLM.int8() 混合精度分解示意图：自动分离离群特征（FP16）与常规特征（INT8）</em>
</p>

在此基础上，BNB 进一步演进，成为了微调技术 **QLoRA** 的核心依赖。它引入了 **NormalFloat4 (NF4)** 数据类型，这是一种专门为正态分布权重量身定制的 4-bit 类型，比标准的 INT4 具有更高的信噪比。如今，通过 `BitsAndBytesConfig`，我们可以轻松调用这些技术，在单张消费级显卡上不仅能加载大模型，还能进行高效的微调。

## 四、Qwen2.5 模型推理量化实战

了解了量化的核心原理后，让我们进入实战环节。本节将演示如何使用 `llmcompressor` 库对 `Qwen/Qwen2.5-1.5B-Instruct` 模型分别进行 **GPTQ** 和 **AWQ** 量化，并对比它们的效果。

> [本节完整代码](https://github.com/datawhalechina/base-nlp/blob/main/code/C13/01_qwen2.5_llmcompressor.ipynb)

### 4.1 环境准备

#### 4.1.1 `llmcompressor` 简介

`LLM Compressor` [^7] 是一个易于使用的库，目标是优化大语言模型以便使用 `vLLM` 进行部署。它能够实现高达 5 倍的推理速度提升，并显著降低成本。作为一个综合性的工具包，提供了以下核心功能：

- **算法支持丰富**：支持包括 GPTQ、AWQ、SmoothQuant、SparseGPT 等在内的多种权重量化、激活量化和剪枝算法。
- **无缝集成**：与 Hugging Face 的 Transformers、Models 和 Datasets 深度集成，使用体验流畅。
- **vLLM 友好**：支持基于 `safetensors` 的压缩模型存储格式，可直接被 `vLLM` 加载。
- **高效处理**：借助 `accelerate` 库，支持对超大模型进行高性能压缩。

如图 13-1-6 所示，LLM Compressor 的工作流程首先**输入**准备好的 Hugging Face 模型和（可选的）校准数据集；接着在**压缩**阶段，使用 `llmcompressor` 库应用量化算法（如 GPTQ、AWQ 等）；随后**输出**压缩后的模型检查点（Compressed Model Checkpoint）；最后在**部署**阶段，将压缩模型直接加载到 vLLM 中进行高效推理，最终服务于上层应用。

<p align="center">
  <img src="./images/13_1_6.png" width="90%" alt="LLM Compressor Workflow" />
  <br />
  <em>图 13-1-6 LLM Compressor 工作流程图</em>
</p>

#### 4.1.2 环境安装

接下来先安装 `llmcompressor` 库，它提供了一套统一的 API 来执行各种量化算法，简化了 `auto-gptq` 或 `autoawq` 等底层库的调用。

```bash
pip install llmcompressor
```

验证安装是否成功：

```bash
pip show llmcompressor
```

如果得到类似这样的输出就说明安装成功了：
```bash
Name: llmcompressor
Version: 0.9.0
Summary: A library for compressing large language models utilizing the latest techniques and research in the field for both training aware and post training techniques. The library is designed to be flexible and easy to use on top of PyTorch and HuggingFace Transformers, allowing for quick experimentation.
Home-page: https://github.com/vllm-project/llm-compressor
Author: Neuralmagic, Inc.
Author-email: support@neuralmagic.com
License: Apache
Location: c:\users\dalvqw\.conda\envs\peft\lib\site-packages
Requires: accelerate, auto-round, compressed-tensors, datasets, loguru, numpy, nvidia-ml-py, pillow, pyyaml, requests, torch, tqdm, transformers
Required-by:
```

### 4.2 GPTQ 量化实战

#### 4.2.1 初始化环境

这里我们不仅需要 `transformers` 来加载模型，还需要从 `llmcompressor` 中导入量化修饰器 `GPTQModifier` 和一键量化函数 `oneshot`。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

# 基础配置
base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
```

#### 4.2.2 定义量化策略

接下来是量化的核心步骤。需要通过 `GPTQModifier` 来定义量化的具体策略。

```python
# 量化后模型输出目录
gptq_out_dir = "models/qwen2.5-1.5b-instruct-gptq-llmc"

# 定义 GPTQ 量化策略
gptq_recipe = [
    GPTQModifier(
        scheme="W4A16",      # 权重 4bit，激活保持 16bit
        targets="Linear",    # 只量化线性层
        ignore=["lm_head"],  # 保持输出头的高精度，避免性能损失
    ),
]
```

- **`scheme="W4A16"`**：**W4** 代表权重（Weights）被量化为 4-bit 整数，将模型体积压缩为原来的 1/4；**A16** 代表激活值（Activations）在计算时保持 16-bit 浮点精度（FP16/BF16）。这种组合既降低了显存占用，又利用了 GPU 的 INT4 Tensor Core 进行加速，同时保持了较高的计算精度。
- **`targets="Linear"`**：指定量化仅应用于线性层（Linear Layers）。Transformer 模型的大部分参数都集中在这些全连接层中。
- **`ignore=["lm_head"]`**：这是一个**必须注意**的细节。模型的输出头（LM Head）负责将高维特征映射回词表空间，对数值精度极其敏感。对其进行 4-bit 量化往往会导致输出乱码或逻辑崩坏，所以通常将其排除在量化范围之外。

#### 4.2.3 执行 One-Shot 量化

定义好策略后，就可以开始执行量化了。`llmcompressor` 提供的 `oneshot` 函数将加载模型、应用算法并保存结果，全流程一气呵成。

```python
oneshot(
    model=base_model_id,
    dataset="open_platypus",       # 使用公开数据集进行校准
    recipe=gptq_recipe,            # 传入定义好的量化策略
    output_dir=gptq_out_dir,
    max_seq_length=2048,
    num_calibration_samples=128,   # 128个样本通常足够计算准确的统计信息
)
```

执行该函数后，终端会打印出详细的量化进度日志。我们可以看到 `llmcompressor` 正在逐层（`model.layers.0`, `model.layers.1`...）对模型的线性模块（`q_proj`, `k_proj`...）进行压缩：

```bash
2025-12-19T03:57:34.951606+0800 | compress_modules | INFO - Quantizing model.layers.0.self_attn.q_proj using 128 samples
2025-12-19T03:57:36.204668+0800 | compress | METRIC - time 1.25s
2025-12-19T03:57:36.206047+0800 | compress | METRIC - error 1758.54
...
2025-12-19T03:57:36.264447+0800 | compress_modules | INFO - Quantizing model.layers.0.self_attn.k_proj using 128 samples
```

在这个过程中，**校准（Calibration）** 是不可或缺的一环。与微调不同，GPTQ 是一种 Post-Training Quantization (PTQ) 方法，它不需要全量训练，但需要少量的真实数据来“观察”模型的激活值分布。

- **`dataset="open_platypus"`**：GPTQ 依赖于计算海森矩阵来判断权重的重要性。我们需要输入一些具有代表性的文本数据。当前选用的 `open_platypus` 是一个高质量的指令微调数据集，涵盖了逻辑推理、代码生成等多种任务，能够很好地代表真实场景的输入分布，防止量化后的模型在特定领域能力退化。
- **`num_calibration_samples=128`**：通常情况下，128 到 512 个样本就足以计算出准确的统计信息。过多的样本不仅增加耗时，边际收益也递减。

#### 4.2.4 加载与效果验证

量化完成后，生成的模型本质上还是一个 Transformer 模型，但其内部的权重层结构发生了变化。我们可以像加载普通模型一样加载它，并检查其结构。

```python
# 加载 GPTQ 量化后的检查点做推理

gptq_tokenizer = AutoTokenizer.from_pretrained(gptq_out_dir, trust_remote_code=True)
if gptq_tokenizer.pad_token_id is None:
    gptq_tokenizer.pad_token = gptq_tokenizer.eos_token

gptq_model = AutoModelForCausalLM.from_pretrained(
    gptq_out_dir,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
gptq_model.eval()

# 打印 tokenizer 的特殊 token 信息，确保 pad_token 设置正确
gptq_tokenizer.pad_token, gptq_tokenizer.eos_token, gptq_tokenizer.pad_token_id, gptq_tokenizer.eos_token_id
```

输出如下：
```bash
('<|endoftext|>', '<|im_end|>', 151643, 151645)
```

加载完成后，我们检查一下模型结构。

```python
# 检查第 0 层的 q_proj，确认量化是否生效
layer = gptq_model.model.layers[0].self_attn.q_proj
print(f"GPTQ Layer Type: {type(layer)}")
```

输出如下：

```bash
GPTQ q_proj layer type: <class 'compressed_tensors.linear.compressed_linear.CompressedLinear'>
GPTQ quantization_config: CompressedTensorsConfig {
  "config_groups": {
    "group_0": {
      "format": "pack-quantized",
      "input_activations": null,
      "output_activations": null,
      "targets": [
        "Linear"
      ],
      "weights": {
        "actorder": "static",
        "block_structure": null,
        "dynamic": false,
        "group_size": 128,
        "num_bits": 4,
        "observer": "minmax",
        "observer_kwargs": {},
        "scale_dtype": null,
        "strategy": "group",
        "symmetric": true,
        "type": "int",
        "zp_dtype": null
      }
    }
...
  "kv_cache_scheme": null,
  "quantization_status": "compressed"
}
```

可以看到输出的层类型变成了 `CompressedLinear`，说明原本庞大的 FP16 线性层已经被成功替换为支持压缩张量计算的专用层。同时，`quantization_config` 中清晰地记录了量化策略：`weights` 部分显示 `num_bits: 4` 和 `group_size: 128`，确认模型已按照 W4A16 的分组量化策略进行了压缩。为了验证量化后的模型没有“变傻”，我们可以进行一次简单的推理测试。

```python
gptq_tokenizer = AutoTokenizer.from_pretrained(gptq_out_dir)

@torch.no_grad()
def gptq_chat(question: str) -> str:
    msgs = [
        {"role": "system", "content": "你是一名 AI 助手，回答准确、简洁。"},
        {"role": "user", "content": question},
    ]
    input_ids = gptq_tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(gptq_model.device)

    gen_ids = gptq_model.generate(
        input_ids=input_ids,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=gptq_tokenizer.eos_token_id,
        pad_token_id=gptq_tokenizer.pad_token_id,
    )
    out_ids = gen_ids[0, input_ids.shape[-1]:]
    return gptq_tokenizer.decode(out_ids, skip_special_tokens=True).strip()

print(gptq_chat("用两三句话解释一下什么是量子计算？"))
```

输出如下：

```bash
量子计算是利用量子位（qubits）代替经典比特进行信息处理的一种计算方式。它利用量子叠加、纠缠等特性来实现超越传统计算机的并行处理能力，可以高效地解决某些特定问题。通过使用量子比特...
```

可以看到模型依然能够生成逻辑清晰、内容准确的回答，证明 4-bit 量化在大幅压缩模型体积的同时，依然很好地保留了模型的核心能力。

### 4.3 AWQ 量化实战

#### 4.3.1 定义 AWQ 策略

AWQ 的配置流程与 GPTQ 非常相似，主要区别在于使用 `AWQModifier`。

```python
from llmcompressor.modifiers.awq import AWQModifier

awq_out_dir = "models/qwen2.5-1.5b-instruct-awq-llmc"

awq_recipe = [
    AWQModifier(
        scheme="W4A16",
        targets="Linear",
        ignore=["lm_head"],
    ),
]
```

虽然参数看起来一样，但就像前文所介绍的那样，它们背后的算法逻辑是截然不同的。AWQ 不像 GPTQ 那样依赖海森矩阵的逆，而是依据**激活值的幅度**来搜索最优的**缩放因子（Scaling Factor）**，将那些对应较大激活值（即更重要）的权重通道进行“放大”保护。这种机制使得 AWQ 在某些特定场景（如代码生成或逻辑推理）下能保留更多的细节能力。

#### 4.3.2 执行量化与对比

```python
oneshot(
    model=base_model_id,
    dataset="open_platypus",
    recipe=awq_recipe,
    output_dir=awq_out_dir,
    max_seq_length=2048,
    num_calibration_samples=128,
)
```

整个过程同样依赖校准数据集来统计激活值的幅度。执行后，会看到类似如下的日志输出，特别是在每一层（`model.layers.*`）的注意力机制中，系统会进行“平滑”操作（`SmoothQuant`）并应用量化：

```bash
2025-12-19T04:08:32.146092+0800 | _set_resolved_mappings | WARNING - skipping AWQ for model.layers.0.self_attn.v_proj ...
...
2025-12-19T04:08:32.160737+0800 | compress_modules | INFO - Running SmoothQuant for model.layers.0.self_attn.q_proj
2025-12-19T04:08:32.161737 | compress_modules | INFO - Running SmoothQuant for model.layers.0.self_attn.k_proj
```

在 AWQ 量化时，日志中出现的 **SmoothQuant** 或 **Smoothing** 阶段，是 `llmcompressor` 在实现 AWQ 时对平滑/缩放步骤采用的内部命名，算法本身依然是 AWQ（与 SmoothQuant 论文中的方法不同）。这个过程本质上是在执行一次**网格搜索（Grid Search）**，所以我们会发现 AWQ 的量化过程比 GPTQ 要慢很多。因为 GPTQ 的核心计算（海森矩阵求逆）是确定性的数学解析过程，计算量相对固定；而 AWQ 需要针对每一层，在校准数据上反复尝试不同的缩放因子 $s$，来找到让量化误差最小的最优解。这个迭代搜索的过程自然比单次矩阵运算更耗时。不过，这种额外的时间投入换来的是对**离群点**的鲁棒性。大模型中常存在极少数数值巨大的激活值（尖峰），如果不加处理直接量化，会带来巨大的精度损失。AWQ 的 Smoothing 过程相当于将这些“尖峰”的压力平滑地分摊到了权重上，从而在不增加推理计算量的前提下，显著降低了量化噪声。而且，AWQ 的这种设计使其生成的模型在 **vLLM** 等推理引擎中往往能获得更好的原生加速支持。量化完成后，可以使用与 GPTQ 相同的方式加载和测试模型。

---

## 参考文献

[^1]: [LeCun, Y., Denker, J. S., & Solla, S. A. (1989). *Optimal Brain Damage*. Advances in Neural Information Processing Systems, 2.](http://yann.lecun.com/exdb/publis/pdf/lecun-90b.pdf)

[^2]: [Frankle, J., & Carbin, M. (2019). *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*. ICLR 2019.](https://arxiv.org/abs/1803.03635)

[^3]: [Frantar, E., et al. (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. ICLR 2023.](https://arxiv.org/abs/2210.17323)

[^4]: [Lin, J., et al. (2024). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. arXiv preprint arXiv:2306.00978.](https://arxiv.org/abs/2306.00978)

[^5]: [Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023.](https://arxiv.org/abs/2305.14314)

[^6]: [Dettmers, T., et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS 2022.](https://arxiv.org/abs/2208.07339)

[^7]: [vLLM Team. *LLM Compressor Documentation*.](https://docs.vllm.ai/projects/llm-compressor/en/latest/)
