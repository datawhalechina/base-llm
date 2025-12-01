# 第二节 MoE 架构解析

我们在上一节详细解析了 Llama2 的架构。像 Llama2、GPT-3 这样的模型，通常被称为**稠密模型（Dense Model）**。这意味着，对于每一个输入的 Token，模型中**所有的**参数（从第一层到最后一层）都会参与计算。

但是，随着模型规模向万亿级别迈进，全量参数计算带来的算力成本变得不可承受。这就引出了本节的主角——**混合专家模型（Mixture of Experts, MoE）**。MoE 技术通过一种 **“稀疏激活”** 的机制，兼具了大规模参数的知识容量与极低的推理成本。Mistral 8x7B 等模型的出现，更是证明了 MoE 在开源大模型领域的巨大潜力，使其成为当前最受关注的技术方向之一。

## 一、MoE 的前世今生

### 1.1 自适应局部专家混合

最早的 MoE 思想可以追溯到 1991 年 Michael Jordan 和 Geoffrey Hinton 发表的经典论文《Adaptive Mixture of Local Experts》 [^1]。这篇论文不仅提出了分治的架构，更重要的是从理论层面解决了神经网络在多任务学习中的根本性难题。

#### 1.1.1 干扰效应与分治思想

在传统的单体神经网络中，如果我们尝试让一个网络同时学习多个截然不同的子任务（例如既学做菜又学修车），往往会出现“**强干扰效应（Strong Interference Effects）**”。这是因为网络的所有权重都参与了所有任务的计算，当网络调整参数以适应任务 A 时，可能会破坏它在任务 B 上已经学到的特征表示。从而导致学习速度变慢，泛化能力变差。

为了解决这个问题，论文提出了一种基于 **“分治（Divide and Conquer）”** 策略的系统架构：

-  **专家网络**：系统包含多个独立的神经网络（可以是简单的前馈网络）。每个专家不再需要处理全局任务，只需专注于输入空间中的一个局部区域或一类特定的子任务。
-  **门控网络**：充当协调者的角色。它接收与专家相同的输入 $x$，并输出一组**混合比例（Mixing Proportions）**$p_i$，即选择每个专家的概率。它就像一个软性的随机开关，决定当前的输入案例应该由哪位专家来主导处理。

#### 1.1.2 损失函数设计

该论文最重要的贡献在于设计了特定的机制来鼓励专家之间的**竞争（Competition）**，而非合作。如果仅仅将所有专家的输出简单线性相加来逼近目标 $\mathbf{y}$，即预测值 $\hat{\mathbf{y}} = \sum p_i \mathbf{E}_i$，并在最终输出上计算误差，专家们会倾向于“合作”。

$$ Loss_{coop} = || \mathbf{y} - \sum_{i} p_i \mathbf{E}_i ||^2 \tag{6.2} $$

在这种合作模式下，为了减小总误差，每个专家都会试图去弥补其他专家的残差。继而导致每个案例都牵动所有专家的权重，依然无法解决“干扰效应”。

为了实现解耦，作者提出将系统视为一个随机生成器，并采用**负对数似然**作为目标函数：

$$ Loss_{comp} = - \log \sum_{i} p_i e^{-\frac{1}{2} || \mathbf{y} - \mathbf{E}_i ||^2} \tag{6.3} $$

其中：
-   $\mathbf{y}$ 是我们希望模型输出的**真实目标**。
-   $\mathbf{E}_i$ 是第 $i$ 个**专家**的输出。
-   $p_i$ 是门控网络分配给第 $i$ 个专家的**权重**（概率）。

在这个目标函数下，系统倾向于**“赢家通吃”**。当某个专家 $\mathbf{E}_i$ 的输出非常接近目标 $\mathbf{y}$ 时（即误差项 $|| \mathbf{y} - \mathbf{E}_i ||^2$ 很小），指数项 $e^{-\dots}$ 会变得很大。为了最小化总 Loss，门控网络会倾向于显著增加这个“表现好”的专家的权重 $p_i$，而忽略其他专家。

这一机制实现了**权重的解耦**。误差反向传播时，只有被选中的“胜出者”和门控网络的权重会被更新，其他专家几乎不受影响。**有效缓解了**任务间的干扰，实现了“让专业的专家干专业的事”。

> **MoE vs. 集成学习**
> 
> 虽然结构看似相似，但两者有本质区别。**集成学习**（如随机森林）通常假设基模型是独立或互补的，预测时所有模型都参与，通过投票或加权平均得出结果。而 **MoE** 强调**动态的条件计算**，它根据输入数据本身（Data-driven）动态地划分任务空间，不同的输入激活不同的子网络路径。

### 1.2 深度神经网络中的 MoE

2013 年，Ilya Sutskever 等人发表了论文《Learning Factored Representations in a Deep Mixture of Experts》 [^2]，将 MoE 与深度学习进行了开创性的结合。

#### 1.2.1 从浅层到深层的变革

在此之前，MoE 通常作为一种独立的浅层模型存在。Ilya 等人的工作打破了这一局限，他们提出 **Deep Mixture of Experts（DMoE）**，将 MoE 结构“模块化”并嵌入到深度神经网络的多个层级中。

意味着 MoE 不再是一个孤立的架构，而成为了一种**可插拔的层**。我们可以在一个深层网络的不同位置（例如第 1 层和第 2 层）分别插入 MoE 模块，每一层都有自己独立的门控网络和专家集合。

- **层级化的门控（Hierarchical Gating）**：输入 $x$ 首先经过第一层的门控 $g^1$，被路由到第一层的专家 $f^1_i$。第一层的输出 $z^1$ 接着作为第二层门控 $g^2$ 的输入，再次被路由到第二层的专家 $f^2_j$。
- **指数级增长的组合路径**：通过这种堆叠，网络能够表达的有效“专家组合”数量呈指数级增长。如果第一层有 $N$ 个专家，第二层有 $M$ 个专家，那么网络潜在的组合路径就高达 $N \times M$ 种。每个输入样本都会根据其特性，动态地选择一条最适合的处理路径。

#### 1.2.2 学习分解的特征表示

论文的标题强调了“Factored Representations”。通过在不同层级引入混合专家，模型能够自发地在不同层级学习到数据的不同维度的特征。论文在“Jittered MNIST”（带随机平移的手写数字）数据集上观察到了有趣的现象：

- **第一层专家**倾向于根据数字的**位置（Location）**进行分工，成为了“Where Experts”。
- **第二层专家**倾向于根据数字的**类别（Class）**进行分工，成为了“What Experts”。

这种自动的特征解耦证明了深度 MoE 能够有效地利用其深层结构，将复杂任务分解为多个正交的子问题进行处理，为后来 MoE 在 Transformer 中的广泛应用奠定了重要的理论基础。

### 1.3 稀疏门控 MoE

Google Brain 团队（包括 Geoffrey Hinton 和 Jeff Dean 等）于 2017 年发表了论文《Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer》 [^3]，正式将 MoE 带入了超大规模模型（百亿参数级）的时代。

#### 1.3.1 条件计算

当时，深度学习面临一个两难困境。模型容量（参数量）越大，预测准确率通常越高，但计算成本和训练时间也会呈平方级增长。传统的 LSTM 等网络受限于梯度消失和计算资源，很难无限制地加深或加宽。这就引出了**条件计算（Conditional Computation）**的概念，我们**能否在不增加计算量的前提下，大幅增加模型的参数量？**

这篇论文给出了肯定的答案。他们设计了一种**稀疏门控混合专家层（Sparsely-Gated MoE Layer）**，可以在每个样本处理过程中只激活网络的一小部分。

- **1000 倍的模型容量**：通过引入多达 1370 亿个参数的 MoE 层（包含数千个专家），该模型在语言建模和机器翻译任务上取得了显著优于 SOTA 的结果，而其计算效率仅有微小的损失。

- **LSTM + MoE 架构**：如图 6-3 所示，研究者将 MoE 层嵌入到堆叠的 LSTM 层之间。左侧展示了该层被卷积式地应用于每个时间步；右侧细节显示，MoE 层接收前一层的输入，通过门控网络（绿色框）计算稀疏权重，仅激活少数专家（灰色框）参与计算，其余专家（白色框）保持闲置。各专家的输出经加权求和后，传递给下一层。对于输入序列中的每一个位置（Token），MoE 都会动态选择不同的专家组合进行处理。

<p align="center">
  <img src="./images/6_2_1.png" width="70%" alt="LSTM + MoE 架构图" />
  <br />
  <em>图 6-3：稀疏门控 MoE 层的架构示意图</em>
</p>

#### 1.3.2 稀疏性与负载均衡

为了让这一构想落地，论文重点解决了**“如何稀疏选择”**以及**“如何防止专家崩塌”**这两个关键挑战。

-   **引入噪声的稀疏门控**：
    传统的 Softmax 门控通常会给所有专家分配非零的权重，意味着所有专家都要参与计算，无法节省算力。为此，论文引入了一种带噪声的门控机制：在门控输入中加入可训练的高斯噪声，计算后仅保留权重最大的 $k$ 个专家（例如 $k=4$），将其余所有专家的权重强制置为 $-\infty$（即概率为 0）。

    如图 6-4 所示，这种稀疏性带来了巨大的收益。在保持计算预算（每步约 800 万次运算）基本不变的情况下，随着专家数量从 4 个增加到 4096 个（横轴），模型参数量剧增，但测试集困惑度（纵轴）显著下降。说明条件计算可以在不增加推理成本的前提下，利用海量参数大幅提升性能。

    <p align="center">
       <img src="./images/6_2_2.png" width="50%" alt="模型容量与效果对比图" />
       <br />
       <em>图 6-4：模型容量与测试集困惑度的关系</em>
    </p>

-   **避免“赢家通吃”的负载均衡**：
    稀疏选择机制很容易引发**“马太效应”（Rich get richer）**。在训练初期，某些专家可能仅仅因为初始化权重的随机差异而“运气好”被选中。被选中意味着得到了梯度更新，它们变得“更聪明”，从而更有可能在下一次被再次选中。反之，其他落选的专家因得不到训练而持续“愚钝”，最终导致大部分专家“饿死”，模型退化为只有少数活跃专家的稠密模型。

    为了解决这个问题，作者在总损失函数中加入了额外的**辅助损失（Auxiliary Loss）**，包含 **Importance Loss** 和 **Load Loss**。这些损失函数并不直接服务于预测准确率，而是专门用来惩罚“分配不均”的现象，强制门控网络“雨露均沾”，确保所有专家都能接收到大致相等的样本量，从而得到充分的训练。

## 二、大模型时代的 MoE

进入 Transformer 时代后，MoE 技术成为了突破模型规模瓶颈的关键。Google 在这一领域进行了密集的探索，通过 GShard、Switch Transformer 和 GLaM 等一系列工作，确立了现代大规模 MoE 的技术范式。

### 2.1 GShard 迈向六千亿参数

2020 年，Google 提出了 **GShard** [^4]，旨在将 Transformer 模型扩展到 **6000 亿（600B）** 参数级别。GShard 不仅仅是一个模型，更是一套支持超大规模稀疏模型并行的训练框架。

它通过**数据并行（Data Parallelism）**与**模型并行（Model Parallelism）**的优雅结合，解决了超大模型无法装入单卡显存、通信开销过大等训练难题。

#### 2.1.1 MoE 层的构建与门控机制

在 GShard 中，MoE 层的应用变得更加标准化。它并没有将所有的层都转换为 MoE 层，而是采用了**“隔层替换”**的策略：

-   **保留 Attention**：Transformer 的 Self-Attention 层保持不变，因为其参数量相对较小且计算关键。
-   **替换 FFN**：将 Transformer Block 中的**前馈神经网络**替换为 MoE 层。
-   **隔层设置**：通常采用“隔层替换”的策略（例如第 1、3、5 层使用 MoE，第 2、4、6 层保留标准 FFN），在增加容量和保持稳定性之间取得平衡。

对于 MoE 层的计算，GShard 明确了输入 Token $\mathbf{x}$ 的输出 $\mathbf{y}$ 是由门控网络 $\mathcal{G}$ 选择的专家输出的加权和：

$$ \mathbf{y} = \sum_{i=1}^{N} p_i(\mathbf{x}) \cdot \mathbf{E}_i(\mathbf{x}) \tag{6.4} $$

其中：
-   $p_i(\mathbf{x})$ 是门控网络（Router）计算出的第 $i$ 个专家的**权重**（通常是 Softmax 后的 Top-k 概率，其余为 0）。
-   $\mathbf{E}_i(\mathbf{x})$ 是第 $i$ 个**专家网络**（Expert FFN）对输入 $\mathbf{x}$ 的处理结果。

每个专家 $\mathbf{E}_i$ 内部通常就是一个标准的双层全连接网络：

$$ \mathbf{E}_i(\mathbf{x}) = \mathbf{W}_{out} \cdot \text{ReLU}(\mathbf{W}_{in} \cdot \mathbf{x}) \tag{6.5} $$

在此基础上，GShard 提出了 **Top-2 Gating（Top-2 门控）** 策略。此前的研究通常使用 Top-k（如 k=4）或更复杂的门控，但 GShard 发现，**每个 Token 只路由给 2 个专家**（Top-2）就足够了：

-   **第一专家**：选择权重最高的专家，保证主要任务的处理。
-   **第二专家**：根据权重概率随机选择或直接选择第二高权重的专家，引入辅助处理和一定的随机性，有助于负载均衡。

这种 Top-2 策略成为了后来（包括 Mistral）的标准配置。

#### 2.1.2 分布式并行策略

单张 GPU/TPU 显然无法装下 6000 亿参数。GShard 创造性地结合了**数据并行（Data Parallelism）**与**模型并行（Model Parallelism）**，解决了超大模型的存储与通信难题。

-   **非 MoE 层（如 Attention）**：采用**复制（Replicated）** 策略。所有设备持有相同的副本，进行标准的数据并行训练。
-   **MoE 层**：采用**分片（Sharded）** 策略。专家网络被切分并分布在不同设备上（例如 2048 个专家分布在 2048 个 TPU 核上）。

当一个 Token 需要被路由到不在当前设备的专家时，系统会通过高效的 **All-to-All** 通信原语，将该 Token 发送到目标设备。计算完成后，再将结果传回。

如图 6-5 展示了从标准 Transformer 到分布式 MoE 的演进。(a) 标准 Transformer 编码器堆叠了 Self-Attention 和 FFN 层；(b) MoE Transformer 将每隔一个 FFN 层替换为 MoE 层；(c) 在跨设备扩展时，Attention 层（黄色）在所有设备间复制，而 MoE 层（红色）则被分片存储。这种“复制与分片”结合的策略，既保证了非 MoE 层的高效计算，又通过分片突破了单设备的显存限制。

<p align="center">
  <img src="./images/6_2_3.png" width="80%" alt="GShard MoE Transformer Encoder 架构图" />
  <br />
  <em>图 6-5：GShard MoE Transformer Encoder 架构与并行策略</em>
</p>

通过这种设计，GShard 实现了**亚线性（Sub-linear）** 的计算成本增长。模型参数量增加 16 倍（从 37.5B 到 600B），训练算力成本仅增加了不到 4 倍。

### 2.2 Switch Transformer

虽然 GShard 验证了规模化路径，但大规模 MoE 的训练仍面临复杂性高、通信成本大和训练不稳定等挑战。2021 年 Google 推出的 **Switch Transformer** [^5] 通过简化路由算法和改进训练技术，成功将参数量推向了 **1.6 万亿**（Trillion）级别，同时实现了 4 倍于 T5-XXL 的训练速度提升。通过表 6-1 可以看出，在相同的算力预算下，Switch Transformer 无论是在质量（负对数困惑度）还是在速度上，都全面超越了传统的 T5 模型以及早期的 MoE 模型。

<div align="center">

<table border="1" style="margin: 0 auto;">
<tr>
  <td style="text-align: center;"><strong>Model</strong></td>
  <td style="text-align: center;"><strong>Capacity Factor</strong></td>
  <td style="text-align: center;"><strong>Quality after 100k steps (↑)<br>(Neg. Log Perp.)</strong></td>
  <td style="text-align: center;"><strong>Time to Quality Threshold (↓)<br>(hours)</strong></td>
  <td style="text-align: center;"><strong>Speed (↑)<br>(examples/sec)</strong></td>
</tr>
<tr>
  <td style="text-align: center;">T5-Base</td>
  <td style="text-align: center;">—</td>
  <td style="text-align: center;">-1.731</td>
  <td style="text-align: center;">Not achieved†</td>
  <td style="text-align: center;">1600</td>
</tr>
<tr>
  <td style="text-align: center;">T5-Large</td>
  <td style="text-align: center;">—</td>
  <td style="text-align: center;">-1.550</td>
  <td style="text-align: center;">131.1</td>
  <td style="text-align: center;">470</td>
</tr>
<tr>
  <td style="text-align: center;">MoE-Base</td>
  <td style="text-align: center;">2.0</td>
  <td style="text-align: center;">-1.547</td>
  <td style="text-align: center;">68.7</td>
  <td style="text-align: center;">840</td>
</tr>
<tr>
  <td style="text-align: center;">Switch-Base</td>
  <td style="text-align: center;">2.0</td>
  <td style="text-align: center;">-1.554</td>
  <td style="text-align: center;">72.8</td>
  <td style="text-align: center;">860</td>
</tr>
<tr>
  <td style="text-align: center;">MoE-Base</td>
  <td style="text-align: center;">1.25</td>
  <td style="text-align: center;">-1.559</td>
  <td style="text-align: center;">80.7</td>
  <td style="text-align: center;">790</td>
</tr>
<tr>
  <td style="text-align: center;">Switch-Base</td>
  <td style="text-align: center;">1.25</td>
  <td style="text-align: center;">-1.553</td>
  <td style="text-align: center;">65.0</td>
  <td style="text-align: center;">910</td>
</tr>
<tr>
  <td style="text-align: center;">MoE-Base</td>
  <td style="text-align: center;">1.0</td>
  <td style="text-align: center;">-1.572</td>
  <td style="text-align: center;">80.1</td>
  <td style="text-align: center;">860</td>
</tr>
<tr>
  <td style="text-align: center;">Switch-Base</td>
  <td style="text-align: center;">1.0</td>
  <td style="text-align: center;">-1.561</td>
  <td style="text-align: center;">62.8</td>
  <td style="text-align: center;">1000</td>
</tr>
<tr>
  <td style="text-align: center;">Switch-Base+</td>
  <td style="text-align: center;">1.0</td>
  <td style="text-align: center;">-1.534</td>
  <td style="text-align: center;">67.6</td>
  <td style="text-align: center;">780</td>
</tr>
</table>

<p><em>表 6-1 Switch Transformer 与 MoE 及 T5 的性能对比</em></p>

</div>

#### 2.2.1 简化稀疏路由

Switch Transformer 的主要创新在于提出了 **Switch Layer**。如图 6-6 所示，它用稀疏的 Switch FFN 层（浅蓝色区域）替换了标准 Transformer 中的稠密 FFN 层。在该层中，对于输入序列中的每个 Token（例如图中的 "More" 和 "Parameters"），路由器（Router）会计算其路由概率，并将其分发给**唯一**的一个专家（实线箭头）进行处理。

这种 **Top-1 Routing（单专家路由）** 机制是 Switch Transformer 与传统 MoE（通常路由给 Top-k 个专家，k > 1）最大的区别。尽管看似激进，但它带来了显著的优势：

1.  **减少路由计算**：路由决策更简单。
2.  **降低通信成本**：每个 Token 只需发送到一个目的地。
3.  **减小专家批量**：每个专家需要处理的 Token 数量（Expert Capacity）至少减半。

虽然直觉上 $k=1$ 可能限制了专家的协作，但实验证明这种简化不仅保持了模型质量，还显著提高了计算效率。

<p align="center">
  <img src="./images/6_2_4.png" width="80%" alt="Switch Transformer Encoder 架构图" />
  <br />
  <em>图 6-6：Switch Transformer Encoder 架构图</em>
</p>

#### 2.2.2 高效稀疏路由与负载均衡

为了在硬件（如 TPU）上高效运行，Switch Transformer 必须解决动态路由带来的负载不均问题。由于硬件通常要求静态的 Tensor 形状，模型必须预设每个专家能处理的最大 Token 数量，即**专家容量（Expert Capacity）**：

$$ 
\text{Capacity} = \left( \frac{\text{TotalTokens}}{\text{NumExperts}} \right) \times \text{CapacityFactor} \tag{6.6}
$$

-   **Capacity Factor（容量因子）**：通常设置为大于 1.0（如 1.0 或 1.25），这一机制的作用如图 6-7 所示。图中每个方块代表专家的处理槽位，Capacity Factor > 1.0 为专家提供了额外的缓冲空间（图中白色空槽位），以应对 Token 分配不均的情况。
-   **Token Dropping（丢弃机制）**：当路由到某个专家的 Token 数量超过其容量上限（即公式计算出的 $Capacity$）时（图中红色虚线所示的溢出部分），就会触发丢弃机制。这些多余的 Token 将不会被该专家处理，而是直接通过残差连接传递到下一层。这虽然保证了并行计算的静态形状要求，但也可能导致信息损失，因此合理的容量设置至关重要。

<p align="center">
  <img src="./images/6_2_5.png" width="90%" alt="Token Routing Dynamics" />
  <br />
  <em>图 6-7：Token 路由动态与专家容量示意图</em>
</p>

同时，Switch Transformer 还引入了一个辅助损失函数来尽量减少 Token 的丢弃，鼓励 Token 均匀分布到所有专家：

$$ 
Loss_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i \tag{6.7}
$$

其中：
-   $f_i$ 是**实际**分发给专家 $i$ 的 Token 比例（实际上有多少人去了专家 $i$ 那里）。
-   $P_i$ 是**预期**路由给专家 $i$ 的概率总和（门控网络觉得专家 $i$ 应该接收多少人）。

这个公式希望“实际去的”和“计划去的”向量的点积最小。只有当两者都均匀分布（即所有专家的负载都相等）时，这个 Loss 才会达到最小值。这迫使 Router 不偏科，雨露均沾。

#### 2.2.3 改进的训练与微调技术

大规模稀疏模型训练极易不稳定，Switch Transformer 提出了一系列改进方案：

-   **Router z-loss**：
    为了提高训练稳定性，Switch Transformer 引入了 z-loss 来惩罚门控网络中过大的 logit 值。这有助于减少数值溢出问题，使训练过程更加平稳。

-   **选择性精度（Selective Precision）**：
    在混合精度训练（通常用 bfloat16）中，路由器的 Softmax 计算容易导致数值不稳定。Switch Transformer 创新地在**局部路由计算部分使用 float32**，而在其他部分保持 bfloat16。这既保证了稳定性，又没有增加昂贵的 float32 通信成本。

-   **更小的初始化方差**：
    将权重初始化的高斯分布标准差缩减为原来的 $1/10$（例如 $s=0.1$ 而非 $1.0$），显著提升了训练初期的稳定性。

-   **专家正则化（Expert Regularization）**：
    在微调阶段，为了防止过拟合（特别是专家层参数量巨大），模型对**专家层内部**采用了更高的 Dropout 比率（如 0.4），而非专家层保持较低比率（如 0.1）。

### 2.3 GLaM 高能效通用语言模型

**GLaM （Generalist Language Model）** [^6] 是 Google 在 2021 年推出的通用大语言模型。与 Switch Transformer 采用的 Encoder-Decoder 架构不同，GLaM 采用了与 GPT-3 相同的 **Decoder-only** 架构，这使其更适合于 Few-shot 和 Zero-shot 生成任务。

GLaM 将参数规模推向了 **1.2 万亿**，是 GPT-3（175B）的 7 倍。不过，得益于稀疏 MoE 架构，它在保证超大规模参数容量的同时，实现了比 GPT-3 更高的训练和推理效率。

#### 2.3.1 架构特点

GLaM 展示了如何将 MoE 层有效地应用于 **Decoder-only** 的语言模型中。如图 6-8 它采用了隔层替换策略，即在标准的 Transformer 堆叠中，每隔一个层（upper block）将其中的 FFN 替换为 MoE 层（bottom block）。

在 MoE 层中，Gating 模块会根据输入 Token（例如 "roses"）的特性，从 64 个专家中动态选择出**最相关**的 2 个专家（蓝色网格所示）。随后，这两个专家的输出经过加权平均后，传递给下一层的 Transformer 模块。这种机制确保了模型在拥有巨大参数量的同时，每次推理仅需激活极少部分的参数。

<p align="center">
  <img src="./images/6_2_6.png" width="40%" alt="GLaM 模型架构图" />
  <br />
  <em>图 6-8：GLaM 模型架构图</em>
</p>

-   **隔层稀疏**：类似于 GShard，GLaM 采用隔层替换策略，将每隔一个 Transformer 层中的前馈网络（FFN）替换为 MoE 层。
-   **Top-2 路由**：每个 MoE 层包含 64 个专家，对于每个输入 Token，门控网络会选择权重最高的 **2 个专家**进行处理。
-   **活跃参数**：尽管总参数量高达 1.2T，但对于每个 Token，仅激活 **966 亿（96.6B）** 参数（约占总量的 8%）。这意味着在推理时，GLaM 的计算量（FLOPs）仅为 GPT-3（175B 全激活）的约一半。

此外，GLaM 的研究团队发现，**高质量的数据**对于大模型的性能至关重要。他们开发了一套高质量的文本质量分类器，对原始网页数据进行了严格的过滤。实验表明，使用过滤后的高质量数据训练的模型，在各项任务上的表现均优于使用未过滤海量数据的模型。

#### 2.3.2 性能与能效对比

GLaM 的主要贡献在于证明了稀疏模型可以在减少计算资源消耗的同时，超越同等规模甚至更大规模的稠密模型。

如图 6-9，GLaM（绿色）在 Zero-shot (a)、One-shot (b) 和 Few-shot (c) 设置下，绝大多数任务上都全面优于 GPT-3（橙色）。特别是在图 (d) 的成本对比中，GLaM 的推理计算量仅为 GPT-3 的一半，训练总能耗更是大幅下降至 1/3。

<p align="center">
  <img src="./images/6_2_7.png" width="100%" alt="GLaM 与 GPT-3 性能与成本对比" />
  <br />
  <em>图 6-9：GLaM 与 GPT-3 在各任务性能及训练/推理成本上的详细对比</em>
</p>

表 6-2 进一步列出了具体的数值对比，直观地证明了 MoE 架构在实现高性能的同时，显著降低了算力成本。

<div align="center">

<table border="1" style="margin: 0 auto;">
<tr>
  <td style="text-align: center;"><strong>对比维度</strong></td>
  <td style="text-align: center;"><strong>指标</strong></td>
  <td style="text-align: center;"><strong>GPT-3 (175B)</strong></td>
  <td style="text-align: center;"><strong>GLaM (1.2T)</strong></td>
  <td style="text-align: center;"><strong>变化幅度</strong></td>
</tr>
<tr>
  <td rowspan="2" style="text-align: center;"><strong>成本 (Cost)</strong></td>
  <td style="text-align: center;">推理计算量 (FLOPs/token)</td>
  <td style="text-align: center;">350 G</td>
  <td style="text-align: center;">180 G</td>
  <td style="text-align: center;"><strong>-48.6%</strong></td>
</tr>
<tr>
  <td style="text-align: center;">训练能耗 (Energy)</td>
  <td style="text-align: center;">1287 MWh</td>
  <td style="text-align: center;">456 MWh</td>
  <td style="text-align: center;"><strong>-64.6%</strong></td>
</tr>
<tr>
  <td rowspan="3" style="text-align: center;"><strong>平均准确率<br>(Accuracy)</strong></td>
  <td style="text-align: center;">Zero-shot</td>
  <td style="text-align: center;">56.9</td>
  <td style="text-align: center;">62.7</td>
  <td style="text-align: center;"><strong>+10.2%</strong></td>
</tr>
<tr>
  <td style="text-align: center;">One-shot</td>
  <td style="text-align: center;">61.6</td>
  <td style="text-align: center;">65.5</td>
  <td style="text-align: center;"><strong>+6.3%</strong></td>
</tr>
<tr>
  <td style="text-align: center;">Few-shot</td>
  <td style="text-align: center;">65.2</td>
  <td style="text-align: center;">68.1</td>
  <td style="text-align: center;"><strong>+4.4%</strong></td>
</tr>
</table>

<p><em>表 6-2：GLaM 与 GPT-3 在成本与性能上的对比</em></p>

</div>

**MoE 架构为实现“更大、更强、更环保”的模型提供了一条极具潜力的技术路径**。这打破了以往“性能提升必须依靠堆砌更多算力”的固有认知。

## 三、MoE 架构的创新与实践

随着开源社区的活跃，MoE 技术不再是科技巨头的专属。**Mistral 8x7B** 和 **DeepSeek-R1** 的出现，分别在中等规模和超大规模上证明了开源 MoE 模型的强大实力，标志着 MoE 技术进入了全面普及和深度创新的新阶段。

### 3.1 Mistral 8x7B 如何以小博大

#### 3.1.1 架构与性能概览

**Mistral 8x7B (Mixtral)** [^7] 在开源大语言模型中成功实践了 MoE 架构，有力地证明了合理设计的稀疏模型即使不需要万亿参数，也能超越同量级的稠密模型。

-   **架构参数**：它拥有约 **470 亿（47B）** 的总参数量（Sparse Parameters），但对于每个 Token，仅激活 **130 亿（13B）** 参数（Active Parameters）。这使得它在推理时拥有 13B 模型的计算速度，却能发挥出 47B 模型的知识容量。**需要注意的是，虽然计算量较小，但由于所有专家参数都需要加载到内存中，其显存占用（VRAM Usage）依然是 47B 模型级别的。**
-   **路由机制**：每一层包含 **8 个专家**（Experts），采用标准的 **Top-2 Routing** 策略。如图 6-10 所示，每个输入 Token 会被 Router 网络分配给 8 个专家中的 2 个，这两个专家的输出经过加权求和后作为该层的最终输出。这种机制巧妙地在增加模型容量（更多专家）的同时，保持了极低的推理成本（只激活 2 个）。

    <p align="center">
      <img src="./images/6_2_8.png" width="90%" alt="Mixture of Experts Layer" />
      <br />
      <em>图 6-10：Mistral 8x7B 的 Top-2 路由机制示意图</em>
    </p>
-   **性能表现**：在 GSM8K（数学）、MMLU（综合知识）、HumanEval（代码）等基准测试上，Mistral 8x7B 以 13B 的活跃参数量超越了稠密的 **Llama 2 70B** 以及 **GPT-3.5**。如图 6-11，Mistral 8x7B（黄色柱状图）在几乎所有任务上都包围或持平了 Llama 2 70B（绿色柱状图），特别是在数学和代码生成任务上，其优势尤为显著。
-   **长上下文能力**：Mistral 8x7B 支持 **32k** 的上下文长度，并且在长文本信息检索（Passkey Retrieval）任务中表现出了 100% 的召回率，证明了 MoE 架构在处理长序列时依然稳健。

    <p align="center">
      <img src="./images/6_2_9.png" width="90%" alt="Mistral 8x7B 性能对比" />
      <br />
      <em>图 6-11：Mistral 8x7B 与 Llama 2 系列在各基准测试上的性能对比</em>
    </p>

#### 3.1.2 路由机制分析

Mistral 团队对 Router 选择专家的行为进行了深入分析，得到了一个令人惊讶的结论。**专家并没有按预想的那样根据“学科领域”（如生物、数学、哲学）进行分工**。

他们统计了不同领域数据（如 arXiv, PubMed, Wikipedia 等）在不同层（Layer 0, 15, 31）的专家分配比例。如图 6-12 可以看出，同一行（即同一个专家）在不同列（不同数据集）上的颜色深浅非常接近。这说明，**无论输入文本属于哪个领域，Router 选择各专家的概率分布几乎是一样的**。专家似乎更多地是根据**语法**和**Token 结构**（如缩进、介词）来分工，而非人类定义的知识领域。

<p align="center">
  <img src="./images/6_2_10.png" width="90%" alt="Mistral 8x7B 路由专家分布" />
  <br />
  <em>图 6-12：不同领域数据在 Mistral 8x7B 各层中的专家路由分布（显示出无领域偏差的特性）</em>
</p>

### 3.2 DeepSeekMoE 与 DeepSeek-R1

如果说 Mistral 开启了开源 MoE 模型的大门，那么 **DeepSeek-R1** [^8]（及其基座 **DeepSeek-V3** [^9]）则将开源 MoE 模型的性能推向了与当时顶尖闭源模型（如 OpenAI o1）比肩的高度。

DeepSeek 在 MoE 架构上进行了更深度的创新，提出了 **DeepSeekMoE** [^10] 架构，目标是解决传统 Top-k 路由中的“知识冗余”和“专业化不足”问题。

#### 3.2.1 细粒度专家与共享专家

与 Mistral 采用的“粗粒度”专家不同，如图 6-13 所示，DeepSeekMoE 引入了两个关键策略。

<p align="center">
  <img src="./images/6_2_11.png" width="80%" alt="DeepSeekMoE 架构图" />
  <br />
  <em>图 6-13：DeepSeekMoE 架构演进：(a) 传统 Top-2 路由; (b) 细粒度专家分割; (c) 细粒度 + 共享专家隔离（最终架构）</em>
</p>

-  **细粒度专家分割（Fine-Grained Expert Segmentation）**：
    DeepSeek 将一个标准的大专家拆分为多个更小的专家。对比图 6-13 的 (a) 和 (b) 可以看到，原本的专家 1 被进一步拆分为更小的专家 1 和 2。为了保持总计算量不变，激活的专家数量 $K$ 也相应倍增（从 $K=2$ 变为 $K=4$）。这种变化使得组合的可能性呈指数级增加，让模型能更灵活地组合不同的“知识碎片”来应对复杂输入，从而实现了更高的专家专业化。

-  **共享专家隔离（Shared Expert Isolation）**：
    这是 DeepSeekMoE 的核心创新。如图 6-13(c) 所示，专家 1 被指定为绿色的**共享专家（Shared Expert）**。它不再经过 Router 选择，而是通过一条独立的通路直接接收输入（Input Hidden），对所有 Token 总是被激活。Router 仅负责从剩余的路由专家中选择 $K=3$ 个进行补充。
    
    这种设计让共享专家负责捕获通用的、跨任务的知识（如语法），而路由专家则专注于特定的领域知识。通过这种“通用+专用”的分离，有效减少了路由专家中重复学习通用知识的冗余，显著提升了参数效率。

    可以用公式统一表示为：

    $$ \mathbf{y} = \underbrace{\sum_{i \in \mathcal{S}} \mathbf{E}_i(\mathbf{x})}_{\text{Shared Experts}} + \underbrace{\sum_{j \in \text{TopK}(\mathcal{R})} p_j(\mathbf{x}) \cdot \mathbf{E}_j(\mathbf{x})}_{\text{Routed Experts}} \tag{6.8} $$

    其中 $\mathcal{S}$ 代表**共享专家**集合（总是被激活），$\mathcal{R}$ 代表**路由专家**集合（仅选择性激活）。这种双路径结构是其区别于传统 MoE（公式 6.4）的关键。

#### 3.2.2 性能里程碑

DeepSeek-R1 不仅在常规任务上表现出色，更通过**大规模强化学习** 具备了强大的逻辑推理能力。

DeepSeek-R1 在 AIME 2024（数学竞赛）上 Pass@1 准确率达到 79.8%，稍高于 OpenAI-o1-1217；在 MATH-500 上达到 97.3%，与 o1 持平。在 Codeforces 编程竞赛中，其 Elo 等级分达到 2029，超过了 96.3% 的人类参赛者。如图 6-14 所示，DeepSeek-R1（深蓝色柱状图）在多个推理密集型基准测试中均展现出了与顶尖闭源模型（如 OpenAI-o1-1217，灰色柱状图）分庭抗礼的实力。

<p align="center">
  <img src="./images/6_2_12.png" width="80%" alt="DeepSeek-R1 性能对比" />
  <br />
  <em>图 6-14：DeepSeek-R1 在数学、代码及知识类基准测试上的性能表现</em>
</p>

---

## 参考文献

[^1]: [Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). *Adaptive mixtures of local experts*.](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)

[^2]: [Eigen, D., Ranzato, M., & Sutskever, I. (2013). *Learning Factored Representations in a Deep Mixture of Experts*.](https://arxiv.org/abs/1312.4314)

[^3]: [Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*.](https://arxiv.org/abs/1701.06538)

[^4]: [Lepikhin, D., Lee, H., Xu, Y., et al. (2020). *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*.](https://arxiv.org/abs/2006.16668)

[^5]: [Fedus, W., Zoph, B., & Shazeer, N. (2021). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*.](https://arxiv.org/abs/2101.03961)

[^6]: [Du, N., Huang, Y., Dai, A. M., et al. (2021). *GLaM: Efficient Scaling of Language Models with Mixture-of-Experts*.](https://arxiv.org/abs/2112.06905)

[^7]: [Jiang, A. Q., Sablayrolles, A., Roux, A., et al. (2024). *Mixtral of Experts*.](https://arxiv.org/abs/2401.04088)

[^8]: [DeepSeek-AI. (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*.](https://arxiv.org/abs/2501.12948)

[^9]: [DeepSeek-AI. (2024). *DeepSeek-V3 Technical Report*.](https://arxiv.org/abs/2412.19437)

[^10]: [Dai, D., Deng, C., Zhao, C., et al. (2024). *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*.](https://arxiv.org/abs/2401.06066)

