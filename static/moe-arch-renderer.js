/**
 * MoE Architecture Renderer — D3-based Mixture-of-Experts transformer diagram.
 *
 * Renders a vertical MoE Transformer architecture diagram, replacing the standard
 * dense FFN block with an MoE block (router + expert columns).
 *
 * Public API (attached to window):
 *   renderMoeArchitecture(g, model, options) — draw the MoE diagram into a D3 group
 *   calcMoeArchHeight(config, availableWidth)  — compute needed height
 *
 * Reads MoE params from model.config:
 *   model_type, num_experts, moe_router_topk, num_moe_layers,
 *   moe_ffn_hidden_size, has_shared_expert, d_model, d_ffn, num_heads
 */

(function () {
  "use strict";

  // ═══════════════════════════════════════════════════════════════
  // Layout constants — design-space coordinates, uniformly scaled
  // ═══════════════════════════════════════════════════════════════

  var MOE_DESIGN = {
    W: 540,          // design canvas width
    CX: 300,         // center X
    BOX_W: 275,      // wide block width (embeddings, output)
    NARROW_W: 140,   // narrow block width (LayerNorm, Add)
    BLOCK_W: 260,    // sub-block width (Attention, MoE)
    SUB_W: 185,      // inner sub-item width
    H_SM: 28,        // small block height (LN, Add)
    H_MD: 36,        // medium block height (embeddings, output)
    H_ATTN: 100,     // attention sub-block height
    H_MOE: 240,      // base MoE block height (grows with experts)
    ARROW_S: 14,     // arrow icon size
    // Tensor grid
    TENSOR_W: 120,
    TENSOR_H: 120,
    TENSOR_X: 0,   // left-aligned: grid 0-120, card starts at 150 (gap=30)
    TENSOR_Y: 58,  // aligned with transformer card top (Y_HEADER=56)
    // Y-layout chain
    Y_EMBED: 0,
    Y_HEADER: 56,
    Y_LN1: 80,
    Y_ATTN: 120,
    Y_ADD1: 232,
    Y_LN2: 272,
    Y_MOE: 312,
    // Y_ADD2 and below are computed dynamically based on MoE block height
    Y_TF_END: 660,
    Y_LN3: 680,
    Y_OUTPUT: 720,
  };

  // ═══════════════════════════════════════════════════════════════
  // Tooltip definitions — MoE-specific architecture concept tips
  // ═══════════════════════════════════════════════════════════════

  var _MOE_TIPS = {
    transformer_card: {
      t: "Transformer层（堆叠N次）",
      te: "Transformer Layer (stacked N times)",
      d: "整个Transformer模型由N个相同的层堆叠而成。对于MoE模型，部分层将标准FFN替换为MoE FFN（稀疏激活），其余层保持Dense FFN。图中num_moe_layers指示MoE层数。",
      m: "Layer_i(x) = AddNorm_MoE-FFN/FFN(AddNorm_Attn(x)) for i ∈ [1, N]",
      v: "DeepSeek-R1: 61层, MoE层58层 | Mixtral 8×7B: 32层, MoE层全交替",
    },
    embeddings: {
      t: "位置编码 + 词嵌入 + Dropout",
      te: "Positional Encoding + Word Embeddings + Dropout",
      d: "将离散token ID映射为稠密向量（维度d_model），注入位置信息后应用Dropout正则化。现代模型使用可学习位置编码或RoPE。",
      m: "H_0 = Dropout(WordEmbed(token_ids) + PositionEncode(positions))",
      v: "d_model: 7168(DeepSeek-R1) / 4096(Mixtral 8×7B)",
    },
    layer_norm_1: {
      t: "层归一化（注意力前）",
      te: "Layer Normalization (pre-attention)",
      d: "Pre-LN架构中的第一个Layer Norm。对每个token特征向量独立归一化（均值为0，方差为1），再通过可学习γ/β参数仿射变换。",
      m: "LN(x) = γ ⊙ ((x - μ) / √(σ² + ε)) + β",
      v: "参数量: 2 × d_model per LN",
    },
    multi_head_attention: {
      t: "多头自注意力机制",
      te: "Multi-Head Self-Attention",
      d: "自注意力是Transformer的核心，允许每个token关注所有其他token捕获上下文依赖。多头机制将Q/K/V投影到多个子空间并行计算，最后拼接输出。",
      m: "MultiHead(Q,K,V) = Concat(head_1,...,head_h)W_O, head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)",
      v: "head数: 128(DeepSeek-R1) / 32(Mixtral)",
    },
    add_1: {
      t: "残差加法（注意力残差）",
      te: "Residual addition (attention residual)",
      d: "将注意力输出与输入逐元素相加。残差连接为梯度提供直通路径（identity shortcut），是深层Transformer训练的关键。",
      m: "x = x + MHA(LN_1(x))",
      v: "",
    },
    layer_norm_2: {
      t: "层归一化（MoE/FFN前）",
      te: "Layer Normalization (pre-MoE/FFN)",
      d: "第二个Layer Norm，位于MoE/FFN子层之前。将残差加法后的隐表示归一化到标准分布，为后续前馈网络提供数值稳定的输入。",
      m: "LN(x) = γ ⊙ ((x - μ) / √(σ² + ε)) + β",
      v: "",
    },
    moe_ffn_block: {
      t: "MoE前馈网络块",
      te: "Mixture-of-Experts FFN Block",
      d: "MoE层用稀疏激活的前馈网络替代标准Dense FFN。每个token通过路由器选择Top-K个专家进行计算，各专家独立输出后加权求和。相比Dense FFN，MoE在相同计算量下可大幅提升模型容量（参数量）。",
      m: "MoE-FFN(x) = Σ_{i∈TopK(x)} softmax(Router(x))_i · Expert_i(x)",
      v: "DeepSeek-R1: 256专家×Top-8, 参数量685B(激活37B) | Mixtral: 8专家×Top-2",
    },
    moe_router: {
      t: "MoE路由器（Top-K + Softmax）",
      te: "MoE Router — Top-K Selection + Softmax",
      d: "路由器是一个轻量级线性层，输入token隐表示输出每个专家的「亲和度」分数。通过Top-K筛选激活分数最高的K个专家，再Softmax归一化为路由权重。未被选中的专家在本token计算中完全不参与（稀疏激活）。",
      m: "Router(x) = softmax(TopK(W_r · x)), W_r ∈ R^{d_model × num_experts}",
      v: "Top-K: 8(DeepSeek-R1) / 2(Mixtral) | 路由参数: d_model × num_experts",
    },
    moe_expert: {
      t: "MoE专家（独立FFN）",
      te: "MoE Expert (independent FFN)",
      d: "每个专家是一个独立的标准FFN：Linear上投影→GeLU激活→Linear下投影→Dropout。各专家参数独立、互不共享。被选中的专家并行计算，输出按路由权重加权求和送至后续Add。",
      m: "Expert_i(x) = Dropout(W_i^{down}(GeLU(W_i^{up} x)))",
      v: "每个专家FFN维度: 2048(DeepSeek-R1) | 单个专家参数量: 3×d_model×d_ffn",
    },
    moe_shared_expert: {
      t: "共享专家（始终激活）",
      te: "Shared Expert (always active)",
      d: "共享专家对所有token始终激活，不经过路由器选择。其输出直接与Top-K路由专家的加权输出相加。共享专家的引入可缓解某些token「无专家可选」的问题，并为所有token提供公共知识表示。",
      m: "MoE-FFN(x) = Shared(x) + Σ_{i∈TopK(x)} w_i · Expert_i(x)",
      v: "DeepSeek-R1: 1个共享专家 | 共享专家与路由专家结构相同",
    },
    add_2: {
      t: "残差加法（MoE/FFN残差）",
      te: "Residual addition (MoE/FFN residual)",
      d: "将MoE/FFN输出与输入逐元素相加。本层的第二个残差连接，与注意力残差共同构成Transformer层的梯度高速通道。",
      m: "x = x + MoE-FFN(LN_2(x))",
      v: "",
    },
    final_layer_norm: {
      t: "最终层归一化（输出前）",
      te: "Final Layer Normalization (pre-output)",
      d: "所有Transformer层堆叠完成后的最后一个Layer Norm，不属于任何单层内部。归一化后传递给输出层。",
      m: "H_final = LayerNorm(H_N)",
      v: "",
    },
    output_layer: {
      t: "输出层与损失函数",
      te: "Output Layer & Loss",
      d: "将d_model维度的隐表示映射到vocab_size维度得到logits。训练时计算交叉熵损失。LM Head权重常与输入Embedding共享（weight tying）。",
      m: "logits = H_final · W_embed^T, Loss = -Σ y_i log(softmax(logits_i))",
      v: "vocab_size: 129280(DeepSeek-R1)",
    },
    tensor_grid: {
      t: "张量并行网格",
      te: "Tensor Parallelism grid",
      d: "将TP维度可视化。TP将单层参数矩阵按列或行切分到多个NPU，各NPU独立计算分片后通过AllReduce通信聚合。橙色高亮格表示当前NPU分片。",
      m: "",
      v: "切分方式: 列切分/行切分 | 通信: AllReduce",
    },
    residual_connection: {
      t: "残差连接（Skip Connection）",
      te: "Residual / Skip Connection",
      d: "图中橙色虚线标注的残差连接，从子层输入分叉直接连接到Add节点。残差连接为反向传播梯度提供恒等映射路径，使深层模型可训练。",
      m: "output = SubLayer(x) + x",
      v: "",
    },
  };

  // ═══════════════════════════════════════════════════════════════
  // Helper — draw a labeled rounded rectangle
  // ═══════════════════════════════════════════════════════════════

  function drawBox(g, x, y, w, h, label, fill, stroke, textColor, cls, tipId, hoverHelper) {
    var bg = g.append("g");
    var rect = bg.append("rect")
      .attr("x", x).attr("y", y)
      .attr("width", w).attr("height", h)
      .attr("rx", 4).attr("ry", 4)
      .attr("fill", fill)
      .attr("stroke", stroke)
      .attr("stroke-width", 1.2)
      .attr("class", (cls || "") + " moe-node-rect");
    if (tipId && hoverHelper) hoverHelper(rect, 1.2, tipId);
    bg.append("text")
      .attr("x", x + w / 2)
      .attr("y", y + h / 2 + 1)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .attr("font-size", "11px")
      .attr("font-family", "var(--font-sans)")
      .attr("font-weight", "500")
      .attr("fill", textColor)
      .text(label);
    return bg;
  }

  // ═══════════════════════════════════════════════════════════════
  // Helper — draw a sub-block with title and inner items
  // ═══════════════════════════════════════════════════════════════

  function drawSubBlock(g, x, y, w, h, title, subItems, fill, stroke, titleColor, itemBg, itemStroke, tipId, hoverHelper) {
    var sg = g.append("g");
    var rect = sg.append("rect")
      .attr("x", x).attr("y", y)
      .attr("width", w).attr("height", h)
      .attr("rx", 4).attr("ry", 4)
      .attr("fill", fill)
      .attr("stroke", stroke)
      .attr("stroke-width", 1.2)
      .attr("class", "moe-node-rect");
    if (tipId && hoverHelper) hoverHelper(rect, 1.2, tipId);

    sg.append("text")
      .attr("x", x + w / 2)
      .attr("y", y + 14)
      .attr("text-anchor", "middle")
      .attr("font-size", "10px")
      .attr("font-family", "var(--font-sans)")
      .attr("font-weight", "600")
      .attr("fill", titleColor)
      .text(title);

    var innerX = x + (w - MOE_DESIGN.SUB_W) / 2;
    var innerStartY = y + 26;
    var innerH = 20;
    var innerGap = 4;

    subItems.forEach(function (item, i) {
      var iy = innerStartY + i * (innerH + innerGap);
      sg.append("rect")
        .attr("x", innerX).attr("y", iy)
        .attr("width", MOE_DESIGN.SUB_W).attr("height", innerH)
        .attr("rx", 3).attr("ry", 3)
        .attr("fill", itemBg)
        .attr("stroke", itemStroke)
        .attr("stroke-width", 0.8)
        .attr("stroke-opacity", 0.5);
      sg.append("text")
        .attr("x", innerX + MOE_DESIGN.SUB_W / 2)
        .attr("y", iy + innerH / 2 + 1)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .attr("font-size", "9px")
        .attr("font-family", "var(--font-mono)")
        .attr("font-weight", "500")
        .attr("fill", "var(--text-secondary)")
        .text(item);
    });
    return sg;
  }

  // ═══════════════════════════════════════════════════════════════
  // Helper — draw a vertical flow arrow between blocks
  // ═══════════════════════════════════════════════════════════════

  function drawFlowArrow(g, cx, topY, bottomY, color) {
    color = color || "var(--green)";
    var midY = (topY + bottomY) / 2;
    // Arrow head (triangle pointing down)
    var arrowH = 8;
    var arrowW = 5;
    g.append("line")
      .attr("x1", cx).attr("y1", topY)
      .attr("x2", cx).attr("y2", bottomY - arrowH)
      .attr("stroke", color)
      .attr("stroke-width", 1.2)
      .attr("stroke-opacity", 0.7);
    g.append("polygon")
      .attr("points",
        cx + "," + (bottomY) + " " +
        (cx - arrowW) + "," + (bottomY - arrowH) + " " +
        (cx + arrowW) + "," + (bottomY - arrowH))
      .attr("fill", color)
      .attr("opacity", 0.7);
  }

  // ═══════════════════════════════════════════════════════════════
  // Helper — dashed line
  // ═══════════════════════════════════════════════════════════════

  function drawDashedLine(g, x1, y1, x2, y2, color) {
    g.append("line")
      .attr("x1", x1).attr("y1", y1)
      .attr("x2", x2).attr("y2", y2)
      .attr("stroke", color)
      .attr("stroke-width", 1.2)
      .attr("stroke-dasharray", "4 3")
      .attr("class", "moe-residual-line");
  }

  // ═══════════════════════════════════════════════════════════════
  // Pre-compute MoE block height in design space (no rendering)
  // Used by both renderMoeArchitecture and calcMoeArchHeight so the
  // transformer-layer card can be drawn before the inner blocks.
  // ═══════════════════════════════════════════════════════════════

  function calcMoeBlockDesignH(config) {
    var hasShared = config.has_shared_expert === true || config.has_shared_expert === 1;
    var routerH = 56;
    var expertH = 75;
    var sharedExpertH = 50;
    var moePadTop = 22;
    var moePadBottom = 14;
    return routerH + 8 + expertH + moePadTop + moePadBottom + (hasShared ? sharedExpertH + 8 : 0);
  }

  // ═══════════════════════════════════════════════════════════════
  // Render the MoE FFN block (router + expert columns)
  // ═══════════════════════════════════════════════════════════════

  function renderMoEBlock(g, moeX, moeY, moeW, config, scale, hoverHelper) {
    scale = scale || 1;
    var numExperts = config.num_experts || 8;
    var topK = config.moe_router_topk || 2;
    var ffnDim = config.moe_ffn_hidden_size || config.d_ffn || 0;
    var hasShared = config.has_shared_expert === true || config.has_shared_expert === 1;
    var dModel = config.d_model || 4096;
    var CX = moeX + moeW / 2;  // scaled center of the MoE block

    // Format ffn dimension label
    var ffnLabel = ffnDim ? ffnDim : "4h";
    if (typeof ffnLabel === "number" && ffnLabel > 999) {
      ffnLabel = (ffnLabel / 1024).toFixed(1) + "K";
    }

    // ── Determine how many expert columns to show ──
    var displayExperts = numExperts;
    var showEllipsis = false;
    if (numExperts > 4) {
      displayExperts = 2; // Show Expert 1, ..., Expert E
      showEllipsis = true;
    }

    // ── Compute MoE block height (scaled) ──
    var s = scale;
    var routerH = 56 * s;  // Top-K + Softmax
    var expertH = 75 * s; // Each expert column
    var moePadTop = 22 * s;
    var moePadBottom = 14 * s;
    var sharedExpertH = 50 * s;
    var moeInnerH = routerH + 8 * s + expertH + moePadTop + moePadBottom;
    if (hasShared) moeInnerH += sharedExpertH + 8 * s;

    // ── Outer MoE block rect ──
    var moeOuterRect = g.append("rect")
      .attr("x", moeX).attr("y", moeY)
      .attr("width", moeW).attr("height", moeInnerH)
      .attr("rx", 4 * s).attr("ry", 4 * s)
      .attr("fill", "#1a1114")
      .attr("stroke", "var(--red)")
      .attr("stroke-width", 1.2 * s)
      .attr("class", "moe-node-rect");
    if (hoverHelper) hoverHelper(moeOuterRect, 1.2 * s, "moe_ffn_block");

    g.append("text")
      .attr("x", CX)
      .attr("y", moeY + 14 * s)
      .attr("text-anchor", "middle")
      .attr("font-size", Math.max(8, 10 * s) + "px")
      .attr("font-family", "var(--font-sans)")
      .attr("font-weight", "600")
      .attr("fill", "var(--red)")
      .text("Mixture-of-Experts (MoE) FFN Block");

    // ── Router: Top-K + Softmax ──
    var routerW = 210 * s;
    var routerX = CX - routerW / 2;
    var routerItemH = 22 * s;
    var routerY = moeY + moePadTop;

    // Top-K
    g.append("rect")
      .attr("x", routerX).attr("y", routerY)
      .attr("width", routerW).attr("height", routerItemH)
      .attr("rx", 3 * s).attr("ry", 3 * s)
      .attr("fill", "#1a1528")
      .attr("stroke", "var(--purple)")
      .attr("stroke-width", 0.8 * s)
      .call(function (r) { if (hoverHelper) hoverHelper(r, 0.8 * s, "moe_router"); });
    g.append("text")
      .attr("x", CX).attr("y", routerY + routerItemH / 2 + 1)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .attr("font-size", Math.max(7, 9 * s) + "px")
      .attr("font-family", "var(--font-mono)")
      .attr("fill", "var(--purple)")
      .text("Top-K Selection  (K=" + topK + ")");

    // Softmax
    var softmaxY = routerY + routerItemH + 4 * s;
    g.append("rect")
      .attr("x", routerX).attr("y", softmaxY)
      .attr("width", routerW).attr("height", routerItemH)
      .attr("rx", 3 * s).attr("ry", 3 * s)
      .attr("fill", "#1a1528")
      .attr("stroke", "var(--purple)")
      .attr("stroke-width", 0.8 * s)
      .call(function (r) { if (hoverHelper) hoverHelper(r, 0.8 * s, "moe_router"); });
    g.append("text")
      .attr("x", CX).attr("y", softmaxY + routerItemH / 2 + 1)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .attr("font-size", Math.max(7, 9 * s) + "px")
      .attr("font-family", "var(--font-mono)")
      .attr("fill", "var(--purple)")
      .text("Softmax");

    // ── Small arrow from Softmax down to experts ──
    var expertStartY = softmaxY + routerItemH + 10 * s;
    var _arrowTop = softmaxY + routerItemH;
    var _arrowBot = expertStartY;
    // Simple vertical line + small triangle
    g.append("line")
      .attr("x1", CX).attr("y1", _arrowTop)
      .attr("x2", CX).attr("y2", _arrowBot - 6 * s)
      .attr("stroke", "var(--purple)")
      .attr("stroke-width", 1 * s)
      .attr("stroke-opacity", 0.7);
    g.append("polygon")
      .attr("points",
        CX + "," + _arrowBot + " " +
        (CX - 4 * s) + "," + (_arrowBot - 6 * s) + " " +
        (CX + 4 * s) + "," + (_arrowBot - 6 * s))
      .attr("fill", "var(--purple)")
      .attr("opacity", 0.7);

    // ── Expert columns ──
    var maxExpertW = 80 * s;
    var expertW = Math.min(maxExpertW, (moeW - 40 * s) / Math.max(displayExperts, 1));
    var expertGap = showEllipsis ? 50 * s : 8 * s;
    var totalExpertW = displayExperts * expertW + (displayExperts - 1) * expertGap;
    var expertStartX = CX - totalExpertW / 2;

    var expertItems = [
      "Linear (h → " + ffnLabel + ")",
      "GeLU",
      "Linear (" + ffnLabel + " → h)",
      "Dropout",
    ];

    for (var e = 0; e < displayExperts; e++) {
      var ex = expertStartX + e * (expertW + expertGap);
      // Highlight: when collapsed, no selection; otherwise first topK are cyan
      var isSelected = showEllipsis ? false : (e < topK);

      // Expert column rect
      var expertRect = g.append("rect")
        .attr("x", ex).attr("y", expertStartY)
        .attr("width", expertW).attr("height", expertH)
        .attr("rx", 4 * s).attr("ry", 4 * s)
        .attr("fill", isSelected ? "#1a2030" : "#151018")
        .attr("stroke", isSelected ? "var(--cyan)" : "var(--red)")
        .attr("stroke-width", (isSelected ? 1.5 : 0.8) * s)
        .attr("class", isSelected ? "moe-expert-selected" : "moe-expert-unselected");
      if (hoverHelper) hoverHelper(expertRect, (isSelected ? 1.5 : 0.8) * s, "moe_expert");

      // Expert label
      var eLabel;
      if (e === displayExperts - 1 && showEllipsis) {
        eLabel = "Expert " + numExperts;
      } else {
        eLabel = "Expert " + (e + 1);
      }
      g.append("text")
        .attr("x", ex + expertW / 2)
        .attr("y", expertStartY + 12 * s)
        .attr("text-anchor", "middle")
        .attr("font-size", Math.max(6, 8 * s) + "px")
        .attr("font-family", "var(--font-sans)")
        .attr("font-weight", "600")
        .attr("fill", isSelected ? "var(--cyan)" : "var(--red)")
        .text(eLabel);

      // Expert sub-items
      var itemH = Math.max(6, 10 * s);
      var itemGap = Math.max(0, 1 * s);
      var itemStartY_d = expertStartY + 18 * s;
      expertItems.forEach(function (item, i) {
        var iy = itemStartY_d + i * (itemH + itemGap);
        g.append("text")
          .attr("x", ex + expertW / 2)
          .attr("y", iy + itemH / 2)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("font-size", Math.max(5, 7 * s) + "px")
          .attr("font-family", "var(--font-mono)")
          .attr("fill", "var(--text-secondary)")
          .text(item);
      });

      // Ellipsis between Expert 1 and Expert E
      if (showEllipsis && e === 0) {
        g.append("text")
          .attr("x", ex + expertW + expertGap / 2)
          .attr("y", expertStartY + expertH / 2)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("font-size", Math.max(7, 10 * s) + "px")
          .attr("font-family", "var(--font-mono)")
          .attr("fill", "var(--text-muted)")
          .text("......");
      }
    }

    // ── Shared Expert (if enabled) — second row, wider/shorter, 2-column items ──
    if (hasShared) {
      var seW = totalExpertW;
      var seH = sharedExpertH;
      var seX = expertStartX;
      var seY = expertStartY + expertH + 8 * s;

      // Shared expert box
      var seRect = g.append("rect")
        .attr("x", seX).attr("y", seY)
        .attr("width", seW).attr("height", seH)
        .attr("rx", 4 * s).attr("ry", 4 * s)
        .attr("fill", "#1a2a18")
        .attr("stroke", "var(--green)")
        .attr("stroke-width", 1.5 * s)
        .attr("class", "moe-node-rect");
      if (hoverHelper) hoverHelper(seRect, 1.5 * s, "moe_shared_expert");

      // Label
      g.append("text")
        .attr("x", seX + seW / 2)
        .attr("y", seY + 12 * s)
        .attr("text-anchor", "middle")
        .attr("font-size", Math.max(6, 8 * s) + "px")
        .attr("font-family", "var(--font-sans)")
        .attr("font-weight", "600")
        .attr("fill", "var(--green)")
        .text("Shared Expert");

      // Items in 2-column × 2-row layout
      var seItemStartY = seY + 16 * s;
      var seItemH = Math.max(6, 10 * s);
      var seHalfW = seW / 2;
      expertItems.forEach(function (item, i) {
        var col = i % 2;
        var row = Math.floor(i / 2);
        g.append("text")
          .attr("x", seX + col * seHalfW + seHalfW / 2)
          .attr("y", seItemStartY + row * seItemH + seItemH / 2)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("font-size", Math.max(5, 7 * s) + "px")
          .attr("font-family", "var(--font-mono)")
          .attr("fill", "var(--text-secondary)")
          .text(item);
      });
    }

    // Return the total height of the MoE block
    return moeInnerH;
  }

  // ═══════════════════════════════════════════════════════════════
  // Render residual / skip connections (right side)
  // All coordinates must be PRE-SCALED by the caller using sx()/sy()
  // ═══════════════════════════════════════════════════════════════

  function renderResiduals(g, scaledCX, scaledBoxHalfW, scaledNarrowHalfW, scaledSkipRight,
                           skip1StartY, skip1EndY, skip2StartY, skip2EndY, scale) {
    var skipColor = "var(--orange)";
    scale = scale || 1;

    // Skip 1: from before LN1 to Add1
    var skip1EndX = scaledCX + scaledNarrowHalfW;
    drawDashedLine(g, scaledCX, skip1StartY, scaledSkipRight, skip1StartY, skipColor);
    drawDashedLine(g, scaledSkipRight, skip1StartY, scaledSkipRight, skip1EndY, skipColor);
    drawDashedLine(g, scaledSkipRight, skip1EndY, skip1EndX + 6 * scale, skip1EndY, skipColor);

    // Arrow at end
    g.append("polygon")
      .attr("points",
        skip1EndX + "," + skip1EndY + " " +
        (skip1EndX - 5 * scale) + "," + (skip1EndY - 3 * scale) + " " +
        (skip1EndX - 5 * scale) + "," + (skip1EndY + 3 * scale))
      .attr("fill", skipColor)
      .attr("opacity", 0.7);

    var skip1MidY = skip1StartY + (skip1EndY - skip1StartY) / 2;
    g.append("text")
      .attr("text-anchor", "middle")
      .attr("font-size", Math.max(7, 8 * scale) + "px")
      .attr("font-family", "var(--font-sans)")
      .attr("font-style", "italic")
      .attr("fill", skipColor)
      .attr("transform", "translate(" + (scaledSkipRight + 9 * scale) + ", " + skip1MidY + ") rotate(-90)")
      .text("residual");

    // Skip 2: from before LN2 to Add2
    var skip2EndX = scaledCX + scaledNarrowHalfW;
    drawDashedLine(g, scaledCX, skip2StartY, scaledSkipRight, skip2StartY, skipColor);
    drawDashedLine(g, scaledSkipRight, skip2StartY, scaledSkipRight, skip2EndY, skipColor);
    drawDashedLine(g, scaledSkipRight, skip2EndY, skip2EndX + 6 * scale, skip2EndY, skipColor);

    g.append("polygon")
      .attr("points",
        skip2EndX + "," + skip2EndY + " " +
        (skip2EndX - 5 * scale) + "," + (skip2EndY - 3 * scale) + " " +
        (skip2EndX - 5 * scale) + "," + (skip2EndY + 3 * scale))
      .attr("fill", skipColor)
      .attr("opacity", 0.7);

    var skip2MidY = skip2StartY + (skip2EndY - skip2StartY) / 2;
    g.append("text")
      .attr("text-anchor", "middle")
      .attr("font-size", Math.max(7, 8 * scale) + "px")
      .attr("font-family", "var(--font-sans)")
      .attr("font-style", "italic")
      .attr("fill", skipColor)
      .attr("transform", "translate(" + (scaledSkipRight + 9 * scale) + ", " + skip2MidY + ") rotate(-90)")
      .text("residual");
  }

  // ═══════════════════════════════════════════════════════════════
  // Render tensor grid (left side)
  // ═══════════════════════════════════════════════════════════════

  function renderTensorGrid(g, tensorX, tensorY, tensorW, tensorH, tpCount, ppCount, highlightTpIdx, highlightPpIdx) {
    var effectiveTp = tpCount || 4;
    var gridCols = Math.ceil(Math.sqrt(effectiveTp));
    var gridRows = Math.ceil(effectiveTp / gridCols);
    var cellW = tensorW / gridCols;
    var cellH = tensorH / gridRows;

    // Border
    g.append("rect")
      .attr("x", tensorX).attr("y", tensorY)
      .attr("width", tensorW).attr("height", tensorH)
      .attr("rx", 4).attr("ry", 4)
      .attr("fill", "var(--bg-surface)")
      .attr("stroke", "var(--text-muted)")
      .attr("stroke-width", 1);

    // Title
    g.append("text")
      .attr("x", tensorX + tensorW / 2)
      .attr("y", tensorY - 8)
      .attr("text-anchor", "middle")
      .attr("font-size", "9px")
      .attr("font-family", "var(--font-sans)")
      .attr("font-weight", "500")
      .attr("fill", "var(--text-secondary)")
      .text("TP切分 Tensor 映射");

    // Cells — render exactly effectiveTp cells (same as dense model)
    var hasHighlight = highlightTpIdx != null && highlightTpIdx >= 0;
    function renderCell(cellIdx, isHL) {
      var col = cellIdx % gridCols;
      var row = Math.floor(cellIdx / gridCols);
      var cx = tensorX + col * cellW;
      var cy = tensorY + row * cellH;
      g.append("rect")
        .attr("x", cx).attr("y", cy)
        .attr("width", cellW).attr("height", cellH)
        .attr("fill", isHL ? "var(--cyan)" : "#161c24")
        .attr("stroke", isHL ? "#ff8f40" : "var(--text-muted)")
        .attr("stroke-width", isHL ? 2 : 0.5)
        .attr("stroke-dasharray", isHL ? "3 2" : "none");
      g.append("text")
        .attr("x", cx + cellW / 2)
        .attr("y", cy + cellH / 2 + 1)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .attr("font-size", effectiveTp > 16 ? "7px" : "9px")
        .attr("font-family", "var(--font-mono)")
        .attr("font-weight", "600")
        .attr("fill", isHL ? "#0a0e14" : "var(--text-secondary)")
        .text(cellIdx + 1);
    }
    // Draw non-highlighted cells first, then highlighted on top
    for (var cellIdx = 0; cellIdx < effectiveTp; cellIdx++) {
      if (hasHighlight && cellIdx === highlightTpIdx) continue;
      renderCell(cellIdx, false);
    }
    if (hasHighlight) {
      renderCell(highlightTpIdx, true);
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // Render legend (bottom-left)
  // ═══════════════════════════════════════════════════════════════

  function renderLegend(g, legendX, legendY, hasSharedExpert) {
    var items = [
      { color: "var(--yellow)", label: "Embeddings & Output" },
      { color: "var(--green)", label: "Layer Normalization" },
      { color: "var(--cyan)", label: "Self-Attention & Stack" },
      { color: "var(--red)", label: "MoE / FFN Block" },
      { color: "var(--purple)", label: "MoE Router (Top-K + Softmax)" },
      { color: "var(--orange)", label: "Skip / Residual", dash: true },
    ];

    if (hasSharedExpert) {
      items.push({ color: "var(--green)", label: "Shared Expert" });
    }

    var rowH = 16;
    var swatchSize = 8;

    // Title
    g.append("text")
      .attr("x", legendX)
      .attr("y", legendY)
      .attr("font-size", "9px")
      .attr("font-family", "var(--font-sans)")
      .attr("font-weight", "600")
      .attr("fill", "var(--text-muted)")
      .attr("letter-spacing", "1px")
      .text("Legend");

    items.forEach(function (item, i) {
      var ly = legendY + 14 + (i + 1) * rowH;
      if (item.dash) {
        g.append("line")
          .attr("x1", legendX).attr("y1", ly - 2)
          .attr("x2", legendX + 10).attr("y2", ly - 2)
          .attr("stroke", item.color)
          .attr("stroke-width", 1.5)
          .attr("stroke-dasharray", "3 2");
      } else {
        g.append("rect")
          .attr("x", legendX).attr("y", ly - swatchSize + 2)
          .attr("width", swatchSize).attr("height", swatchSize)
          .attr("rx", 2).attr("ry", 2)
          .attr("fill", item.color);
      }
      g.append("text")
        .attr("x", legendX + 14).attr("y", ly + 4)
        .attr("font-size", "8px")
        .attr("font-family", "var(--font-sans)")
        .attr("fill", "var(--text-secondary)")
        .text(item.label);
    });
  }

  // ═══════════════════════════════════════════════════════════════
  // Public: compute diagram height for canvas layout
  // ═══════════════════════════════════════════════════════════════

  function calcMoeArchHeight(config, availableWidth, ppCount) {
    var scale = Math.min(1, (availableWidth - 16) / MOE_DESIGN.W);
    var D = MOE_DESIGN;
    var moeDesignH = calcMoeBlockDesignH(config);
    var actualY_Add2 = D.Y_MOE + moeDesignH + 10;
    var actualY_TFEnd = actualY_Add2 + D.H_SM + 14;
    var actualY_LN3 = actualY_TFEnd + 20;
    var actualY_Output = actualY_LN3 + D.H_SM + 10;
    var outputBottomY = actualY_Output + D.H_MD;
    // Legend: bottom aligns with output bottom
    var _numLegendItems = (config.has_shared_expert ? 7 : 6);
    var _legendTotalH = 14 + (_numLegendItems + 1) * 16 + 8;
    // PP→layer mapping table height (only if topology data is available)
    var _tableBottomY = D.TENSOR_Y + D.TENSOR_H;
    if (ppCount > 0 && config.num_layers > 0) {
      var COL_PP = 50, COL_RANGE = 74;
      var ROW_H = 14, HEADER_H = 15;
      _tableBottomY = D.TENSOR_Y + D.TENSOR_H + 18 + 14 + HEADER_H + ppCount * ROW_H + 36;
    }
    var legendY = Math.max(_tableBottomY + 16, outputBottomY - _legendTotalH);
    return Math.ceil((legendY + 160) * scale) + 80;
  }

  // ═══════════════════════════════════════════════════════════════
  // Public: main render function
  // ═══════════════════════════════════════════════════════════════

  /**
   * Render MoE architecture diagram into a D3 group element.
   *
   * @param {d3.Selection} g       - D3 <g> element (zoom-layer child)
   * @param {object}        model  - Model entry { config, computed, layers }
   * @param {object}        opts   - { x, y, areaW, scale, labelColor,
   *                                   tpCount, ppCount,
   *                                   highlightTpIdx, highlightPpIdx,
   *                                   highlightInputOutput }
   */
  function renderMoeArchitecture(g, model, opts) {
    opts = opts || {};
    var cfg = model && model.config ? model.config : {};
    var comp = model && model.computed ? model.computed : {};
    var numLayers = cfg.num_layers || 1;
    var D = MOE_DESIGN;

    // ── Guard against missing layout params ──
    var _ox = Number.isFinite(opts.x) ? opts.x : 0;
    var _oy = Number.isFinite(opts.y) ? opts.y : 0;
    var _aw = Number.isFinite(opts.areaW) && opts.areaW > 0 ? opts.areaW : D.W;

    // Debug: log if any value was NaN/missing
    if (!Number.isFinite(opts.x) || !Number.isFinite(opts.y) || !Number.isFinite(opts.areaW)) {
      console.warn('[MoE] Bad layout opts, using fallbacks. Raw opts:', JSON.stringify({
        x: opts.x, y: opts.y, areaW: opts.areaW, scale: opts.scale,
        tpCount: opts.tpCount, ppCount: opts.ppCount
      }), 'Fallback:', { _ox: _ox, _oy: _oy, _aw: _aw });
    }

    // ── Scale ──
    var scale = (opts.scale != null && opts.scale > 0)
      ? opts.scale
      : Math.min(1, (_aw - 16) / D.W);
    var offX = _ox + (_aw - D.W * scale) / 2;

    // ── Helper to map design-X to scaled-X ──
    function sx(v) { return offX + v * scale; }
    function sy(v) { return _oy + v * scale; }
    function sw(v) { return v * scale; }

    var CX = sx(D.CX);

    // Colors
    var lc = opts.labelColor || "var(--cyan)";
    var _fp = opts.filterPrefix || "";

    // ── Hover tooltip helpers (matching dense model pattern) ──
    function _showMoeTip(show, tipId, event) {
      var tipEl = document.getElementById("model-tooltip");
      if (!tipEl) return;
      if (!show || !tipId || !_MOE_TIPS[tipId]) {
        tipEl.classList.remove("visible");
        return;
      }
      var tip = _MOE_TIPS[tipId];
      var html = '<div class="tip-title">' + tip.t + "</div>";
      html += '<div class="tip-eng">' + tip.te + "</div>";
      html += '<div class="tip-explain">' + tip.d + "</div>";
      if (tip.m) html += '<div class="tip-math">' + tip.m + "</div>";
      if (tip.v) html += '<div class="tip-values">' + tip.v + "</div>";
      tipEl.innerHTML = html;
      tipEl.classList.add("visible");
      if (event) {
        var x = event.clientX + 16;
        var y = event.clientY - 10;
        var tw = tipEl.offsetWidth || 300;
        var th = tipEl.offsetHeight || 200;
        if (x + tw > window.innerWidth - 20) x = event.clientX - tw - 16;
        if (y + th > window.innerHeight - 20) y = event.clientY - th - 10;
        if (x < 10) x = 10;
        if (y < 10) y = 10;
        tipEl.style.left = x + "px";
        tipEl.style.top = y + "px";
      }
    }

    function _addMoeHover(rect, origSw, tipId) {
      rect
        .classed("model-node", true)
        .on("mouseenter", function (event) {
          d3.select(this)
            .attr("stroke-width", origSw * 2.2)
            .attr("filter", "url(#" + _fp + "model-hover-glow)");
          _showMoeTip(true, tipId, event);
        })
        .on("mousemove", function (event) {
          _showMoeTip(true, tipId, event);
        })
        .on("mouseleave", function () {
          d3.select(this).attr("stroke-width", origSw).attr("filter", null);
          _showMoeTip(false);
        });
    }

    // ── Pre-compute MoE height and dynamic Y positions ──
    var _moeDesignH = calcMoeBlockDesignH(cfg);
    var actualY_Add2 = D.Y_MOE + _moeDesignH + 10;
    var actualY_TFEnd = actualY_Add2 + D.H_SM + 14;
    var actualY_LN3 = actualY_TFEnd + 20;
    var actualY_Output = actualY_LN3 + D.H_SM + 10;

    // ── Transformer Layer card (stacked shadows + wrapper) ──
    var _tfCardY = D.Y_HEADER;
    var _tfCardH = actualY_TFEnd - _tfCardY;
    var _stackCount = 5;
    var _stackGap = 9;
    for (var _si = 0; _si < _stackCount; _si++) {
      var _off = (_stackCount - _si) * _stackGap;
      g.append("rect")
        .attr("x", sx(D.CX - D.BOX_W / 2 + _off))
        .attr("y", sy(_tfCardY + _off))
        .attr("width", sw(D.BOX_W))
        .attr("height", sw(_tfCardH))
        .attr("rx", 6 * scale).attr("ry", 6 * scale)
        .attr("fill", "#0d131a")
        .attr("stroke", lc)
        .attr("stroke-width", 1 * scale)
        .attr("opacity", 0.3 + _si * 0.25)
        .attr("class", "moe-stack-shadow");
    }

    // Main card rect
    var _tfCardRect = g.append("rect")
      .attr("x", sx(D.CX - D.BOX_W / 2))
      .attr("y", sy(_tfCardY))
      .attr("width", sw(D.BOX_W))
      .attr("height", sw(_tfCardH))
      .attr("rx", 6 * scale).attr("ry", 6 * scale)
      .attr("fill", "#0d131a")
      .attr("stroke", lc)
      .attr("stroke-width", 1.5 * scale)
      .attr("class", "moe-stack-card");
    _addMoeHover(_tfCardRect, 1.5 * scale, "transformer_card");

    // Card header: "Transformer Layer  (×N)"
    g.append("text")
      .attr("x", sx(D.CX - D.BOX_W / 2 + 12))
      .attr("y", sy(_tfCardY + 16))
      .attr("font-size", Math.max(8, 10 * scale) + "px")
      .attr("font-family", "var(--font-sans)")
      .attr("font-weight", "600")
      .attr("fill", lc)
      .text("Transformer Layer  (×N)");

    // Right-aligned layer count
    var numMoeLayers = cfg.num_moe_layers;
    var _layerLabel;
    if (numMoeLayers != null && numMoeLayers > 0 && numMoeLayers < numLayers) {
      _layerLabel = "×" + numLayers + "  MoE: " + numMoeLayers + "/" + numLayers;
    } else {
      _layerLabel = "×" + numLayers;
    }
    g.append("text")
      .attr("x", sx(D.CX + D.BOX_W / 2 - 12))
      .attr("y", sy(_tfCardY + 16))
      .attr("text-anchor", "end")
      .attr("font-size", Math.max(8, 10 * scale) + "px")
      .attr("font-family", "var(--font-mono)")
      .attr("font-weight", "600")
      .attr("fill", "var(--text-muted)")
      .text(_layerLabel);

    // ── 1. Embeddings ──
    drawBox(g, sx(D.CX - D.BOX_W / 2), sy(D.Y_EMBED), sw(D.BOX_W), sw(D.H_MD),
      "Position + Word Embeddings & Dropout",
      "#1a1a10", "var(--yellow)", "var(--yellow)", "moe-embed",
      "embeddings", _addMoeHover);

    // ── 2. Layer Norm 1 ──
    drawBox(g, sx(D.CX - D.NARROW_W / 2), sy(D.Y_LN1), sw(D.NARROW_W), sw(D.H_SM),
      "Layer Norm",
      "#111a13", "var(--green)", "var(--green)", "moe-ln",
      "layer_norm_1", _addMoeHover);

    // ── 3. Multi-Head Self-Attention ──
    drawSubBlock(g, sx(D.CX - D.BLOCK_W / 2), sy(D.Y_ATTN), sw(D.BLOCK_W), sw(D.H_ATTN),
      "Multi-Head Self-Attention",
      ["Self Attention", "Linear (h → h)", "Dropout"],
      "#111922", "var(--cyan)", "var(--cyan)", "#15202b", "var(--cyan)",
      "multi_head_attention", _addMoeHover);

    // ── 4. Add 1 ──
    drawBox(g, sx(D.CX - D.NARROW_W / 2), sy(D.Y_ADD1), sw(D.NARROW_W), sw(D.H_SM),
      "Add",
      "var(--bg-surface)", "var(--text-muted)", "var(--text-secondary)", "moe-add",
      "add_1", _addMoeHover);

    // ── 5. Layer Norm 2 ──
    drawBox(g, sx(D.CX - D.NARROW_W / 2), sy(D.Y_LN2), sw(D.NARROW_W), sw(D.H_SM),
      "Layer Norm",
      "#111a13", "var(--green)", "var(--green)", "moe-ln",
      "layer_norm_2", _addMoeHover);

    // ── 6. MoE FFN Block ──
    renderMoEBlock(g, sx(D.CX - D.BLOCK_W / 2), sy(D.Y_MOE), sw(D.BLOCK_W), cfg, scale, _addMoeHover);
    // actualY_Add2…actualY_Output pre-computed above

    // ── 7. Add 2 ──
    drawBox(g, sx(D.CX - D.NARROW_W / 2), sy(actualY_Add2), sw(D.NARROW_W), sw(D.H_SM),
      "Add",
      "var(--bg-surface)", "var(--text-muted)", "var(--text-secondary)", "moe-add",
      "add_2", _addMoeHover);

    // ── 8. Final Layer Norm ──
    drawBox(g, sx(D.CX - D.NARROW_W / 2), sy(actualY_LN3), sw(D.NARROW_W), sw(D.H_SM),
      "Final Layer Norm",
      "#111a13", "var(--green)", "var(--green)", "moe-ln",
      "final_layer_norm", _addMoeHover);

    // ── 9. Output Layer ──
    drawBox(g, sx(D.CX - D.BOX_W / 2), sy(actualY_Output), sw(D.BOX_W), sw(D.H_MD),
      "Output Layer & Loss",
      "#1a140f", "var(--orange)", "var(--orange)", "moe-output",
      "output_layer", _addMoeHover);

    // ── Main flow arrows ──
    var arrowColor = "var(--green)";
    drawFlowArrow(g, CX, sy(D.Y_EMBED + D.H_MD), sy(D.Y_LN1), arrowColor);
    drawFlowArrow(g, CX, sy(D.Y_LN1 + D.H_SM), sy(D.Y_ATTN), arrowColor);
    drawFlowArrow(g, CX, sy(D.Y_ATTN + D.H_ATTN), sy(D.Y_ADD1), arrowColor);
    drawFlowArrow(g, CX, sy(D.Y_ADD1 + D.H_SM), sy(D.Y_LN2), arrowColor);
    drawFlowArrow(g, CX, sy(D.Y_LN2 + D.H_SM), sy(D.Y_MOE), arrowColor);
    drawFlowArrow(g, CX, sy(D.Y_MOE + _moeDesignH), sy(actualY_Add2), arrowColor);
    drawFlowArrow(g, CX, sy(actualY_Add2 + D.H_SM), sy(actualY_LN3), arrowColor);
    drawFlowArrow(g, CX, sy(actualY_LN3 + D.H_SM), sy(actualY_Output), arrowColor);

    // ── Skip / Residual connections ──
    // Compute Y positions in design space first, then scale them
    var _skip1StartY_d = D.Y_EMBED + D.H_MD + (D.Y_LN1 - D.Y_EMBED - D.H_MD) / 2;
    var _skip1EndY_d = D.Y_ADD1 + D.H_SM / 2;
    var _skip2StartY_d = D.Y_ADD1 + D.H_SM + (D.Y_LN2 - D.Y_ADD1 - D.H_SM) / 2;
    var _skip2EndY_d = actualY_Add2 + D.H_SM / 2;
    var _scaledSkipRight = CX + sw(D.BOX_W / 2 + 28);
    renderResiduals(g, CX, sw(D.BOX_W / 2), sw(D.NARROW_W / 2), _scaledSkipRight,
      sy(_skip1StartY_d), sy(_skip1EndY_d),
      sy(_skip2StartY_d), sy(_skip2EndY_d),
      scale);

    // ── Tensor grid (left side, if tpCount available) ──
    var tpCount = opts.tpCount || 0;
    var ppCount = opts.ppCount || 0;
    var _tableBottomDesignY = D.TENSOR_Y + D.TENSOR_H; // will grow if table is rendered
    if (tpCount > 0) {
      renderTensorGrid(
        g,
        sx(D.TENSOR_X), sy(D.TENSOR_Y),
        sw(D.TENSOR_W), sw(D.TENSOR_H),
        tpCount, ppCount,
        opts.highlightTpIdx, opts.highlightPpIdx
      );

      // ── PP → layer mapping table (matches dense model) ──
      if (ppCount > 0 && numLayers > 0) {
        var _mapTitleY = D.TENSOR_Y + D.TENSOR_H + 18;
        g.append("text")
          .attr("x", sx(D.TENSOR_X + D.TENSOR_W / 2))
          .attr("y", sy(_mapTitleY))
          .attr("text-anchor", "middle")
          .attr("font-size", Math.max(7, 9 * scale) + "px")
          .attr("font-family", "var(--font-sans)")
          .attr("font-weight", "500")
          .attr("fill", lc)
          .text("PP切分模型层映射");

        var layersPerPp = Math.floor(numLayers / ppCount);
        var remainder = numLayers % ppCount;
        var COL_PP = 50, COL_RANGE = 74;
        var ROW_H = 14, HEADER_H = 15;
        var tableW_d = COL_PP + COL_RANGE;
        var tableX_d = D.TENSOR_X + (D.TENSOR_W - tableW_d) / 2;
        var tableY_d = _mapTitleY + 14;
        var tableH_d = HEADER_H + ppCount * ROW_H;
        var hasPpHL = opts.highlightPpIdx != null && opts.highlightPpIdx >= 0;
        var sepX1 = tableX_d + COL_PP;

        // Header background
        g.append("rect")
          .attr("x", sx(tableX_d)).attr("y", sy(tableY_d))
          .attr("width", sw(tableW_d)).attr("height", sw(HEADER_H))
          .attr("fill", "#21262d");

        // Row backgrounds (highlighted row last)
        for (var pi = 0; pi < ppCount; pi++) {
          if (hasPpHL && opts.highlightPpIdx === pi) continue;
          var ry = tableY_d + HEADER_H + pi * ROW_H;
          g.append("rect")
            .attr("x", sx(tableX_d)).attr("y", sy(ry))
            .attr("width", sw(tableW_d)).attr("height", sw(ROW_H))
            .attr("fill", pi % 2 === 0 ? "var(--bg-surface)" : "#161b22");
        }
        if (hasPpHL) {
          var hlPi = opts.highlightPpIdx;
          var hlRy = tableY_d + HEADER_H + hlPi * ROW_H;
          g.append("rect")
            .attr("x", sx(tableX_d)).attr("y", sy(hlRy))
            .attr("width", sw(tableW_d)).attr("height", sw(ROW_H))
            .attr("fill", "var(--cyan)")
            .attr("fill-opacity", 0.2)
            .attr("stroke", "#ff8f40")
            .attr("stroke-width", 2 * scale)
            .attr("stroke-dasharray", "3 2");
        }

        // Vertical separator
        g.append("line")
          .attr("x1", sx(sepX1)).attr("y1", sy(tableY_d))
          .attr("x2", sx(sepX1)).attr("y2", sy(tableY_d + tableH_d))
          .attr("stroke", "var(--text-muted)")
          .attr("stroke-width", 0.6 * scale);

        // Horizontal separators + outer border
        for (var hi = 0; hi <= ppCount; hi++) {
          var hy = tableY_d + HEADER_H + hi * ROW_H;
          g.append("line")
            .attr("x1", sx(tableX_d)).attr("y1", sy(hy))
            .attr("x2", sx(tableX_d + tableW_d)).attr("y2", sy(hy))
            .attr("stroke", "var(--text-muted)")
            .attr("stroke-width", 0.6 * scale);
        }
        g.append("rect")
          .attr("x", sx(tableX_d)).attr("y", sy(tableY_d))
          .attr("width", sw(tableW_d)).attr("height", sw(tableH_d))
          .attr("fill", "none")
          .attr("stroke", "var(--text-muted)")
          .attr("stroke-width", 1 * scale)
          .attr("rx", 2 * scale);

        // Header text
        var _hdrTextY = tableY_d + 11;
        g.append("text")
          .attr("x", sx(tableX_d + COL_PP / 2))
          .attr("y", sy(_hdrTextY))
          .attr("text-anchor", "middle")
          .attr("font-size", Math.max(6, 8 * scale) + "px")
          .attr("font-family", "var(--font-sans)")
          .attr("font-weight", "600")
          .attr("fill", "var(--text-secondary)")
          .text("PP索引");
        g.append("text")
          .attr("x", sx(tableX_d + COL_PP + COL_RANGE / 2))
          .attr("y", sy(_hdrTextY))
          .attr("text-anchor", "middle")
          .attr("font-size", Math.max(6, 8 * scale) + "px")
          .attr("font-family", "var(--font-sans)")
          .attr("font-weight", "600")
          .attr("fill", "var(--text-secondary)")
          .text("模型起止层数");

        // Row text
        for (var pi2 = 0; pi2 < ppCount; pi2++) {
          var rowY2 = tableY_d + HEADER_H + pi2 * ROW_H;
          // Distribute remainder layers: first `remainder` PPs get one extra layer
          var layerStart = pi2 * layersPerPp + Math.min(pi2, remainder);
          var layerEnd = layerStart + layersPerPp + (pi2 < remainder ? 1 : 0) - 1;
          var rty = rowY2 + 10;
          g.append("text")
            .attr("x", sx(tableX_d + COL_PP / 2))
            .attr("y", sy(rty))
            .attr("text-anchor", "middle")
            .attr("font-size", Math.max(6, 8 * scale) + "px")
            .attr("font-family", "var(--font-mono)")
            .attr("fill", "var(--text-primary)")
            .text(pi2);
          g.append("text")
            .attr("x", sx(tableX_d + COL_PP + COL_RANGE / 2))
            .attr("y", sy(rty))
            .attr("text-anchor", "middle")
            .attr("font-size", Math.max(6, 8 * scale) + "px")
            .attr("font-family", "var(--font-mono)")
            .attr("fill", "var(--text-primary)")
            .text(layerStart + "~" + layerEnd);
        }

        _tableBottomDesignY = tableY_d + tableH_d + 36;
      }
    }

    // ── Legend (bottom-left) — bottom aligns with model output bottom ──
    var _numLegendItems = (cfg.has_shared_expert ? 7 : 6);
    var _legendTotalH = 14 + (_numLegendItems + 1) * 16 + 8;
    var _outputBottomY = actualY_Output + D.H_MD;
    var _legendDesignY = Math.max(_tableBottomDesignY + 16, _outputBottomY - _legendTotalH);
    var _legendY = sy(_legendDesignY);
    renderLegend(
      g,
      sx(D.TENSOR_X), _legendY,
      cfg.has_shared_expert
    );

    // Return rendered height
    return Math.ceil(_legendY + sw(160)) - _oy;
  }

  // ═══════════════════════════════════════════════════════════════
  // Expose public API on window
  // ═══════════════════════════════════════════════════════════════

  window.calcMoeArchHeight = calcMoeArchHeight;
  window.renderMoeArchitecture = renderMoeArchitecture;
  window.MOE_DESIGN = MOE_DESIGN; // expose for height calculations elsewhere

})();
