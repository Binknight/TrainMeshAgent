/**
 * Topology Renderer — D3-based mesh topology visualization module.
 *
 * Merges: incremental DP switch, D3 zoom, batch SSE rendering, debounced resize,
 * tooltip system, estimation engine, simulation data fetch, text overflow fix.
 *
 * Public API (also attached to window for inline onclick handlers):
 *   loadMeshData(topoData)   — ingest topology JSON from SSE / API
 *   meshRebuild()            — re-render the current topology
 *   meshSwitchDp(idx)        — single-view DP selector
 *   meshSwitchDpOrig(idx)    — compare-view original DP selector
 *   meshSwitchDpEq(idx)      — compare-view equivalent DP selector
 *   fetchSimulationData()    — pull simulation results from REST and re-render
 */

// ── Model config & estimation engine ──

var _MODEL_CFG = {
  'A2': { model_params: 7e9, hidden_dim: 4096, num_layers: 32 },
  'A3': { model_params: 20e9, hidden_dim: 6144, num_layers: 48 },
  'A5': { model_params: 70e9, hidden_dim: 8192, num_layers: 64 }
};
var _SEQ_LEN = 4096;
var _MICRO_BATCH = 1;
var _BYTES_PER_PARAM = 2;

function estimateCardMetrics(deviceType, totalNodes, dp, tp, pp) {
  var cfg = _MODEL_CFG[deviceType] || _MODEL_CFG['A3'];
  var p = cfg.model_params;

  var flops = 6 * p * _SEQ_LEN * _MICRO_BATCH / (dp * tp * pp);

  var param_mem = p * _BYTES_PER_PARAM / tp;
  var optim_mem = 2 * p * _BYTES_PER_PARAM / dp;
  var act_mem = cfg.hidden_dim * _SEQ_LEN * _MICRO_BATCH * _BYTES_PER_PARAM * cfg.num_layers / pp;
  var hbm = (param_mem + optim_mem + act_mem) / 1e9;

  var tp_comm = tp > 1 ? cfg.num_layers * 2 * cfg.hidden_dim * _SEQ_LEN * _MICRO_BATCH * _BYTES_PER_PARAM * (tp - 1) / tp / 1e9 : 0;
  var pp_comm = pp > 1 ? 2 * (pp - 1) / pp * cfg.hidden_dim * _SEQ_LEN * _MICRO_BATCH * _BYTES_PER_PARAM / 1e6 : 0;
  var dp_comm = dp > 1 ? 2 * p * _BYTES_PER_PARAM * (dp - 1) / dp / 1e9 : 0;

  return {
    flops_per_card: flops,
    hbm_gb: hbm,
    tp_comm_gb_per_micro: tp_comm,
    pp_comm_mb_per_micro: pp_comm,
    dp_comm_gb_per_step: dp_comm
  };
}

function formatFlops(value) {
  if (value == null) return '—';
  if (value === 0) return '0';
  var exp = Math.floor(Math.log10(Math.abs(value)));
  var mantissa = value / Math.pow(10, exp);
  return mantissa.toFixed(2) + '×10^' + exp;
}

function computeAllEstimates(deviceType, totalNodes, dp, tp, pp) {
  var est = {};
  for (var r = 0; r < totalNodes; r++) {
    est[r] = estimateCardMetrics(deviceType, totalNodes, dp, tp, pp);
  }
  return est;
}

// ── State ──

var meshOriginal = null;       // { name, device_type, tp, pp, dp, total_nodes }
var meshEquivalent = null;
var meshOrigDp = 0;
var meshEqDp = 0;
var meshEstimateOrig = {};     // { global_rank: { flops_per_card, hbm_gb, ... } }
var meshEstimateEq = {};
var meshActualOrig = {};       // { global_rank: CardMetrics } from REST simulation data
var meshActualEq = {};
var meshPinnedRank = null;     // { side: "orig"|"eq", globalRank: number }

// Tracks the last rendered configuration for fast DP-switch path
var _renderState = {
  mode: null,
  orig: { tp: 0, pp: 0, dp: 0, activeDp: -1 },
  eq: { tp: 0, pp: 0, dp: 0, activeDp: -1 },
  width: 0,
  height: 0,
};

// ── Layout constants ──

var MESH_CARD = {
  dpShadowOffset: [[90, 90], [60, 60], [30, 30]],
  tpPadX: 16,
  tpPadY: 16,
  headerH: 42,
  ppGap: 12,
  ppHeaderH: 24,
  tpW: 82,
  tpH: 42,
  tpGap: 14,
  maxPpDisplay: 8,
};

var meshWidth = 600;
var meshHeight = 400;

// ── Internal helpers ──

function meshUpdateSize() {
  var wrap = document.getElementById("canvas-svg-wrap");
  meshWidth = wrap.clientWidth || 600;
  meshHeight = wrap.clientHeight || 400;
}

function meshBuildDisplayList(ppList) {
  var total = ppList.length;
  if (total <= MESH_CARD.maxPpDisplay) {
    return ppList.map(function (p) { return { type: "pp", data: p }; });
  }
  var result = [];
  for (var i = 0; i < 3; i++) result.push({ type: "pp", data: ppList[i] });
  result.push({ type: "ellipsis", hiddenStart: 3, hiddenEnd: total - 5, total: total });
  for (var j = total - 4; j < total; j++) result.push({ type: "pp", data: ppList[j] });
  return result;
}

function meshBuildData(tp, pp, dpCount, activeDp) {
  var ranksPerDp = pp * tp;
  var dpBase = activeDp * ranksPerDp;
  return {
    config: { tpCount: tp, ppCount: pp, dpCount: dpCount, activeDp: activeDp },
    dp: { id: activeDp, label: "DP " + activeDp, isTop: true, stackCount: dpCount, stackIndex: activeDp },
    pp: d3.range(pp).map(function (pi) {
      return {
        id: pi,
        label: "PP" + pi,
        tps: d3.range(tp).map(function (ti) {
          var gr = dpBase + pi * tp + ti;
          return { id: ti, label: "TP" + ti, rank: "Rank" + gr, globalRank: gr };
        }),
      };
    }),
  };
}

function _meshCalcDims(tp, pp) {
  var displayCount = Math.min(MESH_CARD.maxPpDisplay, pp);
  var ppW = 115;
  var ellipsisH = 100;
  var ppH =
    MESH_CARD.tpPadY * 2 +
    MESH_CARD.ppHeaderH +
    tp * (MESH_CARD.tpH + MESH_CARD.tpGap) -
    MESH_CARD.tpGap +
    20;
  var ppTotalW = displayCount * ppW + (displayCount - 1) * MESH_CARD.ppGap;
  var dpW = ppTotalW + MESH_CARD.tpPadX * 2;
  var dpH = Math.max(ppH, ellipsisH) + MESH_CARD.headerH + 80;
  return { dpW: dpW, dpH: dpH };
}

// ── Tooltip system ──

function _getMetrics(globalRank, isOrig) {
  var estimate = isOrig ? meshEstimateOrig[globalRank] : meshEstimateEq[globalRank];
  var actual = isOrig ? meshActualOrig[globalRank] : meshActualEq[globalRank];
  var deviceType = isOrig ? (meshOriginal || {}).device_type : (meshEquivalent || {}).device_type;
  return { estimate: estimate, actual: actual, deviceType: deviceType };
}

function _hasActualData(side) {
  var map = side === 'orig' ? meshActualOrig : meshActualEq;
  return Object.keys(map).length > 0;
}

function _buildTooltipHTML(globalRank, metrics) {
  var deviceType = metrics.deviceType || '';
  var estimate = metrics.estimate;
  var actual = metrics.actual;

  var html = '<div class="tooltip-header">Rank ' + globalRank;
  if (deviceType) html += ' <span class="tooltip-device">' + deviceType + '</span>';
  html += '</div><div class="tooltip-body">';

  if (!estimate && !actual) {
    html += '<div class="tooltip-empty">暂无性能数据</div>';
  } else if (!actual || Object.keys(actual).length === 0) {
    html += '<div class="tooltip-section-label">📊 理论估算</div>';
    html += _buildSingleTable(estimate);
  } else {
    html += '<table class="tooltip-table">';
    html += '<tr><th class="col-header left">指标</th><th class="col-header">理论估算值</th><th class="col-header">仿真验证值</th></tr>';
    html += _buildCompareRow('单卡FLOPs', '', estimate, actual, 'flops_per_card', formatFlops);
    html += _buildCompareRow('HBM', 'GB', estimate, actual, 'hbm_gb');
    html += _buildCompareRow('TP通信', 'GB/micro', estimate, actual, 'tp_comm_gb_per_micro');
    html += _buildCompareRow('PP通信', 'MB/micro', estimate, actual, 'pp_comm_mb_per_micro');
    html += _buildCompareRow('DP通信', 'GB/step', estimate, actual, 'dp_comm_gb_per_step');
    html += '</table>';
  }

  html += '<div class="tooltip-pin-hint">🖱 点击固定 · 再点取消</div>';
  html += '</div>';
  return html;
}

function _buildSingleTable(metrics) {
  if (!metrics) return '<div class="tooltip-empty">无数据</div>';
  var html = '<table class="tooltip-table">';
  html += _buildMetricRow('单卡FLOPs', formatFlops(metrics.flops_per_card), '');
  html += _buildMetricRow('HBM', metrics.hbm_gb != null ? Number(metrics.hbm_gb).toFixed(2) : '—', 'GB');
  html += _buildMetricRow('TP通信', metrics.tp_comm_gb_per_micro != null ? Number(metrics.tp_comm_gb_per_micro).toFixed(2) : '—', 'GB/micro');
  html += _buildMetricRow('PP通信', metrics.pp_comm_mb_per_micro != null ? Number(metrics.pp_comm_mb_per_micro).toFixed(2) : '—', 'MB/micro');
  html += _buildMetricRow('DP通信', metrics.dp_comm_gb_per_step != null ? Number(metrics.dp_comm_gb_per_step).toFixed(2) : '—', 'GB/step');
  html += '</table>';
  return html;
}

function _buildMetricRow(label, value, unit) {
  var v;
  if (value == null) { v = '—'; }
  else if (typeof value === 'string') { v = value; }
  else { v = Number(value).toFixed(2); }
  var uv = unit ? ' ' + unit : '';
  return '<tr><td class="metric-label">' + label + '</td><td class="metric-val">' + v + uv + '</td></tr>';
}

function _buildCompareRow(label, unit, estimate, actual, key, fmt) {
  fmt = fmt || function (v) { return v != null ? Number(v).toFixed(2) : '—'; };
  var ev = estimate && estimate[key] != null ? fmt(estimate[key]) : '—';
  var av = actual && actual[key] != null ? fmt(actual[key]) : '—';
  var delta = '';
  var deltaClass = '';
  if (estimate && actual && estimate[key] != null && actual[key] != null) {
    var diff = actual[key] - estimate[key];
    var pct = estimate[key] !== 0 ? (diff / estimate[key] * 100) : 0;
    var sign = diff >= 0 ? '+' : '';
    delta = sign + Number(pct).toFixed(1) + '%';
    deltaClass = Math.abs(pct) <= 5 ? 'positive' : 'negative';
  }
  var uv = unit ? ' ' + unit : '';
  return '<tr><td class="metric-label">' + label + '</td>' +
    '<td class="metric-val">' + ev + uv + '</td>' +
    '<td class="metric-val actual">' + av + uv + '</td>' +
    '<td class="metric-delta ' + deltaClass + '">' + delta + '</td></tr>';
}

function showTooltip(event, globalRank, isOrig) {
  var tip = document.getElementById('rank-tooltip');
  var metrics = _getMetrics(globalRank, isOrig);
  tip.innerHTML = _buildTooltipHTML(globalRank, metrics);
  tip.classList.add('visible');
  _positionTooltip(event, tip);
}

function _positionTooltip(event, tip) {
  var x = event.clientX + 16;
  var y = event.clientY - 10;
  var tw = tip.offsetWidth;
  var th = tip.offsetHeight;
  if (x + tw > window.innerWidth - 8) x = event.clientX - tw - 16;
  if (y + th > window.innerHeight - 8) y = event.clientY - th - 10;
  if (x < 8) x = 8;
  if (y < 8) y = 8;
  tip.style.left = x + 'px';
  tip.style.top = y + 'px';
}

function moveTooltip(event) {
  var tip = document.getElementById('rank-tooltip');
  if (!tip.classList.contains('pinned')) {
    _positionTooltip(event, tip);
  }
}

function hideTooltip() {
  var tip = document.getElementById('rank-tooltip');
  if (!tip.classList.contains('pinned')) {
    tip.classList.remove('visible');
    tip.innerHTML = '';
    d3.selectAll('.tp-rect.pinned').classed('pinned', false);
    meshPinnedRank = null;
  }
}

// ── Fetch simulation data from REST ──

function fetchSimulationData() {
  if (!sessionId) return;
  try {
    fetch(API + '/session/' + sessionId + '/simulation').then(function (resp) {
      if (!resp.ok) return;
      return resp.json().then(function (data) {
        var changed = false;
        if (data.original_simulation && data.original_simulation.cards) {
          meshActualOrig = {};
          data.original_simulation.cards.forEach(function (c) {
            meshActualOrig[c.global_rank] = c;
          });
          changed = true;
        }
        if (data.equivalent_simulation && data.equivalent_simulation.cards) {
          meshActualEq = {};
          data.equivalent_simulation.cards.forEach(function (c) {
            meshActualEq[c.global_rank] = c;
          });
          changed = true;
        }
        if (changed && (meshOriginal || meshEquivalent)) {
          meshRebuild();
        }
      });
    }).catch(function (e) {
      console.warn('Failed to fetch simulation data:', e);
    });
  } catch (e) {
    console.warn('Failed to fetch simulation data:', e);
  }
}

// ── Rendering ──

function _meshBuildView(parentG, data, dpSelectId, switchFn, viewX, viewY, viewW, viewH, sharedScale, isOrig) {
  if (isOrig === undefined) isOrig = !!meshOriginal;
  var cfg = data.config;
  var ppList = data.pp;
  var displayList = meshBuildDisplayList(ppList);
  var tpCount = ppList[0].tps.length;
  var displayCount = displayList.length;

  var ppW = 115;
  var ellipsisH = 100;
  var ppH =
    MESH_CARD.tpPadY * 2 +
    MESH_CARD.ppHeaderH +
    tpCount * (MESH_CARD.tpH + MESH_CARD.tpGap) -
    MESH_CARD.tpGap +
    20;
  var ppTotalW = displayCount * ppW + (displayCount - 1) * MESH_CARD.ppGap;
  var dpW = ppTotalW + MESH_CARD.tpPadX * 2;
  var dpH = Math.max(ppH, ellipsisH) + MESH_CARD.headerH + 80;

  var scale = sharedScale || Math.min(1, (viewW * 0.72) / dpW, (viewH * 0.72) / dpH);
  var dpX = viewX + (viewW - dpW * scale) / 2;
  var dpY = viewY + 8 * scale;

  // DP Stack shadows
  var shadowG = parentG.append("g").attr("class", "dp-shadows");
  MESH_CARD.dpShadowOffset.forEach(function (off, i) {
    shadowG
      .append("rect")
      .attr("x", dpX + off[0] * scale)
      .attr("y", dpY + off[1] * scale)
      .attr("width", dpW * scale)
      .attr("height", dpH * scale)
      .attr("rx", 14 * scale)
      .attr("ry", 14 * scale)
      .attr("class", "dp-shadow")
      .attr("opacity", 0.4 + i * 0.15);
  });

  // Main DP Card
  var dpG = parentG.append("g").attr("class", "dp-card-group");
  dpG
    .append("rect")
    .attr("x", dpX)
    .attr("y", dpY)
    .attr("width", dpW * scale)
    .attr("height", dpH * scale)
    .attr("rx", 14 * scale)
    .attr("ry", 14 * scale)
    .attr("class", "dp-card");

  // Header
  var hdrH = MESH_CARD.headerH * scale;
  dpG
    .append("rect")
    .attr("x", dpX)
    .attr("y", dpY)
    .attr("width", dpW * scale)
    .attr("height", hdrH)
    .attr("rx", 14 * scale)
    .attr("ry", 14 * scale)
    .attr("class", "dp-header");
  dpG
    .append("rect")
    .attr("x", dpX)
    .attr("y", dpY + hdrH / 2)
    .attr("width", dpW * scale)
    .attr("height", hdrH / 2)
    .attr("class", "dp-header");

  // DP select dropdown
  var fo = dpG
    .append("foreignObject")
    .attr("x", dpX + 20 * scale)
    .attr("y", dpY + 6 * scale)
    .attr("width", 200 * scale)
    .attr("height", 40 * scale);
  fo.append("xhtml:div")
    .style("display", "inline-block")
    .html(
      '<select id="' +
        dpSelectId +
        '" onchange="' +
        switchFn +
        '(parseInt(this.value))" style="font-size:' +
        14 * scale +
        "px;min-width:" +
        Math.max(100, 120 * scale) +
        "px;max-width:" +
        200 * scale +
        "px;background:" +
        (scale < 0.7 ? "#11161d" : "#0d1117") +
        ';color:#58a6ff;border:1px solid #58a6ff;border-radius:4px;padding:2px 6px;font-family:sans-serif;cursor:pointer;overflow:hidden;text-overflow:ellipsis;"></select>'
    );

  // Info text — textLength to prevent overflow
  var infoMaxWidth = dpW * scale - (20 + 200 + 30) * scale;
  if (infoMaxWidth < 60 * scale) infoMaxWidth = 60 * scale;
  var npuTotal = cfg.tpCount * cfg.ppCount * cfg.dpCount;
  var infoText = "TP" + cfg.tpCount + "×PP" + cfg.ppCount + "×DP" + cfg.dpCount + " | " + npuTotal + " NPUs";
  dpG
    .append("text")
    .attr("x", dpX + dpW * scale - 20 * scale)
    .attr("y", dpY + 25 * scale)
    .attr("text-anchor", "end")
    .attr("class", "info-text")
    .attr("font-size", Math.max(8, 11 * scale) + "px")
    .attr("textLength", infoMaxWidth)
    .attr("lengthAdjust", "spacingAndGlyphs")
    .text(infoText)
    .append("title").text(infoText);

  // PP Cards
  var ppStartX = dpX + MESH_CARD.tpPadX * scale;
  var ppStartY = dpY + hdrH + 20 * scale;

  displayList.forEach(function (item, di) {
    var px = ppStartX + di * (ppW + MESH_CARD.ppGap) * scale;
    var py = ppStartY;

    if (item.type === "ellipsis") {
      var eY = py + ((ppH - ellipsisH) * scale) / 2;
      var eG = parentG.append("g").attr("class", "pp-ellipsis-group");
      eG
        .append("rect")
        .attr("x", px)
        .attr("y", eY)
        .attr("width", ppW * scale)
        .attr("height", ellipsisH * scale)
        .attr("rx", 8 * scale)
        .attr("ry", 8 * scale)
        .attr("class", "pp-ellipsis");
      eG
        .append("text")
        .attr("x", px + (ppW * scale) / 2)
        .attr("y", eY + (ellipsisH * scale) / 2 + 6 * scale)
        .attr("text-anchor", "middle")
        .attr("class", "pp-ellipsis-text")
        .attr("font-size", Math.max(10, 18 * scale) + "px")
        .text("...");
      eG.append("title").text("PP" + item.hiddenStart + " ~ PP" + item.hiddenEnd);
      return;
    }

    var pp = item.data;
    var ppG = parentG.append("g").attr("class", "pp-group");
    ppG
      .append("rect")
      .attr("x", px)
      .attr("y", py)
      .attr("width", ppW * scale)
      .attr("height", ppH * scale)
      .attr("rx", 8 * scale)
      .attr("ry", 8 * scale)
      .attr("class", "pp-card");
    ppG
      .append("rect")
      .attr("x", px)
      .attr("y", py)
      .attr("width", ppW * scale)
      .attr("height", MESH_CARD.ppHeaderH * scale)
      .attr("rx", 8 * scale)
      .attr("ry", 8 * scale)
      .attr("class", "pp-header");
    ppG
      .append("rect")
      .attr("x", px)
      .attr("y", py + (MESH_CARD.ppHeaderH * scale) / 2)
      .attr("width", ppW * scale)
      .attr("height", (MESH_CARD.ppHeaderH * scale) / 2)
      .attr("class", "pp-header");
    ppG
      .append("text")
      .attr("x", px + (ppW * scale) / 2)
      .attr("y", py + 17 * scale)
      .attr("text-anchor", "middle")
      .attr("class", "pp-label")
      .attr("font-size", Math.max(9, 13 * scale) + "px")
      .attr("textLength", ppW * scale - 12 * scale)
      .attr("lengthAdjust", "spacingAndGlyphs")
      .text(pp.label);

    var tpX = px + ((ppW - MESH_CARD.tpW) * scale) / 2;
    var tpY = py + MESH_CARD.ppHeaderH * scale + MESH_CARD.tpPadY * scale;
    pp.tps.forEach(function (tp, ti) {
      var ty = tpY + ti * (MESH_CARD.tpH + MESH_CARD.tpGap) * scale;
      var side = isOrig ? 'orig' : 'eq';
      var hasActual = _hasActualData(side);
      var rectClass = 'tp-rect' + (hasActual ? ' has-data' : '');
      ppG
        .append("rect")
        .attr("x", tpX)
        .attr("y", ty)
        .attr("width", MESH_CARD.tpW * scale)
        .attr("height", MESH_CARD.tpH * scale)
        .attr("rx", 4 * scale)
        .attr("ry", 4 * scale)
        .attr("class", rectClass)
        .attr("data-rank", tp.globalRank)
        .attr("data-side", side)
        .on("mouseover", function (event) {
          if (meshPinnedRank) return;
          d3.select(this).attr("stroke", "#fff").attr("stroke-width", 2);
          showTooltip(event, tp.globalRank, isOrig);
        })
        .on("mousemove", function (event) {
          moveTooltip(event);
        })
        .on("mouseout", function () {
          if (meshPinnedRank) return;
          d3.select(this).attr("stroke", null).attr("stroke-width", null);
          hideTooltip();
        })
        .on("click", function (event) {
          event.stopPropagation();
          var alreadyPinned = meshPinnedRank && meshPinnedRank.globalRank === tp.globalRank && meshPinnedRank.side === side;
          d3.selectAll('.tp-rect.pinned').classed('pinned', false);
          if (alreadyPinned) {
            hideTooltip();
            document.getElementById('rank-tooltip').classList.remove('pinned');
            meshPinnedRank = null;
          } else {
            d3.select(this).classed('pinned', true);
            var metrics = _getMetrics(tp.globalRank, isOrig);
            var tip = document.getElementById('rank-tooltip');
            tip.innerHTML = _buildTooltipHTML(tp.globalRank, metrics);
            tip.classList.add('visible', 'pinned');
            _positionTooltip(event, tip);
            meshPinnedRank = { side: side, globalRank: tp.globalRank };
          }
        });
      ppG
        .append("text")
        .attr("x", tpX + MESH_CARD.tpW * scale - 4 * scale)
        .attr("y", ty + 11 * scale)
        .attr("text-anchor", "end")
        .attr("class", "tp-label")
        .attr("font-size", Math.max(7, 10 * scale) + "px")
        .attr("textLength", MESH_CARD.tpW * scale * 0.4)
        .attr("lengthAdjust", "spacingAndGlyphs")
        .text(tp.label);
      ppG
        .append("text")
        .attr("x", tpX + (MESH_CARD.tpW * scale) / 2)
        .attr("y", ty + (MESH_CARD.tpH * scale) / 2 + 4 * scale)
        .attr("text-anchor", "middle")
        .attr("class", "rank-label")
        .attr("font-size", Math.max(8, 12 * scale) + "px")
        .attr("textLength", MESH_CARD.tpW * scale - 8 * scale)
        .attr("lengthAdjust", "spacingAndGlyphs")
        .text(tp.rank);
    });

    // Arrow between PP cards
    if (di < displayList.length - 1) {
      var ax1 = px + ppW * scale + 2 * scale;
      var ax2 = px + (ppW + MESH_CARD.ppGap) * scale - 2 * scale;
      var ay = py + (ppH * scale) / 2;
      parentG
        .append("line")
        .attr("x1", ax1)
        .attr("y1", ay)
        .attr("x2", ax2 - 6 * scale)
        .attr("y2", ay)
        .attr("class", "arrow-line");
      parentG
        .append("polygon")
        .attr(
          "points",
          ax2 +
            "," +
            ay +
            " " +
            (ax2 - 5 * scale) +
            "," +
            (ay - 3 * scale) +
            " " +
            (ax2 - 5 * scale) +
            "," +
            (ay + 3 * scale)
        )
        .attr("class", "arrow-head");
    }
  });
}

function _populateDpSelect(selId, dpCount, activeDp) {
  var sel = document.getElementById(selId);
  if (!sel) return;
  sel.innerHTML = "";
  for (var i = 0; i < dpCount; i++) {
    var opt = document.createElement("option");
    opt.value = i;
    opt.textContent = "DP " + i;
    if (i === activeDp) opt.selected = true;
    sel.appendChild(opt);
  }
}

function _meshNpuTotal(entry) {
  return entry.tp * entry.pp * entry.dp;
}

function _meshUpdateRanks(parentG, tp, pp, oldDp, newDp) {
  var delta = (newDp - oldDp) * pp * tp;
  if (delta === 0) return;
  parentG.selectAll(".rank-label").each(function () {
    var currentRank = parseInt(this.textContent.replace("Rank", ""));
    this.textContent = "Rank" + (currentRank + delta);
  });
}

function _updateDpSelect(selId, activeDp) {
  var sel = document.getElementById(selId);
  if (sel) sel.value = activeDp;
}

// ── DP switch handlers (exposed globally for inline onchange) ──

function meshSwitchDp(dpIndex) {
  meshOrigDp = dpIndex;
  meshRebuild();
}

function meshSwitchDpOrig(dpIndex) {
  meshOrigDp = dpIndex;
  meshRebuild();
}

function meshSwitchDpEq(dpIndex) {
  meshEqDp = dpIndex;
  meshRebuild();
}

// ── Main render ──

function meshRebuild() {
  var container = d3.select("#mesh-container");
  // Dismiss pinned tooltip on rebuild
  var tip = document.getElementById('rank-tooltip');
  tip.classList.remove('visible', 'pinned');
  tip.innerHTML = '';
  meshPinnedRank = null;
  container.selectAll("svg").remove();
  meshUpdateSize();

  if (!meshOriginal && !meshEquivalent) return;

  var toolbar = document.getElementById("mesh-config-toolbar");
  toolbar.classList.add("visible");
  document.getElementById("canvas-placeholder").classList.add("hidden");

  var newMode = meshOriginal && meshEquivalent ? "compare" : "single";

  // ── Fast path: DP-only switch (same mode, same tp/pp/dp counts) ──
  if (_renderState.mode === newMode) {
    if (newMode === "single") {
      var entry = meshOriginal || meshEquivalent;
      if (
        entry.tp === _renderState.orig.tp &&
        entry.pp === _renderState.orig.pp &&
        entry.dp === _renderState.orig.dp &&
        meshOrigDp !== _renderState.orig.activeDp
      ) {
        _meshUpdateRanks(container.select("svg"), entry.tp, entry.pp, _renderState.orig.activeDp, meshOrigDp);
        _updateDpSelect("mesh-dp-select", meshOrigDp);
        _renderState.orig.activeDp = meshOrigDp;
        return;
      }
    } else {
      var origSameShape =
        meshOriginal.tp === _renderState.orig.tp &&
        meshOriginal.pp === _renderState.orig.pp &&
        meshOriginal.dp === _renderState.orig.dp;
      var eqSameShape =
        meshEquivalent.tp === _renderState.eq.tp &&
        meshEquivalent.pp === _renderState.eq.pp &&
        meshEquivalent.dp === _renderState.eq.dp;
      var origDpChanged = meshOrigDp !== _renderState.orig.activeDp;
      var eqDpChanged = meshEqDp !== _renderState.eq.activeDp;

      if ((origSameShape && origDpChanged) || (eqSameShape && eqDpChanged)) {
        var childGs = container.select("svg").select(".zoom-layer").selectAll(":scope > g").nodes();

        if (origSameShape && origDpChanged && childGs.length >= 2) {
          _meshUpdateRanks(d3.select(childGs[0]),
            meshOriginal.tp, meshOriginal.pp, _renderState.orig.activeDp, meshOrigDp);
          _updateDpSelect("mesh-dp-sel-orig", meshOrigDp);
          _renderState.orig.activeDp = meshOrigDp;
        }
        if (eqSameShape && eqDpChanged && childGs.length >= 2) {
          _meshUpdateRanks(d3.select(childGs[1]),
            meshEquivalent.tp, meshEquivalent.pp, _renderState.eq.activeDp, meshEqDp);
          _updateDpSelect("mesh-dp-sel-eq", meshEqDp);
          _renderState.eq.activeDp = meshEqDp;
        }
        return;
      }
    }
  }

  // ── Full rebuild ──

  if (meshOriginal && meshEquivalent) {
    // ── Compare Mode ──
    document.getElementById("mesh-tpInput").parentElement.style.display = "none";
    document.getElementById("mesh-ppInput").parentElement.style.display = "none";
    document.getElementById("mesh-dpInput").parentElement.style.display = "none";
    toolbar.querySelector("button").style.display = "none";
    document.getElementById("mesh-npu-count").textContent =
      "原始模型 " + _meshNpuTotal(meshOriginal) + " NPUs  |  最小等效模型 " + _meshNpuTotal(meshEquivalent) + " NPUs";
    document.getElementById("canvas-label").textContent =
      (meshOriginal.name || "原始模型") + "  vs  " + (meshEquivalent.name || "最小等效模型");

    var titleH = 26;
    var gap = 24;
    var availableW = meshWidth - gap;
    var contentH = meshHeight - titleH;

    var dimsOrig = _meshCalcDims(meshOriginal.tp, meshOriginal.pp);
    var dimsEq = _meshCalcDims(meshEquivalent.tp, meshEquivalent.pp);
    var origShare = dimsOrig.dpW / (dimsOrig.dpW + dimsEq.dpW);
    origShare = Math.max(0.45, Math.min(0.6, origShare));
    var origW = availableW * origShare;
    var eqW = availableW * (1 - origShare);

    var scaleOrig = Math.min(1, (origW * 0.72) / dimsOrig.dpW, (contentH * 0.72) / dimsOrig.dpH);
    var scaleEq = Math.min(1, (eqW * 0.72) / dimsEq.dpW, (contentH * 0.72) / dimsEq.dpH);
    var sharedScale = Math.min(scaleOrig, scaleEq);

    var svg = container
      .append("svg")
      .attr("viewBox", "0 0 " + meshWidth + " " + meshHeight)
      .attr("preserveAspectRatio", "xMidYMid meet");

    var zoomLayer = svg.append("g").attr("class", "zoom-layer");

    svg.call(
      d3.zoom()
        .scaleExtent([0.3, 3])
        .on("zoom", function (event) {
          zoomLayer.attr("transform", event.transform);
        })
    );

    zoomLayer
      .append("text")
      .attr("x", origW / 2)
      .attr("y", titleH - 8)
      .attr("text-anchor", "middle")
      .attr("fill", "#58a6ff")
      .attr("font-family", "sans-serif")
      .attr("font-size", "13px")
      .attr("font-weight", "bold")
      .text(meshOriginal.name || "原始模型");
    zoomLayer
      .append("text")
      .attr("x", origW + gap + eqW / 2)
      .attr("y", titleH - 8)
      .attr("text-anchor", "middle")
      .attr("fill", "#58a6ff")
      .attr("font-family", "sans-serif")
      .attr("font-size", "13px")
      .attr("font-weight", "bold")
      .text(meshEquivalent.name || "最小等效模型");

    _meshBuildView(
      zoomLayer.append("g"),
      meshBuildData(meshOriginal.tp, meshOriginal.pp, meshOriginal.dp, meshOrigDp),
      "mesh-dp-sel-orig",
      "meshSwitchDpOrig",
      0,
      titleH,
      origW,
      contentH,
      sharedScale,
      true
    );
    _populateDpSelect("mesh-dp-sel-orig", meshOriginal.dp, meshOrigDp);

    _meshBuildView(
      zoomLayer.append("g"),
      meshBuildData(meshEquivalent.tp, meshEquivalent.pp, meshEquivalent.dp, meshEqDp),
      "mesh-dp-sel-eq",
      "meshSwitchDpEq",
      origW + gap,
      titleH,
      eqW,
      contentH,
      sharedScale,
      false
    );
    _populateDpSelect("mesh-dp-sel-eq", meshEquivalent.dp, meshEqDp);

    _renderState.mode = "compare";
    _renderState.orig.tp = meshOriginal.tp;
    _renderState.orig.pp = meshOriginal.pp;
    _renderState.orig.dp = meshOriginal.dp;
    _renderState.orig.activeDp = meshOrigDp;
    _renderState.eq.tp = meshEquivalent.tp;
    _renderState.eq.pp = meshEquivalent.pp;
    _renderState.eq.dp = meshEquivalent.dp;
    _renderState.eq.activeDp = meshEqDp;
  } else {
    // ── Single View Mode ──
    var entry = meshOriginal || meshEquivalent;
    if (meshOrigDp >= entry.dp) meshOrigDp = entry.dp - 1;

    document.getElementById("mesh-tpInput").value = entry.tp;
    document.getElementById("mesh-ppInput").value = entry.pp;
    document.getElementById("mesh-dpInput").value = entry.dp;
    document.getElementById("mesh-tpInput").parentElement.style.display = "";
    document.getElementById("mesh-ppInput").parentElement.style.display = "";
    document.getElementById("mesh-dpInput").parentElement.style.display = "";
    toolbar.querySelector("button").style.display = "";
    document.getElementById("mesh-npu-count").textContent = "Total NPUs: " + _meshNpuTotal(entry);
    document.getElementById("canvas-label").textContent =
      (entry.name || "组网拓扑") +
      " | " +
      (entry.device_type || "") +
      "  DP" +
      entry.dp +
      "×TP" +
      entry.tp +
      "×PP" +
      entry.pp;

    var data = meshBuildData(entry.tp, entry.pp, entry.dp, meshOrigDp);
    var svg = container
      .append("svg")
      .attr("viewBox", "0 0 " + meshWidth + " " + meshHeight)
      .attr("preserveAspectRatio", "xMidYMid meet");
    var zoomLayer = svg.append("g").attr("class", "zoom-layer");
    svg.call(
      d3.zoom()
        .scaleExtent([0.3, 3])
        .on("zoom", function (event) {
          zoomLayer.attr("transform", event.transform);
        })
    );
    _meshBuildView(zoomLayer, data, "mesh-dp-select", "meshSwitchDp", 0, 0, meshWidth, meshHeight);
    _populateDpSelect("mesh-dp-select", entry.dp, meshOrigDp);

    _renderState.mode = "single";
    _renderState.orig.tp = entry.tp;
    _renderState.orig.pp = entry.pp;
    _renderState.orig.dp = entry.dp;
    _renderState.orig.activeDp = meshOrigDp;
  }

  _renderState.width = meshWidth;
  _renderState.height = meshHeight;
}

// ── Public API ──

function loadMeshData(topoData) {
  var tp = topoData.tp_size || topoData.tp || 4;
  var pp = topoData.pp_size || topoData.pp || 4;
  var dp = topoData.dp_size || topoData.dp || 4;
  var totalNodes = topoData.total_nodes || (dp * tp * pp);
  var deviceType = topoData.device_type || '';
  var rawName = topoData.name || "";
  var name =
    rawName.indexOf("原始") !== -1
      ? "原始模型"
      : rawName.indexOf("等效") !== -1
      ? "最小等效模型"
      : rawName;
  var entry = {
    name: name,
    device_type: deviceType,
    tp: tp,
    pp: pp,
    dp: dp,
    total_nodes: totalNodes,
  };

  if (name.indexOf("原始") !== -1) {
    meshOriginal = entry;
    meshOrigDp = 0;
    meshEstimateOrig = computeAllEstimates(deviceType, totalNodes, dp, tp, pp);
  } else {
    meshEquivalent = entry;
    meshEqDp = 0;
    meshEstimateEq = computeAllEstimates(deviceType, totalNodes, dp, tp, pp);
  }
  meshRebuild();
}

// ── Attach to window for inline onclick / onchange handlers ──

window.loadMeshData = loadMeshData;
window.meshRebuild = meshRebuild;
window.meshSwitchDp = meshSwitchDp;
window.meshSwitchDpOrig = meshSwitchDpOrig;
window.meshSwitchDpEq = meshSwitchDpEq;
window.fetchSimulationData = fetchSimulationData;

// ── Resize handler (debounced) ──

var _resizeTimer = null;
window.addEventListener("resize", function () {
  if (!meshOriginal && !meshEquivalent) return;
  if (_resizeTimer) clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(function () {
    _resizeTimer = null;
    meshRebuild();
  }, 200);
});

// ── Canvas background click dismisses pinned tooltip ──

document.getElementById('mesh-container').addEventListener('click', function (e) {
  if (e.target.tagName === 'svg' || e.target.id === 'mesh-container') {
    var tip = document.getElementById('rank-tooltip');
    tip.classList.remove('visible', 'pinned');
    tip.innerHTML = '';
    d3.selectAll('.tp-rect.pinned').classed('pinned', false);
    meshPinnedRank = null;
  }
});

// ── Escape key dismisses pinned tooltip ──

document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape' && meshPinnedRank) {
    var tip = document.getElementById('rank-tooltip');
    tip.classList.remove('visible', 'pinned');
    tip.innerHTML = '';
    d3.selectAll('.tp-rect.pinned').classed('pinned', false);
    meshPinnedRank = null;
  }
});
