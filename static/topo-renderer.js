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

// ── Estimation engine — fetched from backend ──

var _estimateFetchAbort = {};  // track abort controllers per side to cancel stale requests

async function fetchEstimates(deviceType, totalNodes, dp, tp, pp, side, numLayers, hiddenDim) {
  // Cancel any in-flight request for this side
  if (_estimateFetchAbort[side]) {
    _estimateFetchAbort[side].abort();
  }
  var ctrl = new AbortController();
  _estimateFetchAbort[side] = ctrl;
  var timeout = setTimeout(function () { ctrl.abort(); }, 10000);

  try {
    var body = {
      device_type: deviceType,
      total_nodes: totalNodes,
      dp: dp, tp: tp, pp: pp
    };
    if (numLayers != null) body.num_layers = numLayers;
    if (hiddenDim != null) body.hidden_dim = hiddenDim;

    var resp = await fetch(API + '/session/estimate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: ctrl.signal
    });
    clearTimeout(timeout);
    if (!resp.ok) throw new Error('Estimate API returned ' + resp.status);
    var data = await resp.json();
    var est = {};
    data.cards.forEach(function (c) {
      est[c.global_rank] = c;
    });
    return est;
  } finally {
    clearTimeout(timeout);
    if (_estimateFetchAbort[side] === ctrl) {
      delete _estimateFetchAbort[side];
    }
  }
}

function formatFlops(value) {
  if (value == null) return '—';
  if (value === 0) return '0';
  var exp = Math.floor(Math.log10(Math.abs(value)));
  var mantissa = value / Math.pow(10, exp);
  return mantissa.toFixed(2) + '×10^' + exp;
}

// ── State ──

var meshOriginal = null;       // { name, device_type, tp, pp, dp, total_nodes }
var meshEquivalent = null;
var meshOrigDp = 0;
var meshEqDp = 0;
var meshModelOrig = {};        // { num_layers, hidden_dim } — filled by model_json SSE or REST
var meshModelEq = {};
var meshEstimateOrig = {};     // { global_rank: { flops_per_card, hbm_gb, ... } }
var meshEstimateEq = {};
var meshActualOrig = {};       // { global_rank: CardMetrics } from REST simulation data
var meshActualEq = {};
var meshPinnedRank = null;     // { side: "orig"|"eq", globalRank: number }
var meshPinnedTpInfo = null; // { side: "orig"|"eq", tpIndex: number, ppIndex: number, globalRank: number }

// ── Flowing border animation (RAF-based, more reliable than CSS @keyframes on SVG) ──
var _flowAnimId = null;
function _startFlowAnimation() {
  if (_flowAnimId) return;
  var start = null;
  // dash patterns both 3+2=5, use 4× cycle for seamless wrap
  var cyclePx = 20;
  var cycleMs = 600;
  function tick(ts) {
    if (!start) start = ts;
    var offset = -((ts - start) * cyclePx / cycleMs) % cyclePx;
    var els = document.querySelectorAll('.tp-rect.pinned, .tensor-cell.pinned, .pp-row.pinned');
    for (var i = 0; i < els.length; i++) {
      els[i].setAttribute('stroke-dashoffset', offset);
    }
    _flowAnimId = requestAnimationFrame(tick);
  }
  _flowAnimId = requestAnimationFrame(tick);
}

var modelOriginal = null;    // TrainingModel from SSE model_json
var modelEquivalent = null;
var _formulaCardReady = false;  // true when original mesh is loaded, shows formula card between orig and eq

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
  tpPadX: 18,
  tpPadY: 18,
  headerH: 44,
  ppGap: 18,
  ppHeaderH: 36,
  tpW: 100,
  tpH: 46,
  tpGap: 16,
  maxPpDisplay: 8,
};

// PP color palette — three position types: first, middle, last
// First and last PPs on both sides map to each other; middle PPs map to each other
var _PP_COLORS = ['#58a6ff', '#ff8f40', '#3fb950'];
var _PP_COLORS_HOVER = ['#79c0ff', '#ffb366', '#4ae168'];

var meshWidth = 600;
var meshHeight = 400;

// ── Internal helpers ──

function meshUpdateSize() {
  var wrap = document.getElementById("canvas-svg-wrap");
  meshWidth = wrap.clientWidth || 600;
  meshHeight = Math.max(480, Math.min(wrap.clientHeight * 0.68, 700));
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

// Returns a color index for a PP so that original and equivalent PPs that map to
// each other share the same color. The mapping groups PPs into up to three buckets:
// first (0), middle (1), and last (2). When either side has 2 or fewer PPs the
// "last" bucket collapses into the "middle" bucket (since both map to the same index).
function _getPpColorIndex(ppIdx, ppCount, otherPpCount) {
  if (ppCount <= 1) return 1;
  if (ppIdx === 0) return 0;
  // Last forms a distinct group only when both sides have 3+ PPs
  var hasThreeGroups = ppCount >= 3 && otherPpCount >= 3;
  if (ppIdx === ppCount - 1 && hasThreeGroups) return 2;
  return 1;
}

function _mapRankToOtherSide(side, globalRank) {
  var srcTopo = side === 'orig' ? meshOriginal : meshEquivalent;
  var dstTopo = side === 'orig' ? meshEquivalent : meshOriginal;
  if (!srcTopo || !dstTopo) return null;

  var srcTp = srcTopo.tp, srcPp = srcTopo.pp;
  var srcRanksPerDp = srcTp * srcPp;

  // Extract structural position (PP index, TP index) regardless of DP copy
  var rankInDp = globalRank % srcRanksPerDp;
  if (rankInDp < 0) rankInDp += srcRanksPerDp;
  var srcPpIdx = Math.floor(rankInDp / srcTp);
  var tpIdx = rankInDp % srcTp;

  var dstTp = dstTopo.tp, dstPp = dstTopo.pp;
  var dstPpIdx;
  if (srcPpIdx === 0) {
    dstPpIdx = 0;
  } else if (srcPpIdx === srcPp - 1) {
    dstPpIdx = dstPp - 1;
  } else {
    dstPpIdx = Math.min(1, dstPp - 1);
  }

  var dstTpIdx = Math.min(tpIdx, dstTp - 1);

  // Map result using round-robin DP: source DP → target DP via modulo
  var srcDp = Math.floor(globalRank / srcRanksPerDp);
  var dstDp = srcDp % dstTopo.dp;
  var dstRanksPerDp = dstTp * dstPp;
  return dstDp * dstRanksPerDp + dstPpIdx * dstTp + dstTpIdx;
}

function _getRoundRobinDp(srcDp, dstDpCount) {
  if (!dstDpCount || dstDpCount <= 0) return 0;
  return srcDp % dstDpCount;
}

function _meshCalcDims(tp, pp) {
  var displayCount = Math.min(MESH_CARD.maxPpDisplay, pp);
  var ppW = 120;
  var ellipsisH = 100;
  var ppH =
    MESH_CARD.tpPadY * 2 +
    MESH_CARD.ppHeaderH +
    tp * (MESH_CARD.tpH + MESH_CARD.tpGap) -
    MESH_CARD.tpGap +
    24;
  var ppTotalW = displayCount * ppW + (displayCount - 1) * MESH_CARD.ppGap;
  var dpW = ppTotalW + MESH_CARD.tpPadX * 2;
  var dpH = Math.max(ppH, ellipsisH) + MESH_CARD.headerH + 88;
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

function _buildTooltipBody(globalRank, metrics, isOrig, showHint) {
  var estimate = metrics.estimate;
  var actual = metrics.actual;
  var html = '';

  if (!estimate && !actual) {
    html += '<div class="tooltip-empty">暂无性能数据</div>';
  } else if (!actual || Object.keys(actual).length === 0) {
    html += '<div class="tooltip-section-label">📊 理论估算</div>';
    html += _buildSingleTable(estimate);
  } else {
    html += '<table class="tooltip-table">';
    html += '<tr><th class="col-header left">指标</th><th class="col-header">理论估算值</th><th class="col-header">仿真验证值</th><th class="col-header">差异</th><th class="col-header metric-link-col">详情</th></tr>';
    html += _buildCompareRow('单卡FLOPs', '', estimate, actual, 'flops_per_card', formatFlops, globalRank, isOrig);
    html += _buildCompareRow('HBM', 'GB', estimate, actual, 'hbm_gb', null, globalRank, isOrig);
    html += _buildCompareRow('TP通信', 'GB/micro', estimate, actual, 'tp_comm_gb_per_micro', null, globalRank, isOrig);
    html += _buildCompareRow('PP通信', 'MB/micro', estimate, actual, 'pp_comm_mb_per_micro', null, globalRank, isOrig);
    html += _buildCompareRow('DP通信', 'GB/step', estimate, actual, 'dp_comm_gb_per_step', null, globalRank, isOrig);
    html += '</table>';

    if (actual && Object.keys(actual).length > 0) {
      html += '<button class="tooltip-detail-btn" onclick="event.stopPropagation();meshPinnedRank={side:\'' + (isOrig ? 'orig' : 'eq') + '\',globalRank:' + globalRank + '};switchCanvasTab(\'results\')">📊 仿真详情</button>';
    }
  }

  if (showHint !== false) {
    html += '<div class="tooltip-pin-hint">🖱 点击固定 · 再点取消</div>';
  }
  return html;
}

function _buildTooltipHTML(globalRank, metrics, isOrig) {
  var deviceType = metrics.deviceType || '';
  var html = '<div class="tooltip-header">Rank ' + globalRank;
  if (deviceType) html += ' <span class="tooltip-device">' + deviceType + '</span>';
  html += '</div><div class="tooltip-body">';
  html += _buildTooltipBody(globalRank, metrics, isOrig);
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

function _buildCompareRow(label, unit, estimate, actual, key, fmt, globalRank, isOrig) {
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

  var linkHtml = '';
  if (key === 'flops_per_card') {
    linkHtml = '<a class="metric-link" href="javascript:void(0)" onclick="event.stopPropagation();toggleTooltipDetail(' + globalRank + ',' + (isOrig ? 'true' : 'false') + ',\'flops\')">查看</a>';
  } else {
    var linkType = key.replace(/_per_micro|_per_step|_gb|_mb/g, '');
    if (key === 'hbm_gb') linkType = 'hbm';
    else if (key === 'tp_comm_gb_per_micro') linkType = 'tp-comm';
    else if (key === 'pp_comm_mb_per_micro') linkType = 'pp-comm';
    else if (key === 'dp_comm_gb_per_step') linkType = 'dp-comm';
    linkHtml = '<a class="metric-link" href="javascript:void(0)" onclick="event.stopPropagation();toggleTooltipDetail(' + globalRank + ',' + (isOrig ? 'true' : 'false') + ',\'' + linkType + '\')">查看</a>';
  }

  return '<tr><td class="metric-label">' + label + '</td>' +
    '<td class="metric-val">' + ev + uv + '</td>' +
    '<td class="metric-val actual">' + av + uv + '</td>' +
    '<td class="metric-delta ' + deltaClass + '">' + delta + '</td>' +
    '<td class="metric-val" style="text-align:center">' + linkHtml + '</td></tr>';
}

function _buildLinkedTooltipHTML(origRank, origMetrics, eqRank, eqMetrics) {
  var html = '';

  // Original section
  html += '<div class="tooltip-header linked">';
  html += '<span class="linked-badge orig">原始组网</span> Rank ' + origRank;
  var origDevice = origMetrics.deviceType || '';
  if (origDevice) html += ' <span class="tooltip-device">' + origDevice + '</span>';
  html += '</div>';
  html += '<div class="tooltip-body">';
  html += _buildTooltipBody(origRank, origMetrics, true, false);
  html += '</div>';

  // Divider
  html += '<div class="tooltip-linked-divider"></div>';

  // Equivalent section
  html += '<div class="tooltip-header linked">';
  html += '<span class="linked-badge eq">等效组网</span> Rank ' + eqRank;
  var eqDevice = eqMetrics.deviceType || '';
  if (eqDevice) html += ' <span class="tooltip-device">' + eqDevice + '</span>';
  html += '</div>';
  html += '<div class="tooltip-body">';
  html += _buildTooltipBody(eqRank, eqMetrics, false, false);
  html += '</div>';

  html += '<div class="tooltip-pin-hint">🖱 点击固定 · 再点取消</div>';
  return html;
}

function showTooltip(event, globalRank, isOrig) {
  var tip = document.getElementById('rank-tooltip');
  var metrics = _getMetrics(globalRank, isOrig);
  tip.innerHTML = _buildTooltipHTML(globalRank, metrics, isOrig);
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

function _positionTooltipAtElement(tip, el) {
  var box = el.getBoundingClientRect();
  var x = box.right + 8;
  var y = box.top;
  var tw = tip.offsetWidth;
  var th = tip.offsetHeight;
  if (x + tw > window.innerWidth - 8) x = box.left - tw - 8;
  if (y + th > window.innerHeight - 8) y = window.innerHeight - th - 8;
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
  var tipMapped = document.getElementById('rank-tooltip-mapped');
  if (!tip.classList.contains('pinned')) {
    tip.classList.remove('visible');
    tip.innerHTML = '';
    if (tipMapped) { tipMapped.classList.remove('visible'); tipMapped.innerHTML = ''; }
    d3.selectAll('.tp-rect.pinned').classed('pinned', false);
    meshPinnedRank = null;
  }
}

function _clearBothTooltips() {
  var tip = document.getElementById('rank-tooltip');
  var tipMapped = document.getElementById('rank-tooltip-mapped');
  tip.classList.remove('visible', 'pinned');
  tip.innerHTML = '';
  if (tipMapped) { tipMapped.classList.remove('visible', 'pinned'); tipMapped.innerHTML = ''; }
  _closeDetailPanels();
  d3.selectAll('.tp-rect.pinned').classed('pinned', false);
  meshPinnedRank = null;
  meshPinnedTpInfo = null;
  modelRebuild();
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
        // Mark simulation as completed if results exist
        if (data.original_simulation || data.equivalent_simulation) {
          window._simCompleted = true;
          if (typeof checkSimReady === 'function') checkSimReady();
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

  var ppW = 120;
  var ellipsisH = 100;
  var ppH =
    MESH_CARD.tpPadY * 2 +
    MESH_CARD.ppHeaderH +
    tpCount * (MESH_CARD.tpH + MESH_CARD.tpGap) -
    MESH_CARD.tpGap +
    24;
  var ppTotalW = displayCount * ppW + (displayCount - 1) * MESH_CARD.ppGap;
  var dpW = ppTotalW + MESH_CARD.tpPadX * 2;
  var dpH = Math.max(ppH, ellipsisH) + MESH_CARD.headerH + 88;

  var scale = sharedScale != null ? sharedScale : 0.5;
  var dpX = viewX + (viewW - dpW * scale) / 2;
  var dpY = viewY + 8 * scale;

  // DP Stack shadows — 5 layers for original, 3 for equivalent
  var dpStackCount = isOrig ? 5 : 3;
  var dpMaxOff = 90, dpMinOff = isOrig ? 10 : 30;
  var dpShadows = [];
  for (var si = 0; si < dpStackCount; si++) {
    var off = dpMaxOff - si * (dpMaxOff - dpMinOff) / (dpStackCount - 1);
    dpShadows.push([off, off]);
  }
  var shadowG = parentG.append("g").attr("class", "dp-shadows");
  dpShadows.forEach(function (off, i) {
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
  var selectW = Math.max(64, 80 * scale);
  var foY = dpY + 6 * scale;
  var foH = hdrH - 6 * scale;
  var selectH = Math.min(foH - 6, Math.max(22, 30 * scale));
  var foW = selectW + 8;
  dpG
    .append("foreignObject")
    .attr("x", dpX + 20 * scale)
    .attr("y", foY)
    .attr("width", foW)
    .attr("height", foH)
    .append("xhtml:div")
    .style("width", "100%")
    .style("height", "100%")
    .html((function () {
      var opts = '';
      for (var oi = 0; oi < cfg.dpCount; oi++) {
        opts += '<option value="' + oi + '"' + (oi === cfg.activeDp ? ' selected' : '') + '>DP ' + oi + '</option>';
      }
      var ff = Math.max(12, 13 * scale);
      return '<div class="dp-select-wrap" style="width:100%;height:100%;display:flex;align-items:center;">' +
        '<select id="' + dpSelectId + '" class="dp-select" onchange="' + switchFn + '(parseInt(this.value))" style="' +
        'font-size:' + ff + 'px;width:100%;">' + opts + '</select></div>';
    })());

  // Info text
  var npuTotal = cfg.tpCount * cfg.ppCount * cfg.dpCount;
  var infoText = "TP" + cfg.tpCount + "×PP" + cfg.ppCount + "×DP" + cfg.dpCount + " | " + npuTotal + " NPUs";
  dpG
    .append("text")
    .attr("x", dpX + dpW * scale - 16 * scale)
    .attr("y", dpY + MESH_CARD.headerH * scale / 2)
    .attr("dy", "0.35em")
    .attr("text-anchor", "end")
    .attr("class", "info-text")
    .attr("font-size", Math.max(11, 12 * scale) + "px")
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

    // Assign color by PP position so mapped groups share the same color across sides
    var otherPpCount = isOrig ? (meshEquivalent ? meshEquivalent.pp : ppList.length) : (meshOriginal ? meshOriginal.pp : ppList.length);
    var ppColorIdx = _getPpColorIndex(pp.id, ppList.length, otherPpCount);
    var ppColor = _PP_COLORS[ppColorIdx];
    var ppColorHover = _PP_COLORS_HOVER[ppColorIdx];

    ppG
      .append("rect")
      .attr("x", px)
      .attr("y", py)
      .attr("width", ppW * scale)
      .attr("height", ppH * scale)
      .attr("rx", 8 * scale)
      .attr("ry", 8 * scale)
      .attr("class", "pp-card")
      .attr("style", "stroke: " + ppColor);
    ppG
      .append("rect")
      .attr("x", px)
      .attr("y", py)
      .attr("width", ppW * scale)
      .attr("height", MESH_CARD.ppHeaderH * scale)
      .attr("rx", 8 * scale)
      .attr("ry", 8 * scale)
      .attr("class", "pp-header")
      .attr("style", "fill: " + ppColor);
    ppG
      .append("rect")
      .attr("x", px)
      .attr("y", py + (MESH_CARD.ppHeaderH * scale) / 2)
      .attr("width", ppW * scale)
      .attr("height", (MESH_CARD.ppHeaderH * scale) / 2)
      .attr("class", "pp-header")
      .attr("style", "fill: " + ppColor);
    var ppLabelSize = Math.max(10, 13 * scale);
    var ppLabelY = py + (MESH_CARD.ppHeaderH * scale) / 2 + ppLabelSize * 0.35;
    ppG
      .append("text")
      .attr("x", px + (ppW * scale) / 2)
      .attr("y", ppLabelY)
      .attr("text-anchor", "middle")
      .attr("class", "pp-label")
      .attr("font-size", ppLabelSize + "px")
      .text(pp.label);

    // Hover glow via JS (per-color filters replace the old single #pp-card-glow)
    ppG.on("mouseenter", function () {
      d3.select(this).select(".pp-card")
        .attr("style", "stroke: " + ppColorHover + "; stroke-width: 1.5; filter: url(#pp-glow-" + ppColorIdx + ")");
    }).on("mouseleave", function () {
      d3.select(this).select(".pp-card")
        .attr("style", "stroke: " + ppColor);
    });

    var tpX = px + ((ppW - MESH_CARD.tpW) * scale) / 2;
    var tpY = py + MESH_CARD.ppHeaderH * scale + MESH_CARD.tpPadY * scale;
    pp.tps.forEach(function (tp, ti) {
      var ty = tpY + ti * (MESH_CARD.tpH + MESH_CARD.tpGap) * scale;
      var side = isOrig ? 'orig' : 'eq';
      var hasActual = _hasActualData(side);
      var isPinned = meshPinnedRank && (
        (meshPinnedRank.side === side && meshPinnedRank.globalRank === tp.globalRank) ||
        (meshPinnedRank.mappedRank != null && meshPinnedRank.side !== side && meshPinnedRank.mappedRank === tp.globalRank)
      );
      var rectClass = 'tp-rect' + (hasActual ? ' has-data' : '') + (isPinned ? ' pinned' : '');
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
          showTooltip(event, parseInt(this.getAttribute("data-rank")), isOrig);
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
          var globalRank = parseInt(this.getAttribute("data-rank"));
          var isCompare = !!(meshOriginal && meshEquivalent);
          var mappedRank = isCompare ? _mapRankToOtherSide(side, globalRank) : null;
          var otherSide = side === 'orig' ? 'eq' : 'orig';
          var alreadyPinned = meshPinnedRank && meshPinnedRank.globalRank === globalRank && meshPinnedRank.side === side;

          d3.selectAll('.tp-rect.pinned').classed('pinned', false);
          var tip = document.getElementById('rank-tooltip');
          var tipMapped = document.getElementById('rank-tooltip-mapped');

          if (alreadyPinned) {
            _clearBothTooltips();
          } else {
            d3.select(this).classed('pinned', true);
            if (mappedRank != null) {
              d3.selectAll('.tp-rect[data-rank="' + mappedRank + '"][data-side="' + otherSide + '"]').classed('pinned', true);
            }

            if (isCompare && mappedRank != null) {
              var origRank = side === 'orig' ? globalRank : mappedRank;
              var eqRank = side === 'orig' ? mappedRank : globalRank;
              var origMetrics = _getMetrics(origRank, true);
              var eqMetrics = _getMetrics(eqRank, false);

              // Main tooltip: the side the user clicked on
              var clickedIsOrig = side === 'orig';
              tip.innerHTML = _buildTooltipHTML(clickedIsOrig ? origRank : eqRank, clickedIsOrig ? origMetrics : eqMetrics, clickedIsOrig);
              tip.classList.add('visible', 'pinned');
              _positionTooltip(event, tip);

              // Mapped tooltip: the other side, positioned near its rank rect
              var otherRank = clickedIsOrig ? eqRank : origRank;
              var otherIsOrig = !clickedIsOrig;
              var otherMetrics = _getMetrics(otherRank, otherIsOrig);
              tipMapped.innerHTML = _buildTooltipHTML(otherRank, otherMetrics, otherIsOrig);
              tipMapped.classList.add('visible', 'pinned');
              var mappedRect = document.querySelector('.tp-rect[data-rank="' + mappedRank + '"][data-side="' + otherSide + '"]');
              if (mappedRect) {
                _positionTooltipAtElement(tipMapped, mappedRect);
              } else {
                _positionTooltip(event, tipMapped);
              }
            } else {
              var metrics = _getMetrics(globalRank, isOrig);
              tip.innerHTML = _buildTooltipHTML(globalRank, metrics, isOrig);
              tip.classList.add('visible', 'pinned');
              _positionTooltip(event, tip);
            }
            meshPinnedRank = { side: side, globalRank: globalRank, mappedRank: mappedRank };
            meshPinnedTpInfo = { side: side, tpIndex: ti, ppIndex: pp.id, globalRank: globalRank };
            modelRebuild();
          }
        });
      ppG
        .append("text")
        .attr("x", tpX + 6 * scale)
        .attr("y", ty + 16 * scale)
        .attr("text-anchor", "start")
        .attr("class", "tp-label")
        .attr("font-size", Math.max(7, 9 * scale) + "px")
        .attr("pointer-events", "none")
        .text(tp.label);
      ppG
        .append("text")
        .attr("x", tpX + 6 * scale)
        .attr("y", ty + 34 * scale)
        .attr("text-anchor", "start")
        .attr("class", "rank-label")
        .attr("font-size", "10px")
        .attr("pointer-events", "none")
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
  // Options already generated; just update selected index
  sel.value = activeDp;
}

function _meshNpuTotal(entry) {
  return entry.tp * entry.pp * entry.dp;
}

function _meshUpdateRanks(parentG, tp, pp, oldDp, newDp) {
  var delta = (newDp - oldDp) * pp * tp;
  if (delta === 0) return;
  parentG.selectAll(".tp-rect").each(function () {
    var currentRank = parseInt(this.getAttribute("data-rank"));
    this.setAttribute("data-rank", currentRank + delta);
  });
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

// ── Formula Card (static formulas, no computation) ──

function _renderFormulaCard(parentG, viewX, viewY, viewW, viewH) {
  var cardG = parentG.append("g").attr("class", "formula-card-group");
  var pad = 14;

  // Spacing tuned for 340px card
  var lineH_label = 20;
  var lineH_formula = 16;
  var lineH_desc = 14;
  var sectGap = 8;

  // Section definitions (long formulas split to fit 280px content area)
  var sections = [
    {
      label: "▸ 单卡FLOPs",
      lines: [
        "[(72·B·S·H² + 12·B·S²·H) ÷ TP] · L÷PP"
      ],
      desc: "计算量与 TP 成反比，与 L/PP 成正比"
    },
    {
      label: "▸ HBM 占用",
      lines: [
        "[L·(12H²+4H)÷(TP·PP) + B·S·H·L÷PP",
        " + L·(12H²+4H)÷(TP·PP)] · a ÷ 1e9"
      ],
      desc: "权重+梯度+优化器 / 激活值 / 通信缓冲"
    },
    {
      label: "▸ 通信流量",
      lines: [
        "TP: L÷PP · 32·B·S·H ÷ 1e6 GB/micro",
        "DP: 8·L·(4H²+12H²)÷(TP·PP)",
        "    ÷ 1e9 GB/step",
        "PP: 4·B·S·H ÷ 1e9 GB/step"
      ],
      desc: "TP/DP通信与 TP·PP 成反比，PP通信仅取决于模型规模"
    },
  ];

  // Compute card intrinsic height
  var titleFont = 16;
  var cardH = pad + titleFont + 10; // top pad + title area + title gap
  sections.forEach(function(sec) {
    cardH += lineH_label;
    cardH += sec.lines.length * lineH_formula;
    cardH += lineH_desc + sectGap;
  });
  cardH += pad - sectGap; // remove trailing gap, add bottom pad

  // Center card vertically in the available space
  var cardY = viewY + (viewH - cardH) / 2;
  if (cardY < viewY) cardY = viewY;

  // Card background
  cardG.append("rect")
    .attr("x", viewX).attr("y", cardY)
    .attr("width", viewW).attr("height", cardH)
    .attr("rx", 8).attr("ry", 8)
    .attr("class", "formula-card-rect");

  // Title
  cardG.append("text")
    .attr("x", viewX + viewW / 2).attr("y", cardY + pad + titleFont)
    .attr("text-anchor", "middle")
    .attr("fill", "#39bae6").attr("font-weight", "bold")
    .attr("font-size", titleFont + "px").attr("font-family", "var(--font-sans)")
    .text("📐 等效机制分析");

  var curY = cardY + pad + titleFont + 10;

  sections.forEach(function(sec) {
    // Section label
    cardG.append("text")
      .attr("x", viewX + pad).attr("y", curY + 15)
      .attr("fill", "var(--teal)").attr("font-weight", "600")
      .attr("font-size", "12px").attr("font-family", "var(--font-sans)")
      .text(sec.label);
    curY += lineH_label;

    // Formula lines
    sec.lines.forEach(function(line) {
      cardG.append("text")
        .attr("x", viewX + pad + 4).attr("y", curY + 14)
        .attr("fill", "var(--text-primary)").attr("font-size", "11px")
        .attr("font-family", "var(--font-mono)")
        .text(line);
      curY += lineH_formula;
    });

    // Description
    cardG.append("text")
      .attr("x", viewX + pad + 4).attr("y", curY + 12)
      .attr("fill", "var(--text-muted)").attr("font-size", "10px")
      .attr("font-family", "var(--font-sans)")
      .text(sec.desc);
    curY += lineH_desc + sectGap;
  });
}

// ── DP switch handlers (exposed globally for inline onchange) ──

function meshSwitchDp(dpIndex) {
  meshOrigDp = dpIndex;
  meshRebuild();
}

function meshSwitchDpOrig(dpIndex) {
  meshOrigDp = dpIndex;
  if (meshEquivalent && meshEquivalent.dp > 0) {
    meshEqDp = _getRoundRobinDp(dpIndex, meshEquivalent.dp);
  }
  meshRebuild();
}

function meshSwitchDpEq(dpIndex) {
  meshEqDp = dpIndex;
  if (meshOriginal && meshOriginal.dp > 0) {
    meshOrigDp = _getRoundRobinDp(dpIndex, meshOriginal.dp);
  }
  meshRebuild();
}

// ── Main render ──

function meshRebuild() {
  // Dismiss pinned tooltip on rebuild
  var tip = document.getElementById('rank-tooltip');
  tip.classList.remove('visible', 'pinned');
  tip.innerHTML = '';
  var tipMapped = document.getElementById('rank-tooltip-mapped');
  if (tipMapped) { tipMapped.classList.remove('visible', 'pinned'); tipMapped.innerHTML = ''; }
  _closeDetailPanels();
  meshPinnedRank = null;
  meshPinnedTpInfo = null;
  meshUpdateSize();

  var hasTopo = !!(meshOriginal || meshEquivalent);
  var model = modelOriginal || modelEquivalent;
  var hasModel = !!(model && model.layers && model.layers.length);

  if (!hasTopo) {
    canvasRebuild();
    return;
  }

  // ── Fast path: DP-only switch (same mode, same tp/pp/dp counts, topo-only) ──
  if (!hasModel) {
    var container = d3.select("#canvas-section");
    var newMode = (meshOriginal && meshEquivalent) ? "compare" : "single";

    if (_renderState.mode === newMode) {
      if (newMode === "single") {
        var entry = meshOriginal || meshEquivalent;
        if (
          entry.tp === _renderState.orig.tp &&
          entry.pp === _renderState.orig.pp &&
          entry.dp === _renderState.orig.dp &&
          meshOrigDp !== _renderState.orig.activeDp
        ) {
          var svgRoot = container.select("svg");
          if (!svgRoot.empty()) {
            _meshUpdateRanks(svgRoot, entry.tp, entry.pp, _renderState.orig.activeDp, meshOrigDp);
            _updateDpSelect("mesh-dp-select", meshOrigDp);
            _renderState.orig.activeDp = meshOrigDp;
            return;
          }
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
          if (origDpChanged || eqDpChanged) return;
        }
      }
    }
  }

  // Full rebuild
  canvasRebuild();
}

// ── Public API ──

function canvasRebuild() {
  _closeDetailPanels();
  meshUpdateSize();

  var hasTopo = !!(meshOriginal || meshEquivalent);
  var model = modelOriginal || modelEquivalent;
  var hasModel = !!(model && model.layers && model.layers.length);

  if (!hasTopo && !hasModel) {
    document.getElementById("canvas-section").style.display = 'none';
    document.getElementById("canvas-placeholder").classList.remove("hidden");
    document.getElementById("mesh-config-toolbar").classList.remove("visible");
    document.getElementById("canvas-label").textContent = '等待组网数据...';
    _renderState.mode = null;
    return;
  }

  document.getElementById("canvas-placeholder").classList.add("hidden");
  var toolbar = document.getElementById("mesh-config-toolbar");
  if (hasTopo) toolbar.classList.add("visible");
  else toolbar.classList.remove("visible");

  var container = d3.select("#canvas-section");
  container.selectAll("svg").remove();
  document.getElementById("canvas-section").style.display = '';

  // ── Calculate topology height (content can exceed meshHeight for large tp/pp) ──
  var topoH = 0;
  if (hasTopo) {
    if (meshOriginal && meshEquivalent) {
      var _cmpTitleH = 26;
      var dO = _meshCalcDims(meshOriginal.tp, meshOriginal.pp);
      var dE = _meshCalcDims(meshEquivalent.tp, meshEquivalent.pp);
      var maxDpH = Math.max(dO.dpH, dE.dpH);
      // DP card at scale 0.5 + shadow offset (90*0.5) + title + small margin
      topoH = _cmpTitleH + Math.ceil((maxDpH + 90) * 0.5) + 4;
    } else {
      var entry = meshOriginal || meshEquivalent;
      var dims = _meshCalcDims(entry.tp, entry.pp);
      topoH = Math.ceil((dims.dpH + 90) * 0.5) + 8;
    }
  }

  var modelH = hasModel ? 520 : 0;
  var sectionGap = (hasTopo && hasModel) ? 6 : 0;
  var totalH = topoH + sectionGap + modelH;

  var svg = container.append("svg")
    .attr("viewBox", "0 0 " + meshWidth + " " + totalH)
    .attr("preserveAspectRatio", "xMidYMid meet");
  var zoomLayer = svg.append("g").attr("class", "zoom-layer");

  svg.call(
    d3.zoom()
      .scaleExtent([0.3, 3])
      .filter(function (event) {
        if (event.type === 'wheel' && !event.ctrlKey) return false;
        return event.type !== 'dblclick';
      })
      .wheelDelta(function(event) {
        // Smoother, finer zoom steps on Ctrl+scroll
        var factor = event.deltaMode === 1 ? 0.02 : 0.001;
        return -event.deltaY * factor;
      })
      .on("zoom", function (event) {
        zoomLayer.attr("transform", event.transform);
      })
  );

  // Arrow marker defs and hover filters
  if (hasModel || hasTopo) {
    var defs = svg.append('defs');

    if (hasModel) {
      defs.append('marker')
        .attr('id', 'arrow-dataflow')
        .attr('viewBox', '0 0 10 10')
        .attr('refX', 5).attr('refY', 5)
        .attr('markerWidth', 6).attr('markerHeight', 6)
        .attr('orient', 'auto-start-reverse')
        .append('path')
        .attr('d', 'M 0 0 L 10 5 L 0 10 z')
        .attr('fill', 'var(--text-muted)').attr('opacity', 0.5);

      // Hover glow filter for model diagram
      var modelFilter = defs.append('filter')
        .attr('id', 'model-hover-glow')
        .attr('x', '-30%').attr('y', '-30%')
        .attr('width', '160%').attr('height', '160%');
      modelFilter.append('feDropShadow')
        .attr('dx', 0).attr('dy', 3)
        .attr('stdDeviation', 4)
        .attr('flood-color', '#39bae6')
        .attr('flood-opacity', 0.45);
    }

    // Hover glow filter for formula card
    var cardFilter = defs.append('filter')
      .attr('id', 'formula-card-glow')
      .attr('x', '-30%').attr('y', '-30%')
      .attr('width', '160%').attr('height', '160%');
    cardFilter.append('feDropShadow')
      .attr('dx', 0).attr('dy', 4)
      .attr('stdDeviation', 10)
      .attr('flood-color', '#39bae6')
      .attr('flood-opacity', 0.55);

    // Hover glow filter for DP card
    var dpCardFilter = defs.append('filter')
      .attr('id', 'dp-card-glow')
      .attr('x', '-20%').attr('y', '-20%')
      .attr('width', '140%').attr('height', '140%');
    dpCardFilter.append('feDropShadow')
      .attr('dx', 0).attr('dy', 3)
      .attr('stdDeviation', 8)
      .attr('flood-color', '#58a6ff')
      .attr('flood-opacity', 0.5);

    // Hover glow filters for PP cards — one per position-type color
    _PP_COLORS.forEach(function(color, i) {
      var ppFilter = defs.append('filter')
        .attr('id', 'pp-glow-' + i)
        .attr('x', '-25%').attr('y', '-25%')
        .attr('width', '150%').attr('height', '150%');
      ppFilter.append('feDropShadow')
        .attr('dx', 0).attr('dy', 2)
        .attr('stdDeviation', 6)
        .attr('flood-color', color)
        .attr('flood-opacity', 0.5);
    });

    // Hover glow filter for TP rank rects
    var tpRectFilter = defs.append('filter')
      .attr('id', 'tp-rect-glow')
      .attr('x', '-40%').attr('y', '-40%')
      .attr('width', '180%').attr('height', '180%');
    tpRectFilter.append('feDropShadow')
      .attr('dx', 0).attr('dy', 2)
      .attr('stdDeviation', 5)
      .attr('flood-color', '#3fb950')
      .attr('flood-opacity', 0.7);

    // Hover glow filter for tensor grid cells
    var tensorCellFilter = defs.append('filter')
      .attr('id', 'tensor-cell-glow')
      .attr('x', '-40%').attr('y', '-40%')
      .attr('width', '180%').attr('height', '180%');
    tensorCellFilter.append('feDropShadow')
      .attr('dx', 0).attr('dy', 2)
      .attr('stdDeviation', 4)
      .attr('flood-color', '#ff8f40')
      .attr('flood-opacity', 0.6);

    defs.append('style')
      .attr('type', 'text/css')
      .text([
        '.model-node { cursor: pointer; transition: stroke-width 0.2s ease, filter 0.2s ease; }',
        '.formula-card-rect { cursor: default; transition: filter 0.3s ease, stroke 0.3s ease, transform 0.3s ease; }',
        '.formula-card-rect:hover { filter: url(#formula-card-glow); stroke: #39bae6; stroke-width: 1.5; transform: translateY(-2px); }',
        '.formula-card-group text { pointer-events: none; }',
        '.dp-card { transition: filter 0.3s ease, stroke 0.3s ease; }',
        '.dp-card-group:hover .dp-card { filter: url(#dp-card-glow); stroke: #79c0ff; stroke-width: 2.5; }',
        '.dp-card-group:hover .dp-shadow { filter: url(#dp-card-glow); }',
        '.pp-card { transition: filter 0.3s ease, stroke 0.3s ease; }',
        '.tp-rect { transition: filter 0.2s ease, stroke 0.2s ease, fill 0.2s ease; }',
        '.tp-rect:hover { filter: url(#tp-rect-glow); stroke: #4ae168; stroke-width: 1.5; fill: #1a2e1f; }',
        '.tensor-cell { transition: filter 0.2s ease, stroke 0.2s ease; }',
        '.tensor-cell:hover { filter: url(#tensor-cell-glow); stroke: #ff8f40; stroke-width: 1.2; }',
        '.tensor-cell-label { pointer-events: none; }',
      ].join(' '));
  }

  // ── Layout dimensions used by both topology and model sections ──
  var _topoLayout = null; // { mode, origW, cardW, eqW, gap1, gap2, titleH, contentH }

  // ═══ Render topology ═══
  if (hasTopo) {
    if (meshOriginal && meshEquivalent) {
      // ── Three-Part Mode: Orig + Card + Eq ──
      document.getElementById("mesh-tpInput").parentElement.style.display = "none";
      document.getElementById("mesh-ppInput").parentElement.style.display = "none";
      document.getElementById("mesh-dpInput").parentElement.style.display = "none";
      toolbar.querySelector("button").style.display = "none";
      document.getElementById("mesh-npu-count").textContent =
        "原始组网 " + _meshNpuTotal(meshOriginal) + " NPUs  |  最小等效组网 " + _meshNpuTotal(meshEquivalent) + " NPUs";
      document.getElementById("canvas-label").textContent =
        (meshOriginal.name || "原始组网") + "  vs  " + (meshEquivalent.name || "最小等效组网");

      var _tH = 26, _gap = 10;
      var _cardW = 340; // fixed card width, centered between topologies
      var _availForTopos = meshWidth - _cardW - _gap * 2;
      var dimsOrig = _meshCalcDims(meshOriginal.tp, meshOriginal.pp);
      var dimsEq = _meshCalcDims(meshEquivalent.tp, meshEquivalent.pp);
      var origShare = dimsOrig.dpW / (dimsOrig.dpW + dimsEq.dpW);
      origShare = Math.max(0.4, Math.min(0.6, origShare));
      var _origW = _availForTopos * origShare;
      var _eqW = _availForTopos * (1 - origShare);
      var _contentH = topoH - _tH;
      var _sharedScale = 0.5;

      zoomLayer.append("text")
        .attr("x", _origW / 2).attr("y", _tH - 8)
        .attr("text-anchor", "middle").attr("fill", "#58a6ff")
        .attr("font-family", "sans-serif").attr("font-size", "13px").attr("font-weight", "bold")
        .text(meshOriginal.name || "原始组网");
      zoomLayer.append("text")
        .attr("x", _origW + _gap + _cardW + _gap + _eqW / 2).attr("y", _tH - 8)
        .attr("text-anchor", "middle").attr("fill", "#58a6ff")
        .attr("font-family", "sans-serif").attr("font-size", "13px").attr("font-weight", "bold")
        .text(meshEquivalent.name || "最小等效组网");

      _meshBuildView(
        zoomLayer.append("g"),
        meshBuildData(meshOriginal.tp, meshOriginal.pp, meshOriginal.dp, meshOrigDp),
        "mesh-dp-sel-orig", "meshSwitchDpOrig",
        0, _tH, _origW, _contentH, _sharedScale, true
      );
      _populateDpSelect("mesh-dp-sel-orig", meshOriginal.dp, meshOrigDp);

      _renderFormulaCard(zoomLayer.append("g"),
        _origW + _gap, _tH, _cardW, _contentH);

      _meshBuildView(
        zoomLayer.append("g"),
        meshBuildData(meshEquivalent.tp, meshEquivalent.pp, meshEquivalent.dp, meshEqDp),
        "mesh-dp-sel-eq", "meshSwitchDpEq",
        _origW + _gap + _cardW + _gap, _tH, _eqW, _contentH, _sharedScale, false
      );
      _populateDpSelect("mesh-dp-sel-eq", meshEquivalent.dp, meshEqDp);

      _topoLayout = { mode: "three", origW: _origW, cardW: _cardW, eqW: _eqW, gap: _gap, titleH: _tH, cardX: _origW + _gap };
      _renderState.mode = "compare";
      _renderState.orig.tp = meshOriginal.tp; _renderState.orig.pp = meshOriginal.pp;
      _renderState.orig.dp = meshOriginal.dp; _renderState.orig.activeDp = meshOrigDp;
      _renderState.eq.tp = meshEquivalent.tp; _renderState.eq.pp = meshEquivalent.pp;
      _renderState.eq.dp = meshEquivalent.dp; _renderState.eq.activeDp = meshEqDp;
    } else if (_formulaCardReady && meshOriginal) {
      // ── Two-Part Mode: Orig + Card ──
      var _tH2 = 26, _gap2 = 10;
      var _cardW2 = 340; // fixed card width
      var _origW2 = meshWidth - _cardW2 - _gap2;
      var _contentH2 = topoH - _tH2;
      var _sharedScale2 = 0.5;
      var entry2 = meshOriginal;

      if (meshOrigDp >= entry2.dp) meshOrigDp = entry2.dp - 1;

      document.getElementById("mesh-tpInput").parentElement.style.display = "none";
      document.getElementById("mesh-ppInput").parentElement.style.display = "none";
      document.getElementById("mesh-dpInput").parentElement.style.display = "none";
      toolbar.querySelector("button").style.display = "none";
      document.getElementById("mesh-npu-count").textContent = "Total NPUs: " + _meshNpuTotal(entry2);
      document.getElementById("canvas-label").textContent =
        (entry2.name || "组网拓扑") + " | " + (entry2.device_type || "") +
        "  DP" + entry2.dp + "×TP" + entry2.tp + "×PP" + entry2.pp;

      zoomLayer.append("text")
        .attr("x", _origW2 / 2).attr("y", _tH2 - 8)
        .attr("text-anchor", "middle").attr("fill", "#58a6ff")
        .attr("font-family", "sans-serif").attr("font-size", "13px").attr("font-weight", "bold")
        .text(entry2.name || "原始组网");

      _meshBuildView(
        zoomLayer.append("g"),
        meshBuildData(entry2.tp, entry2.pp, entry2.dp, meshOrigDp),
        "mesh-dp-select", "meshSwitchDp",
        0, _tH2, _origW2, _contentH2, _sharedScale2, true
      );
      _populateDpSelect("mesh-dp-select", entry2.dp, meshOrigDp);

      _renderFormulaCard(zoomLayer.append("g"),
        _origW2 + _gap2, _tH2, _cardW2, _contentH2);

      _topoLayout = { mode: "two", origW: _origW2, cardW: _cardW2, gap: _gap2, titleH: _tH2, cardX: _origW2 + _gap2 };
      _renderState.mode = "single_card";
      _renderState.orig.tp = entry2.tp; _renderState.orig.pp = entry2.pp;
      _renderState.orig.dp = entry2.dp; _renderState.orig.activeDp = meshOrigDp;
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
        (entry.name || "组网拓扑") + " | " + (entry.device_type || "") +
        "  DP" + entry.dp + "×TP" + entry.tp + "×PP" + entry.pp;

      var data = meshBuildData(entry.tp, entry.pp, entry.dp, meshOrigDp);
      _meshBuildView(zoomLayer, data, "mesh-dp-select", "meshSwitchDp", 0, 0, meshWidth, topoH);
      _populateDpSelect("mesh-dp-select", entry.dp, meshOrigDp);

      _renderState.mode = "single";
      _renderState.orig.tp = entry.tp; _renderState.orig.pp = entry.pp;
      _renderState.orig.dp = entry.dp; _renderState.orig.activeDp = meshOrigDp;
    }
  }

  // ═══ Render model ═══
  if (hasModel) {
    var modelTopY = topoH + sectionGap;

    // ── Calculate model layout aligned to DP card ──
    var modelX0, modelAreaW;
    var modelX0Eq, modelAreaWEq;
    if (_topoLayout && _topoLayout.mode === "three") {
      // Three-part: align models to topology columns, use nearly full column width
      var _moW3 = _topoLayout.origW;
      var _meW3 = _topoLayout.eqW;
      modelAreaW = _moW3 - 16;
      modelX0 = 8;
      modelAreaWEq = _meW3 - 16;
      modelX0Eq = _moW3 + _topoLayout.gap + _topoLayout.cardW + _topoLayout.gap + 8;
    } else if (_topoLayout && _topoLayout.mode === "two") {
      // Two-part: only original model, use full topology width
      var _moW2 = _topoLayout.origW;
      modelAreaW = _moW2 - 16;
      modelX0 = 8;
    } else if (meshOriginal && meshEquivalent) {
      var _mgap = 24;
      var _mavailW = meshWidth - _mgap;
      var _mdO = _meshCalcDims(meshOriginal.tp, meshOriginal.pp);
      var _mdE = _meshCalcDims(meshEquivalent.tp, meshEquivalent.pp);
      var _mshare = _mdO.dpW / (_mdO.dpW + _mdE.dpW);
      _mshare = Math.max(0.45, Math.min(0.6, _mshare));
      var _moW = _mavailW * _mshare;
      var _meW = _mavailW * (1 - _mshare);
      modelAreaW = _moW - 16;
      modelX0 = 8;
      modelAreaWEq = _meW - 16;
      modelX0Eq = _moW + _mgap + 8;
    } else if (hasTopo) {
      modelAreaW = meshWidth - 32;
      modelX0 = 16;
    } else {
      modelAreaW = meshWidth - 32;
      modelX0 = 16;
    }

    var hasBothModels = !!(modelOriginal && modelOriginal.layers && modelOriginal.layers.length
      && modelEquivalent && modelEquivalent.layers && modelEquivalent.layers.length);

    // ── Resolve topology TP/PP for tensor grid and TP/PP highlight ──
    var origTp = meshOriginal ? meshOriginal.tp : 0;
    var origPp = meshOriginal ? meshOriginal.pp : 0;
    var eqTp = meshEquivalent ? meshEquivalent.tp : 0;
    var eqPp = meshEquivalent ? meshEquivalent.pp : 0;
    var highlightOrigTp = null, highlightOrigPp = null;
    var highlightEqTp = null, highlightEqPp = null;
    if (meshPinnedTpInfo) {
      var clickedSide = meshPinnedTpInfo.side;
      // Highlight the clicked side directly
      if (clickedSide === 'orig') {
        highlightOrigTp = meshPinnedTpInfo.tpIndex;
        highlightOrigPp = meshPinnedTpInfo.ppIndex;
      } else {
        highlightEqTp = meshPinnedTpInfo.tpIndex;
        highlightEqPp = meshPinnedTpInfo.ppIndex;
      }
      // Compute mapped TP/PP for the other side via _mapRankToOtherSide
      var mappedRank = _mapRankToOtherSide(clickedSide, meshPinnedTpInfo.globalRank);
      if (mappedRank != null) {
        var otherSide = clickedSide === 'orig' ? 'eq' : 'orig';
        var otherMesh = otherSide === 'orig' ? meshOriginal : meshEquivalent;
        if (otherMesh) {
          var dstTp = otherMesh.tp, dstPp = otherMesh.pp;
          var rankInDp = mappedRank % (dstTp * dstPp);
          if (rankInDp < 0) rankInDp += dstTp * dstPp;
          var otherTpIdx = rankInDp % dstTp;
          var otherPpIdx = Math.floor(rankInDp / dstTp);
          if (clickedSide === 'orig') {
            highlightEqTp = otherTpIdx;
            highlightEqPp = otherPpIdx;
          } else {
            highlightOrigTp = otherTpIdx;
            highlightOrigPp = otherPpIdx;
          }
        }
      }
    }

    if (hasBothModels) {
      zoomLayer.append('text')
        .attr('x', modelX0 + modelAreaW / 2).attr('y', modelTopY + 40)
        .attr('text-anchor', 'middle').attr('fill', 'var(--cyan)')
        .attr('font-size', '13px').attr('font-family', 'var(--font-sans)')
        .attr('font-weight', 600).text('原始模型');
      zoomLayer.append('text')
        .attr('x', modelX0Eq + modelAreaWEq / 2).attr('y', modelTopY + 40)
        .attr('text-anchor', 'middle').attr('fill', 'var(--teal)')
        .attr('font-size', '13px').attr('font-family', 'var(--font-sans)')
        .attr('font-weight', 600).text('最小等效模型');

      var sharedScale = Math.min(
        Math.min(1, (modelAreaW - 16) / _TM_DESIGN.W),
        Math.min(1, (modelAreaWEq - 16) / _TM_DESIGN.W)
      );
      _renderOneModel(zoomLayer, modelOriginal, modelX0, modelTopY + 44, modelAreaW, false, sharedScale, null, 'var(--cyan)', origTp, origPp, highlightOrigTp, highlightOrigPp);
      _renderOneModel(zoomLayer, modelEquivalent, modelX0Eq, modelTopY + 44, modelAreaWEq, false, sharedScale, null, 'var(--teal)', eqTp, eqPp, highlightEqTp, highlightEqPp);
    } else {
      var singleTp = (meshOriginal || meshEquivalent) ? (meshOriginal || meshEquivalent).tp : 0;
      var singlePp = (meshOriginal || meshEquivalent) ? (meshOriginal || meshEquivalent).pp : 0;
      var singleHlTp = null, singleHlPp = null;
      if (meshPinnedTpInfo) {
        singleHlTp = meshPinnedTpInfo.tpIndex;
        singleHlPp = meshPinnedTpInfo.ppIndex;
      }
      _renderOneModel(zoomLayer, model, modelX0, modelTopY, modelAreaW, true, null, null, 'var(--cyan)', singleTp, singlePp, singleHlTp, singleHlPp);
    }
  }

  _renderState.width = meshWidth;
  _renderState.height = topoH;
}

async function loadMeshData(topoData) {
  var tp = topoData.tp_size || topoData.tp || 4;
  var pp = topoData.pp_size || topoData.pp || 4;
  var dp = topoData.dp_size || topoData.dp || 4;
  var totalNodes = topoData.total_nodes || (dp * tp * pp);
  var deviceType = topoData.device_type || '';
  var rawName = topoData.name || "";
  var name =
    rawName.indexOf("原始") !== -1
      ? "原始组网"
      : rawName.indexOf("等效") !== -1
      ? "最小等效组网"
      : rawName;
  var entry = {
    name: name,
    device_type: deviceType,
    tp: tp,
    pp: pp,
    dp: dp,
    total_nodes: totalNodes,
  };

  var isOrig = name.indexOf("原始") !== -1;
  var side = isOrig ? 'orig' : 'eq';

  // Store model config from topoData if provided (e.g. from REST restore)
  if (topoData.num_layers != null) {
    if (isOrig) {
      meshModelOrig = { num_layers: topoData.num_layers, hidden_dim: topoData.hidden_dim };
    } else {
      meshModelEq = { num_layers: topoData.num_layers, hidden_dim: topoData.hidden_dim };
    }
  }

  if (isOrig) {
    meshOriginal = entry;
    meshOrigDp = 0;
  } else {
    meshEquivalent = entry;
    meshEqDp = 0;
  }
  if (typeof checkSimReady === 'function') checkSimReady();

  // Skip estimate fetch if already populated (prevent duplicate calls on re-entry)
  var existing = isOrig ? meshEstimateOrig : meshEstimateEq;
  if (Object.keys(existing).length === 0 || existing[0] == null) {
    try {
      var modelSide = isOrig ? modelOriginal : modelEquivalent;
      var numLayers = topoData.num_layers != null ? topoData.num_layers
        : ((isOrig ? meshModelOrig : meshModelEq).num_layers
          || (modelSide && modelSide.config ? modelSide.config.num_layers : null));
      var hiddenDim = topoData.hidden_dim != null ? topoData.hidden_dim
        : ((isOrig ? meshModelOrig : meshModelEq).hidden_dim
          || (modelSide && modelSide.config ? modelSide.config.d_model : null));
      var estimates = await fetchEstimates(deviceType, totalNodes, dp, tp, pp, side, numLayers, hiddenDim);
      if (isOrig) {
        meshEstimateOrig = estimates;
      } else {
        meshEstimateEq = estimates;
      }
    } catch (e) {
      console.warn('Estimate fetch failed, using empty estimates:', e);
      if (isOrig) {
        meshEstimateOrig = {};
      } else {
        meshEstimateEq = {};
      }
    }
  }

  // canvasRebuild is called by the caller after loadMeshData completes
}

// ── Attach to window for inline onclick / onchange handlers ──

window.loadMeshData = loadMeshData;
window.meshRebuild = meshRebuild;
window.canvasRebuild = canvasRebuild;
window.meshSwitchDp = meshSwitchDp;
window.meshSwitchDpOrig = meshSwitchDpOrig;
window.meshSwitchDpEq = meshSwitchDpEq;
window.fetchSimulationData = fetchSimulationData;
window.toggleTooltipDetail = toggleTooltipDetail;
window._closeDetailPanels = _closeDetailPanels;
window.loadModelData = loadModelData;
window.modelRebuild = modelRebuild;

// ── Resize handler (debounced) ──

var _resizeTimer = null;
window.addEventListener("resize", function () {
  if (!meshOriginal && !meshEquivalent && !modelOriginal && !modelEquivalent) return;
  if (_resizeTimer) clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(function () {
    _resizeTimer = null;
    canvasRebuild();
  }, 200);
});

// ── Canvas background click dismisses pinned tooltip ──

document.getElementById('canvas-svg-wrap').addEventListener('click', function (e) {
  if (e.target.tagName === 'svg' || e.target.id === 'canvas-svg-wrap') {
    _clearBothTooltips();
  }
});

// ── Escape key dismisses detail panels first, then pinned tooltip ──

document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape') {
    if (tooltipDetailState) {
      _closeDetailPanels();
    } else if (meshPinnedRank) {
      _clearBothTooltips();
    }
  }
});

// ═══════════════════════════════════════════════════════════════════
// Metric Detail — inline panel attached to tooltip (replaces modal)
// ═══════════════════════════════════════════════════════════════════

var tooltipDetailState = null; // { metricType, origRank, eqRank }

function _metricTypeLabel(type) {
  var map = { 'flops': '单卡FLOPs详情', 'hbm': 'HBM 占用详情', 'tp-comm': 'TP通信详情', 'pp-comm': 'PP通信详情', 'dp-comm': 'DP通信详情' };
  return map[type] || '指标详情';
}

async function toggleTooltipDetail(globalRank, isOrig, metricType) {
  if (metricType === 'flops') {
    _closeDetailPanels();
    var pinned = meshPinnedRank;
    var hasLinked = pinned && pinned.mappedRank != null;
    var origRank = hasLinked ? (pinned.side === 'orig' ? pinned.globalRank : pinned.mappedRank) : (isOrig ? globalRank : null);
    var eqRank = hasLinked ? (pinned.side === 'orig' ? pinned.mappedRank : pinned.globalRank) : (isOrig ? null : globalRank);
    tooltipDetailState = { metricType: metricType, origRank: origRank, eqRank: eqRank };
  
    var detail = document.getElementById('rank-tooltip-detail');
    var detailMapped = document.getElementById('rank-tooltip-detail-mapped');
    var tip = document.getElementById('rank-tooltip');
    var tipMapped = document.getElementById('rank-tooltip-mapped');
  
    if (hasLinked) {
      var mainIsOrig = pinned.side === 'orig';
      var mainRank = mainIsOrig ? origRank : eqRank;
      var mappedRank = mainIsOrig ? eqRank : origRank;
      _showDetailPanel(detail, tip, mainRank, _generateFlopsMockData(_getCardFlops(mainRank, mainIsOrig)), metricType);
      _showDetailPanel(detailMapped, tipMapped, mappedRank, _generateFlopsMockData(_getCardFlops(mappedRank, !mainIsOrig)), metricType);
    } else {
      if (origRank != null) {
        _showDetailPanel(detail, tip, origRank, _generateFlopsMockData(_getCardFlops(origRank, true)), metricType);
      } else if (eqRank != null) {
        _showDetailPanel(detail, tip, eqRank, _generateFlopsMockData(_getCardFlops(eqRank, false)), metricType);
      }
    }
    return;
  }

  // If same metric type already open, close it (toggle off)
  if (tooltipDetailState && tooltipDetailState.metricType === metricType) {
    _closeDetailPanels();
    return;
  }

  _closeDetailPanels();

  var pinned = meshPinnedRank;
  var hasLinked = pinned && pinned.mappedRank != null;

  // Determine ranks for both sides
  var origRank = hasLinked ? (pinned.side === 'orig' ? pinned.globalRank : pinned.mappedRank) : (isOrig ? globalRank : null);
  var eqRank = hasLinked ? (pinned.side === 'orig' ? pinned.mappedRank : pinned.globalRank) : (isOrig ? null : globalRank);

  // Fetch both sides in parallel
  var promises = [];
  promises.push(origRank != null
    ? fetch(API + '/session/' + sessionId + '/simulation/original/' + origRank + '/' + metricType + '-detail').then(function(r) { return r.ok ? r.json() : null; })
    : Promise.resolve(null));
  promises.push(eqRank != null
    ? fetch(API + '/session/' + sessionId + '/simulation/equivalent/' + eqRank + '/' + metricType + '-detail').then(function(r) { return r.ok ? r.json() : null; })
    : Promise.resolve(null));

  try {
    var results = await Promise.all(promises);
    tooltipDetailState = { metricType: metricType, origRank: origRank, eqRank: eqRank };

    var detail = document.getElementById('rank-tooltip-detail');
    var detailMapped = document.getElementById('rank-tooltip-detail-mapped');
    var tip = document.getElementById('rank-tooltip');
    var tipMapped = document.getElementById('rank-tooltip-mapped');

    if (hasLinked) {
      var mainIsOrig = pinned.side === 'orig';
      var mainRank = mainIsOrig ? origRank : eqRank;
      var mainData = mainIsOrig ? results[0] : results[1];
      var mappedRank = mainIsOrig ? eqRank : origRank;
      var mappedData = mainIsOrig ? results[1] : results[0];

      if (mainData) _showDetailPanel(detail, tip, mainRank, mainData, metricType);
      if (mappedData) _showDetailPanel(detailMapped, tipMapped, mappedRank, mappedData, metricType);
    } else {
      if (origRank != null && results[0]) {
        _showDetailPanel(detail, tip, origRank, results[0], metricType);
      } else if (eqRank != null && results[1]) {
        _showDetailPanel(detail, tip, eqRank, results[1], metricType);
      }
    }
  } catch (e) {
    console.error('Fetch detail error:', e);
  }
}

function _getCardFlops(globalRank, isOrig) {
  var actual = (isOrig ? meshActualOrig : meshActualEq)[globalRank];
  var est = (isOrig ? meshEstimateOrig : meshEstimateEq)[globalRank];
  return (actual && actual.flops_per_card) || (est && est.flops_per_card) || 0;
}

function _generateFlopsMockData(cardFlops) {
  var total = cardFlops || 7.5e11;
  var forward = total * 0.34;
  var backwardB = total * 0.33;
  var backwardW = total * 0.33;
  return {
    total_flops: total,
    forward_flops: forward,
    backward_B_flops: backwardB,
    backward_W_flops: backwardW
  };
}

function _renderFlopsDetailHtml(d) {
  var html = '<table class="tooltip-detail-table">';
  html += '<tr><th>参数</th><th>计算量 (FLOPs)</th><th>占比</th></tr>';
  html += '<tr><td class="tooltip-detail-label">总计算量</td><td>' + formatFlops(d.total_flops) + '</td><td>100%</td></tr>';
  html += '<tr><td class="tooltip-detail-label">前向计算量</td><td>' + formatFlops(d.forward_flops) + '</td><td>' + (d.forward_flops / d.total_flops * 100).toFixed(1) + '%</td></tr>';
  html += '<tr><td class="tooltip-detail-label">反向传播输入梯度计算量</td><td>' + formatFlops(d.backward_B_flops) + '</td><td>' + (d.backward_B_flops / d.total_flops * 100).toFixed(1) + '%</td></tr>';
  html += '<tr><td class="tooltip-detail-label">反向传播权重梯度计算量</td><td>' + formatFlops(d.backward_W_flops) + '</td><td>' + (d.backward_W_flops / d.total_flops * 100).toFixed(1) + '%</td></tr>';
  html += '</table>';
  return html;
}

function _showDetailPanel(panel, tooltip, globalRank, data, metricType) {
  var html = '<div class="tooltip-detail-header">';
  html += '<span class="tooltip-detail-title">Rank ' + globalRank + ' | ' + _metricTypeLabel(metricType) + '</span>';
  html += '<button class="tooltip-detail-close" onclick="event.stopPropagation();_closeDetailPanels()">&times;</button>';
  html += '</div><div class="tooltip-detail-body">';

  if (metricType === 'hbm') {
    html += _renderHbmDetailHtml(data);
  } else if (metricType === 'flops') {
    html += _renderFlopsDetailHtml(data);
  } else {
    html += _renderCommDetailHtml(data);
  }

  html += '</div>';
  panel.innerHTML = html;
  panel.classList.add('visible');
  _positionDetailPanel(panel, tooltip);
}

function _renderHbmDetailHtml(d) {
  var html = '<table class="tooltip-detail-table">';
  html += '<tr><th>参数</th><th>占用 (GB)</th><th>占比</th></tr>';
  var items = [
    { label: '权重', value: d.weights_gb },
    { label: '梯度', value: d.gradients_gb },
    { label: '优化器', value: d.optimizer_gb },
    { label: '激活', value: d.activations_gb }
  ];
  var total = d.total_hbm_gb || 1;
  items.forEach(function(item) {
    html += '<tr><td class="tooltip-detail-label">' + item.label + '</td><td>' + item.value.toFixed(2) + '</td><td>' + (item.value / total * 100).toFixed(1) + '%</td></tr>';
  });
  html += '<tr class="total-row"><td class="tooltip-detail-label">总计</td><td>' + d.total_hbm_gb.toFixed(2) + '</td><td>100%</td></tr>';
  html += '</table>';
  return html;
}

function _renderCommDetailHtml(d) {
  var typeName = { 'tp': 'TP', 'pp': 'PP', 'dp': 'DP' }[d.comm_type] || d.comm_type.toUpperCase();
  var html = '<table class="tooltip-detail-table">';
  html += '<tr><th>参数</th><th>值</th></tr>';
  html += '<tr><td class="tooltip-detail-label">通信类型</td><td>' + typeName + ' 通信</td></tr>';
  html += '<tr><td class="tooltip-detail-label">通信次数</td><td>' + d.comm_count + ' 次/step</td></tr>';
  html += '<tr><td class="tooltip-detail-label">通信卡数</td><td>' + d.comm_cards + ' 张</td></tr>';
  html += '<tr><td class="tooltip-detail-label">单次通信量</td><td>' + d.comm_size_per_time_gb.toFixed(4) + ' GB</td></tr>';
  html += '<tr class="total-row"><td class="tooltip-detail-label">总通信量</td><td>' + d.total_comm_gb.toFixed(4) + ' GB</td></tr>';
  html += '</table>';
  return html;
}

function _closeDetailPanels() {
  var detail = document.getElementById('rank-tooltip-detail');
  var detailMapped = document.getElementById('rank-tooltip-detail-mapped');
  if (detail) { detail.classList.remove('visible'); detail.innerHTML = ''; }
  if (detailMapped) { detailMapped.classList.remove('visible'); detailMapped.innerHTML = ''; }
  tooltipDetailState = null;
}

function _positionDetailPanel(panel, tooltip) {
  var tipBox = tooltip.getBoundingClientRect();
  var panelW = 300;
  var x = tipBox.right + 6;
  var y = tipBox.top;
  if (x + panelW > window.innerWidth - 8) {
    x = tipBox.left - panelW - 6;
  }
  if (x < 8) x = 8;
  var maxH = 420;
  if (y + maxH > window.innerHeight - 8) {
    y = window.innerHeight - maxH - 8;
  }
  if (y < 8) y = 8;
  panel.style.left = x + 'px';
  panel.style.top = y + 'px';
}

// Model visualization — Transformer training model structure diagram
// ═══════════════════════════════════════════════════════════════════

var MODEL_COLORS = {
  input_embedding:  { fill: '#1a1a10', stroke: '#ffb454', label: 'Embeddings', text: '#ffb454' },
  layer_norm:       { fill: '#111a13', stroke: '#7fd962', label: 'Layer Norm', text: '#7fd962' },
  mha:              { fill: '#111922', stroke: '#39bae6', label: 'Multi-Head Attention', text: '#39bae6' },
  mha_sub:          { fill: '#15202b', stroke: '#39bae6', text: '#7b8ca3' },
  ffn:              { fill: '#1a1114', stroke: '#f26d78', label: 'Feed-Forward Network', text: '#f26d78' },
  ffn_sub:          { fill: '#1f1518', stroke: '#f26d78', text: '#7b8ca3' },
  add:              { fill: '#161c24', stroke: '#4a5568', label: 'Add', text: '#7b8ca3' },
  output:           { fill: '#1a140f', stroke: '#ff8f40', label: 'Output Layer', text: '#ff8f40' },
  skip:             { stroke: '#ff8f40' },
  transformer_card: { fill: '#111820', stroke: '#39bae6' },
};

// ── Model structure concept tips (from transformer_tip.json) ──
var _MODEL_TIPS = {
  container_transformer_layer: { t:'Transformer层（堆叠N次）', te:'Transformer Layer (stacked N times)', d:'整个Transformer模型由N个完全相同的层堆叠而成（如GPT-3为96层）。每一层包含两个核心子层：多头自注意力和前馈网络，通过残差连接和层归一化组合。', m:'Layer_i(x) = AddNorm_FFN(AddNorm_Attn(x)) for i ∈ [1, N]', v:'GPT-3: 96层 | LLaMA-7B: 32层 | LLaMA-70B: 80层' },
  embeddings: { t:'位置编码 + 词嵌入 + Dropout', te:'Positional Encoding + Word Embeddings + Dropout', d:'将离散token ID映射为稠密向量（维度d_model），注入位置信息后应用Dropout正则化。现代模型使用可学习位置编码或RoPE。', m:'H_0 = Dropout(WordEmbed(token_ids) + PositionEncode(positions))', v:'d_model: 768(GPT-2) / 4096(LLaMA-7B) / 8192(LLaMA-70B)' },
  layer_norm_1: { t:'层归一化（注意力前）', te:'Layer Normalization (pre-attention)', d:'Pre-LN架构中的第一个Layer Norm。对每个token特征向量独立归一化（均值为0，方差为1），再通过可学习γ/β参数仿射变换。Pre-LN相比原始Post-LN训练更稳定、梯度流更平滑。', m:'LN(x) = γ ⊙ ((x - μ) / √(σ² + ε)) + β', v:'ε = 1e-5 ~ 1e-6 | 参数量: d_model per LN' },
  multi_head_self_attention: { t:'多头自注意力机制', te:'Multi-Head Self-Attention', d:'自注意力是Transformer的核心，允许每个token关注所有其他token捕获上下文依赖。多头机制将Q/K/V投影到多个子空间并行计算，最后拼接输出。Decoder-only模型使用因果掩码确保自回归约束。', m:'MultiHead(Q,K,V) = Concat(head_1,...,head_h)W_O, head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)', v:'head数: 12(GPT-2) / 32(LLaMA-7B) / 64(LLaMA-70B) | d_k: 64~128' },
  self_attention: { t:'自注意力计算', te:'Self Attention computation', d:'核心注意力计算：Q/K/V投影→计算QK^T注意力分数→除以√d_k缩放→因果掩码→softmax→对V加权求和，得到每个token的上下文表示。', m:'Attention(Q,K,V) = softmax(QK^T/√d_k + mask)V', v:'' },
  linear_projection: { t:'线性输出投影', te:'Linear output projection', d:'将多头注意力的拼接结果通过线性变换W_O投影回原始隐藏维度d_model。让不同注意力头的信息融合，是MHA子层参数量最大的部分。', m:'output = Concat(heads) · W_O, W_O ∈ R^{d_model × d_model}', v:'' },
  dropout_attention: { t:'注意力Dropout', te:'Attention Dropout', d:'在注意力权重上应用Dropout正则化，随机置零部分注意力连接。迫使模型不依赖单一注意力模式，增强泛化能力。', m:'', v:'' },
  add_1: { t:'残差加法（注意力残差）', te:'Residual addition (attention residual)', d:'将注意力输出与输入逐元素相加。残差连接为梯度提供直通路径（identity shortcut），是深层Transformer训练的关键。图中橙色虚线标注了这条residual路径。', m:'x = x + MHA(LN_1(x))', v:'' },
  layer_norm_2: { t:'层归一化（FFN前）', te:'Layer Normalization (pre-FFN)', d:'第二个Layer Norm，位于FFN子层之前。将残差加法后的隐表示归一化到标准分布，为FFN提供数值稳定的输入。两个子层各有独立的LN参数。', m:'LN(x) = γ ⊙ ((x - μ) / √(σ² + ε)) + β', v:'' },
  feed_forward_network: { t:'前馈神经网络', te:'Feed-Forward Network', d:'FFN是每层第二个核心子层：将d_model扩展到4×d_model（中间层），应用GeLU激活函数引入非线性，再投影回d_model。为模型提供位置无关的非线性变换能力，参数量约为注意力子层的2倍。', m:'FFN(x) = Dropout(W_2(GeLU(W_1 x + b_1)) + b_2)', v:'中间层: 4×d_model | FFN参数占比: ~67%' },
  linear_up_projection: { t:'线性上投影（升维）', te:'Linear up-projection', d:'将输入从d_model映射到4×d_model维度（如4096→16384），极大增加表示容量。W_1 ∈ R^{d_model × 4d_model}。', m:'h_up = W_1 x + b_1', v:'' },
  gelu_activation: { t:'高斯误差线性单元', te:'Gaussian Error Linear Unit (GeLU)', d:'GPT-2/3使用的平滑非线性激活函数，相比ReLU更平滑，有助于梯度稳定传播。现代模型（LLaMA）使用SwiGLU替代以提升性能。', m:'GeLU(x) = x · Φ(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))', v:'' },
  linear_down_projection: { t:'线性下投影（降维）', te:'Linear down-projection', d:'将4×d_model维度的中间表示压缩回d_model维度，使FFN输出可无缝通过残差连接与注意力输出相加。W_2 ∈ R^{4d_model × d_model}。', m:'h_out = W_2 h_ge + b_2', v:'' },
  dropout_ffn: { t:'FFN Dropout', te:'FFN Dropout', d:'在线性降维投影之后应用Dropout，随机置零输出特征。与注意力Dropout类似但是作用在FFN子层的线性输出上。', m:'', v:'' },
  add_2: { t:'残差加法（FFN残差）', te:'Residual addition (FFN residual)', d:'将FFN输出与输入逐元素相加。本层的第二个残差连接。两个残差连接共同构成Transformer层的梯度高速通道。', m:'x = x + FFN(LN_2(x))', v:'' },
  layer_norm_3: { t:'最终层归一化（输出前）', te:'Final Layer Normalization (pre-output)', d:'所有Transformer层堆叠完成后的最后一个Layer Norm，不属于任何单层内部。归一化后传递给输出层，确保输入到LM Head的隐表示数值稳定。', m:'H_final = LayerNorm(H_N)', v:'' },
  output_layer_and_loss: { t:'输出层与损失函数', te:'Output Layer & Loss', d:'将d_model维度的隐表示映射到vocab_size维度得到logits。训练时计算交叉熵损失。GPT系列中LM Head权重常与输入Embedding共享（weight tying）。', m:'logits = H_final · W_embed^T, Loss = -Σ y_i log(softmax(logits_i))', v:'' },
  tensor_parallelism_grid: { t:'张量并行网格（4×4 TP）', te:'Tensor Parallelism grid (4×4 = 16 NPUs)', d:'4×4网格可视化Tensor Parallelism（TP）分布。TP将单层参数矩阵按列或行切分到多个NPU，各NPU独立计算分片后通过AllReduce通信聚合。橙色高亮格(tp=0)表示当前NPU分片。适用于单层参数过大无法放入单NPU的场景，要求NPU间高速互联（如HCCS）。', m:'', v:'切分方式: 列切分/行切分 | 通信: AllReduce/AllGather/ReduceScatter' },
  residual_connection_1: { t:'残差连接1：绕过注意力', te:'Residual bypassing attention sublayer', d:'从Embeddings输出分叉，绕过Layer Norm 1和Multi-Head Attention子层，直接连接到第一个Add。保留子层输入的原始信号，梯度可通过恒等映射直接回传。', m:'', v:'' },
  residual_connection_2: { t:'残差连接2：绕过FFN', te:'Residual bypassing FFN sublayer', d:'从第一个Add输出分叉，绕过Layer Norm 2和FFN子层，直接连接到第二个Add。两条残差连接确保梯度流动的双重保障。', m:'', v:'' }
};

function loadModelData(modelData, role) {
  role = role || (modelData._role || ((modelOriginal ? 'equivalent' : 'original')));

  var entry = {
    type: modelData.type,
    config: modelData.config,
    computed: modelData.computed,
    layers: modelData.layers || [],
    output_layer: modelData.output_layer,
  };
  if (role === 'original') {
    modelOriginal = entry;
    if (modelData.config) {
      meshModelOrig = { num_layers: modelData.config.num_layers || modelData.layers_count, hidden_dim: modelData.config.d_model };
    }
    // If mesh already loaded, re-fetch estimates with correct model params
    if (meshOriginal) _refetchMeshEstimate('orig');
  } else {
    modelEquivalent = entry;
    if (modelData.config) {
      meshModelEq = { num_layers: modelData.config.num_layers || modelData.layers_count, hidden_dim: modelData.config.d_model };
    }
    if (meshEquivalent) _refetchMeshEstimate('eq');
  }
  if (typeof checkSimReady === 'function') checkSimReady();
}

async function _refetchMeshEstimate(side) {
  var mesh = side === 'orig' ? meshOriginal : meshEquivalent;
  var model = side === 'orig' ? meshModelOrig : meshModelEq;
  if (!mesh || !model || model.num_layers == null) return;
  try {
    var estimates = await fetchEstimates(mesh.device_type, mesh.total_nodes, mesh.dp, mesh.tp, mesh.pp, side, model.num_layers, model.hidden_dim);
    if (side === 'orig') {
      meshEstimateOrig = estimates;
    } else {
      meshEstimateEq = estimates;
    }
    canvasRebuild();
  } catch (e) {
    console.warn('Estimate re-fetch failed for ' + side, e);
  }
}

// ── Transformer Model Architecture Diagram (vertical flow) ──
// Renders the internal structure of a single transformer layer as a vertical
// column diagram: Embeddings → LayerNorm → Attention → Add → LayerNorm → FFN → Add → LayerNorm → Output
// with a TP tensor grid on the left, skip connections on the right, and a legend.

// Design-base coordinates (rendered at natural size, then scaled uniformly to fit areaW).
var _TM_DESIGN = {
  W: 480, H: 620,         // design canvas
  CX: 246,                 // center X
  BOX_W: 220,              // wide block width
  NARROW_W: 120,           // narrow block width (LayerNorm, Add)
  BLOCK_W: 200,            // sub-block width (ATTN, FFN)
  SUB_W: 150,              // inner sub-item width
  H_SM: 26,                // small block height
  H_MD: 32,                // medium block height
  H_ATTN: 92,              // attention block height
  H_FFN: 106,              // FFN block height
  ARROW_S: 12,             // arrow size
  TENSOR_W: 124,           // tensor grid width
  TENSOR_H: 124,           // tensor grid height
  TENSOR_X: 0,             // tensor grid left edge
  TENSOR_Y: 50,            // tensor grid top (aligned to transformer card top)
  // Vertical layout
  Y_EMBED: 0,
  Y_HEADER: 48,            // transformer card top
  Y_LN1: 72,
  Y_ATTN: 110,
  Y_ADD1: 214,
  Y_LN2: 252,
  Y_FFN: 290,
  Y_ADD2: 408,
  Y_TF_END: 460,           // transformer card bottom
  Y_LN3: 478,
  Y_OUTPUT: 516,
};

function _renderOneModel(g, model, x0, topY, areaW, showHeader, forceScale, _unused2, labelColor, tpCount, ppCount, highlightTpIdx, highlightPpIdx) {
  var cfg = model.config || {};
  var comp = model.computed || {};
  var numLayers = cfg.num_layers || 1;
  var D = _TM_DESIGN;

  // ── Scale to fit allocated width (use forced scale when comparing two models side-by-side) ──
  var scale = (forceScale != null && forceScale > 0) ? forceScale : Math.min(1, (areaW - 16) / D.W);
  var useW = D.W * scale;
  var useH = D.H * scale;
  var offX = x0 + (areaW - useW) / 2;

  // ── Config header ──
  var headerH = 0;
  if (showHeader) {
    headerH = 30;
    var cfgText = 'd_model=' + cfg.d_model + '  num_heads=' + cfg.num_heads + '  d_ffn=' + cfg.d_ffn
      + '  d_head=' + comp.d_head + '  params=' + (comp.total_params_billions || '—');
    g.append('text')
      .attr('x', x0 + areaW / 2).attr('y', topY + 16)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-secondary)').attr('font-size', '10px').attr('font-family', 'JetBrains Mono, monospace')
      .text((model.type || 'TRANSFORMER_MODEL').toUpperCase() + '  ·  ' + cfg.num_layers + ' layers  ·  ' + cfgText);
  }

  var sg = g.append('g')
    .attr('transform', 'translate(' + offX + ',' + (topY + headerH) + ') scale(' + scale + ')');

  // ── Helper: show/hide model concept tooltip ──
  function showTip(show, tipId, event) {
    var tipEl = document.getElementById('model-tooltip');
    if (!tipEl) return;
    if (!show || !tipId || !_MODEL_TIPS[tipId]) {
      tipEl.classList.remove('visible');
      return;
    }
    var tip = _MODEL_TIPS[tipId];
    var html = '<div class="tip-title">' + tip.t + '</div>';
    html += '<div class="tip-eng">' + tip.te + '</div>';
    html += '<div class="tip-explain">' + tip.d + '</div>';
    if (tip.m) html += '<div class="tip-math">' + tip.m + '</div>';
    if (tip.v) html += '<div class="tip-values">' + tip.v + '</div>';
    tipEl.innerHTML = html;
    tipEl.classList.add('visible');
    // Position near cursor, avoid edge overflow
    var x = event.clientX + 16;
    var y = event.clientY - 10;
    var tw = tipEl.offsetWidth || 300;
    var th = tipEl.offsetHeight || 200;
    if (x + tw > window.innerWidth - 20) x = event.clientX - tw - 16;
    if (y + th > window.innerHeight - 20) y = event.clientY - th - 10;
    if (x < 10) x = 10;
    if (y < 10) y = 10;
    tipEl.style.left = x + 'px';
    tipEl.style.top = y + 'px';
  }

  // ── Helper: bind hover effect (glow + stroke lift + tooltip) ──
  function addHover(rect, origSw, tipId) {
    rect
      .classed('model-node', true)
      .on('mouseenter', function (event) {
        d3.select(this)
          .attr('stroke-width', origSw * 2.2)
          .attr('filter', 'url(#model-hover-glow)');
        showTip(true, tipId, event);
      })
      .on('mousemove', function (event) {
        showTip(true, tipId, event);
      })
      .on('mouseleave', function () {
        d3.select(this)
          .attr('stroke-width', origSw)
          .attr('filter', null);
        showTip(false);
      });
  }

  // ── Helper: draw a simple labeled box ──
  function box(x, y, w, h, label, fill, stroke, textColor, fontSize, tipId) {
    fontSize = fontSize || 11;
    var rect = sg.append('rect')
      .attr('x', x).attr('y', y).attr('width', w).attr('height', h)
      .attr('rx', 4).attr('fill', fill).attr('stroke', stroke).attr('stroke-width', 1.2);
    addHover(rect, 1.2, tipId);
    sg.append('text')
      .attr('x', x + w / 2).attr('y', y + h / 2 + 1)
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
      .attr('font-size', fontSize).attr('font-family', 'DM Sans, sans-serif').attr('font-weight', 500)
      .attr('fill', textColor).text(label);
  }

  // ── Helper: draw a sub-block with title and inner items ──
  function subBlock(ox, oy, ow, oh, title, items, fill, stroke, titleColor, itemBg, itemStroke, tipId, subTipIds) {
    var rect = sg.append('rect')
      .attr('x', ox).attr('y', oy).attr('width', ow).attr('height', oh)
      .attr('rx', 4).attr('fill', fill).attr('stroke', stroke).attr('stroke-width', 1.2);
    addHover(rect, 1.2, tipId);
    sg.append('text')
      .attr('x', ox + ow / 2).attr('y', oy + 13)
      .attr('text-anchor', 'middle').attr('font-size', 10).attr('font-family', 'DM Sans, sans-serif')
      .attr('font-weight', 600).attr('fill', titleColor).text(title);

    var innerW = D.SUB_W;
    var innerX = ox + (ow - innerW) / 2;
    var innerStartY = oy + 23;
    var innerH = 18;
    var innerGap = 3;

    items.forEach(function (item, i) {
      var iy = innerStartY + i * (innerH + innerGap);
      var innerRect = sg.append('rect')
        .attr('x', innerX).attr('y', iy).attr('width', innerW).attr('height', innerH)
        .attr('rx', 3).attr('fill', itemBg).attr('stroke', itemStroke)
        .attr('stroke-width', 0.8).attr('stroke-opacity', 0.5);
      addHover(innerRect, 0.8, subTipIds ? subTipIds[i] : null);
      sg.append('text')
        .attr('x', innerX + innerW / 2).attr('y', iy + innerH / 2 + 1)
        .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
        .attr('font-size', 9).attr('font-family', 'JetBrains Mono, monospace').attr('font-weight', 500)
        .attr('fill', 'var(--text-secondary)').text(item);
    });
  }

  // ── Helper: arrow between two vertical positions ──
  function arrow(cx, y1, y2) {
    var midY = (y1 + y2) / 2;
    sg.append('polygon')
      .attr('points', cx + ',' + (midY - D.ARROW_S / 2) + ' '
        + (cx - D.ARROW_S / 2) + ',' + (midY + D.ARROW_S / 2) + ' '
        + (cx + D.ARROW_S / 2) + ',' + (midY + D.ARROW_S / 2))
      .attr('fill', '#3fb950');
  }

  // ── Helper: dashed line ──
  function dashLine(x1, y1, x2, y2, color) {
    sg.append('line')
      .attr('x1', x1).attr('y1', y1).attr('x2', x2).attr('y2', y2)
      .attr('stroke', color).attr('stroke-width', 1).attr('stroke-dasharray', '3 3');
  }

  // ══════════════════════════════════════════════
  // 1. Embeddings
  // ══════════════════════════════════════════════
  box(D.CX - D.BOX_W / 2, D.Y_EMBED, D.BOX_W, D.H_MD,
    'Position + Word Embeddings & Dropout',
    MODEL_COLORS.input_embedding.fill, MODEL_COLORS.input_embedding.stroke, MODEL_COLORS.input_embedding.text, 10, 'embeddings');

  // ══════════════════════════════════════════════
  // 2. Transformer Layer card (with stacked depth shadows)
  // ══════════════════════════════════════════════
  var tfH = D.Y_TF_END - D.Y_HEADER;
  var isOriginalModel = (model === modelOriginal);
  var stackOffsets = isOriginalModel
    ? [[24, 20], [20, 16], [16, 12], [12, 8], [8, 4]]   // 5 layers
    : [[24, 20], [16, 14], [8, 7]];                       // 3 layers
  stackOffsets.forEach(function (off, i) {
    sg.append('rect')
      .attr('x', D.CX - D.BOX_W / 2 + off[0])
      .attr('y', D.Y_HEADER + off[1])
      .attr('width', D.BOX_W).attr('height', tfH)
      .attr('rx', 8)
      .attr('fill', '#0d131a')
      .attr('stroke', MODEL_COLORS.transformer_card.stroke)
      .attr('stroke-width', 1).attr('opacity', 0.35 + i * 0.25);
  });

  var tfCard = sg.append('rect')
    .attr('x', D.CX - D.BOX_W / 2).attr('y', D.Y_HEADER)
    .attr('width', D.BOX_W).attr('height', tfH)
    .attr('rx', 8).attr('fill', MODEL_COLORS.transformer_card.fill)
    .attr('stroke', MODEL_COLORS.transformer_card.stroke).attr('stroke-width', 1.8);
  addHover(tfCard, 1.8, 'container_transformer_layer');

  sg.append('text')
    .attr('x', D.CX - D.BOX_W / 2 + 12).attr('y', D.Y_HEADER + 18)
    .text('Transformer Layer  (×N)').attr('font-size', 11)
    .attr('font-weight', 600).attr('font-family', 'DM Sans, sans-serif')
    .attr('fill', MODEL_COLORS.transformer_card.stroke);

  sg.append('text')
    .attr('x', D.CX + D.BOX_W / 2 - 12).attr('y', D.Y_HEADER + 18)
    .text('×' + numLayers).attr('font-size', 11)
    .attr('font-weight', 600).attr('font-family', 'JetBrains Mono, monospace')
    .attr('fill', 'var(--text-muted)').attr('text-anchor', 'end');

  // ══════════════════════════════════════════════
  // 3. Layer Norm 1
  // ══════════════════════════════════════════════
  box(D.CX - D.NARROW_W / 2, D.Y_LN1, D.NARROW_W, D.H_SM, 'Layer Norm',
    MODEL_COLORS.layer_norm.fill, MODEL_COLORS.layer_norm.stroke, MODEL_COLORS.layer_norm.text, null, 'layer_norm_1');

  // ══════════════════════════════════════════════
  // 4. Multi-Head Self-Attention
  // ══════════════════════════════════════════════
  subBlock(D.CX - D.BLOCK_W / 2, D.Y_ATTN, D.BLOCK_W, D.H_ATTN,
    'Multi-Head Self-Attention',
    ['Self Attention', 'Linear (h → h)', 'Dropout'],
    MODEL_COLORS.mha.fill, MODEL_COLORS.mha.stroke, MODEL_COLORS.mha.text,
    MODEL_COLORS.mha_sub.fill, MODEL_COLORS.mha_sub.stroke,
    'multi_head_self_attention', ['self_attention', 'linear_projection', 'dropout_attention']);

  // ══════════════════════════════════════════════
  // 5. Add 1
  // ══════════════════════════════════════════════
  box(D.CX - D.NARROW_W / 2, D.Y_ADD1, D.NARROW_W, D.H_SM, 'Add',
    MODEL_COLORS.add.fill, MODEL_COLORS.add.stroke, MODEL_COLORS.add.text, null, 'add_1');

  // ══════════════════════════════════════════════
  // 6. Layer Norm 2
  // ══════════════════════════════════════════════
  box(D.CX - D.NARROW_W / 2, D.Y_LN2, D.NARROW_W, D.H_SM, 'Layer Norm',
    MODEL_COLORS.layer_norm.fill, MODEL_COLORS.layer_norm.stroke, MODEL_COLORS.layer_norm.text, null, 'layer_norm_2');

  // ══════════════════════════════════════════════
  // 7. Feed-Forward Network
  // ══════════════════════════════════════════════
  subBlock(D.CX - D.BLOCK_W / 2, D.Y_FFN, D.BLOCK_W, D.H_FFN,
    'Feed-Forward Network',
    ['Linear (h → 4h)', 'GeLU', 'Linear (4h → h)', 'Dropout'],
    MODEL_COLORS.ffn.fill, MODEL_COLORS.ffn.stroke, MODEL_COLORS.ffn.text,
    MODEL_COLORS.ffn_sub.fill, MODEL_COLORS.ffn_sub.stroke,
    'feed_forward_network', ['linear_up_projection', 'gelu_activation', 'linear_down_projection', 'dropout_ffn']);

  // ══════════════════════════════════════════════
  // 8. Add 2
  // ══════════════════════════════════════════════
  box(D.CX - D.NARROW_W / 2, D.Y_ADD2, D.NARROW_W, D.H_SM, 'Add',
    MODEL_COLORS.add.fill, MODEL_COLORS.add.stroke, MODEL_COLORS.add.text, null, 'add_2');

  // ══════════════════════════════════════════════
  // 9. Layer Norm 3
  // ══════════════════════════════════════════════
  box(D.CX - D.NARROW_W / 2, D.Y_LN3, D.NARROW_W, D.H_SM, 'Layer Norm',
    MODEL_COLORS.layer_norm.fill, MODEL_COLORS.layer_norm.stroke, MODEL_COLORS.layer_norm.text, null, 'layer_norm_3');

  // ══════════════════════════════════════════════
  // 10. Output Layer
  // ══════════════════════════════════════════════
  box(D.CX - D.BOX_W / 2, D.Y_OUTPUT, D.BOX_W, D.H_MD,
    'Output Layer & Loss',
    MODEL_COLORS.output.fill, MODEL_COLORS.output.stroke, MODEL_COLORS.output.text, 10, 'output_layer_and_loss');

  // ══════════════════════════════════════════════
  // Main flow arrows (between components)
  // ══════════════════════════════════════════════
  var gaps = [
    [D.Y_EMBED + D.H_MD, D.Y_LN1],
    [D.Y_LN1 + D.H_SM, D.Y_ATTN],
    [D.Y_ATTN + D.H_ATTN, D.Y_ADD1],
    [D.Y_ADD1 + D.H_SM, D.Y_LN2],
    [D.Y_LN2 + D.H_SM, D.Y_FFN],
    [D.Y_FFN + D.H_FFN, D.Y_ADD2],
    [D.Y_ADD2 + D.H_SM, D.Y_LN3],
    [D.Y_LN3 + D.H_SM, D.Y_OUTPUT],
  ];
  gaps.forEach(function (g) { arrow(D.CX, g[0], g[1]); });

  // ══════════════════════════════════════════════
  // Skip / Residual connections (right side)
  // ══════════════════════════════════════════════
  var skipColor = MODEL_COLORS.skip.stroke;
  var skipRight = D.CX + D.BOX_W / 2 + 48;

  function drawSkip(startY, endX, endY) {
    dashLine(D.CX, startY, skipRight, startY, skipColor);
    dashLine(skipRight, startY, skipRight, endY, skipColor);
    dashLine(skipRight, endY, endX + 6, endY, skipColor);
    // Arrow at endpoint
    sg.append('polygon')
      .attr('points', endX + ',' + (endY - D.ARROW_S / 2) + ' '
        + (endX + D.ARROW_S / 2) + ',' + endY + ' '
        + endX + ',' + (endY + D.ARROW_S / 2))
      .attr('fill', skipColor);
  }

  // Skip 1: Embeddings→LN1 gap → Add1
  var skip1StartY = D.Y_EMBED + D.H_MD + (D.Y_LN1 - D.Y_EMBED - D.H_MD) / 2;
  var skip1EndX = D.CX + D.NARROW_W / 2;
  var skip1EndY = D.Y_ADD1 + D.H_SM / 2;
  drawSkip(skip1StartY, skip1EndX, skip1EndY);

  sg.append('text')
    .attr('text-anchor', 'middle').attr('font-size', 8)
    .attr('font-family', 'DM Sans, sans-serif').attr('font-style', 'italic')
    .attr('fill', skipColor)
    .attr('transform', 'translate(' + (skipRight + 10) + ',' + (skip1StartY + (skip1EndY - skip1StartY) / 2) + ') rotate(-90)')
    .text('residual');

  // Skip 2: Add1→LN2 gap → Add2
  var skip2StartY = D.Y_ADD1 + D.H_SM + (D.Y_LN2 - D.Y_ADD1 - D.H_SM) / 2;
  var skip2EndX = D.CX + D.NARROW_W / 2;
  var skip2EndY = D.Y_ADD2 + D.H_SM / 2;
  drawSkip(skip2StartY, skip2EndX, skip2EndY);

  sg.append('text')
    .attr('text-anchor', 'middle').attr('font-size', 8)
    .attr('font-family', 'DM Sans, sans-serif').attr('font-style', 'italic')
    .attr('fill', skipColor)
    .attr('transform', 'translate(' + (skipRight + 10) + ',' + (skip2StartY + (skip2EndY - skip2StartY) / 2) + ') rotate(-90)')
    .text('residual');

  // ══════════════════════════════════════════════
  // TP Tensor Grid (left side, inside transformer card area)
  // Dynamic grid: auto-generates from TP count, defaults to no fill
  // ══════════════════════════════════════════════
  var effectiveTp = tpCount || 16;
  var gridCols = Math.ceil(Math.sqrt(effectiveTp));
  var gridRows = Math.ceil(effectiveTp / gridCols);
  var cellW = D.TENSOR_W / gridCols, cellH = D.TENSOR_H / gridRows;

  var tensorOuter = sg.append('rect')
    .attr('x', D.TENSOR_X).attr('y', D.TENSOR_Y)
    .attr('width', D.TENSOR_W).attr('height', D.TENSOR_H)
    .attr('rx', 4).attr('fill', 'var(--bg-surface)')
    .attr('stroke', 'var(--text-muted)').attr('stroke-width', 1);
  addHover(tensorOuter, 1, 'tensor_parallelism_grid');

  var hasHighlight = highlightTpIdx != null && highlightTpIdx >= 0;
  function renderCell(cellIdx, isHighlighted) {
    var col = cellIdx % gridCols;
    var row = Math.floor(cellIdx / gridCols);
    var tcx = D.TENSOR_X + col * cellW;
    var tcy = D.TENSOR_Y + row * cellH;
    var cellClass = 'tensor-cell' + (isHighlighted ? ' pinned' : '');
    var cellRect = sg.append('rect')
      .attr('x', tcx).attr('y', tcy).attr('width', cellW).attr('height', cellH)
      .attr('fill', 'var(--bg-surface)')
      .attr('stroke', isHighlighted ? '#ff8f40' : 'var(--text-muted)')
      .attr('stroke-width', isHighlighted ? 2 : 0.5)
      .attr('stroke-dasharray', isHighlighted ? '3 2' : 'none')
      .attr('class', cellClass);
    addHover(cellRect, 0.5, 'tensor_parallelism_grid');
    sg.append('text')
      .attr('x', tcx + cellW / 2).attr('y', tcy + cellH / 2 + 1)
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
      .attr('font-size', effectiveTp > 16 ? 7 : 9).attr('font-family', 'JetBrains Mono, monospace').attr('font-weight', 600)
      .attr('fill', 'var(--text-secondary)').text(cellIdx + 1)
      .attr('class', 'tensor-cell-label');
  }
  for (var cellIdx = 0; cellIdx < effectiveTp; cellIdx++) {
    if (hasHighlight && cellIdx === highlightTpIdx) continue;
    renderCell(cellIdx, false);
  }
  if (hasHighlight) {
    renderCell(highlightTpIdx, true);
  }

  sg.append('text')
    .attr('x', D.TENSOR_X + D.TENSOR_W / 2).attr('y', D.TENSOR_Y - 10)
    .attr('text-anchor', 'middle').attr('font-size', 9)
    .attr('font-family', 'DM Sans, sans-serif').attr('font-weight', 500)
    .attr('fill', labelColor).text('TP切分Tensor映射');

  // ── PP → layer mapping table ──
  var mapTableTitleY = D.TENSOR_Y + D.TENSOR_H + 18;
  sg.append('text')
    .attr('x', D.TENSOR_X + D.TENSOR_W / 2).attr('y', mapTableTitleY)
    .attr('text-anchor', 'middle').attr('font-size', 9)
    .attr('font-family', 'DM Sans, sans-serif').attr('font-weight', 500)
    .attr('fill', labelColor).text('PP切分模型层映射');

  var mapTableY = mapTableTitleY + 14;
  var legendTopY = mapTableY; // will be updated if table renders

  if (ppCount && cfg.num_layers) {
    var layersPerPp = Math.floor(cfg.num_layers / ppCount);
    var COL_PP = 40, COL_START = 42, COL_END_W = 42;
    var ROW_H = 14, HEADER_H = 15;
    var tableW = COL_PP + COL_START + COL_END_W;
    var tableX = D.TENSOR_X + (D.TENSOR_W - tableW) / 2;

    var tableH = HEADER_H + ppCount * ROW_H;
    var gridStroke = 'var(--text-muted)';
    var gridStrokeW = 0.6;

    // Column boundary X positions
    var sepX1 = tableX + COL_PP;
    var sepX2 = tableX + COL_PP + COL_START;
    var colSeps = [sepX1, sepX2];

    // ── Cell backgrounds (drawn first, below grid lines) ──
    // Header row background
    sg.append('rect')
      .attr('x', tableX).attr('y', mapTableY)
      .attr('width', tableW).attr('height', HEADER_H)
      .attr('fill', '#21262d');

    // Data row backgrounds (highlighted row rendered last to stay on top)
    for (var pi = 0; pi < ppCount; pi++) {
      if (hasHighlight && highlightPpIdx === pi) continue;
      var rowY = mapTableY + HEADER_H + pi * ROW_H;
      sg.append('rect')
        .attr('x', tableX).attr('y', rowY)
        .attr('width', tableW).attr('height', ROW_H)
        .attr('fill', (pi % 2 === 0 ? 'var(--bg-surface)' : '#161b22'))
        .attr('class', 'pp-row');
    }
    if (hasHighlight) {
      var hlPi = highlightPpIdx;
      var hlRowY = mapTableY + HEADER_H + hlPi * ROW_H;
      sg.append('rect')
        .attr('x', tableX).attr('y', hlRowY)
        .attr('width', tableW).attr('height', ROW_H)
        .attr('fill', (hlPi % 2 === 0 ? 'var(--bg-surface)' : '#161b22'))
        .attr('stroke', '#ff8f40')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '3 2')
        .attr('class', 'pp-row pinned');
    }

    // ── Vertical column separator lines (full table height) ──
    colSeps.forEach(function (sx) {
      sg.append('line')
        .attr('x1', sx).attr('y1', mapTableY).attr('x2', sx).attr('y2', mapTableY + tableH)
        .attr('stroke', gridStroke).attr('stroke-width', gridStrokeW);
    });

    // ── Horizontal row separator lines ──
    for (var hi = 0; hi <= ppCount; hi++) {
      var hy = mapTableY + HEADER_H + hi * ROW_H;
      sg.append('line')
        .attr('x1', tableX).attr('y1', hy).attr('x2', tableX + tableW).attr('y2', hy)
        .attr('stroke', gridStroke).attr('stroke-width', gridStrokeW);
    }

    // ── Table outer border ──
    sg.append('rect')
      .attr('x', tableX).attr('y', mapTableY)
      .attr('width', tableW).attr('height', tableH)
      .attr('fill', 'none')
      .attr('stroke', gridStroke).attr('stroke-width', 1).attr('rx', 2);

    // ── Header text ──
    var headerY = mapTableY + 11;
    sg.append('text').attr('x', tableX + COL_PP / 2).attr('y', headerY)
      .attr('text-anchor', 'middle').attr('font-size', 8)
      .attr('font-family', 'DM Sans, sans-serif').attr('font-weight', 600)
      .attr('fill', 'var(--text-secondary)').text('PP');
    sg.append('text').attr('x', tableX + COL_PP + COL_START / 2).attr('y', headerY)
      .attr('text-anchor', 'middle').attr('font-size', 8)
      .attr('font-family', 'DM Sans, sans-serif').attr('font-weight', 600)
      .attr('fill', 'var(--text-secondary)').text('Start');
    sg.append('text').attr('x', tableX + COL_PP + COL_START + COL_END_W / 2).attr('y', headerY)
      .attr('text-anchor', 'middle').attr('font-size', 8)
      .attr('font-family', 'DM Sans, sans-serif').attr('font-weight', 600)
      .attr('fill', 'var(--text-secondary)').text('End');

    // ── Data row text ──
    for (var pi2 = 0; pi2 < ppCount; pi2++) {
      var rowY2 = mapTableY + HEADER_H + pi2 * ROW_H;
      var layerStart2 = pi2 * layersPerPp;
      var layerEnd2 = layerStart2 + layersPerPp - 1;

      var rowTextY = rowY2 + 10;
      sg.append('text').attr('x', tableX + COL_PP / 2).attr('y', rowTextY)
        .attr('text-anchor', 'middle').attr('font-size', 8)
        .attr('font-family', 'JetBrains Mono, monospace')
        .attr('fill', 'var(--text-primary)').text(pi2);
      sg.append('text').attr('x', tableX + COL_PP + COL_START / 2).attr('y', rowTextY)
        .attr('text-anchor', 'middle').attr('font-size', 8)
        .attr('font-family', 'JetBrains Mono, monospace')
        .attr('fill', 'var(--text-primary)').text(layerStart2);
      sg.append('text').attr('x', tableX + COL_PP + COL_START + COL_END_W / 2).attr('y', rowTextY)
        .attr('text-anchor', 'middle').attr('font-size', 8)
        .attr('font-family', 'JetBrains Mono, monospace')
        .attr('fill', 'var(--text-primary)').text(layerEnd2);
    }

    legendTopY = mapTableY + tableH + 36;
  }

  // ══════════════════════════════════════════════
  // Legend (bottom-left, below tensor area)
  // ══════════════════════════════════════════════
  var legendItems = [
    { color: MODEL_COLORS.input_embedding.stroke, label: 'Embeddings' },
    { color: MODEL_COLORS.layer_norm.stroke, label: 'Layer Norm' },
    { color: MODEL_COLORS.mha.stroke, label: 'Self-Attention' },
    { color: MODEL_COLORS.ffn.stroke, label: 'FFN' },
  ];
  var lx = D.TENSOR_X;
  var LEGEND_ROW = 20;
  var legendTitleY = Math.max(D.Y_TF_END - (legendItems.length + 1) * LEGEND_ROW, legendTopY);
  sg.append('text')
    .attr('x', lx).attr('y', legendTitleY + 4)
    .text('Legend').attr('font-size', 11)
    .attr('font-family', 'DM Sans, sans-serif').attr('font-weight', 600)
    .attr('fill', 'var(--text-muted)').attr('letter-spacing', '1px');

  legendItems.forEach(function (item, i) {
    var ly = legendTitleY + 8 + (i + 1) * LEGEND_ROW;
    sg.append('rect')
      .attr('x', lx).attr('y', ly - 5).attr('width', 9).attr('height', 9)
      .attr('rx', 2).attr('fill', item.color);
    sg.append('text')
      .attr('x', lx + 14).attr('y', ly + 5)
      .attr('font-size', 10).attr('font-family', 'DM Sans, sans-serif')
      .attr('fill', 'var(--text-secondary)').text(item.label);
  });
}

function modelRebuild() {
  canvasRebuild();
}

_startFlowAnimation();
