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

async function fetchEstimates(deviceType, totalNodes, dp, tp, pp, side, numLayers) {
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

function _buildTooltipHTML(globalRank, metrics, isOrig) {
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
    html += '<tr><th class="col-header left">指标</th><th class="col-header">理论估算值</th><th class="col-header">仿真验证值</th><th class="col-header">差异</th><th class="col-header metric-link-col">详情</th></tr>';
    html += _buildCompareRow('单卡FLOPs', '', estimate, actual, 'flops_per_card', formatFlops, globalRank, isOrig);
    html += _buildCompareRow('HBM', 'GB', estimate, actual, 'hbm_gb', null, globalRank, isOrig);
    html += _buildCompareRow('TP通信', 'GB/micro', estimate, actual, 'tp_comm_gb_per_micro', null, globalRank, isOrig);
    html += _buildCompareRow('PP通信', 'MB/micro', estimate, actual, 'pp_comm_mb_per_micro', null, globalRank, isOrig);
    html += _buildCompareRow('DP通信', 'GB/step', estimate, actual, 'dp_comm_gb_per_step', null, globalRank, isOrig);
    html += '</table>';
  }

  if (actual && Object.keys(actual).length > 0) {
    html += '<button class="tooltip-detail-btn" onclick="event.stopPropagation();openDetailPopup(' + globalRank + ',' + (isOrig ? 'true' : 'false') + ')">📊 仿真详情</button>';
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
    linkHtml = '<span class="metric-link-na">查看</span>';
  } else {
    var linkType = key.replace(/_per_micro|_per_step|_gb|_mb/g, '');
    if (key === 'hbm_gb') linkType = 'hbm';
    else if (key === 'tp_comm_gb_per_micro') linkType = 'tp-comm';
    else if (key === 'pp_comm_mb_per_micro') linkType = 'pp-comm';
    else if (key === 'dp_comm_gb_per_step') linkType = 'dp-comm';
    linkHtml = '<a class="metric-link" href="javascript:void(0)" onclick="event.stopPropagation();openMetricDetail(' + globalRank + ',' + (isOrig ? 'true' : 'false') + ',\'' + linkType + '\')">查看</a>';
  }

  return '<tr><td class="metric-label">' + label + '</td>' +
    '<td class="metric-val">' + ev + uv + '</td>' +
    '<td class="metric-val actual">' + av + uv + '</td>' +
    '<td class="metric-delta ' + deltaClass + '">' + delta + '</td>' +
    '<td class="metric-val" style="text-align:center">' + linkHtml + '</td></tr>';
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
            tip.innerHTML = _buildTooltipHTML(tp.globalRank, metrics, isOrig);
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

async function loadMeshData(topoData) {
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

  // Skip estimate fetch if already populated (prevent duplicate calls on re-entry)
  var existing = isOrig ? meshEstimateOrig : meshEstimateEq;
  if (Object.keys(existing).length === 0 || existing[0] == null) {
    try {
      var numLayers = topoData.num_layers != null ? topoData.num_layers : (isOrig ? meshModelOrig : meshModelEq).num_layers;
      var estimates = await fetchEstimates(deviceType, totalNodes, dp, tp, pp, side, numLayers);
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

  meshRebuild();
}

// ── Attach to window for inline onclick / onchange handlers ──

window.loadMeshData = loadMeshData;
window.meshRebuild = meshRebuild;
window.meshSwitchDp = meshSwitchDp;
window.meshSwitchDpOrig = meshSwitchDpOrig;
window.meshSwitchDpEq = meshSwitchDpEq;
window.fetchSimulationData = fetchSimulationData;
window.openMetricDetail = openMetricDetail;
window.closeMetricDetailPopup = closeMetricDetailPopup;

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

// ═══════════════════════════════════════════════════════════════════
// Device Simulation Detail Popup
// ═══════════════════════════════════════════════════════════════════

var detailData = null;
var detailSide = '';
var detailGlobalRank = -1;
var detailActiveTab = 'timeline';

async function openDetailPopup(globalRank, isOrig) {
  detailSide = isOrig ? 'original' : 'equivalent';
  detailGlobalRank = globalRank;
  try {
    var resp = await fetch(API + '/session/' + sessionId + '/simulation/' + detailSide + '/' + globalRank + '/detail');
    if (!resp.ok) { alert('获取仿真详情失败'); return; }
    detailData = await resp.json();
    _showDetailPopup();
  } catch (e) {
    console.error('Fetch device detail error:', e);
    alert('获取仿真详情异常: ' + e.message);
  }
}

function _showDetailPopup() {
  var overlay = document.getElementById('detail-popup-overlay');
  var d = detailData;

  document.getElementById('detail-popup-title').textContent = '📊 Rank ' + d.global_rank + ' 仿真详情';
  document.getElementById('detail-popup-subtitle').textContent =
    d.topology_name + ' | ' + d.device_type + ' | DP' + d.dp_rank + '/TP' + d.tp_rank + '/PP' + d.pp_rank;

  var t = d.timeline;
  var summary = document.getElementById('detail-popup-summary');
  summary.innerHTML =
    '<span class="detail-summary-chip">总耗时 <strong>' + t.total_time_ms.toFixed(2) + ' ms</strong></span>' +
    '<span class="detail-summary-chip">计算 <strong>' + t.compute_pct + '%</strong> (' + t.compute_time_ms.toFixed(2) + ' ms)</span>' +
    '<span class="detail-summary-chip">通信 <strong>' + t.comm_pct + '%</strong> (' + t.comm_time_ms.toFixed(2) + ' ms)</span>' +
    '<span class="detail-summary-chip">总FLOPs <strong>' + formatFlops(t.total_flops) + '</strong></span>' +
    '<span class="detail-summary-chip">总通信 <strong>' + t.total_comm_gb.toFixed(4) + ' GB</strong></span>' +
    '<span class="detail-summary-chip">算子数 <strong>' + d.operators.length + '</strong></span>';

  document.querySelectorAll('.detail-tab').forEach(function(el) { el.classList.remove('active'); });
  var activeTab = document.querySelector('.detail-tab[data-tab="timeline"]');
  if (activeTab) activeTab.classList.add('active');
  detailActiveTab = 'timeline';

  overlay.style.display = 'flex';
  _renderActiveTab();
}

function closeDetailPopup() {
  document.getElementById('detail-popup-overlay').style.display = 'none';
  detailData = null;
}

function switchDetailTab(tab) {
  detailActiveTab = tab;
  document.querySelectorAll('.detail-tab').forEach(function(el) { el.classList.remove('active'); });
  var btn = document.querySelector('.detail-tab[data-tab="' + tab + '"]');
  if (btn) btn.classList.add('active');
  _renderActiveTab();
}

function _renderActiveTab() {
  if (!detailData) return;
  var body = document.getElementById('detail-popup-body');
  body.innerHTML = '';
  if (detailActiveTab === 'timeline') _renderTimeline(detailData.operators, body);
  else if (detailActiveTab === 'traffic') _renderTraffic(detailData.operators, detailData.timeline, body);
  else if (detailActiveTab === 'tracing') _renderTracing(detailData.operators, detailData.timeline, body);
}

// ── Drag support for the popup ──

(function() {
  var popup = document.getElementById('detail-popup');
  var header = document.getElementById('detail-popup-header');
  if (!popup || !header) return;
  var dragging = false, offX = 0, offY = 0;
  header.addEventListener('mousedown', function(e) {
    dragging = true; offX = e.clientX - popup.offsetLeft; offY = e.clientY - popup.offsetTop;
    e.preventDefault();
  });
  document.addEventListener('mousemove', function(e) {
    if (!dragging) return;
    popup.style.left = (e.clientX - offX) + 'px';
    popup.style.top = (e.clientY - offY) + 'px';
    popup.style.margin = '0';
  });
  document.addEventListener('mouseup', function() { dragging = false; });
})();

// Close popup on overlay click
document.getElementById('detail-popup-overlay').addEventListener('click', function(e) {
  if (e.target === this) closeDetailPopup();
});

// ── Timeline (Gantt Chart) ──

function _renderTimeline(operators, container) {
  var wrap = document.createElement('div');
  wrap.className = 'detail-timeline-wrap';
  container.appendChild(wrap);

  var totalTime = operators.length > 0 ? operators[operators.length - 1].start_us + operators[operators.length - 1].duration_us : 1;
  var rowH = 16, rowGap = 2, labelW = 170;
  var barAreaLeft = labelW + 8;

  var groups = [];
  var currentCat = '';
  var catStart = 0, catEnd = 0;
  operators.forEach(function(op, i) {
    if (op.category !== currentCat) {
      if (currentCat !== '') groups.push({cat: currentCat, start: catStart, end: catEnd});
      currentCat = op.category;
      catStart = op.start_us;
    }
    catEnd = op.start_us + op.duration_us;
  });
  groups.push({cat: currentCat, start: catStart, end: catEnd});

  var catColors = {fwd: 'rgba(57,186,230,0.04)', bwd: 'rgba(186,161,255,0.04)', optimizer: 'rgba(127,217,98,0.04)'};
  var catLabels = {fwd: 'Forward', bwd: 'Backward', optimizer: 'Optimizer'};

  var totalRows = operators.length;
  var svgH = totalRows * (rowH + rowGap) + 40;
  var svgW = Math.max(barAreaLeft + 900, 800);

  var svg = d3.select(wrap).append('svg')
    .attr('width', svgW).attr('height', svgH);

  var xScale = d3.scaleLinear().domain([0, totalTime]).range([barAreaLeft, svgW - 20]);

  groups.forEach(function(g) {
    var x0 = xScale(g.start), x1 = xScale(g.end);
    var y0 = 0, y1 = svgH - 20;
    var firstIdx = operators.findIndex(function(op) { return op.category === g.cat; });
    var lastIdx = -1;
    for (var i = operators.length - 1; i >= 0; i--) { if (operators[i].category === g.cat) { lastIdx = i; break; } }
    if (firstIdx >= 0 && lastIdx >= 0) {
      y0 = firstIdx * (rowH + rowGap);
      y1 = (lastIdx + 1) * (rowH + rowGap);
    }
    svg.append('rect').attr('x', x0).attr('y', y0).attr('width', x1 - x0).attr('height', y1 - y0)
      .attr('fill', catColors[g.cat] || 'transparent').attr('rx', 2);
    svg.append('text').attr('x', x0 + 4).attr('y', y0 + 12)
      .attr('fill', 'var(--text-muted)').attr('font-size', '9px').attr('font-family', 'var(--font-mono)')
      .text(catLabels[g.cat] || g.cat);
  });

  operators.forEach(function(op, i) {
    var y = i * (rowH + rowGap);
    var bx = xScale(op.start_us), bw = Math.max(xScale(op.duration_us) - xScale(0), 2);

    svg.append('rect')
      .attr('class', 'timeline-bar ' + op.op_type)
      .attr('x', bx).attr('y', y).attr('width', bw).attr('height', rowH);

    svg.append('text')
      .attr('x', barAreaLeft - 4).attr('y', y + 11)
      .attr('text-anchor', 'end')
      .attr('fill', 'var(--text-secondary)').attr('font-size', '9px').attr('font-family', 'var(--font-mono)')
      .text(op.op_name);

    svg.append('title').text(
      op.op_name + ' | ' + op.op_type + ' | ' + op.category + '\n' +
      'Start: ' + (op.start_us / 1000).toFixed(2) + ' ms\n' +
      'Duration: ' + (op.duration_us / 1000).toFixed(2) + ' ms\n' +
      (op.flops > 0 ? 'FLOPs: ' + formatFlops(op.flops) + '\n' : '') +
      (op.comm_bytes > 0 ? 'Comm: ' + (op.comm_bytes / 1e6).toFixed(2) + ' MB' : '')
    );
  });

  var xAxis = d3.axisTop(xScale).ticks(10).tickFormat(function(v) { return (v / 1000).toFixed(1) + 'ms'; });
  svg.append('g').attr('class', 'timeline-axis').attr('transform', 'translate(0, ' + (svgH - 18) + ')').call(xAxis);

  var legendY = svgH - 8;
  var legend = [{label: '计算', cls: 'computation'}, {label: '通信', cls: 'communication'}, {label: '集合通信', cls: 'collective'}];
  var lx = barAreaLeft;
  legend.forEach(function(l) {
    svg.append('rect').attr('x', lx).attr('y', legendY - 8).attr('width', 12).attr('height', 8)
      .attr('class', 'timeline-bar ' + l.cls);
    svg.append('text').attr('x', lx + 16).attr('y', legendY).attr('fill', 'var(--text-muted)').attr('font-size', '9px').text(l.label);
    lx += 80;
  });
}

// ── Traffic Breakdown ──

function _renderTraffic(operators, timeline, container) {
  container.innerHTML = '';

  var tbl = '<table class="detail-stat-table">' +
    '<tr><td class="stat-label">总耗时</td><td class="stat-val">' + timeline.total_time_ms.toFixed(2) + ' ms</td></tr>' +
    '<tr><td class="stat-label">计算耗时</td><td class="stat-val">' + timeline.compute_time_ms.toFixed(2) + ' ms (' + timeline.compute_pct + '%)</td></tr>' +
    '<tr><td class="stat-label">通信耗时</td><td class="stat-val">' + timeline.comm_time_ms.toFixed(2) + ' ms (' + timeline.comm_pct + '%)</td></tr>' +
    '<tr><td class="stat-label">总计算量</td><td class="stat-val">' + formatFlops(timeline.total_flops) + '</td></tr>' +
    '<tr><td class="stat-label">总通信量</td><td class="stat-val">' + timeline.total_comm_gb.toFixed(4) + ' GB</td></tr>' +
    '</table>';
  container.innerHTML += tbl;

  var pieTitle = document.createElement('div');
  pieTitle.className = 'traffic-chart-title';
  pieTitle.textContent = '计算 vs 通信 耗时占比';
  container.appendChild(pieTitle);

  var pieW = 260, pieH = 180;
  var pieSvg = d3.select(container).append('svg').attr('width', pieW).attr('height', pieH);
  var pieData = [
    {label: '计算', value: timeline.compute_time_ms, color: 'var(--cyan)'},
    {label: '通信', value: timeline.comm_time_ms, color: 'var(--orange)'}
  ];
  var pie = d3.pie().value(function(d) { return d.value; })(pieData);
  var arc = d3.arc().innerRadius(45).outerRadius(80);
  var g = pieSvg.append('g').attr('transform', 'translate(110,95)');
  g.selectAll('path').data(pie).enter().append('path')
    .attr('d', arc).attr('fill', function(d) { return d.data.color; }).attr('stroke', 'var(--bg-deep)').attr('stroke-width', 2);
  g.selectAll('text').data(pie).enter().append('text')
    .attr('transform', function(d) { return 'translate(' + arc.centroid(d) + ')'; })
    .attr('fill', '#fff').attr('font-size', '10px').attr('text-anchor', 'middle')
    .text(function(d) { return d.data.value > 0 ? d.data.label + '\n' + (d.data.value / timeline.total_time_ms * 100).toFixed(1) + '%' : ''; });

  var lx2 = 30;
  [['计算', 'var(--cyan)'], ['通信', 'var(--orange)']].forEach(function(l) {
    pieSvg.append('rect').attr('x', lx2).attr('y', pieH - 22).attr('width', 10).attr('height', 10).attr('fill', l[1]).attr('rx', 2);
    pieSvg.append('text').attr('x', lx2 + 14).attr('y', pieH - 13).attr('fill', 'var(--text-secondary)').attr('font-size', '10px').text(l[0]);
    lx2 += 70;
  });

  var barTitle = document.createElement('div');
  barTitle.className = 'traffic-chart-title';
  barTitle.textContent = '算子类型耗时分布 (Top 15)';
  barTitle.style.marginTop = '16px';
  container.appendChild(barTitle);

  var aggMap = {};
  operators.forEach(function(op) {
    if (!aggMap[op.op_name]) aggMap[op.op_name] = {duration: 0, type: op.op_type};
    aggMap[op.op_name].duration += op.duration_us;
  });
  var agg = Object.entries(aggMap).map(function(e) { return {name: e[0], duration: e[1].duration, type: e[1].type}; });
  agg.sort(function(a, b) { return b.duration - a.duration; });
  var top15 = agg.slice(0, 15);

  var barW = 700, barH = top15.length * 20 + 30;
  var barSvg = d3.select(container).append('svg').attr('width', barW).attr('height', barH);
  var maxDur = d3.max(top15, function(d) { return d.duration; });
  var barX = d3.scaleLinear().domain([0, maxDur]).range([130, barW - 20]);

  top15.forEach(function(d, i) {
    var y = i * 20;
    barSvg.append('text').attr('x', 126).attr('y', y + 13).attr('text-anchor', 'end')
      .attr('fill', 'var(--text-secondary)').attr('font-size', '9px').attr('font-family', 'var(--font-mono)').text(d.name);
    barSvg.append('rect').attr('x', barX(0)).attr('y', y + 2).attr('width', barX(d.duration) - barX(0)).attr('height', 14)
      .attr('fill', d.type === 'communication' || d.type === 'collective' ? 'var(--orange)' : 'var(--cyan)').attr('opacity', 0.7).attr('rx', 2);
    barSvg.append('text').attr('x', barX(d.duration) + 4).attr('y', y + 13)
      .attr('fill', 'var(--text-muted)').attr('font-size', '8px').attr('font-family', 'var(--font-mono)')
      .text((d.duration / 1000).toFixed(2) + ' ms');
  });
}

// ── Operator Tracing (Flame Graph) ──

function _renderTracing(operators, timeline, container) {
  var totalTime = operators.length > 0 ? operators[operators.length - 1].start_us + operators[operators.length - 1].duration_us : 1;

  var treeW = 860, rowH = 18;
  var marginLeft = 20;

  var catColors = {fwd: 'var(--cyan)', bwd: 'var(--purple)', optimizer: 'var(--green)'};
  var catBg = {fwd: 'rgba(57,186,230,0.08)', bwd: 'rgba(186,161,255,0.08)', optimizer: 'rgba(127,217,98,0.08)'};

  var blockGroups = [];
  var currentBlock = null;
  operators.forEach(function(op) {
    var parts = op.parent_op.split('/');
    var blockKey = parts[0] + '/' + (parts[1] || '');
    if (blockKey !== currentBlock) {
      currentBlock = blockKey;
      blockGroups.push({key: blockKey, start: op.start_us, end: op.start_us + op.duration_us, cat: op.category});
    } else {
      var bg = blockGroups[blockGroups.length - 1];
      bg.end = Math.max(bg.end, op.start_us + op.duration_us);
    }
  });

  var svgH = blockGroups.length * (rowH + 4) + 60;
  var svgW = treeW;
  var xScale2 = d3.scaleLinear().domain([0, totalTime]).range([marginLeft, svgW - 20]);

  var svg = d3.select(container).append('svg').attr('width', svgW).attr('height', svgH);

  blockGroups.forEach(function(bg, bi) {
    var y = bi * (rowH + 4);
    var x0 = xScale2(bg.start), x1 = xScale2(bg.end);

    svg.append('rect').attr('x', x0).attr('y', y).attr('width', Math.max(x1 - x0, 1)).attr('height', rowH)
      .attr('fill', catBg[bg.cat] || 'transparent').attr('rx', 2).attr('stroke', 'var(--border)').attr('stroke-width', '0.5');

    svg.append('text').attr('x', x0 + 4).attr('y', y + 12)
      .attr('fill', catColors[bg.cat] || 'var(--text-secondary)').attr('font-size', '9px').attr('font-family', 'var(--font-mono)')
      .text(bg.key);

    var durMs = ((bg.end - bg.start) / 1000).toFixed(2);
    svg.append('text').attr('x', x1 - 4).attr('y', y + 12).attr('text-anchor', 'end')
      .attr('fill', 'var(--text-muted)').attr('font-size', '8px').text(durMs + ' ms');

    var pctW = (bg.end - bg.start) / totalTime * (svgW - 20 - marginLeft);
    if (pctW > 0.5) {
      svg.append('rect').attr('x', x0).attr('y', y).attr('width', pctW).attr('height', rowH)
        .attr('fill', (bg.cat === 'fwd' ? 'var(--cyan)' : bg.cat === 'bwd' ? 'var(--purple)' : 'var(--green)'))
        .attr('opacity', 0.2).attr('rx', 2);
    }
  });

  var xAxis2 = d3.axisTop(xScale2).ticks(8).tickFormat(function(v) { return (v / 1000).toFixed(1) + 'ms'; });
  svg.append('g').attr('class', 'timeline-axis').attr('transform', 'translate(0, ' + (svgH - 18) + ')').call(xAxis2);

  svg.append('text').attr('x', marginLeft).attr('y', svgH - 2)
    .attr('fill', 'var(--text-muted)').attr('font-size', '9px')
    .text('火焰图: 横轴=时间占比, 每个色块=一个算子组 (Forward/Backward/Optimizer)');
}

// ═══════════════════════════════════════════════════════════════════
// Metric Detail Popup (HBM, TP/PP/DP Comm breakdown)
// ═══════════════════════════════════════════════════════════════════

var metricDetailData = null;
var metricDetailType = '';

function _metricTypeLabel(type) {
  var map = { 'hbm': 'HBM 占用详情', 'tp-comm': 'TP通信详情', 'pp-comm': 'PP通信详情', 'dp-comm': 'DP通信详情' };
  return map[type] || '指标详情';
}

async function openMetricDetail(globalRank, isOrig, metricType) {
  if (metricType === 'flops') return;
  var side = isOrig ? 'original' : 'equivalent';
  try {
    var resp = await fetch(API + '/session/' + sessionId + '/simulation/' + side + '/' + globalRank + '/' + metricType + '-detail');
    if (!resp.ok) { alert('获取详情失败'); return; }
    metricDetailData = await resp.json();
    metricDetailType = metricType;
    _showMetricDetailPopup(globalRank);
  } catch (e) {
    console.error('Fetch metric detail error:', e);
    alert('获取详情异常: ' + e.message);
  }
}

function _showMetricDetailPopup(globalRank) {
  var overlay = document.getElementById('metric-detail-overlay');
  var title = document.getElementById('metric-detail-title');
  var body = document.getElementById('metric-detail-body');

  title.textContent = 'Rank ' + globalRank + ' | ' + _metricTypeLabel(metricDetailType);

  if (metricDetailType === 'hbm') {
    _renderHbmDetail(body);
  } else {
    _renderCommDetail(body);
  }

  overlay.style.display = 'flex';
}

function _renderHbmDetail(container) {
  var d = metricDetailData;
  var html = '<table class="metric-detail-table">';
  html += '<tr><th>参数</th><th>占用 (GB)</th><th>占比</th></tr>';
  var items = [
    { label: '权重', value: d.weights_gb },
    { label: '梯度', value: d.gradients_gb },
    { label: '优化器', value: d.optimizer_gb },
    { label: '激活', value: d.activations_gb }
  ];
  var total = d.total_hbm_gb || 1;
  items.forEach(function(item) {
    html += '<tr><td class="metric-detail-label">' + item.label + '</td><td>' + item.value.toFixed(2) + '</td><td>' + (item.value / total * 100).toFixed(1) + '%</td></tr>';
  });
  html += '<tr class="total-row"><td class="metric-detail-label">总计</td><td>' + d.total_hbm_gb.toFixed(2) + '</td><td>100%</td></tr>';
  html += '</table>';
  container.innerHTML = html;
}

function _renderCommDetail(container) {
  var d = metricDetailData;
  var typeName = { 'tp': 'TP', 'pp': 'PP', 'dp': 'DP' }[d.comm_type] || d.comm_type.toUpperCase();
  var html = '<table class="metric-detail-table">';
  html += '<tr><th>参数</th><th>值</th></tr>';
  html += '<tr><td class="metric-detail-label">通信类型</td><td>' + typeName + ' 通信</td></tr>';
  html += '<tr><td class="metric-detail-label">通信次数</td><td>' + d.comm_count + ' 次/step</td></tr>';
  html += '<tr><td class="metric-detail-label">通信卡数</td><td>' + d.comm_cards + ' 张</td></tr>';
  html += '<tr><td class="metric-detail-label">单次通信量</td><td>' + d.comm_size_per_time_gb.toFixed(4) + ' GB</td></tr>';
  html += '<tr class="total-row"><td class="metric-detail-label">总通信量</td><td>' + d.total_comm_gb.toFixed(4) + ' GB</td></tr>';
  html += '</table>';
  container.innerHTML = html;
}

function closeMetricDetailPopup() {
  document.getElementById('metric-detail-overlay').style.display = 'none';
  metricDetailData = null;
  metricDetailType = '';
}

// Close metric popup on overlay click
document.getElementById('metric-detail-overlay').addEventListener('click', function(e) {
  if (e.target === this) closeMetricDetailPopup();
});

// Close metric popup on Escape
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape' && metricDetailData) {
    closeMetricDetailPopup();
  }
});
