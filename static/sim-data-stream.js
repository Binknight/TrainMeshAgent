/**
 * SimDataStream — decoupled streaming data pipeline for simulation results.
 *
 * Architecture:
 *   Provider (data source) → DataStream (event bus) → Renderers (UI)
 *
 * To swap mock for a real MCP/API backend, implement a new Provider with the
 * same pipeTo() + start() interface — no renderer changes needed.
 */

// ── Tiny event emitter ──────────────────────────────────────────────────────
function Emitter() {
  this._handlers = {};
}
Emitter.prototype.on = function (ev, fn) {
  (this._handlers[ev] = this._handlers[ev] || []).push(fn);
  return this;
};
Emitter.prototype.off = function (ev, fn) {
  var list = this._handlers[ev];
  if (!list) return this;
  if (fn) { this._handlers[ev] = list.filter(function (f) { return f !== fn; }); }
  else { delete this._handlers[ev]; }
  return this;
};
Emitter.prototype.emit = function (ev, payload) {
  var list = this._handlers[ev];
  if (!list) return;
  for (var i = 0; i < list.length; i++) list[i](payload);
};

// ── DataStream — owns the event bus + cancellation ─────────────────────────
function SimDataStream() {
  Emitter.call(this);
  this._cancelled = false;
  this._running = false;
}
SimDataStream.prototype = Object.create(Emitter.prototype);

SimDataStream.prototype.cancel = function () {
  this._cancelled = true;
  this._running = false;
};
SimDataStream.prototype.reset = function () {
  this._cancelled = false;
  this._running = false;
  this._handlers = {};
};
SimDataStream.prototype.isCancelled = function () { return this._cancelled; };
SimDataStream.prototype.isRunning = function () { return this._running; };
SimDataStream.prototype.markRunning = function () { this._running = true; };

// ── Base Provider (interface) ───────────────────────────────────────────────
function SimDataProvider() {}
SimDataProvider.prototype.pipeTo = function (stream) { this._stream = stream; };

// ── MockProvider — emits pre-generated ops over totalDuration ms ────────────
// opts: { totalDuration: 60000 }
function MockSimProvider(rankInfo, opts) {
  SimDataProvider.call(this);
  this._rankInfo = rankInfo;
  this._totalDuration = (opts && opts.totalDuration) || 60000;
  this._timer = null;
}
MockSimProvider.prototype = Object.create(SimDataProvider.prototype);

MockSimProvider.prototype._generateAll = function () {
  // Use the existing genMockWorkload (defined globally in index.html).
  // It returns { ops, loadSeq, totalUs, hbm }.
  // We only use ops + totalUs for timeline/opspec streaming.
  var data = typeof genMockWorkload === "function"
    ? genMockWorkload(this._rankInfo)
    : { ops: [], totalUs: 0 };
  this._ops = data.ops;
  this._totalUs = data.totalUs;
};

MockSimProvider.prototype.start = function () {
  if (!this._stream || this._stream.isCancelled()) return;
  this._generateAll();
  var N = this._ops.length;
  if (N === 0) {
    this._stream.emit("complete");
    return;
  }
  this._stream.markRunning();
  var self = this;
  var idx = 0;
  var delay = Math.max(200, Math.floor(this._totalDuration / N));

  // Emit totalUs upfront so renderers can pre-size the SVG
  this._stream.emit("init", { totalUs: this._totalUs, totalOps: N });

  function tick() {
    if (self._stream.isCancelled()) return;
    self._stream.emit("op", { op: self._ops[idx], index: idx, total: N });
    idx++;
    self._stream.emit("progress", { index: idx, total: N });
    if (idx >= N) {
      self._stream.emit("complete");
      return;
    }
    self._timer = setTimeout(tick, delay);
  }
  // Emit first op immediately, rest on timer
  tick();
};

MockSimProvider.prototype.stop = function () {
  if (this._timer) { clearTimeout(this._timer); this._timer = null; }
};

// ── Factory ─────────────────────────────────────────────────────────────────
function createSimStream(rankInfo, opts) {
  var stream = new SimDataStream();
  var provider = new MockSimProvider(rankInfo, opts);
  provider.pipeTo(stream);
  return { stream: stream, provider: provider, start: function () { provider.start(); } };
}
