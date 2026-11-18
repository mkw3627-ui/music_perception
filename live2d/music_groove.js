/*!
 * music_groove.js — 音频驱动的 Live2D 律动引擎（为她所听）
 *
 * 风格系统（sway / bounce / nod / lean / step）参考并改编自
 * hamletroyophelia/midi-dance-avatar（GPL-2.0-only）的编舞数学，
 * 但输入从 MIDI 音符换成了连续音频特征，并重写为无依赖 UMD 模块。
 *
 * 输入特征（每帧）:
 *   energy 0..1   overall loudness
 *   low/mid/high  band energy 0..1
 *   beat    0..1  onset strength (decays)
 *   bpm     number
 *   pitch   0..1  spectral centroid proxy
 *   silent  bool
 *
 * 输出是一个归一化的动作帧；由调用方映射到 Live2D 参数。
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.MusicGroove = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var STYLES = ['dance', 'sway', 'bounce', 'nod', 'lean', 'step'];
  var STYLE_WEIGHTS = {
    dance:  { sway: 0.70, bounce: 0.62, nod: 0.35, lean: 0.42, step: 0.52 },
    sway:   { sway: 1.00, bounce: 0.20, nod: 0.10, lean: 0.56, step: 0.08 },
    bounce: { sway: 0.15, bounce: 1.00, nod: 0.40, lean: 0.18, step: 0.10 },
    nod:    { sway: 0.18, bounce: 0.34, nod: 1.00, lean: 0.16, step: 0.08 },
    lean:   { sway: 0.38, bounce: 0.16, nod: 0.10, lean: 1.00, step: 0.08 },
    step:   { sway: 0.48, bounce: 0.32, nod: 0.15, lean: 0.26, step: 1.00 }
  };
  var BASE_GROOVE = { sway: 0.58, bounce: 0.44, nod: 0.22, lean: 0.30, step: 0.34 };

  // 节奏状态：按「BPM + 鼓点规整度 + 打击感」分档，每档的动作结构不一样。
  //   head   抒情/慢歌/没有鼓点的段：慢慢左右摇，幅度小、身体几乎不动。
  //          注意这里不是「只点头」——慢歌使劲点头比不动还出戏。
  //   sway   中速：头为主，身体只有一点点重心
  //   groove 快且鼓点稳、打击感强：全身律动，幅度最大
  // 数值是相对权重：sway/bounce/nod/lean/step 乘在风格权重上，
  // body 缩放所有身体动作，head 缩放头部左右转，tilt 缩放歪头。
  var MODES = {
    head:   { sway: 0.92, bounce: 0.12, nod: 0.20, lean: 0.30, step: 0.04, body: 0.16, head: 0.72, tilt: 0.95 },
    sway:   { sway: 0.82, bounce: 0.55, nod: 0.50, lean: 0.55, step: 0.40, body: 0.70, head: 1.00, tilt: 0.90 },
    groove: { sway: 1.00, bounce: 1.00, nod: 0.80, lean: 1.00, step: 1.00, body: 1.00, head: 1.10, tilt: 1.00 }
  };

  // ---- 舞段（组合）----
  // 一段音乐里不停地重复「上下上下 + 歪一下头」就是循环动画的观感来源。
  // 这里把动作按「头左右摇 / 点头 / 身体」的配比分成长短不同的舞段：
  //   shake    大幅左右摇头（2 拍一个来回）+ 身体摆
  //   nod      大幅上下点头 + 身体跟着上下
  //   bodyOnly 身体大幅摆动，头只轻轻跟
  //   groove   均衡全身律动
  //   subtle   收着跳，给强拍留出对比
  // 每个舞段持续 7~13 秒（按拍数折算，快歌慢歌时长一致），中间不换；
  // 换段时用 1.5s 的平滑把配比揉过去，而不是直接切到另一套动作。
  //   hx   头部左右摆动（ParamAngleX 的转动幅度）
  //   sx   0 = 慢的 4 拍一摆，1 = 每拍换边的摇头（两个正弦按比例混合）
  //   hy   点头 / 起伏幅度；body 身体幅度；tilt 歪头幅度
  //   sw   身体每拍左右扭一下的幅度（辅助，不是主角）
  //   roll 扇形摇头：头以脖子为轴左右「划弧」的幅度（ParamAngleZ 侧倾）
  //   ct   头的反向补偿比例（身体扭动时把脸转回来，默认不用）
  var MOVES = {
    groove:   { hx: 1.00, sx: 0.20, hy: 1.00, body: 1.00, tilt: 1.00, sw: 0.00, ct: 0.00, roll: 0.00 },
    shake:    { hx: 0.30, sx: 0.00, hy: 0.60, body: 1.05, tilt: 1.00, sw: 0.55, ct: 0.00, roll: 1.40 },
    nod:      { hx: 0.55, sx: 0.05, hy: 1.45, body: 1.12, tilt: 0.65, sw: 0.00, ct: 0.00, roll: 0.00 },
    bodyOnly: { hx: 0.40, sx: 0.00, hy: 0.55, body: 1.38, tilt: 0.45, sw: 0.00, ct: 0.00, roll: 0.00 },
    subtle:   { hx: 0.70, sx: 0.10, hy: 0.78, body: 0.72, tilt: 1.10, sw: 0.00, ct: 0.00, roll: 0.00 }
  };
  var MOVE_FADE = 1.5;     // 换段过渡的固定时长（秒），两端速度都为 0
  var STYLE_FADE = 1.6;    // 换风格 / 换方向的过渡时长（秒），同样两端速度为 0
  var PROF_FADE = 1.2;     // 慢歌/中速/律动三档之间的过渡时长（秒）
  var PHRASE_BEATS = 16;   // 风格/方向的一个乐句 = 16 拍（128BPM 约 7.5 秒）

  // ---- 歌曲段落 ----
  // 「前奏 / 渐强 / 主歌 / 副歌 / 尾奏」：拿不到谱子，只能从响度走向推。
  // 前奏要收着、主歌平稳、渐强一点点往上推、副歌放开、尾奏慢慢收。
  // 每个段落只改四个乘数，并且用 3 秒定点过渡（两端速度为 0），
  // 所以段落切换在画面上是「慢慢变成另一种力度」，不是「啪」一下换档。
  //   amp    整体幅度
  //   rate   律动/扇面摇头的快慢（慢歌慢摇、副歌略快）
  //   body   身体参与量
  //   move   大动作（大甩头、大幅点头）的权重
  //   accent 强拍点缀的幅度
  var SECTIONS = {
    intro:  { amp: 0.66, rate: 0.84, body: 0.50, move: 0.00, accent: 0.38 },
    build:  { amp: 0.94, rate: 0.97, body: 0.90, move: 0.55, accent: 0.86 },
    verse:  { amp: 1.00, rate: 1.00, body: 1.00, move: 0.75, accent: 1.00 },
    chorus: { amp: 1.20, rate: 1.12, body: 1.12, move: 1.00, accent: 1.28 },
    outro:  { amp: 0.74, rate: 0.86, body: 0.55, move: 0.18, accent: 0.42 }
  };
  var SECT_FADE = 3.0;     // 段落切换的过渡时长（秒），两端速度都为 0

  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
  function finite(v, d) { return typeof v === 'number' && isFinite(v) ? v : d; }
  // 平滑限幅：0.7 倍以内原样输出，再往上用 tanh 慢慢压住。
  // 硬 clamp 会在极限处切出一段「平顶」—— 头摆到头就停住几帧，离开时像被推一把，
  // 那一下特别像卡顿。平滑限幅没有平顶，极限处速度自然降到 0 再回来。
  function softLim(v, lim) {
    var a = Math.abs(v) / lim;
    if (a <= 0.7) return v;
    var r = lim * (0.7 + 0.3 * Math.tanh((a - 0.7) / 0.3));
    return v < 0 ? -r : r;
  }
  function round(v) { return Math.round(v * 1e6) / 1e6; }
  function smoothstep(lo, hi, v) {
    var t = clamp((v - lo) / Math.max(hi - lo, 1e-9), 0, 1);
    return t * t * (3 - 2 * t);
  }
  // 起音-衰减包络：attack 秒起手、decay 秒收尾，峰值归一化到 1。
  // 纯指数衰减一上来就是最大值，等于「每个拍点抽一下」；加了 attack
  // 之后动作先有几十毫秒的「准备」，再落到极值 —— 打击感还在，但不抽搐。
  function swell(t, attack, decay) {
    if (!(t > 0)) return 0;
    var ratio = decay / attack;
    var peak = (1 - 1 / (1 + ratio)) * Math.exp(-Math.log(1 + ratio) / ratio);
    return (1 - Math.exp(-t / attack)) * Math.exp(-t / decay) / (peak || 1);
  }
  function stableHash(s) {
    var h = 2166136261;
    for (var i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }

  function styleWeights(style, blendBase) {
    var m = STYLE_WEIGHTS[style] || STYLE_WEIGHTS.dance;
    if (!blendBase) return m;
    var out = {};
    for (var k in m) out[k] = clamp(BASE_GROOVE[k] + m[k] * 0.46, 0, 1);
    return out;
  }

  function MusicGroove(opts) {
    opts = opts || {};
    this.mode = opts.style || 'auto';
    this.intensity = clamp(finite(opts.intensity, 1.0), 0, 1.6);
    this.variation = clamp(finite(opts.variation, 0.8), 0, 1);
    this.seed = String(opts.seed || '');
    this.phraseRest = clamp(finite(opts.phraseRest, 0.8), 0.2, 4);
    this.reset();
  }

  MusicGroove.prototype.reset = function () {
    this.beatPhase = 0;
    this.phraseStart = 0;
    this.phraseIndex = 0;
    this.style = 'sway';
    this.direction = 1;
    this._dirS = 1;
    this._wCur = null;
    this._wFrom = null;
    this._wStyle = '';
    this._wFade = 0;
    this._dirFrom = 1;
    this._dirTo = 1;
    this._dirFade = 9;
    this.styleIntensity = 1;
    this._lastBeat = 0;
    this._beatFix = 0;
    this._resumed = true;
    this._silence = 0;
    this._styleTime = 0;
    this._accent = null;
    this._accentCooldown = 0;
    this._accentX = this._accentY = this._accentZ = 0;
    // 三轴当前速度（度/秒）：强拍点缀靠它判断「头现在正往哪边去」，
    // 先顺着惯性推一小段再反向大甩，衔接才连贯。
    this._velX = 0;
    this._velY = 0;
    this._velZ = 0;
    this._lastHR = null;
    this._lastHY = null;
    this._lastHZ = null;
    this._modeState = 'sway';
    this._modeCand = null;
    this._modeCandT = 0;
    this._modeHold = 0;
    // 反循环：慢速起伏（9~18s）、每拍微随机、每乐句整体幅度，三者周期互不整除。
    // 这里再叠一层「舞段」配比（头摇/点头/身体的分配），每 7~13 秒换一次。
    this._move = 'groove';
    this._moveT = 0;
    this._moveDur = 0;      // 0 = 开唱后第一帧就选一段
    this._moveSeq = 0;
    this._moveK = { hx: 1, sx: 0.2, hy: 1, body: 1, tilt: 1, sw: 0, ct: 0, roll: 0 };
    this._moveFrom = null;
    this._moveFade = 0;
    this._moveLast = '';
    this._slowT = 0;
    this._slowPeriod = 11;
    this._slowOff = 0;
    this._slowKS = 1;
    // 扇形摇头自己的相位（弧度）。它按「相位增量 × 当前频率」累加，
    // 不直接用 rollW * ph —— 见 update() 里的说明。
    this._fanPh = 0;
    this._lastPh = null;
    this._leadS = 0.1;
    this._styleIntS = 1;
    this._beatGain = 1;
    this._beatGainS = 1;
    this._ampPhrase = 1;
    this._ampPhraseS = 1;
    // 身体参与量：轻节奏时从 0 开始，随节奏/音量慢慢涨起来
    this._bodyMix = 0;
    // 「这首歌现在有多猛」：音量 × 打击感（起音有多硬）。慢歌/抒情段接近 0，
    // 副歌鼓点一进来就接近 1。所有幅度都受它调制，而不是靠随机数决定。
    this._punchS = 0.45;
    this._punchMax = 0.45;
    this._powerS = 0.45;
    this._powMove = 0.45;
    // 头部摆幅的慢平滑值（不能跟着单拍起音跳）
    this._headAmpS = 8;
    // 突发重音（平静里突然的一声）
    this._hitPulse = 0;
    this._hitT = 0;
    this._hitLast = 0;
    this._prof = null;
    this._profFrom = null;
    this._profFade = 0;
    // 从「慢」开始估，身体参与量也从 0 起步：开场不会先晃两秒身体再收回去。
    this._bpmS = 60;
    this._regS = 0.5;
    // 速度可不可信（监听器的拍点相位集中度）：噪声/散拍时压住「快」，别乱摇
    this._trustS = 0;
    // 歌曲段落：响度包络、这首歌自己的峰值、走向，以及当前的段落参数
    this._eS = 0;
    this._ePrev = 0;
    this._eAvg = 0.30;
    this._eTrend = 0;
    this._ePeak = 0.30;
    this._songT = 0;
    this._sect = 'intro';
    this._sectK = {};
    for (var sk0 in SECTIONS.intro) this._sectK[sk0] = SECTIONS.intro[sk0];
    this._sectFrom = null;
    this._sectFade = 0;
    this._sectCand = '';
    this._sectCandT = 0;
    this._sectHold = 0;
    this._smoothed = {
      bounce: 0, sway: 0, nod: 0, lean: 0, step: 0, headBob: 0,
      bodyRotation: 0, headRotation: 0, headTilt: 0, bodyAct: 0.7,
      positionX: 0, positionY: 0, scale: 1
    };
    this.frame = this._frame(0);
    return this;
  };

  MusicGroove.prototype._pickStyle = function (time) {
    // 每个乐句换一次主导风格；静音后重新开始。
    var h = stableHash(this.seed + '|' + Math.round(this.phraseStart * 1000));
    var style = this.mode === 'auto'
      ? (this.variation > 0 ? STYLES[h % STYLES.length] : 'sway')
      : this.mode;
    this.style = style;
    this.direction = this.variation > 0 && (h & 1) === 0 ? -1 : 1;
    this.styleIntensity = 1 + ((((h >>> 20) % 13) - 6) / 100) * this.variation;
    // 每个乐句换一个慢速起伏的周期/相位和整体幅度：三者的周期互不整除，
    // 动作就不会形成「每小节原样重播」的循环感。
    // 幅度随机只留 ±10%：以前 ±18% 跟别的随机叠在一起能差 40%，
    // 结果「高潮那句反而比主歌还小」——看起来完全不合常理。
    this._slowPeriod = 9 + ((h >>> 4) % 900) / 100;
    this._slowOff = ((h >>> 12) % 628) / 100;
    this._ampPhrase = 0.90 + ((h >>> 26) % 20) / 100;
    this._styleTime = time;
  };

  /**
   * 换一个舞段（动作组合）。权重随能量和身体参与度走：
   * 安静段多是 subtle / groove，副歌里才会出现大幅摇头、大幅点头、
   * 身体大动头小动这几种。同一个舞段不会连着来（权重压到 8%）。
   */
  MusicGroove.prototype._pickMove = function (energy, bpm) {
    // 用户把「变化」调到最低时就不轮换了，固定用均衡律动。
    if (!(this.variation > 0.001)) { this._move = 'groove'; this._moveDur = 1e9; return; }
    var rock = clamp(energy, 0, 1);
    var bodyMix = clamp(this._bodyMix || 0, 0, 1);
    // 打击感：大动作只在这种歌里出现。抒情段就算音量不小，也不会突然大甩头。
    var pow = clamp(this._powMove || 0.45, 0, 1);
    // 段落决定「这一段该不该放得开」：前奏几乎不放大动作，副歌才给足。
    var secMove = 0.30 + 0.85 * clamp((this._sectK && this._sectK.move) || 0.75, 0, 1);
    var pool = [];
    function add(id, w) { if (w > 0.01) pool.push({ id: id, w: w }); }
    add('groove', 1.0);
    add('subtle', (0.45 + 0.85 * (1 - rock)) * (1.30 - 0.60 * pow));
    // 大动作的重量用 pow² —— 抒情段几乎是 0，猛的地方迅速放大。
    // 线性的话「软歌 40% 猛」也还有一半权重，会时不时冒出一段大摇头。
    add('shake', (rock > 0.40 ? 0.35 + 1.25 * rock : 0.02) * (0.06 + 1.25 * pow * pow) * secMove);
    add('nod', (0.25 + 1.05 * rock) * (0.18 + 1.05 * pow * pow) * secMove);
    add('bodyOnly', (bodyMix > 0.42 ? 0.35 + 0.95 * bodyMix : 0.02) * (0.20 + 0.95 * pow * pow) * secMove);
    var total = 0;
    for (var i = 0; i < pool.length; i++) {
      if (pool[i].id === this._move) pool[i].w *= 0.08;
      total += pool[i].w;
    }
    var r = Math.random() * total, pick = pool[pool.length - 1].id;
    for (var j = 0; j < pool.length; j++) {
      r -= pool[j].w;
      if (r <= 0) { pick = pool[j].id; break; }
    }
    // 记下换段这一刻正在用的配比：过渡从「现在这个姿势」出发，
    // 而不是从新舞段的标准姿势倒推回去（那会在换段瞬间弹一下）。
    var from = {};
    for (var fk in this._moveK) from[fk] = this._moveK[fk];
    this._moveFrom = from;
    this._moveFade = 0;
    this._move = pick;
    this._moveLast = pick;
    // 7~13 秒换算成拍数：不管快歌慢歌，一段舞的观感时长都差不多，
    // 不会出现「快歌一秒换一次、慢歌二十秒不动」。
    this._moveDur = (7 + Math.random() * 6) * clamp(bpm, 60, 200) / 60;
    this._moveSeq = (this._moveSeq || 0) + 1;
  };

  /**
   * 舞段计时 + 配比揉合。换段只在整拍上发生，并且配比用 1.5s 平滑过去，
   * 所以观众看到的是「动作慢慢变成另一种」，而不是「啪」地换一套。
   */
  MusicGroove.prototype._updateMove = function (dt, bpm, energy, silent) {
    var k = this._moveK;
    if (!silent) {
      this._moveT += dt * bpm / 60;
      var off = this.beatPhase - Math.round(this.beatPhase);
      if (this._moveT >= this._moveDur && Math.abs(off) < 0.12) {
        this._pickMove(energy, bpm);
        this._moveT = 0;
      }
    }
    var tgt = MOVES[this._move] || MOVES.groove;
    // 换段用「从当前配比出发、smoothstep 走到新配比」的定点过渡，
    // 而不是指数追赶：
    //   · 起点速度是 0 —— 换段那一帧不会突然提速，看起来就是「接着刚才的势头转过去」
    //   · 终点速度也是 0 —— 到位的瞬间不会「顿」一下
    //   · 时长固定 1.5s，不会像指数那样前 0.5 秒走完 60%，后半段拖很久
    if (this._moveFrom) {
      this._moveFade += dt;
      var u = clamp(this._moveFade / MOVE_FADE, 0, 1);
      var e = smoothstep(0, 1, u);
      for (var kk in k) k[kk] = this._moveFrom[kk] + (tgt[kk] - this._moveFrom[kk]) * e;
      if (u >= 1) this._moveFrom = null;
    } else {
      for (var k2 in k) k[k2] = tgt[k2];
    }
  };

  /**
   * 按 BPM / 鼓点规整度 / 能量选节奏状态，并把状态参数平滑过去。
   * 阈值带迟滞（已在 groove 时门槛低 6BPM，已在 head 时高 6BPM），
   * 免得在边界上来回闪；状态之间的过渡用 0.8s 的平滑，不会硬换。
   */
  MusicGroove.prototype._updateMode = function (bpm, energy, regularity, silent, dt) {
    if (!this._prof) {
      this._prof = {};
      for (var k0 in MODES.sway) this._prof[k0] = MODES.sway[k0];
    }
    if (silent) return;   // 停音乐时保持上一个状态，别让休止把状态带偏

    var slow = 1 - Math.exp(-dt / 1.2);
    this._bpmS += (bpm - this._bpmS) * slow;
    this._regS += (regularity - this._regS) * slow;

    var prev = this._modeState || 'sway';
    var fastAt = prev === 'groove' ? 104 : 110;
    var slowAt = prev === 'head' ? 86 : 80;
    var cand = 'sway';
    if (this._bpmS < slowAt) cand = 'head';
    else if (this._bpmS >= fastAt && this._regS > 0.42 && energy > 0.30) cand = 'groove';
    // 没有鼓点、也不响的歌（抒情、轻音乐、氛围）：不管 BPM 多少，都只缓慢摇动。
    // 以前只看 BPM，一首 120BPM 的抒情歌也会被当成摇滚使劲点头。
    var hard0 = clamp(((this._punchS || 0) - 0.18) / 0.45, 0, 1);
    if (cand !== 'head' && hard0 < 0.30 && energy < 0.62) cand = 'head';
    // 反过来：BPM 不高但鼓点很硬、拍子很稳的歌（慢摇滚、trap），
    // 不该因为它"慢"就只轻轻摇头 —— 至少给到 sway 档，让身体跟上。
    if (cand === 'head' && hard0 > 0.62 && energy > 0.42) cand = 'sway';

    // 防抖：候选状态要连续稳定 0.6s 才切，切完至少保持 1.5s。
    // 否则 BPM 在阈值附近轻微漂移时，身体会跟着一会儿动一会儿不动地「抽」。
    this._modeHold = Math.max(0, (this._modeHold || 0) - dt);
    if (cand === this._modeState) {
      this._modeCandT = 0;
    } else {
      if (this._modeCand !== cand) { this._modeCand = cand; this._modeCandT = 0; }
      this._modeCandT = (this._modeCandT || 0) + dt;
      if (this._modeCandT >= 0.6 && this._modeHold <= 0) {
        // 记住切换前的状态参数，做定点过渡（不是从新状态倒推）
        var prevT = MODES[this._modeState] || MODES.sway;
        this._profFrom = {};
        for (var kp in prevT) this._profFrom[kp] = this._prof[kp];
        this._profFade = 0;
        this._modeState = cand;
        this._modeCandT = 0;
        this._modeHold = 1.5;
      }
    }

    // 状态之间同样用 1.2s 定点过渡：起手和到位都是 0 速度。
    // 以前用 0.8s 指数追赶，切换那一帧速度最大，慢歌→律动会「咚」地一下变大。
    var tgt = MODES[this._modeState];
    if (this._profFrom) {
      this._profFade += dt;
      var pu = clamp(this._profFade / PROF_FADE, 0, 1);
      var pe = smoothstep(0, 1, pu);
      for (var k1 in tgt) this._prof[k1] = this._profFrom[k1] + (tgt[k1] - this._profFrom[k1]) * pe;
      if (pu >= 1) this._profFrom = null;
    } else {
      for (var k2 in tgt) this._prof[k2] = tgt[k2];
    }
  };

  MusicGroove.prototype._frame = function (energy) {
    var s = this._smoothed;
    return {
      active: energy > 0.01,
      style: this.style,
      phraseIndex: this.phraseIndex,
      bounce: round(s.bounce), sway: round(s.sway), nod: round(s.nod),
      lean: round(s.lean), step: round(s.step), headBob: round(s.headBob),
      bodyRotation: round(s.bodyRotation), headRotation: round(s.headRotation),
      headTilt: round(s.headTilt),
      bodyAct: round(s.bodyAct),
      mode: this._modeState,
      // 当前舞段（groove / shake / nod / bodyOnly / subtle）：给调试/模拟看
      move: this._move,
      // 当前歌曲段落（intro / build / verse / chorus / outro）：给调试/模拟看
      section: this._sect,
      // 当前节拍相位（拍数，整数=正拍）：给调试/模拟看卡点对不对
      phase: round(this.beatPhase),
      // 随机强拍点缀：不参与平滑，本身已经是快起慢落的包络
      accentX: round(this._accentX || 0), accentY: round(this._accentY || 0),
      accentZ: round(this._accentZ || 0),
      positionX: round(s.positionX), positionY: round(s.positionY),
      scale: round(s.scale),
      energy: round(energy)
    };
  };

  /**
   * @param {object} f   audio features
   * @param {number} dt  seconds since last call
   * @returns {object}   motion frame
   */
  // 目标值平滑的时间常数。0.06s 是折中：再快（0.03）拍点像抽鞭子，
  // 再慢（0.10+）动作明显跟不上鼓点。正弦目标用下面的相位超前把这点
  // 延迟完全抵消掉，所以「跟得上拍」和「动得柔和」可以同时成立。
  var SMOOTH_TAU = 0.06;

  MusicGroove.prototype.update = function (f, dt) {
    f = f || {};
    dt = clamp(finite(dt, 1 / 60), 0, 0.25);
    var energy = clamp(finite(f.energy, 0), 0, 1);
    var beat = clamp(finite(f.beat, 0), 0, 1);
    var bpm = clamp(finite(f.bpm, 100), 40, 220);
    var regularity = clamp(finite(f.regularity, 0.5), 0, 1);
    var pitch = clamp(finite(f.pitch, 0.5), 0, 1);
    var hit = clamp(finite(f.hit, 0), 0, 1);
    // 打击感：鼓点有多硬（监听器给的起音包络峰值）。没有这个字段时按「中等」算，
    // 老版本的 bridge 也能继续用。
    var punch = clamp(finite(f.punch, 0.45), 0, 1);
    var silent = !!f.silent || energy < 0.02;

    // 静音：进入休止，慢慢回到呼吸位。
    if (silent) {
      this._silence += dt;
      this._beatFix = 0;
      if (this._silence > this.phraseRest) {
        this.beatPhase = 0;
        this.phraseStart = this._styleTime = 0;
        // 记一笔：这是「停过一阵」之后的重新开始。第一帧有声时直接对上
        // 监听器锁定的拍点，而不是从 0 慢慢爬过去（那要好几秒才对得上拍）。
        this._resumed = true;
      }
      energy = 0;
    } else {
      this._silence = 0;
      this.beatPhase += dt * bpm / 60;
      // 打击感用 2.2s 的慢包络：一两个装饰音不会让它跳，但副歌一进来
      // 两三秒内就能感觉到「这首歌变猛了」。
      this._punchS += (punch - this._punchS) * (1 - Math.exp(-dt / 2.2));
      if (this._styleTime === 0) { this.phraseStart = 0; this._pickStyle(0); }
      // 每 16 拍换一个乐句风格（128BPM 约 7.5 秒）。原来是 8 拍，
      // 换得太勤，观众一段时间就能背下来「四拍一个套路」。
      if (this.beatPhase >= PHRASE_BEATS) {
        this.beatPhase -= PHRASE_BEATS;
        this.phraseIndex += 1;
        this.phraseStart += PHRASE_BEATS * 60 / bpm;
        this._pickStyle(this.phraseStart);
      }
    }

    // 用监听器锁定的拍相位软对齐引擎自己的相位：每帧按有限速率修正
    // （最多 0.8 拍/秒，约 1 秒收敛），摆动的极值、点头就严格落在鼓点上；
    // 修正过程只是「稍微快一点/慢一点」，不会让动作跳一下。
    var beatPos = finite(f.beatPos, -1);
    var beatConf = clamp(finite(f.beatConf, 0), 0, 1);
    if (!silent && beatPos >= 0 && beatConf >= 0.35) {
      var tgt = beatPos % PHRASE_BEATS;
      if (this._resumed) {
        // 长时间静音后重新有声：幅度此刻还是 0，直接对齐看不出跳，
        // 但能立刻跟上鼓点；否则要按 0.8 拍/秒爬好几秒。
        // 只在「还很轻」的时候允许瞬移 —— 已经跳起来了就不动相位，慢慢拉。
        if (this._powerS < 0.35) {
          this.beatPhase = tgt;
          this._lastPh = null;
        }
        this._resumed = false;
      } else {
        var dp = tgt - this.beatPhase;
        dp -= Math.round(dp / PHRASE_BEATS) * PHRASE_BEATS;
        this.beatPhase += clamp(dp / 0.9, -0.8, 0.8) * dt;
      }
      if (this.beatPhase < 0) this.beatPhase += PHRASE_BEATS;
      if (this.beatPhase >= PHRASE_BEATS) this.beatPhase -= PHRASE_BEATS;
    }

    // 节拍相位有多齐（0=散拍/噪声，1=稳稳踩在鼓点上）。速度和律动都要它把关：
    // 一段没有鼓点的铺底，BPM 也可能被估成 170，但相位集中度只有 0.1 ——
    // 这时候绝不能让人物跟着 170 摇，那正是「慢歌乱快」的来源。
    this._trustS += (clamp((beatConf - 0.22) / 0.30, 0, 1) - this._trustS)
      * (1 - Math.exp(-dt / 1.2));
    var tempoTrust = silent ? 0 : this._trustS;

    // 「这首歌现在有多猛」= 音量 × 打击感。拆成两个量：
    //   rel  这一段是不是这首歌里比较猛的地方（跟这首歌自己的基准比）
    //   hard 这首歌本身是不是有鼓点的歌（抒情/轻音乐 → 低）
    // 只用绝对音量的话，「很响但很软」的歌会被当成摇滚；只用相对值的话，
    // 一首从头到尾都很轻的歌也会被自己比成高潮。两个一起才符合听感。
    this._punchMax += (Math.max(punch, 0.25) - this._punchMax) * (1 - Math.exp(-dt / 20));
    if (this._punchMax < 0.25) this._punchMax = 0.25;
    var relP = clamp(this._punchS / this._punchMax, 0, 1);
    var hard = clamp((this._punchS - 0.18) / 0.45, 0, 1);
    var power = clamp(energy * (0.32 + 0.68 * (0.45 * relP + 0.55 * hard)), 0, 1.15);
    this._powerS = power;
    // 舞段/大动作的门槛：这首歌有鼓点、而且这一段比较猛，才放开大动作
    this._powMove = clamp(0.55 * hard + 0.45 * relP * energy, 0, 1);

    // ---- 歌曲段落：前奏 / 渐强 / 主歌 / 副歌 / 尾奏 ----
    // 拿不到谱子，只能从响度走向推。三个量：
    //   rel   现在和这首歌自己最响的时刻比 —— 这是高潮还是一般段落
    //   loud  绝对响度 —— 一首从头轻到尾的歌不能全程判成高潮
    //   trend 能量走向 —— 在涨是渐强，在落是尾奏
    this._songT = (!silent || this._silence < 12) ? this._songT + dt : 0;
    if (this._silence > 12) {
      // 换歌/停了很久：这首歌的「最响」「近况」「走向」都重新开始
      this._songT = 0;
      this._ePeak = 0.30;
      this._eAvg = 0.30;
      this._eTrend = 0;
    }
    var prevE = this._eS;
    this._eS += (energy - this._eS) * (1 - Math.exp(-dt / 0.9));
    // 峰值只慢慢落：副歌过去一分多钟，同一个位置的大合唱还认得出来
    this._ePeak = Math.max(Math.max(0.28, this._eS), this._ePeak - dt * 0.012);
    // 近 20 秒的平均响度：用它判断「现在比刚才明显响了」。
    // 只看峰值不行 —— 一首歌从头到尾都很响时，主歌也会顶到峰值，
    // 结果整首都判成副歌；跟自己的近况比才知道哪一段是真的抬起来了。
    // 开场 8 秒用快时间常数：从 0 爬到正常音量不是「这首歌在渐强」，
    // 否则每首歌开头都会被判成渐强→副歌。
    var avgTau = this._songT < 8 ? 1.2 : 20;
    this._eAvg += (this._eS - this._eAvg) * (1 - Math.exp(-dt / avgTau));
    this._eTrend += (((this._eS - prevE) / Math.max(dt, 1e-3)) - this._eTrend)
      * (1 - Math.exp(-dt / 2.5));
    var rel = clamp(this._eS / this._ePeak, 0, 1.2);
    var avgRel = clamp(this._eS / Math.max(0.18, this._eAvg), 0, 3);
    var loud = clamp((this._eS - 0.30) / 0.45, 0, 1);
    var heat = clamp(0.45 * rel + 0.55 * loud, 0, 1.1);
    var trend = clamp(this._eTrend / 0.10, -1, 1);
    // 进出副歌用不同门槛（迟滞）：进去要明显够响，出来要明显掉下来。
    // 同一档门槛会被段落里 ±5% 的自然起伏反复越过，动作就一涨一落地「呼吸」。
    var inChorus = this._sect === 'chorus';
    var hot = inChorus ? (heat >= 0.80 || avgRel >= 1.06) : (heat >= 0.90 || avgRel >= 1.16);
    var secCand = 'verse';
    // 顺序有讲究：正在往上冲的时候算「渐强」，冲上去站稳了才算副歌。
    // 否则一到副歌入口那两秒就被判成副歌，整个爬升段都被当成高潮。
    if (this._songT < 20 && heat < 0.62 && this._powerS < 0.50) secCand = 'intro';
    else if (trend > 0.10 && heat < 0.95) secCand = 'build';
    // 副歌要同时满足「够响」+（比近况明显抬起来 或 已经顶到这首歌的响度）
    else if (this._eS >= 0.42 && hot) secCand = 'chorus';
    else if (trend > 0.07) secCand = 'build';
    else if (trend < -0.12 && heat < 0.45 && this._songT > 25) secCand = 'outro';
    // 防抖：候选要稳 1.2 秒才切，切完至少保持 2.5 秒。
    // 没有这一步，副歌里一句轻唱就会被判成尾奏，动作一收一放地「呼吸」。
    this._sectHold = Math.max(0, this._sectHold - dt);
    if (secCand === this._sect) {
      this._sectCandT = 0;
    } else {
      if (this._sectCand !== secCand) { this._sectCand = secCand; this._sectCandT = 0; }
      this._sectCandT = (this._sectCandT || 0) + dt;
      if (this._sectCandT >= 1.2 && this._sectHold <= 0) {
        this._sectFrom = {};
        for (var sk1 in this._sectK) this._sectFrom[sk1] = this._sectK[sk1];
        this._sect = secCand;
        this._sectFade = 0;
        this._sectCandT = 0;
        this._sectHold = 2.5;
      }
    }
    var secT = SECTIONS[this._sect] || SECTIONS.verse;
    if (this._sectFrom) {
      this._sectFade += dt;
      var su = clamp(this._sectFade / SECT_FADE, 0, 1);
      var se = smoothstep(0, 1, su);
      for (var sk2 in secT) this._sectK[sk2] = this._sectFrom[sk2] + (secT[sk2] - this._sectFrom[sk2]) * se;
      if (su >= 1) this._sectFrom = null;
    } else {
      for (var sk3 in secT) this._sectK[sk3] = secT[sk3];
    }
    var sec = this._sectK;

    // 舞段调度：决定这几秒是「大幅摇头 + 身体摆」还是「大幅点头 + 上下」……
    // 换段只在整拍上、并且配比有 1.5s 的揉合。
    this._updateMove(dt, bpm, energy, silent);
    var mk = this._moveK;

    // 起音冲量：C1 连续的衰减包络，避免参数突跳。
    var rising = beat > this._lastBeat;
    if (rising) {
      this._pulse = Math.min(1, this._pulse0 = beat);
      this._pulseT = 0;
      // 相位锁定：每来一个真拍，把节拍相位往整拍拉一点，
      // 免得引擎自己跑的和歌里的鼓点越差越远（那就是「循环动画」的来源）。
      // 修正量不能一帧到位：一帧最多 0.08 拍，等于头在一帧里挪一两个像素，
      // 而且正好发生在拍点（波形在极值处）—— 每个鼓点都「咯」一下。
      // 现在把它存成 _beatFix，用 0.08s 时间常数摊到接下来约 0.25 秒里，
      // 相位照样锁得住，速度是连续的。
      var nearest = Math.round(this.beatPhase);
      var off = nearest - this.beatPhase;
      if (Math.abs(off) <= 0.35) this._beatFix = clamp(off * 0.35, -0.08, 0.08);
    }
    if (this._beatFix && !silent) {
      var fixStep = this._beatFix * (1 - Math.exp(-dt / 0.08));
      this.beatPhase += fixStep;
      this._beatFix -= fixStep;
      if (Math.abs(this._beatFix) < 2e-4) this._beatFix = 0;
    }
    this._lastBeat = beat;
    this._pulseT = (this._pulseT || 0) + dt;
    // 起音包络：不能「一步跳到位再衰减」——那样每个拍点都是突然抽一下，
    // 中间没有任何准备。先 50ms 起手、再 0.19s 收尾，头/身体是「压下去」
    // 而不是「弹一下」；峰值比拍点晚几十毫秒，反而更像打击的余势。
    var onset = this._pulse ? this._pulse * swell(this._pulseT, 0.035, 0.19) : 0;

    // 突发重音：平静里突然的一声，不管节拍时钟锁没锁上都要鼓动一下。
    // bridge 送来的是快速衰减的包络，所以不比较「比上一帧大」，而是
    // 「升到 0.3 以上、且距上一次重音至少 0.25 秒」才算新的一下。
    this._hitT = (this._hitT || 0) + dt;
    var hitStart = hit > 0.3 && (this._hitLast <= 0.3 || this._hitT > 0.25);
    if (hitStart) { this._hitPulse = Math.max(hit, 0.55); this._hitT = 0; }
    this._hitLast = hit;
    // 重音也带 50ms 起手：突然的一声要看得见，但不能让整个人「弹」一下。
    var hitEnv = this._hitPulse ? this._hitPulse * swell(this._hitT, 0.05, 0.26) : 0;
    if (hitEnv < 0.01) hitEnv = 0;
    if (hitEnv > onset) onset = hitEnv;
    // 重音自己的驱动量：不乘 groove，所以「平静的音乐里突然一声」也能明显鼓一下；
    // 越猛的部分这一下越大（0.45 保底），抒情段则收着点。
    var hitDrive = hitEnv * clamp(0.45 + 0.75 * power + 0.25 * energy, 0, 1.2);

    var wTarget = styleWeights(this.style, this.mode === 'auto');
    // 换风格（每 16 拍）不是直接换一套权重，而是「从当前权重出发」做一次
    // 1.6 秒的定点过渡：两端速度都是 0，所以换风格那一拍不会突然提速，
    // 到位时也不会顿一下。指数追赶的问题正是起步瞬间速度最大 —— 那一下
    // 看起来就像被谁推了一把。
    if (!this._wCur) {
      this._wCur = {};
      for (var kk in wTarget) this._wCur[kk] = wTarget[kk];
      this._wStyle = this.style;
    }
    if (this._wStyle !== this.style) {
      this._wFrom = {};
      for (var kf in wTarget) this._wFrom[kf] = this._wCur[kf];
      this._wStyle = this.style;
      this._wFade = 0;
    }
    if (this._wFrom) {
      this._wFade += dt;
      var wu = clamp(this._wFade / STYLE_FADE, 0, 1);
      var we = smoothstep(0, 1, wu);
      for (var kw in wTarget) this._wCur[kw] = this._wFrom[kw] + (wTarget[kw] - this._wFrom[kw]) * we;
      if (wu >= 1) this._wFrom = null;
    } else {
      for (var kw2 in wTarget) this._wCur[kw2] = wTarget[kw2];
    }
    var w = this._wCur;
    // 换乐句时 _pickStyle 会翻转左右方向；直接取反等于把头和身体瞬间拧到另一边。
    // 同样用 1.6s 定点过渡：先减速、再反向加速，像人真的把重心挪过去。
    if (this._dirTo !== this.direction) {
      this._dirFrom = this._dirS;
      this._dirTo = this.direction;
      this._dirFade = 0;
    }
    this._dirFade += dt;
    var du = clamp(this._dirFade / STYLE_FADE, 0, 1);
    this._dirS = this._dirFrom + (this._dirTo - this._dirFrom) * smoothstep(0, 1, du);
    var dir = this._dirS;
    // 节奏状态：慢歌只动头，快且稳的鼓点才全身律动。
    // 规整度再乘上「相位可信度」：没有鼓点的铺底估出来的 BPM 不算数。
    this._updateMode(bpm, energy, regularity * (0.35 + 0.65 * tempoTrust), silent, dt);
    var prof = this._prof;
    // 身体参与量（0=只有头，1=全身）：轻节奏、音量小的时候身体完全不动，
    // 只晃头；节奏和音量慢慢起来时身体再逐渐加入，降下去时再慢慢收掉。
    // 用独立的慢包络（起 1.6s / 落 1.0s），所以是「渐渐胀起来 / 渐渐低下去」，
    // 而不是到某个 BPM 突然开始晃身体。
    var tempoLift = clamp((this._bpmS - 78) / 34, 0, 1);        // 78→112 BPM
    // 慢歌但鼓点很硬（慢摇滚、trap、慢速重拍）：身体也该跟着动。
    // 以前只看 BPM，62 BPM 的重歌身体纹丝不动，只剩头在点 —— 反而更怪。
    tempoLift = Math.max(tempoLift, clamp((power - 0.40) / 0.30, 0, 1) * 0.92);
    var powerLift = clamp((energy - 0.10) / 0.45, 0, 1);        // 音量越足越多
    var bodyT = silent ? 0 : tempoLift * (0.30 + 0.70 * powerLift);
    var bTau = bodyT > this._bodyMix ? 1.6 : 1.0;
    this._bodyMix += (bodyT - this._bodyMix) * (1 - Math.exp(-dt / bTau));
    if (this._bodyMix < 0.002) this._bodyMix = 0;
    // 状态里的 body 仍保留一点区分（head 最含蓄、groove 最放开），但不再直接乘，
    // 否则中间速段（sway 0.7）会把身体幅度压掉一半。
    // 舞段再乘一次 body：有的段落身体大幅摆，有的段落只有头在动。
    // 段落参数再乘一次：前奏/尾奏身体几乎不参与，副歌才真的「全身在动」。
    var bodyAmt = clamp(this._bodyMix * (0.70 + 0.30 * prof.body) * mk.body * sec.body, 0, 1.20);
    // 反循环：慢速起伏（9~18s）、每拍微随机、每乐句整体幅度，三者周期互不整除。
    this._slowT += dt;
    var slowK = 1 + 0.10 * Math.sin((Math.PI * 2 / Math.max(4, this._slowPeriod)) * this._slowT + this._slowOff);
    if (rising) this._beatGain = 0.92 + Math.random() * 0.16;
    this._beatGainS += (this._beatGain - this._beatGainS) * (1 - Math.exp(-dt / 0.18));
    this._ampPhraseS += (this._ampPhrase - this._ampPhraseS) * (1 - Math.exp(-dt / 1.2));
    // 换句时 _slowOff/_slowPeriod/styleIntensity 是直接换新值的，慢波会瞬间跳
    // 十几到三十几个百分点 —— 那一下就是「卡一下」。这里对最终乘数再平滑，
    // 换句变成「起伏慢慢换个节奏」，看不出接缝。
    this._slowKS += (slowK - this._slowKS) * (1 - Math.exp(-dt / 1.4));
    this._styleIntS += (this.styleIntensity - this._styleIntS) * (1 - Math.exp(-dt / 1.4));
    // 幅度直接由 power 决定：又响又硬的段落摆得最大，抒情段自然收回来。
    // 以前幅度只看音量，再加上乐句随机数 ±18%，就会出现「节奏越强、
    // 动作反而越小越慢」的反常观感 —— 现在强弱的差距是单调、可预期的。
    var groove = clamp(power * 1.30, 0, 1.2) * this._styleIntS * this.intensity
      * this._slowKS * this._beatGainS * this._ampPhraseS * sec.amp;
    // 以前是硬 clamp 1.25：主歌和副歌的天花板一样高，高潮只多出一点点。
    // 换成平滑限幅 —— 正常段落原样通过，只有极强的地方才慢慢压住。
    groove = softLim(groove, 1.32);
    // 风格权重再乘上状态权重：状态决定「动哪里」，风格决定「怎么动」。
    var ww = {
      sway: w.sway * prof.sway,
      bounce: w.bounce * prof.bounce,
      nod: w.nod * prof.nod,
      lean: w.lean * prof.lean,
      step: w.step * prof.step
    };

    // 连续律动的起手相位：平滑滤波器本身会把正弦拖后一点（SMOOTH_TAU 秒约等于
    // 1 拍里的 0.06 拍），这里按拍长算出等效延迟并提前，让摆动的极值、点头
    // 的低点严格落在鼓点上。只对正弦类目标做提前，脉冲类目标不能提前。
    var leadW = Math.PI * 2 * bpm / 60;
    // 超前量取决于 BPM。BPM 被重新估出来时会一步变化（比如 100 → 128），
    // 如果直接用，ph 会瞬移零点几拍 —— 头的位置在一帧里挪一下。
    // 0.4s 平滑后，稳态完全一样，只有换速的那一下变成「慢慢对齐」。
    var leadTgt = Math.atan(leadW * SMOOTH_TAU) / (Math.PI * 2);
    this._leadS += (leadTgt - this._leadS) * (1 - Math.exp(-dt / 0.4));
    var ph = this.beatPhase + this._leadS;
    // 这一帧走了多少拍。乐句回绕（beatPhase 15.99 → 0.01）时补 16 拍，
    // 所以它是真实前进量，不是回绕造成的 -16。
    var dPh = this._lastPh == null ? 0 : ph - this._lastPh;
    if (dPh < -PHRASE_BEATS * 0.5) dPh += PHRASE_BEATS;
    else if (dPh > PHRASE_BEATS * 0.5) dPh -= PHRASE_BEATS;
    // 一帧正常最多走 dt*bpm/60 拍（dt 已限到 0.25s）；再多出来的只可能是
    // 静音重开 / 时钟重新对齐造成的整体挪相位。这种「整段跳」不该让扇形摇头
    // 跟着一口气转过去，直接按 0 处理（此刻幅度也是 0，看不出来）。
    var dPhMax = dt * bpm / 60 + 0.12;
    if (dPh > dPhMax) dPh = dPhMax;
    else if (dPh < -0.05) dPh = 0;
    this._lastPh = ph;

    var swayRaw = Math.sin(Math.PI * 0.5 * ph) * groove * ww.sway * dir;
    var stepRaw = Math.sin(Math.PI * ph) * groove * ww.step * dir;
    var bounceRaw = clamp(onset * groove * ww.bounce, 0, 1);
    // 点头强度随舞段变：nod 段（1.45 倍）是真的「大幅度点头」，
    // bodyOnly 段只有 0.55 倍，头只是轻轻跟着点。
    var nodRaw = clamp(onset * groove * ww.nod * mk.hy, 0, 1.5);
    var leanRaw = clamp(((pitch - 0.5) * 2 * 0.7 + swayRaw * 0.3) * groove * ww.lean, -1, 1);
    // 连续点头：每拍一个完整的抬头-低头循环。起音脉冲只负责重拍强调，
    // 单靠它时两拍之间几乎没有头动作，看起来就像只有身体在动。
    // 不再乘 w.nod（这个权重只有 0.2~0.4），否则点头幅度小到看不见。
    // 连续点头：每拍一个完整的抬头-低头循环。低点在拍点上（cos 相位对齐），
    // 所以重拍时头正好压下去，而不是在拍与拍之间晃。
    var bobRaw = clamp((0.5 + 0.5 * Math.cos(Math.PI * 2 * ph))
      * clamp(groove * 1.1, 0, 1) * mk.hy, 0, 1.5);
    // 头部左右：4 拍一个来回的慢摆，和每拍换边的摇头按 sx 混合。
    // sx 高的段落实打实是「左右左右」地甩头，而不是慢慢漂到一边再漂回来；
    // 两个分量权重和为 1，所以 hx 就是幅度，不会被频率混合吃掉。
    // 摇头段主要靠侧倾，扭头分量要让开；但让开的量必须连续变化。
    // 以前写的是 `mk.roll > 0.05 ? ……`，换段时配比跨过 0.05 那一刻会
    // 一步跳 0.18 的权重 —— 头会「咯噔」一下，正是用户说的卡。
    var rampDown = 1 - 0.7 * clamp(mk.roll / 0.5, 0, 1);
    var sxUse = clamp(mk.sx + 0.26 * power * rampDown, 0, 0.9);
    var headXRaw = (Math.cos(Math.PI * 0.5 * ph) * (1 - sxUse)
      + Math.cos(Math.PI * ph) * sxUse) * mk.hx * groove * prof.head;
    var headSwayRaw = Math.cos(Math.PI * 0.5 * ph) * groove * prof.head;
    // 扇形摇头：不让脸「转」过去（那是 ParamAngleX 扭头，看着别扭），
    // 而是让头以脖子为轴左右侧倾（ParamAngleZ），头的重心在扇面上左右划弧。
    // 脸始终朝着观众，只是左右晃 —— 这才是「平移地摇」的观感。
    // 频率跟着「这首歌有多猛」走：抒情段 4 拍慢慢划一次，猛的地方每拍换边。
    // 相位必须自己积分：以前写的是 cos(rollW * ph)，而 ph 每 16 拍回绕一次，
    // rollW*16 并不是 2π 的整数倍 —— 每次换乐句那一帧 cos 的相位就瞬移一大截
    // （实测 cos 值跳 1~2、歪头一下跳 3°），听感上就是「动一会儿咔一下」。
    // 改成按「本帧前进的拍数 × 当前角频率」累加，相位全程连续；强度变化时
    // 也只是频率慢慢变，不会跳相。
    // 扇形摇头的快慢现在真的跟着歌走：62 BPM 的抒情歌约 0.35 Hz（两秒多划
    // 一个来回），128 BPM 约 1 Hz，170 BPM 的舞曲约 1.3 Hz。
    // 以前只看 power，慢歌和快歌摇得一样快 ——「快歌慢歌一个样」就是这么来的。
    // 可信度不够（没鼓点的铺底、散拍）时按 0.25 折算，宁可慢，不要乱摇。
    var tempoN = clamp((this._bpmS - 62) / 100, 0, 1.15) * clamp(0.25 + 0.75 * tempoTrust, 0, 1);
    var rollHz = (0.34 + 0.78 * tempoN + 0.20 * clamp(power, 0, 1)) * sec.rate;
    // 换算成「每拍多少弧度」：相位仍按拍累加，换速时频率连续变化，不会跳相
    var rollW = Math.PI * 2 * rollHz * 60 / clamp(bpm, 40, 220);
    this._fanPh += rollW * dPh;
    var fanRaw = Math.cos(this._fanPh) * groove * mk.sw;
    var rollRaw = Math.cos(this._fanPh) * groove * mk.roll
      * (0.40 + 0.60 * clamp(power, 0, 1));
    var fanDeg = fanRaw * 6.0 * bodyAmt;
    // 大节奏强调：能量越高、起音越重，头左右摆幅越大（6° → 20°）。
    // 副歌/鼓点密的地方会明显甩头，安静段仍然小幅。
    // 起音只在已经有能量时才加权，避免安静歌里偶尔一个跳变就甩头。
    var accentGate = clamp((energy - 0.35) / 0.35, 0, 1);
    var accent = clamp(groove * 0.95 + onset * 0.55 * accentGate, 0, 1);
    // 摆幅必须单独慢平滑：如果让它跟着每一下起音跳，头会在每个拍点上「抽」一下
    // 往外扩，而身体的摇摆不带起音项、看起来是匀的——就成了「头突然大幅度、
    // 身体还是小幅度」。0.65s 平滑后摆幅跟的是整段的强弱，而不是单拍。
    this._headAmpS += (((4 + 2.5 * power) + accent * (6 + 8 * power)) - this._headAmpS)
      * (1 - Math.exp(-dt / 0.65));
    var headAmp = this._headAmpS;

    // ---- 强拍随机点缀 ----
    // 每次强拍随机挑一个头部轴向（左右 / 上下 / 歪头）甩一大下，
    // 但起手方向顺着「这一轴现在正往哪边去」：先在原方向再推一小段（惯性），
    // 然后反方向甩一大段，最后收回正常律动。
    // 这样头不会从左边「啪」地跳到右边，而是连贯地划过去。
    this._accentCooldown = Math.max(0, (this._accentCooldown || 0) - dt);
    // 大动作只在「小节第一拍」起手：随机甩头以前每 2~3 秒来一次，而且大甩的
    // 极值落在拍与拍之间，看起来就是「自己在乱晃、根本没卡点」。现在把起手
    // 绑在小节头，并且把持续时间定成 1.39 拍 —— 包络在 72% 处甩到最大，
    // 正好落在下一拍上，大动作也踩在鼓点上。
    var beatPeriod = 60 / clamp(bpm, 40, 220);
    var nearBeat = Math.abs(this.beatPhase - Math.round(this.beatPhase)) < 0.22;
    var onDownbeat = nearBeat && (Math.round(this.beatPhase) % 4 === 0);
    if (((rising && onDownbeat && this._powerS > 0.45) || (hitStart && hit > 0.5))
        && this._accentCooldown <= 0 && !this._accent) {
      var pick = Math.random();
      var axis = pick < 0.52 ? 'X' : (pick < 0.82 ? 'Y' : 'Z');
      var sm0 = this._smoothed;
      var vel = axis === 'X' ? this._velX : (axis === 'Y' ? this._velY : this._velZ);
      // 视觉上的当前位置：Y 轴是「负角度 = 低头」，和 headBob 反向
      var pos = axis === 'X' ? sm0.headRotation
        : (axis === 'Y' ? -(Math.min(26, sm0.nod * 10 + sm0.headBob * 16)) : sm0.headTilt);
      var dir0;
      if (Math.abs(vel) > 3) dir0 = vel > 0 ? 1 : -1;
      else if (Math.abs(pos) > 1) dir0 = pos > 0 ? 1 : -1;
      else dir0 = Math.random() < 0.5 ? -1 : 1;
      // 大动作持续 1.39 拍，包络在 72% 处甩到最大，正好落在下一拍上。
      var dur = clamp(beatPeriod / 0.72, 0.5, 1.9);
      this._accent = {
        axis: axis,
        dir: dir0,
        // Z（歪头）每度视觉权重最高，幅度收着给；
        // 再整体收 20%：大甩还在，但不再是「一下甩出去半张脸」。
        amp: axis === 'Z' ? (6 + 5 * energy) : (axis === 'Y' ? (9 + 8 * energy) : (8 + 7 * energy)),
        t: 0,
        dur: dur
      };
      // 大动作之间歇 5~10 拍（再叠上「只在小节头起手」），大概 4~6 秒一下，
      // 不会再出现「一直在乱甩」的观感。
      this._accentCooldown = dur + (5 + Math.random() * 5) * beatPeriod;
    }
    var ax = 0, ay = 0, az = 0;
    if (this._accent) {
      var ac = this._accent;
      ac.t += dt;
      var u = ac.t / ac.dur;
      if (u >= 1) {
        this._accent = null;
      } else {
        // 三段式：惯性小推(0~0.22) → 反向大甩(0.22~0.72) → 收回(0.72~1)。
        // 每段都是 smoothstep 起落，位置和速度都连得上，没有「跳一下」。
        // 起手段比大甩段更早开始、幅度更小，看起来是「先被带一下再甩出去」，
        // 有了准备动作，就不是毫无征兆地抽搐。
        var env;
        if (u < 0.22) env = 0.30 * smoothstep(0, 1, u / 0.22);
        else if (u < 0.72) env = 0.30 - 1.30 * smoothstep(0, 1, (u - 0.22) / 0.50);
        else env = -1 + smoothstep(0, 1, (u - 0.72) / 0.28);
        var v = ac.amp * env * ac.dir * clamp(this._powerS * 1.4, 0, 1)
          * clamp(sec.accent, 0.2, 1.4);
        if (ac.axis === 'X') ax = v;
        else if (ac.axis === 'Y') ay = v;
        else az = v;
      }
    }
    this._accentX = ax;
    this._accentY = ay;
    this._accentZ = az;

    var target = {
      // 正弦分量用平滑限幅而不是硬 clamp：硬 clamp 在极限处会切出一段平顶，
      // 头在两侧「贴住」几帧再走 —— 看起来就是一卡一卡。softLim 到极限时
      // 速度自然降到 0，然后再平滑地回来。
      sway: softLim(swayRaw, 1),
      step: softLim(stepRaw, 1),
      // 重音直接加在点头/起伏上，不走 groove 缩放：安静段里的一声也看得见。
      // 系数比之前收了一圈（1.30/1.00/0.62 → 0.95/0.72/0.45）：配合 50ms 起手，
      // 重音是「一下压下去」，而不是整个人突然蹦起 20°。
      bounce: clamp(bounceRaw + hitDrive * 0.95, 0, 1),
      nod: clamp(nodRaw + hitDrive * 0.72, 0, 1.5),
      // 歪头点缀也带一点躯干倾斜（只在身体参与时），不然只有脖子在扭。
      lean: softLim(leanRaw + az * 0.05 * bodyAmt, 1),
      headBob: clamp(bobRaw + hitDrive * 0.45, 0, 1.5),
      // 身体只做重心左右移动；幅度从 3.2→4.2 放大，头甩大动作时躯干跟着带
      // 0.42 倍（只乘 bodyAmt：轻节奏身体不参与时仍然完全不动）。
      // 4.9→6.0：bodyOnly 舞段要能看出「身体大幅摆、头只轻轻跟」。
      // 扇面段里把慢摆让出去一部分，给「每拍换边」的扭动腾幅度，
      // 否则两个分量叠在一起会顶到 ±10° 的硬上限，看起来像卡住。
      bodyRotation: softLim((swayRaw * 6.0 * clamp(1 - 0.45 * mk.sw, 0.30, 1)
        + fanRaw * 6.0 + leanRaw * 1.2 + stepRaw * 0.8) * bodyAmt
        + ax * 0.42 * bodyAmt, 8.5),
      // 头部左右转（ParamAngleX，±30°）：摆幅随大节奏放大到 ±20°。
      // 扇形摇头段里再减去 fanDeg * ct：身体扭过去多少，头就转回来多少，
      // 脸一直朝前，只留下左右平移。
      headRotation: softLim(headXRaw * headAmp * dir - fanDeg * mk.ct
        + (pitch - 0.5) * 2.5, 22),
      // 歪头（ParamAngleZ）；dir 每个乐句翻转，所以左右歪头会交替。
      // 扇形摇头段这里才是主角：每拍左右侧倾一次（rollRaw * 9.5），
      // 头绕着脖子左右划弧、脸不转过去。系数让自然峰值刚好落在下面 ±14 的
      // 上限之内，强拍点缀再加也冲不到 applyToModel 的 ±22 —— 全程不削顶，
      // 摆到两端不会「顿一下」（削顶的平顶一眼就看得出是假的）。
      headTilt: softLim((leanRaw * 6.0 + headSwayRaw * 5.0 * dir) * prof.tilt * mk.tilt
        + fanRaw * 3.2 * mk.tilt + rollRaw * 9.5, 14),
      // 身体到底参与多少（0=只有头，1=全身）。ParamBodyAngleY 用它单独缩放。
      bodyAct: bodyAmt,
      positionX: clamp((swayRaw * 0.026 + stepRaw * 0.012) * bodyAmt, -0.04, 0.04),
      positionY: clamp(-bounceRaw * 0.022 * bodyAmt, -0.03, 0),
      scale: clamp(1 + (bounceRaw * 0.016 - Math.abs(swayRaw) * 0.004) * bodyAmt, 0.982, 1.024)
    };

    var s = this._smoothed;
    // 时间常数见文件头的 SMOOTH_TAU。正弦类目标的延迟已由 lead 抵消，
    // 剩下的是脉冲/重音这类「本来就该有起手」的突变 —— 正是它们被
    // 平滑的几帧，让打击变成了「压下去」而不是「跳一下」。
    var alpha = 1 - Math.exp(-dt / SMOOTH_TAU);
    for (var k in target) s[k] += (target[k] - s[k]) * alpha;

    // 记录三轴「视觉值」的速度，给下一次强拍点缀判断起手方向用。
    // Y 轴实际写进模型的是 -nodDeg，所以这里也取负，保证方向一致。
    var vk = 1 - Math.exp(-dt / 0.10);
    var invDt = 1 / Math.max(dt, 1e-3);
    var dispX = s.headRotation;
    var dispY = -(Math.min(30, s.nod * 10 + s.headBob * 16));
    var dispZ = s.headTilt;
    var lastX = finite(this._lastHR, dispX);
    var lastY = finite(this._lastHY, dispY);
    var lastZ = finite(this._lastHZ, dispZ);
    this._velX += ((dispX - lastX) * invDt - this._velX) * vk;
    this._velY += ((dispY - lastY) * invDt - this._velY) * vk;
    this._velZ += ((dispZ - lastZ) * invDt - this._velZ) * vk;
    this._lastHR = dispX;
    this._lastHY = dispY;
    this._lastHZ = dispZ;

    this.frame = this._frame(silent ? 0 : energy);
    // g56：applyToModel 要用本帧时长做头部限速。帧对象每帧新建，桥同一帧调两次
    // （afterMotionUpdate + beforeModelUpdate 重写）拿到的是同一个引用，可以据此去重。
    this.frame.__dt = dt;
    return this.frame;
  };

  /** 映射到 Cubism 4 参数（调用方负责判断参数是否存在）。 */
  MusicGroove.applyToModel = function (coreModel, frame, opts) {
    opts = opts || {};
    var gain = finite(opts.gain, 1);
    var nod = finite(frame.nod, 0);
    var bob = finite(frame.headBob, 0);
    // 身体参与度：慢歌时接近 0，胸口不会跟着点头一起起伏
    var bodyAct = finite(frame.bodyAct, 1);
    var mix = typeof opts.blend === 'number' ? clamp(opts.blend, 0, 1) : 1;
    // 头部低通的上一帧值（每个 coreModel 一份，换模型就重置）
    var headSt = MusicGroove._headSt || (MusicGroove._headSt = { core: null, m: {} });
    // 所有写入都走这里。blend < 1 时从「模型当前值」过渡到目标值，
    // 而不是从 0 过渡：音乐开始/停下时，头和身体是从待机姿态平滑接过来的，
    // 不会先掉到正中再弹回待机（那正是「硬切回待机」的观感）。
    // 参数适配层：标准 ID 在这个模型上不存在时，换成适配层认出的同义参数；
    // 认不出就跳过。没有加载适配层时按原样写标准 ID。
    var PA = (typeof self !== 'undefined' && self.NekoParamAdapter) || null;
    var prof = PA ? PA.profile(coreModel) : null;
    var STD2CH = { ParamAngleX: 'headX', ParamAngleY: 'headY', ParamAngleZ: 'headZ',
                   ParamBodyAngleX: 'bodyX', ParamBodyAngleY: 'bodyY', ParamBodyAngleZ: 'bodyZ',
                   ParamEyeBallX: 'eyeBallX', ParamEyeBallY: 'eyeBallY', ParamBreath: 'breath' };
    var resolve = function (id) {
      if (!prof) return id;
      var m = prof.map[STD2CH[id]];
      return m ? m.id : null;
    };
    var write = function (id, value) {
      try {
        if (coreModel.getParameterIndex(id) < 0) return;
        if (mix < 0.999) {
          // 读当前值失败（老运行时没有这个 API）就退回「从 0 缩放」，
          // 至少不会在 blend 还是 0 的时候突然写入满幅动作。
          var cur = null;
          try { cur = coreModel.getParameterValueById(id); } catch (e2) { cur = null; }
          value = (typeof cur === 'number' && isFinite(cur))
            ? cur + (value - cur) * mix
            : value * mix;
        }
        // g56：颈部惯性。头一帧跳 3~6°（±30° 参数）在画面上就是「抽一下」——
        // 实测副歌里 AngleY 每帧 4.1~6.1°。真实脖子不可能这么快，用一阶低通
        // 把跳变摊到几帧（tau 45ms），摆幅基本不变、只慢半拍；再兜一道
        // 1.7°/帧 的硬上限（约 102°/s），保证任何断裂都不会一帧弹过去。
        // 同一个 frame 被调两次（桥在 beforeModelUpdate 里重写）时直接复用，
        // 不重复滤第二遍。
        if (id.indexOf('ParamAngle') === 0) {
          if (!headSt.core || headSt.core !== coreModel) headSt.core = coreModel, headSt.m = {};
          var S = headSt.m[id];
          if (S && S.f === frame) {
            value = S.v;
          } else {
            var fdt = clamp(finite(frame.__dt, 1 / 60), 0.004, 0.1);
            var kk = fdt / (0.045 + fdt);
            var prev = S ? S.v : null;
            if (prev == null) {
              try { prev = Number(coreModel.getParameterValueById(id)); } catch (eL) { prev = null; }
            }
            if (typeof prev === 'number' && isFinite(prev)) {
              value = prev + (value - prev) * kk;
              // 硬上限按「秒」算：102°/s（60fps 时 1.7°/帧）。第一次写（还没有
              // 滤过的上一帧）以模型当前值为基准，换模型/切歌也不会一帧弹过去。
              var hiL = 30, loL = -30;
              try {
                var idxL = coreModel.getParameterIndex(id);
                if (idxL >= 0) {
                  hiL = Number(coreModel.getParameterMaximumValue(idxL));
                  loL = Number(coreModel.getParameterMinimumValue(idxL));
                }
              } catch (eL2) { hiL = 30; loL = -30; }
              if (!(hiL > loL)) { hiL = 30; loL = -30; }
              var mStep = 102 * fdt * (hiL - loL) / 60;
              value = clamp(value, prev - mStep, prev + mStep);
            }
            headSt.m[id] = { v: value, f: frame };
          }
        }
        coreModel.setParameterValueById(id, value);
      } catch (e) { /* 模型没有这个参数就跳过 */ }
    };
    // 直接按模型角度写入。身体参数范围通常 ±10°，原值可用。
    var setDegree = write;
    // 头部参数范围通常是 ±30°，同样写 2~3° 只有满幅的 7~10%，肉眼几乎看不见，
    // 而身体 ±10° 写 5~7° 已经是 60~70%。所以头部按参数自身范围的比例驱动，
    // 换模型/换皮也能保持同样的可见幅度。
    var setFraction = function (id, fraction, weight) {
      id = resolve(id);
      if (!id) return;
      var idx = -1;
      try { idx = coreModel.getParameterIndex(id); } catch (e) { idx = -1; }
      if (idx < 0) return;
      try {
        var lo = coreModel.getParameterMinimumValue(idx);
        var hi = coreModel.getParameterMaximumValue(idx);
        if (!(hi > lo)) return;              // 模型把这个参数锁死了
        // 以模型默认值为中心（大多数模型默认 = 区间中点；不对称的模型也不会整体偏移）
        var m = prof && (function () { for (var k in prof.map) if (prof.map[k].id === id) return prof.map[k]; return null; })();
        if (m && PA) write(id, PA.toRaw(m, clamp(fraction * weight, -1, 1)));
        else write(id, (lo + hi) / 2 + clamp(fraction, -1, 1) * (hi - lo) / 2 * weight);
      } catch (e) {
        write(id, fraction * 30);            // 旧运行时没有范围 API，按标准 ±30° 退回
      }
    };

    // 头部 ±30°、身体 ±10°，所以统一用「角度 / 半量程」换算成比例，
    // 换模型也能保持同样的可见幅度（实测这个模型：头部 1°≈15 位移单位，
    // 身体 1°≈31 位移单位，身体每度的视觉权重是头的两倍）。
    // 点头/起伏的上限从 26° 放到 30°：nod 舞段（hy 1.45）是真的「大幅度点头」，
    // 26° 会把它切平，看起来像卡住。
    var nodDeg = softLim(nod * 10 + bob * 16, 30);
    var ax = finite(frame.accentX, 0), ay = finite(frame.accentY, 0), az = finite(frame.accentZ, 0);
    // 强拍点缀是加在平滑之后的，这里如果再用硬 clamp，头一到上限就会「贴住」
    // 几帧再突然松开 —— 那是很明显的卡顿。全部改成平滑限幅：
    // 到极限只是速度自然降到 0，不会出现平顶。
    var yRaw = -nodDeg + ay;
    yRaw = yRaw >= 0 ? softLim(yRaw, 16) : -softLim(-yRaw, 30);
    setFraction('ParamAngleX', softLim(finite(frame.headRotation, 0) + ax, 28) / 30, 1.0 * gain);
    setFraction('ParamAngleY', yRaw / 30, 1.0 * gain);
    setFraction('ParamAngleZ', softLim(finite(frame.headTilt, 0) + az, 22) / 30, 1.0 * gain);
    setFraction('ParamBodyAngleX', finite(frame.bodyRotation, 0) / 10, 1.0 * gain);
    setFraction('ParamBodyAngleY', -(nod * 1.85 + bob * 2.45) * bodyAct / 10, 1.0 * gain);
    setFraction('ParamBodyAngleZ', finite(frame.lean, 0) * 1.6 / 10, 1.0 * gain);
    // 眼球和头/身不是一类参数：头 ±30°、身体 ±10° 可以直接按度写，
    // 眼珠在大多数模型里是**归一化**的（±1），按度写等于把 1° 当成满量程的
    // 1/1——眼珠会直接掉到眼角、被上下眼睑切掉一截（就是「唱歌时眼珠变小、
    // 低音鼓点一来更小」）。桥里虽然还有一道 ±18% 的限幅，但已经落在可见
    // 范围里了。这里统一换算成「占参数区间的比例」：±1 模型里同样的意图
    // 只有 1~2‰，±30 的按度模型行为不变。
    var setGaze = function (id, deg) {
      id = resolve(id);
      if (!id) return;
      var idx = -1;
      try { idx = coreModel.getParameterIndex(id); } catch (e) { idx = -1; }
      if (idx < 0) return;
      var lo, hi;
      try {
        lo = Number(coreModel.getParameterMinimumValue(idx));
        hi = Number(coreModel.getParameterMaximumValue(idx));
      } catch (e2) { lo = -30; hi = 30; }    // 老运行时：按度写回去
      if (!isFinite(lo) || !isFinite(hi) || !(hi > lo)) return;
      if (Math.max(Math.abs(lo), Math.abs(hi)) <= 2) {
        write(id, clamp(deg / 30, -1, 1) * Math.max(hi, -lo));
      } else {
        write(id, deg);
      }
    };
    setGaze('ParamEyeBallX', frame.sway * 0.08 + frame.step * 0.03);
    setGaze('ParamEyeBallY', -nod * 0.06 - bob * 0.05);
    if (typeof opts.breathBase === 'number') {
      // 呼吸单独处理：它不在 frame 里，直接写死 0.45 会在收势那一刻看到胸口「顿」一下。
      // 走同一个 write，按 blend 从当前呼吸相位过渡过去。
      var bid = resolve('ParamBreath');
      if (bid) {
        // 呼吸按模型区间展开（0..1 之外的模型也一样）
        var bm = prof && prof.map.breath;
        var bv = clamp(opts.breathBase + frame.bounce * 0.08, 0.05, 0.95);
        write(bid, bm ? bm.lo + (bm.hi - bm.lo) * bv : bv);
      }
    }
  };

  MusicGroove.STYLES = STYLES.slice();
  MusicGroove.REV = 'r17';
  return MusicGroove;
});
