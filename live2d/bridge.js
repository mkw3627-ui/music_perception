/*!
 * music-groove-bridge.js — 把「为她所听」的频谱接到主程序 Live2D 角色上。
 *
 * 由 templates/index.html 加载，运行在主窗口里；自身只做三件事：
 *   1. 每 100ms 从插件本地端口读一次频谱状态，并用流式起音包络估计 BPM、锁相出拍点；
 *   2. 找到 window.live2dManager 当前模型；
 *   3. 同时管两件事：
 *      · 有频谱律动（为她所听）时，把 sway/bounce 叠加到 ParamAngle* / ParamBodyAngle*；
 *      · 唱歌（干净人声口型轨在跑）时，在 afterMotionUpdate 用绝对写入接管
 *        头/脖子/上身/呼吸，让她整个人跟着乐句摇、跟着字点头；
 *      另外在 beforeModelUpdate 叠加四条表情通道（眉/笑眼/脸颊/眉角）。
 *
 * 插件没开、没有模型、没有音乐时，本脚本什么都不改。
 */
(function () {
  'use strict';
  var BASE = 'http://127.0.0.1:48765';
  var STATIC = '/static/live2d/';
  var VERSION = 81;
  // 一执行就上报：用来区分「页面没加载脚本」和「加载了但模块被拦」。
  try {
    fetch(BASE + '/hello?v=' + VERSION + '&stage=boot', { cache: 'no-store' }).catch(function () {});
  } catch (e) { /* 忽略 */ }
  var engine = null;
  var latest = null;
  var lastEnergy = 0;
  var lastT = 0;
  var blend = 0;
  var hookedModel = null;
  var dbg = { applied: 0, model: '', active: false };

  // ---- 起音节拍时钟 ----
  // 监听器每 50ms 送一小包起音强度，这里维护一条最近 2 秒的滚动包络：
  // 用它找鼓点、估 BPM，再维护一个锁相的节拍时钟，让每一下都落在歌里的鼓点上。
  // 不再等 2 秒整窗：动作延迟从约 2 秒降到约 0.1 秒。
  var PERIOD_MIN = 0.30, PERIOD_MAX = 1.20;    // 50~200 BPM
  // 声音进入扬声器混音到被回环录到、再到父进程打上时间戳，固定差约 10~30ms
  // （WASAPI 共享模式缓冲）。父进程的时间戳是「拿到数据」的时刻，比声音本身晚，
  // 所以要从时间戳往前扣掉它；扣反了拍点就整体偏晚，动作永远慢半拍。
  var CAPTURE_LAG = 0.03;
  var clock = { bpm: 100, nextAt: 0, pulse: 0, enabled: false, onsets: 0, lag: 0, envAt: 0, bpmAt: 0,
                regularity: 0.5, bias: 0, beatIndex: 0, beatPos: 0, phaseConf: 0 };

  // ---- 突发重音 ----
  // 包络每 50ms 更新一次，所以每次轮询只看「上次之后新增的那几帧」：
  // 哪一帧比前面 1.5 秒里最响的还响很多，就当成「平静里突然的一声」，
  // 立刻鼓动一下——不用等节拍时钟，也不需要它是整拍。
  var hitPulse = 0;    // 重音包络（快速衰减）
  var hitAmp = 0;      // 这一下有多突兀（相对基线）
  var envSeenAt = 0;   // 上次轮询看到的 onset_at

  function detectHits(st) {
    var env = st && st.onset_env;
    if (!env || env.length < 6) return;
    var hop = Number(st.onset_hop) || 0.025;
    var at = Number(st.onset_at) || 0;
    if (!envSeenAt) { envSeenAt = at; return; }
    var d = at - envSeenAt;
    envSeenAt = at;
    if (d <= 0 || d > 1.5) return;             // 断了很久，不当成重音
    var n = env.length;
    var k = Math.min(n, Math.max(1, Math.round(d / hop)));
    var base = 0, i, j;
    for (i = 0; i < n - k; i++) base = Math.max(base, Number(env[i]) || 0);
    var peak = 0;
    for (j = n - k; j < n; j++) peak = Math.max(peak, Number(env[j]) || 0);
    // 比刚才最响的还大 35% 以上、绝对值也够大，才算特别响亮的一下；
    // 稳定的鼓点循环里 base 本身就高，不会反复触发。
    if (peak > 0.45 && peak > base * 1.35 + 0.08) {
      hitPulse = 1;
      hitAmp = clamp((peak - base) / 0.45, 0.35, 1);
    }
  }

  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

  /**
   * 抛物线插值求峰的亚 hop 时刻：25ms 一帧只能量到 ±12ms，
   * 插值后大约 ±3ms —— 卡点准不准主要就差这一下。
   * 时间取窗中心而不是窗尾：窗尾会把每个鼓点都系统性地算晚半格（12.5ms），
   * 一直叠加就是「动作总比声音慢一点」。
   */
  function peakTime(env, i, hop, endAt, n) {
    var y0 = Number(env[i - 1]) || 0;
    var y1 = Number(env[i]) || 0;
    var y2 = Number(env[i + 1]) || 0;
    var den = y0 - 2 * y1 + y2;
    var off = Math.abs(den) > 1e-9 ? 0.5 * (y0 - y2) / den : 0;
    return endAt - (n - 1 - (i + clamp(off, -0.5, 0.5)) + 0.5) * hop;
  }

  // ---- 长窗口速度估计 ----
  // 2 秒的窗只装得下慢歌的一两个鼓点，量出来的 BPM 抖得没法用。
  // 这里把每包起音拼成一条 8 秒的时间轴，速度只从这条轴上量。
  var TB_SEC = 8;
  var tb = { t: [], v: [], last: 0 };
  var tempo = { cand: 0, candN: 0, conf: 0 };

  /** 排序取分位，得到「这首歌现在响的起音大概多强」，弱歌/强歌都适用。 */
  function envThreshold(env, frac, floor) {
    var n = env.length;
    if (!n) return floor;
    var step = Math.max(1, Math.floor(n / 48));
    var s = [];
    for (var i = 0; i < n; i += step) s.push(Number(env[i]) || 0);
    s.sort(function (a, b) { return a - b; });
    var ref = s[Math.min(s.length - 1, Math.floor(s.length * 0.90))] || 0;
    return Math.max(floor, ref * frac);
  }

  /** 把一窗包络接进长缓冲；时钟跳了（暂停/切歌/刚打开）就重新攒。 */
  function tbPush(env, hop, endAt) {
    var n = env.length;
    var t0 = endAt - (n - 1) * hop;
    if (tb.last && (endAt > tb.last + 1.5 || endAt < tb.last - 0.6)) {
      tb.t.length = 0; tb.v.length = 0; tb.last = 0;
    }
    // 只追加真正新的那一截，避免同一段反复计入
    var i0 = tb.last ? Math.max(0, Math.round((tb.last - t0) / hop) + 1) : 0;
    for (var i = i0; i < n; i++) {
      tb.t.push(t0 + i * hop);
      tb.v.push(Number(env[i]) || 0);
    }
    if (n) tb.last = t0 + (n - 1) * hop;
    var cut = tb.last - TB_SEC;
    var drop = 0;
    while (drop < tb.t.length && tb.t[drop] < cut) drop++;
    if (drop > 0) { tb.t.splice(0, drop); tb.v.splice(0, drop); }
  }

  /** 长缓冲里的「强起音」：估速度只看打点，不看装饰性的小音。fromT 限定起点（换歌用）。 */
  function tbStrongPeaks(fromT) {
    var n = tb.v.length;
    if (n < 16) return null;
    var thr = envThreshold(tb.v, 0.40, 0.02);
    var pk = [];
    for (var i = 1; i < n - 1; i++) {
      if (fromT && tb.t[i] < fromT) continue;
      var v = tb.v[i];
      if (v >= thr && v >= tb.v[i - 1] && v > tb.v[i + 1]) pk.push({ t: tb.t[i], v: v });
    }
    if (pk.length < 3) return null;
    var maxV = 0;
    for (i = 0; i < pk.length; i++) if (pk[i].v > maxV) maxV = pk[i].v;
    // 只有强弱分明的歌才筛「重拍」，否则一起算（电子乐平铺的踩镲就是拍子）
    var cutv = Math.max(0.10, maxV * 0.55);
    var strong = [];
    for (i = 0; i < pk.length; i++) if (pk[i].v >= cutv) strong.push(pk[i]);
    return strong.length >= 3 ? strong : pk;
  }

  /** 从一组起音算 BPM 和规整度；样本不够返回 null。 */
  function estimateFrom(strong) {
    if (!strong || strong.length < 3) return null;
    // 只用最近的 9 个强起音（8 个间隔）：慢歌约占满 8 秒，快歌只要 3 秒多，
    // 换歌时新歌很快就能顶掉旧值，不会被整段缓冲拖成「两首的平均」。
    var sub = strong.length > 9 ? strong.slice(strong.length - 9) : strong;
    var gaps = [], k;
    for (k = 1; k < sub.length; k++) {
      var g = sub[k].t - sub[k - 1].t;
      while (g > PERIOD_MAX) g /= 2;    // 漏检了半拍
      while (g < PERIOD_MIN) g *= 2;    // 同一拍里的装饰音
      if (g >= PERIOD_MIN && g <= PERIOD_MAX) gaps.push(g);
    }
    if (gaps.length < 2) return null;
    gaps.sort(function (a, b) { return a - b; });
    var med = gaps[gaps.length >> 1];
    // 首尾跨度和拍数拉通：几个拍子摊下来误差不到 1%，比单看相邻间隔准
    var span = sub[sub.length - 1].t - sub[0].t;
    var count = Math.round(span / med);
    if (count >= 2) {
      var fit = span / count;
      if (fit >= PERIOD_MIN && fit <= PERIOD_MAX) med = fit;
    }
    var bpm = 60 / med;
    if (!(bpm >= 50 && bpm <= 200)) return null;
    // 规整度：间隔越一致，速度越可信（用来区分稳稳的鼓点和散拍）
    var mean = 0, j;
    for (j = 0; j < gaps.length; j++) mean += gaps[j];
    mean /= gaps.length;
    var dev = 0;
    for (j = 0; j < gaps.length; j++) { var dd = gaps[j] - mean; dev += dd * dd; }
    dev = Math.sqrt(dev / gaps.length);
    return { bpm: bpm, conf: clamp(1 - dev / (mean * 0.5), 0, 1) };
  }

  /** 用长缓冲估一次 BPM，带倍频保护和换歌确认。 */
  function updateTempo() {
    var est = estimateFrom(tbStrongPeaks());
    if (!est) return;
    // 换歌时 8 秒缓冲里前后两首各占一半，中位数会落在两者中间（量出个
    // 60→140 之间的假值）。不规整就只看最近 5 秒，新歌够稳就采纳新的。
    if (est.conf < 0.55) {
      var rec = estimateFrom(tbStrongPeaks(tb.last - 5));
      if (rec && rec.conf > est.conf + 0.1 && Math.abs(rec.bpm - est.bpm) / est.bpm > 0.06) est = rec;
    }
    var bpm = est.bpm;
    tempo.conf = est.conf;
    clock.regularity = clock.regularity * 0.6 + tempo.conf * 0.4;

    var jump = clock.enabled ? Math.abs(bpm - clock.bpm) / clock.bpm : 0;
    if (!clock.enabled) {
      clock.bpm = bpm;
      clock.enabled = true;
      tempo.cand = 0; tempo.candN = 0;
    } else if (jump <= 0.12) {
      // 小偏差：慢慢跟（换歌/变速时也不会让动作突兀地跳）
      clock.bpm = clock.bpm + (bpm - clock.bpm) * (jump > 0.04 ? 0.35 : 0.18);
    } else if (jump > 0.18) {
      // 可能是倍频/半频（慢歌被当成两倍速、快歌被当成一半速），也可能是换歌：
      // 连续两次量到同一个值才切，避免半拍/整拍误判来回翻。
      if (tempo.cand && Math.abs(bpm - tempo.cand) / tempo.cand < 0.10) tempo.candN++;
      else { tempo.cand = bpm; tempo.candN = 1; }
      if (tempo.candN >= 2) {
        clock.bpm = bpm;
        tempo.cand = 0; tempo.candN = 0;
      }
    } else {
      clock.bpm = clock.bpm + (bpm - clock.bpm) * 0.22;
    }
  }


  /**
   * 从起音包络里找峰值 → 估 BPM → 排定下一个拍点。
   * 包络是最近 2 秒的，最后一帧约等于 state.onset_at，据此换算成本机 rAF 时钟。
   */
  function analyzeEnvelope(st) {
    var env = st && st.onset_env;
    if (!env || !env.length || env.length < 12) return;
    var hop = Number(st.onset_hop) || 0.025;
    var n = env.length;

    var nowPerf = performance.now() / 1000;
    // 后端算完到现在过了多久（同一台机器，时钟一致）；再加上采集本身的固定延迟：
    // 时间戳晚于声音，所以这一帧声音的真实时刻在 now - age - CAPTURE_LAG。
    var lag = clamp((Date.now() - (Number(st.onset_at) || 0) * 1000) / 1000 + CAPTURE_LAG, 0, 2.5);
    clock.lag = lag;
    var endAt = nowPerf - lag;
    // 这一窗接进 8 秒长缓冲：速度要看长窗口才稳（见 updateTempo）。
    tbPush(env, hop, endAt);

    // 起音强度不是统一的：抒情歌整首都在 0.05~0.25，摇滚一下鼓就顶到 1。
    // 以前固定阈值 0.30 —— 慢歌/轻歌一个峰都找不到，节拍时钟永远锁不上，
    // 引擎只能按 100 BPM 兜底，于是「慢歌快歌全一个速度」。
    // 现在阈值跟着这一窗自己的响度走，弱歌也能找到鼓点。
    var thr = envThreshold(env, 0.40, 0.02);
    var peaks = [];
    for (var i = 1; i < n - 1; i++) {
      var v = Number(env[i]) || 0;
      if (v >= thr && v >= (Number(env[i - 1]) || 0) && v > (Number(env[i + 1]) || 0)) peaks.push(i);
    }
    clock.onsets = peaks.length;
    if (!peaks.length) {
      // 这一窗一个起音都没有（间奏、纯人声段）：相位可信度慢慢掉下来。
      // 引擎靠它把速度门控回慢速，而不是继续按上一段的 BPM 摇。
      clock.phaseConf *= 0.97;
      return;
    }

    var lastOnsetAt = peakTime(env, peaks[peaks.length - 1], hop, endAt, n);

    // BPM 每 0.8 秒量一次就够；相位每次轮询（100ms）都修，卡点才跟得住。
    var atNow = Number(st.onset_at) || 0;
    var doTempo = !clock.enabled || atNow - (clock.bpmAt || 0) >= 0.8;
    if (doTempo) {
      clock.bpmAt = atNow;
      updateTempo();
    }
    if (!clock.enabled) return;

    // 相位：下一个拍点 = 最近起音 + n 个周期，落在将来最近的一下。
    var period = 60 / clock.bpm;
    var desired = lastOnsetAt + period;
    while (desired < nowPerf + 0.04) desired += period;
    if (!clock.nextAt) { clock.nextAt = desired; clock.beatIndex = 0; clock.beatPos = 0; return; }

    // 相位对准：窗口里每个强起音按「强度²」投票到拍点网格上，取加权圆周均值。
    // 以前只看最后一个峰 —— 军鼓、踩镲、装饰音都会把参考点带偏，拍子就锁在
    // 那上面，表现出来就是「卡点卡不准、一直在乱摇」。
    var sx = 0, sy = 0, sw = 0;
    // 投票只收「够强的起音」。门槛同样跟着这首歌的响度走，否则轻歌的鼓点
    // 全被 0.35 挡掉，相位一个票都投不出来，卡点就飘。
    var pvThr = envThreshold(env, 0.50, 0.025);
    for (var pi = 0; pi < peaks.length; pi++) {
      var pIdx = peaks[pi];
      var pv = Number(env[pIdx]) || 0;
      if (pv < pvThr) continue;
      var pt = peakTime(env, pIdx, hop, endAt, n);
      var r = (pt - clock.nextAt) / period;
      var ang = Math.PI * 2 * (r - Math.round(r));
      var wgtP = pv * pv;
      sx += wgtP * Math.cos(ang);
      sy += wgtP * Math.sin(ang);
      sw += wgtP;
    }
    if (sw > 0) {
      var mag = Math.sqrt(sx * sx + sy * sy) / sw;      // 0..1：相位集中度（越接近 1 越齐）
      clock.phaseConf = clock.phaseConf * 0.55 + mag * 0.45;
      var delta = Math.atan2(sy, sx) / (Math.PI * 2);   // -0.5..0.5 拍
      if (Math.abs(delta) > 0.42) {
        // 完全错相（换歌 / 换拍 / 很久没锁上）：直接对齐
        clock.nextAt = desired;
        clock.bias = 0;
      } else {
        // 平时不直接挪 nextAt，而是微调接下来的拍长（速度纠偏）：
        // 相位保持连续，修拍的时候动作不会「抽」一下。
        // 偏差会摊到接下来的几拍上（每拍衰减 0.55，总效果约 2.2 拍），
        // 所以系数取 1/2.2 左右还要再小一点：0.45 时环路会自己来回荡
        // （测到的拍点误差是「慢慢偏、突然跳回来」的锯齿），卡点反而飘。
        clock.bias = clamp(delta * 0.25, -0.15, 0.15);
      }
    } else {
      clock.phaseConf *= 0.98;   // 没有够强的起音参与投票：可信度缓慢下降
    }
  }

  /** 每帧推进节拍时钟，返回本帧的起音强度（0..1）；未锁定返回 -1 表示用兜底逻辑。 */
  function beatPulse(dt) {
    if (!clock.enabled) return -1;
    var nowS = performance.now() / 1000;
    var period = 60 / clock.bpm;
    // 时钟跑飞（长时间没收到包络）就重新对齐，别让拍子越拖越远
    if (!clock.nextAt || clock.nextAt < nowS - period * 1.6) clock.nextAt = nowS + period * 0.4;
    if (nowS >= clock.nextAt) {
      clock.nextAt += period * (1 + clamp(clock.bias || 0, -0.2, 0.2));
      clock.bias = (clock.bias || 0) * 0.55;
      if (clock.nextAt < nowS) clock.nextAt = nowS + period;
      clock.pulse = 1;
      clock.beatIndex += 1;
    }
    // 连续拍位置（拍数，整数部分是第几拍，小数部分是这一拍走了多少）：
    // 引擎用它当摆动相位，摆动的极值和点头就能跟鼓点严格对齐。
    var ph = 1 - (clock.nextAt - nowS) / period;
    if (!(ph >= 0 && ph < 1)) ph = 0;
    clock.beatPos = clock.beatIndex + ph;
    var out = clock.pulse;
    clock.pulse *= Math.exp(-dt / 0.13);
    if (clock.pulse < 0.01) clock.pulse = 0;
    dbg.bpm = Math.round(clock.bpm * 10) / 10;
    dbg.onsets = clock.onsets;
    dbg.lag = Math.round(clock.lag * 1000) / 1000;
    dbg.regularity = Math.round(clock.regularity * 100) / 100;
    dbg.tempoConf = Math.round((tempo.conf || 0) * 100) / 100;
    dbg.phaseConf = Math.round((clock.phaseConf || 0) * 100) / 100;
    dbg.tbLen = tb.t.length;
    return out;
  }

  function load(src) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = src;
      s.onload = resolve;
      s.onerror = reject;
      document.head.appendChild(s);
    });
  }

  function currentModel() {
    var m = window.live2dManager;
    return m && (m.currentModel || null);
  }

  // ---- 参数适配层（param_adapter.js）----
  // 所有表情 / 身体 / 嘴型写入都按「语义通道」走适配层：标准 ID 优先，没有就按
  // model3.json 的分组和 cdi3 显示名去认；认不出就跳过，不按编号硬写。数值用
  // 「参考模型单位」（头 ±30°、身体 ±10°、其他参数原值），由适配层按当前模型的
  // 真实区间换算——标准量程模型行为不变，量程不同的模型按比例缩放。
  var PA = null;
  function pa() { return PA || (PA = window.NekoParamAdapter || null); }
  var paInfo = { core: null, named: false, at: 0 };
  function paProfile(core) {
    var A = pa();
    if (!A || !core) return null;
    // 显示名（cdi3）是主程序异步加载的，可能晚于我们第一次挂上模型；没拿到名字时
    // 每秒重试一次，拿到后重建映射（特效类通道全靠名字认）。
    var mgr0 = window.live2dManager;
    var namesReady = !!(mgr0 && mgr0._displayInfo && mgr0._displayInfo.Parameters);
    var now = performance.now();
    if (paInfo.core === core && !paInfo.named && namesReady && now - paInfo.at > 1000) paInfo.core = null;
    if (paInfo.core !== core) {
      paInfo.core = core;
      paInfo.named = namesReady;
      paInfo.at = now;
      var mgr = window.live2dManager, info = { name: '' };
      try {
        info.displayInfo = mgr && mgr._displayInfo || null;
        var fr = mgr && mgr.fileReferences;
        var st = null;
        try { st = hookedModel && hookedModel.internalModel && hookedModel.internalModel.settings; } catch (e0) { st = null; }
        var json = st && st.json;
        info.settings = json ? json : (fr ? { FileReferences: fr } : null);
        if (info.settings && fr && fr.Expressions && !(info.settings.FileReferences || {}).Expressions) {
          info.settings.FileReferences = Object.assign({}, info.settings.FileReferences || {}, { Expressions: fr.Expressions });
        }
        info.name = (mgr && (mgr.modelName || mgr.modelRootPath)) || '';
      } catch (e) { /* 拿不到就只按标准 ID 认 */ }
      var prof = A.attachInfo(core, info);
      try { dbg.adapter = A.describe(core); } catch (e2) { dbg.adapter = null; }
      return prof;
    }
    return A.profile(core);
  }
  // 语义通道 → 是否存在（取代按参数名逐个探测）
  function chHas(core, ch) { var p = paProfile(core); return !!(p && p.map[ch]); }
  function chAdd(core, ch, v) { if (!v || !paProfile(core)) return 0; return pa().addRef(core, ch, v); }
  function chSet(core, ch, v) { if (!paProfile(core)) return 0; return pa().setRef(core, ch, v); }
  function chMap(core, ch) { var p = paProfile(core); return p && p.map[ch] || null; }

  function isActive() {
    // 能量门槛压得很低：安静的段落里也要保持「在听」，否则那一声突然的
    // 重音会被判成「没在放音乐」而完全没反应。只要有重音就保持活跃。
    return !!(latest && latest.ok && latest.enabled !== false && !latest.paused
      && !latest.silent && ((Number(latest.energy) || 0) > 0.02 || hitPulse > 0.05));
  }

  // ---- 为她而唱：逐字口型 ----
  // 成品由插件用 ffplay/winsound 播放，不在主程序的音频管线里，所以
  // app-audio-playback.js 的口型收不到声音。插件另开一个本地口型服务
  // （127.0.0.1:48766 起），这里按「当前唱到第几个字」写嘴部参数。
  var LIPS_PORTS = [48766, 48767, 48768, 48769, 48770];
  var LIPS_BASE = '';
  var lips = { seq: -1, cues: null, env: null, hop: 0.05, mode: '', baseElapsed: 0,
               at: 0, active: false, song: '', duration: 0, rate: null,
               lastOpen: 0, lastForm: 0, tickAt: 0,
               // g55：口型时钟上一帧的值（连续性守卫）+ 干声响度跟随器状态
               tClock: null, loudF: null,
               misses: 0, port: 0, nextFind: 0, owner: null,
               audioEl: null, audioSeq: 0, stem: '', moods: null, t: 0 };
  // 韵母 → 嘴形（-1 = 噘嘴/圆，+1 = 咧宽）。原来全挤在 0.5~1.0，
  // あ/い/う 看起来一模一样，只剩开合；拉开才有「说的是哪个元音」的区别。
  // S = 中性/收口：声母（g/k/h/zh/ch/sh/z/c/s）和韵尾（ng）用，嘴几乎不动。
  // Z = 中性元音（g56）：普通话的「的、了、着、和、这」舌尖元音 / [ɤ]，以及
  //     英语 the / a / of 的 schwa。以前和 S 共用，张口度只有 0.34，
  //     「的/是/实/时」唱起来嘴几乎不张——这就是「嘴巴不张」的来源。
  //     它们不是 [e]，所以 form 仍然是 0（不咧嘴），只是要张到一半。
  var LIPS_FORM = { A: -0.15, I: 0.95, U: -1.0, E: 0.4, O: -0.8, S: 0.0,
                    Z: 0.0, M: 0.0, N: 0.0, F: -0.2, L: 0.0, rest: 0 };
  // 韵母 → 张口度（1 = 最大）。Live2D 上「张多大」比「嘴角形状」显眼得多，
  // 所以 あ 大、い/う 小、え/お 中，光看开合就能分辨唱的是哪个元音。
  // 唱和说不一样：い/う 也只是开口小一号，不是把嘴合上；只有 ん/唇音才接近闭合。
  var LIPS_OPEN = { A: 0.88, O: 0.76, E: 0.66, Z: 0.56, I: 0.52, U: 0.42,
                    S: 0.34, L: 0.52, F: 0.22, M: 0.05, N: 0.06, rest: 0.08 };
  // 元音的「亮/暗」：笑眼和眉毛跟着元音走，あ/え 会弯一点、お/う 收着。
  var LIPS_BRIGHT = { A: 1.0, E: 0.8, I: 0.6, O: 0.15, U: 0.0, S: 0.32,
                      Z: 0.34, L: 0.5, F: 0.2, M: 0.1, N: 0.1, rest: 0.2 };

  // ---- 唱歌时的表情 ----
  // 只动嘴、脸上一点反应都没有最像木偶。这里借四条通道：
  //   眉毛（重音抬、句首换气后挑一下）、笑眼（亮元音和响的乐句把眼睛弯起来）、
  //   脸颊（唱得用力时鼓一点）、眉角（抬眉时眉形跟着挑）。
  // 都在 beforeModelUpdate —— 主程序的表情、眨眼、视线都算完之后 —— 用
  // addParameterValueById 叠加上去，不覆盖她本来的表情；间奏/停唱自然落回 0。
  // 幅度按「参数半量程的百分比」给，换模型（±1 或 0..1 的 rig）都一样看得见。
  var face = { brow: 0, smile: 0, cheek: 0, angle: 0 };
  // 乐句响度（0.35s 平滑）/ 句首脉冲 / 上一帧的演唱 presence
  var faceDyn = { loud: 0, onset: 0, pres: 0 };
  var faceStats = { writes: 0, support: '', last: '', eventAt: 0, support2: '', support3: '' };

  // ---- 情绪层：VAD → FACS → Live2D 参数 ----
  // 逻辑移植自 Soullink Emotion SDK（Apache-2.0，
  // https://github.com/nanlingyin/soullink-emotion-sdk）。它的 VADExpressionMapper
  // 把「正负 V / 唤醒 A / 掌控 D」解成 FACS 动作单元，ActionUnitSolver 再把
  // AU 换算成 browInnerUp / browOuterUp / browDown / eyeOpen / eyeSmile /
  // mouthSmile / headY / headZ 这些脸谱量，STANDARD_PARAM_TABLE 给出各个量到
  // Cubism 参数的缩放（头 30°，压眉 scale -0.85，抬眉 0.9……）。
  // 原 SDK 吃的是文字情绪标签；这里没有字，只有歌，所以输入换成本地能测到的
  // 信号：响度起伏 → 唤醒、音色亮度/段落 → 正负、低频/力度 → 掌控。
  // 和上面那四条「唱的表情」通道分工不同：那四条只回答「这一句多投入」，
  // 这一层回答「这首歌是什么情绪」，管眉头、眼形、视线和头的朝向。
  var EMO = {
    gain: 2.6,       // 总强度：SDK 的原值是给有文字情绪的场景调的，唱歌上翻倍才看得见
    attack: 0.22,    // 情绪起（秒）：g62 从 0.30 压到 0.22，副歌一进来就要看见
    release: 0.95,   // 情绪落（秒）：句尾不能一下变回木头
    maxHead: 8.5,    // 头最多叠几度（g62：5.5 → 8.5，实测这个模型头 ±30°，8.5 还不到三成）
    maxGaze: 0.16,
    mouthBias: 0.42  // 嘴形偏移上限（g62：0.22 → 0.42；ParamMouthForm 实际区间 ±0.8）
  };
  // 笑眼总和的上限（按模型参数区间的比例）：乐句通道 + 情绪层两条叠加之后的
  // 天花板。g48 是 0.80——实测这个模型（Blue cat）的 ParamEyeLSmile 会把上眼睑
  // 压下来盖住虹膜，0.8 等于「眼睛一直弯成半闭」，虹膜被压小；而乐句强弱一变，
  // 总和就在 0.7~0.8 之间来回，虹膜跟着一下大一下小（用户报的「唱歌时眼珠子
  // 一下大一下小」）。g52 压到 0.40：保留「眼睛略微弯一点」，不再盖住虹膜。
  var EYE_SMILE_CAP = 0.12;   // g54：0.40 会把下眼睑拉开一圈，虹膜显小
  var emo = {
    v: 0, a: 0, d: 0, tA: 0, tV: 0, tD: 0, pHi: 0, pLo: 0, hold: 0, blink: 0, pres: 0,
    base: { e: 0.35, b: 0.5, f0: 0, n: 0 },
    f: { browIn: 0, browOut: 0, browDn: 0, eyeSmile: 0, eyeOpen: 0, cheek: 0,
         mouth: 0, gazeX: 0, gazeY: 0, headX: 0, headY: 0, headZ: 0,
         smile2: 0, mouthX: 0, angerMk: 0,
         // g64：贴图特效（整身视角下唯一真看得见的「大幅度表情」）
         blush: 0, tears: 0, flower: 0, darkFace: 0 },
    lyric: null, tag: 'neutral', arch: 'neutral', writes: 0
  };
  // 唱歌态的中性基线：唱歌时表情层一律用这两个全 0 对象，不写 VAD 情绪。
  // 脸部只保留口型（applyLips）与动作层（gest）各自的通道；停唱时再由
  // faceResetToDefault() 把本层写过的参数显式写回默认值。
  var ZERO_FACE = {
    browIn: 0, browOut: 0, browDn: 0, eyeSmile: 0, eyeOpen: 0, cheek: 0,
    mouth: 0, gazeX: 0, gazeY: 0, headX: 0, headY: 0, headZ: 0,
    smile2: 0, mouthX: 0, angerMk: 0, teeth: 0,
    blush: 0, tears: 0, flower: 0, darkFace: 0
  };
  var ZERO_EXTRA = {
    smile2: 0, mouthX: 0, teeth: 0, mk: 0, asym: 0, browSyl: 0,
    blush: 0, tears: 0, flower: 0, darkFace: 0, arc: 0
  };

  // ---- g62：第二套表情通道 ----
  // 主程序/口型轨都不写这几个参数，所以它们能独立表达，不会和口型抢：
  //   ParamMouthForm2  「笑」（-30..30）：真正的嘴角上扬/下垂，比 ParamMouthForm
  //                    直观得多；口型轨每帧绝对写 ParamMouthForm，情绪那一点偏移
  //                    经常被钳到边，这条才是「看得出在笑」的那一条。
  //   ParamMouthForm3  「嘴X」（-30..30）：嘴角左右平移，跟着头转一点点，
  //                    让「转头说话」这件事在嘴上也有落点。
  //   ParamMouthOpenY23「牙」（0..1）：大音量上扬时露一点牙，是最省的高光细节。
  //   Param202 / 203 / 204 / 205 / 206  **一个都不能写**（g63 纠正）：
  //                    cdi3 里这五个的名字全是「变小」，实测是**整只角色切换到
  //                    幼体形态**（头大身短），0.4 就已经明显走形。g62 把它当成
  //                    「猫耳伸出」在副歌写到了 0.72，是读错了扫描结果里
  //                    「drawable 显形」的含义。这条通道已整条删除。
  //                    顺带记一笔：真正的猫耳是 Param88（猫耳，0..1），1 = 耳朵
  //                    **收起**（默认 0 就是露着的），没有「更立」的方向可用。
  //   Param54          「生气」特效（0..30，实测：10 起淡红符号，30 最重）：
  //                    只在 anger 原型且情绪够重时才长出来。
  // 全部先各自平滑再写，和 VAD 的快慢不一致时不会互相拖。
  var extra = {
    smile2: 0, mouthX: 0, teeth: 0, mk: 0,
    asym: 0,              // 左右眉差（一侧略高）
    browSyl: 0,           // 咬字那一下的挑眉
    vibPh: 0,             // 颤音的相位（长音上的微抖）
    arc: 0,               // 乐句弧：一句里面慢慢抬起来、句尾落下去
    blush: 0, tears: 0, flower: 0, darkFace: 0,   // g64 特效表情（平滑后的实际写入量）
    t: 0, lastOpen: 0
  };

  // 歌词情绪：/lips.json 的 moods 行是 [start, end, v, a, d, conf]，
  // 由插件解析「这一句在唱什么词」算出来（中英双语关键词 → VAD）。
  // 它跟音频信号是两条独立证据：音频给的是「唱得多用力」，词给的是
  // 「唱的什么意思」。按 conf 加权掺进目标值，唱完一句后 0.9 秒内淡出。
  var lyric = { rows: null, i: 0, at: -1, cur: null, end: 0 };
  function lyricMood(t) {
    var rows = lips.moods;
    if (rows !== lyric.rows) { lyric.rows = rows; lyric.i = 0; lyric.cur = null; lyric.at = -1; }
    if (!rows || !rows.length) return null;
    // 没在放歌（lips.active 假）时歌词情绪要松开，否则最后的情绪钉死在脸上。
    if (!lips.active) { lyric.cur = null; return null; }
    if (t !== lyric.at) {
      lyric.at = t;
      if (lyric.cur && t < Number(lyric.cur[0])) { lyric.i = 0; lyric.cur = null; }
      while (lyric.i < rows.length && Number(rows[lyric.i][1]) < t) lyric.i++;
      var row = rows[lyric.i];
      if (row && Number(row[0]) <= t && t <= Number(row[1])) {
        lyric.cur = row;
        lyric.end = Number(row[1]);
      }
    }
    if (!lyric.cur) return null;
    var fade = t > lyric.end ? Math.exp(-(t - lyric.end) / 0.9) : 1;
    var conf = (Number(lyric.cur[5]) || 0) * fade;
    if (conf < 0.12) return null;
    return { v: Number(lyric.cur[2]) || 0, a: Number(lyric.cur[3]) || 0,
             d: Number(lyric.cur[4]) || 0, conf: conf };
  }

  function updateEmotion(dt) {
    var stemOn = !!(vox.ok && lips.stem);
    var energy = clamp(stemOn ? 0.55 * mus.energy + 0.45 * vox.energy : mus.energy, 0, 1);
    var bright = clamp(stemOn ? 0.40 * (vox.bright || 0) + 0.60 * mus.bright : mus.bright, 0, 1);
    var low = clamp(mus.low || 0, 0, 1);
    var beat = clamp(mus.beat || 0, 0, 1);
    var sect = clamp(mus.sect || 0, 0, 1);
    var vib = clamp(vox.vib || 0, 0, 1);
    var f0 = Number(vox.f0) || 0;
    // presence：她这一帧到底在不在唱（干声响度 + 口型开合，两个都看）。
    var pres = clamp((stemOn ? vox.energy : 0) * 2.2, 0, 1);
    pres = Math.max(pres, clamp((lips.lastOpen || 0) * 1.6, 0, 1));

    // 自适应基线：每首歌的响度、亮度、音区本来就不一样，用慢速均值当
    // 「这首的常态」；情绪看的是偏离常态多少，不是绝对值高低。
    var kb = 1 - Math.exp(-dt / 6.0);
    emo.base.e += (energy - emo.base.e) * kb;
    emo.base.b += (bright - emo.base.b) * kb;
    if (f0 > 0) emo.base.f0 += (f0 - emo.base.f0) * (1 - Math.exp(-dt / 9.0));
    emo.base.n += 1;
    var seed = emo.base.n < 40 ? 0.55 : 1;  // 开头约 2 秒基线还没收敛，先收着
    var ze = clamp((energy - emo.base.e) * 2.2 * seed, -1, 1);
    var zb = clamp((bright - emo.base.b) * 2.0 * seed, -1, 1);
    var zp = emo.base.f0 > 0 ? clamp((f0 - emo.base.f0) / 220, -1, 1) : 0;

    // 三维情绪的原始估计（权重是移植时按唱歌信号重新配的）
    var tA = clamp(0.55 * ze + 0.22 * beat + 0.16 * sect + 0.18 * zp, -1, 1);
    var tV = clamp(0.52 * zb + 0.24 * sect + 0.16 * vib + 0.12 * ze + 0.04, -1, 1);
    var tD = clamp(0.60 * ze + 0.20 * sect + 0.16 * low, -1, 1);
    if (pres < 0.03) { tA *= 0.2; tV *= 0.2; tD *= 0.2; }  // 间奏：情绪收回大半

    // 这一句的词说了什么（关键词置信度加权，最高占 55%）
    // g69：有 show.json 时换成逐句情绪（LLM/导演给的 V/A + 情绪类别），最高占 78%
    deskTick(dt);
    var lm = desk.on ? deskMood(lips.t) : lyricMood(lips.t);
    if (lm && pres > 0.02) {
      var w = (lm.show ? 0.78 : 0.55) * lm.conf;
      tV = tV * (1 - w) + lm.v * w;
      tA = tA * (1 - w) + lm.a * w;
      tD = tD * (1 - w) + lm.d * w;
      emo.lyric = lm;
    } else {
      emo.lyric = null;
    }
    // 动作层看的是「目标」的移动量：平滑后的 emo.v/a/d 每帧几乎不动，
    // 用它算触发会永远等不到。
    emo.tA = tA; emo.tV = tV; emo.tD = tD;

    var ka = 1 - Math.exp(-dt / (tA > emo.a ? EMO.attack : EMO.release));
    var kv = 1 - Math.exp(-dt / (tV > emo.v ? EMO.attack : EMO.release));
    var kd = 1 - Math.exp(-dt / (tD > emo.d ? EMO.attack : EMO.release));
    emo.a += (tA - emo.a) * ka;
    emo.v += (tV - emo.v) * kv;
    emo.d += (tD - emo.d) * kd;
    emo.pHi = Math.max(0, zp);
    emo.pLo = Math.max(0, -zp);
    emo.hold += ((vib > 0.25 && pres > 0.3 ? 1 : 0) - emo.hold) * (1 - Math.exp(-dt / 0.4));
    // 换气眨眼：从「在唱」掉到「没在唱」的一瞬间闭一下眼，比机械定时眨眼自然。
    if (pres < 0.04 && emo.pres > 0.18) emo.blink = 1;
    emo.pres = pres;
    emo.blink *= Math.exp(-dt / 0.16);
    if (emo.blink < 0.01) emo.blink = 0;

    // g62：先把原型的驻留计时推一帧（以前这一步的 dt 一直是 0，原型永远停在
    // neutral，EMO_ARCH 等于没接上）。必须在 solveEmoFACS 之前，下面的
    // applyArchetype 才是拿「这一帧已经驻留够」的结果。
    latchMood(dt);
    solveEmoFACS();
    // 眼睑/腮帮子（g59）：算出这一帧的开合倍率和鼓脸量，写入在 eyeApply 里
    eyeTick(dt);
    // 动作层：原型换挡 / 目标累计移动 / 到点，三种触发都在这里，写在表情之前
    gestTick(dt);
    // g62：细节层（乐句弧 / 咬字 / 颤音 / 猫耳弹动），写在最后，用到上面全部结果
    extraTick(dt);
  }

  // VAD → FACS：公式沿用 SDK 的 VADExpressionMapper，只把幅度放大，
  // 并补了唱歌特有的几个来源（长音颤音、高音区、段落）。
  function solveEmoFACS() {
    var V = emo.v, A = emo.a, D = emo.d;
    var pos = Math.max(0, V), neg = Math.max(0, -V);
    var ar = Math.max(0, A), calm = Math.max(0, -A);
    var sub = Math.max(0, -D), dom = Math.max(0, D);
    var g = EMO.gain, vib = clamp(vox.vib || 0, 0, 1);
    var sect = clamp(mus.sect || 0, 0, 1);
    var f = emo.f;
    // g62：这一帧「在不在演」——不在唱（间奏、停歌）时耳朵/生气符号这类
    // 会改变轮廓的通道必须收回去，否则会把她的待机样子永久改掉。
    var act = clamp(Math.max(emo.pres, lips.active ? 0.7 : 0) * 1.4, 0, 1);
    f.browIn = clamp((neg * 0.08 + sub * 0.055 + calm * 0.02) * g * 2.4, 0, 0.88);
    f.browOut = clamp((ar * 0.08 + pos * ar * 0.035 + sect * 0.05) * g * 2.4, 0, 0.95);
    f.browDn = clamp((dom * neg * 0.11 + emo.pLo * 0.03) * g * 2.4, 0, 0.75);
    f.eyeSmile = clamp((pos * 0.08 + calm * pos * 0.035) * g * 2.0, 0, 0.45);
    // 脸颊泛红：g62 起不只是「又高兴又害羞」才红——正情绪本身就红，
    // 副歌/长音再补一点。系数压到 0.62 是为了留住层次：如果一直顶在
    // 上限，脸红就只剩「红 / 不红」两档了。
    f.cheek = clamp((pos * 0.40 + pos * sub * 0.30 + f.eyeSmile * 0.40
                     + sect * pos * 0.15) * g * 0.62, 0, 0.85);
    f.eyeOpen = clamp((ar * 0.04 - calm * 0.035 - neg * 0.02
                       - emo.hold * 0.03 - emo.blink * 0.75) * g, -0.85, 0.10);
    f.mouth = clamp((pos * 0.13 + calm * pos * 0.05 + sect * 0.04 - neg * 0.10) * g * 1.3,
                    -EMO.mouthBias, EMO.mouthBias);
    f.gazeX = clamp((-sub * 0.05 + emo.pLo * 0.02) * g, -EMO.maxGaze, EMO.maxGaze);
    f.gazeY = clamp((-sub * 0.05 + dom * 0.025 + emo.pHi * 0.06) * g,
                    -EMO.maxGaze, EMO.maxGaze);
    f.headX = clamp((-neg * 0.03 + emo.pLo * 0.03 - emo.pHi * 0.02) * g * 1.5, -1, 1);
    f.headY = clamp((-sub * 0.035 + dom * 0.025 + emo.pHi * 0.05) * g * 1.5, -1, 1);
    f.headZ = clamp((pos * sub * -0.025 + neg * dom * 0.018 + vib * 0.02) * g * 1.5, -1, 1);
    // ---- g62 新增的几条脸谱量 ----
    // 嘴角（ParamMouthForm2，±30）：正情绪上扬、负情绪下垂，副歌再加一点。
    f.smile2 = clamp((pos * 0.74 - neg * 0.66) * 26 + sect * 4 * act, -24, 28);
    // 嘴的左右位移（ParamMouthForm3，±30）：跟着头的转向挪一点，转头时嘴也跟过去。
    f.mouthX = clamp(f.headX * 5 + emo.pLo * 2.5 - emo.pHi * 1.5, -7, 7);
    // 露牙（ParamMouthOpenY23，0..1）：又开心又唱到高音才露（写入时再看嘴张多开）。
    f.teeth = clamp(pos * ar * 0.34 + emo.hold * 0.10, 0, 0.34);
    // g63：原先这里写 Param202（当成「猫耳」）——cdi3 里它是「变小」，
    // 也就是整只角色切幼体形态。已删除，绝不写。
    // 生气符号（Param54，0..30）：只有 anger 原型才长，而且要有「气」的量。
    f.angerMk = (emo.arch === 'anger')
      ? clamp((Math.max(0, -V) * 0.75 + Math.max(0, A) * 0.35) * 16 * act, 0, 18)
      : 0;
    // ---- g64：贴图特效通道 ----
    // 为什么非要加这一层：整身视角下，眉/嘴角/脸颊这些**形变**参数只动几个像素
    // （_dx 实测：ParamMouthForm 满量程的顶点位移只有 Param202 的 1/30），用户
    // 盯着看也「没有变化」。这个模型自己带了一组画师做的**颜色特效**参数，幅度大、
    // 认得出来，才是「大幅度表情」真正的落点：
    //   Param57 脸红（0..30）、Param64 流泪（0..1）、
    //   Param58 花花（0..1，头发上那朵花变红）、Param56 脸黑（0..30）。
    // 眼里的心形/星星（Param18/19/130/139）**不碰**——那是眼珠通道。
    // 四个都只在「在演」时才长（乘 act），间奏/停歌自动收回。
    f.blush = clamp((pos * 0.80 + sect * pos * 0.22 + f.cheek * 0.30) * act, 0, 1);
    // 流泪要「又难过又没在使劲」：V 很负、但高声量（愤怒式）不给泪。
    // anger 原型直接封零——生气还掉眼泪会读成「委屈」，剧情就错味了。
    f.tears = (emo.arch === 'anger') ? 0
      : clamp((neg * 1.25 - ar * 0.35 - emo.hold * 0.15) * act, 0, 1);
    // 花花只在「高兴 + 激动」同时成立时才开：pos*ar 过 0.30 才开始长。
    f.flower = clamp((pos * ar * 1.45 - 0.30) * 2.0 * act, 0, 1);
    // 脸黑：anger 专用，和生气符号配套。
    f.darkFace = (emo.arch === 'anger')
      ? clamp((Math.max(0, -V) * 1.10 + Math.max(0, A) * 0.25) * act, 0, 1)
      : 0;
    applyArchetype(f);
    emo.tag = (V > 0.15 ? 'pos' : (V < -0.15 ? 'neg' : 'mid'))
            + (A > 0.2 ? '+hi' : (A < -0.2 ? '+lo' : ''))
            + (emo.lyric ? 'L' : '')      // L = 这一句的歌词情绪正在掺入
            + ':' + emo.arch;
  }

  // 情绪原型（EmotionArchetypeRegistry 的中值）：VAD 只能给出「多正 / 多醒」，
  // 具体长什么样靠原型补齐——同样是「高兴」，happy 和 excited 的眉眼完全不同。
  // 用一个较弱的权重往原型中心靠（SDK 里是 0.1~0.46，这里压到 0.35 封顶，
  // 免得盖掉上面按句段算出来的细节）。
  var EMO_ARCH = {
    calm:        { mouth: 0.20, eyeSmile: 0.18, eyeOpen: -0.10, headZ: 0.06, smile2: 5 },
    happy:       { mouth: 0.62, eyeSmile: 0.35, eyeOpen: -0.02, browIn: 0.10, browOut: 0.16,
                   headZ: 0.12, smile2: 14 },
    excited:     { mouth: 0.80, eyeSmile: 0.39, eyeOpen: 0.14, browOut: 0.40, browIn: 0.06,
                   headY: 0.14, headZ: 0.10, smile2: 18 },
    affectionate:{ mouth: 0.46, eyeSmile: 0.34, eyeOpen: -0.08, browIn: 0.16, cheek: 0.35,
                   smile2: 9 },
    sad:         { browIn: 0.52, eyeOpen: -0.22, mouth: -0.44, gazeY: -0.15,
                   headY: -0.12, headZ: -0.08, smile2: -12 },
    concerned:   { browIn: 0.44, eyeOpen: -0.12, mouth: -0.22, headY: -0.08, smile2: -6 },
    anger:       { browDn: 0.52, eyeOpen: -0.13, mouth: -0.40, headY: -0.10, smile2: -12 }
  };
  // 表情「档位」：VAD 连续量决定强度，原型决定长什么样。原型每帧重算会在
  // happy / affectionate / excited 之间来回跳，脸上像在抽；这里给原型一个
  // 1.2 秒的驻留期，新原型得持续成立才切——用户看到的「一会儿高兴一会儿
  // 呆呆的」就是没有驻留造成的。
  //
  // g62 修了两个真问题：
  //   1) 这个函数的 dt 一直是 0（全文件只有 applyArchetype 里的 latchMood(0)
  //      调用），moodSt.t 永远不往前走，`t - since >= 1.2` 永远不成立，
  //      **原型层从此一直是 neutral，EMO_ARCH 一个字都没生效过**。现在
  //      updateEmotion 每帧先调一次 latchMood(dt)，驻留计时才真的在走。
  //   2) 换挡是瞬间的：原型给的量一次跳到位，脸上会「啪」地换一张。加一个
  //      0.35s 的 k 淡入（换挡时清零），换表情就有过渡了。
  var moodSt = { cur: 'neutral', want: 'neutral', since: 0, t: 0, id: 0, k: 1 };
  function latchMood(dt) {
    var want = emoArchetype();
    moodSt.t += dt;
    if (want !== moodSt.want) { moodSt.want = want; moodSt.since = moodSt.t; }
    if (moodSt.want !== moodSt.cur && moodSt.t - moodSt.since >= 1.2) {
      moodSt.cur = moodSt.want;
      moodSt.at = moodSt.t;
      moodSt.id += 1;      // 动作层拿它当「换挡」信号
      moodSt.k = 0;        // 换挡从 0 淡入，别一帧换脸
    }
    if (dt > 0) moodSt.k += (1 - moodSt.k) * (1 - Math.exp(-dt / 0.35));
    if (moodSt.k > 0.999) moodSt.k = 1;
    return moodSt.cur;
  }
  function emoArchetype() {
    var V = emo.v, A = emo.a;
    if (V > 0.22 && A > 0.35) return 'excited';
    if (V > 0.18) return 'happy';
    if (V < -0.20 && A > 0.25) return 'anger';
    if (V < -0.20) return 'sad';
    if (V > -0.05 && A < -0.15) return 'calm';
    if (V > 0.08) return 'affectionate';
    return 'neutral';
  }
  function applyArchetype(f) {
    var name = latchMood(0);
    emo.arch = name;
    var arch = EMO_ARCH[name];
    if (!arch) return;
    var mag = clamp((Math.abs(emo.v) + Math.abs(emo.a) * 0.82 + Math.abs(emo.d) * 0.64) / 2.46, 0, 1);
    // g62：原型权重从 0.35 放到 0.46——情绪一换档，脸要真的换一张。
    // 再乘 moodSt.k（换挡后 0.35s 内从 0 淡到 1），不会一帧跳过去；
    // 再乘 actA（没在唱就收回去），停歌/间奏时不会把她的待机脸改掉。
    var actA = clamp(Math.max(emo.pres, lips.active ? 0.7 : 0) * 1.4, 0, 1);
    var amt = clamp((0.12 + mag * 0.80) * 0.52, 0, 0.46) * moodSt.k * actA;
    function add(key, delta) { f[key] = clamp(f[key] + amt * delta, -1, 1); }
    if (arch.mouth != null) add('mouth', arch.mouth);
    if (arch.eyeSmile != null) add('eyeSmile', arch.eyeSmile);
    if (arch.eyeOpen != null) add('eyeOpen', arch.eyeOpen);
    if (arch.browIn != null) add('browIn', arch.browIn);
    if (arch.browOut != null) add('browOut', arch.browOut);
    if (arch.browDn != null) add('browDn', arch.browDn);
    if (arch.cheek != null) add('cheek', arch.cheek);
    if (arch.gazeY != null) add('gazeY', arch.gazeY);
    if (arch.headY != null) add('headY', arch.headY);
    if (arch.headZ != null) add('headZ', arch.headZ);
    // g62：原型直接给第二套通道定形，不走 add() 的 -1..1 量纲。
    if (arch.smile2 != null) f.smile2 = clamp(f.smile2 + amt * arch.smile2 * 2.2, -24, 28);
    // 原型只做「往中心靠」，不能把下面的幅度推到失真：嘴形和头单独收一下。
    // g62：上限跟着放大（嘴 0.38 → 0.62，眉 0.55 → 0.88）。
    f.mouth = clamp(f.mouth, -0.62, 0.62);
    f.eyeOpen = clamp(f.eyeOpen, -0.85, 0.12);
    f.browIn = clamp(f.browIn, 0, 0.88);
  }

  // ---- g62：把情绪落到「细节」上 ----
  // solveEmoFACS + 原型回答的是「这一小节这张脸该是什么样」，这一层补的是
  // 「同一秒里它在怎么动」。四个来源，幅度都很小（0.02~0.2），但正是这些把
  // 「有表情」和「摆造型」分开：
  //   1) 乐句弧：一句里先抬起来、尾音再落下（起 0.55s / 落 0.70s），
  //      嘴角和脸颊都乘这条增益 —— 同一句里脸的强度是活的；
  //   2) 咬字：嘴每张开一次的增量让眉毛跟着挑一下（重音咬字最明显）；
  //   3) 颤音：长音真的在抖的时候，眉和嘴角按 5.4Hz 轻颤；
  //   （g63 删掉了第 4 条「猫耳弹动」：那个参数其实是「变小」。）
  // 这些量各自走自己的平滑再交给 applyEmo，不会和 VAD 的快慢互相拖。
  function extraTick(dt) {
    var f = emo.f, e = extra;
    e.t += dt;
    var open = lips.lastOpen || 0;
    // 1) 乐句弧
    var want = (emo.pres > 0.16 || open > 0.12) ? 1 : 0;
    e.arc += (want - e.arc) * (1 - Math.exp(-dt / (want ? 0.55 : 0.70)));
    // 2) 咬字：取「嘴张开」的正增量（收口不算）
    var dOpen = Math.max(0, open - e.lastOpen);
    e.lastOpen += (open - e.lastOpen) * (1 - Math.exp(-dt / 0.10));
    e.browSyl = clamp(dOpen * 5.5, 0, 0.14);
    // 3) 颤音
    var vib = clamp(vox.vib || 0, 0, 1);
    e.vibPh += 2 * Math.PI * 5.4 * dt;
    if (e.vibPh > 6.283) e.vibPh -= 6.283;
    var flut = (vib > 0.22 && emo.hold > 0.25)
      ? Math.sin(e.vibPh) * clamp(emo.hold, 0, 1) * clamp((vib - 0.22) / 0.5, 0, 1)
      : 0;
    // 4) g63：猫耳那条已删除（Param202 = 变小，不是耳朵）。

    var arcGain = 0.74 + 0.34 * e.arc;
    var kS = 1 - Math.exp(-dt / 0.24);
    var kM = 1 - Math.exp(-dt / 0.55);
    var kA = 1 - Math.exp(-dt / 0.30);
    e.smile2 += (clamp(f.smile2 * arcGain + flut * 2.2, -26, 30) - e.smile2) * kS;
    e.mouthX += (f.mouthX + flut * 0.8 - e.mouthX) * kS;
    e.mk += (f.angerMk - e.mk) * kM;
    // 露牙要求「嘴真的张开」，不然闭嘴时露出牙会穿帮
    var teethGate = clamp((open - 0.26) / 0.30, 0, 1);
    e.teeth += (f.teeth * teethGate - e.teeth) * kS;
    e.asym += ((f.headX * 0.10 + flut * 0.05) - e.asym) * kA;
    // 5) g64：特效表情的平滑。起手慢一点（0.45~0.55s）、收手更慢（0.7~0.9s），
    //    红脸/眼泪是「一点点上来」的，突然贴上去会像贴图出错。
    var kB = 1 - Math.exp(-dt / (f.blush > e.blush ? 0.45 : 0.70));
    var kT = 1 - Math.exp(-dt / (f.tears > e.tears ? 0.55 : 0.90));
    var kF = 1 - Math.exp(-dt / (f.flower > e.flower ? 0.50 : 0.85));
    e.blush += (f.blush - e.blush) * kB;
    e.tears += (f.tears - e.tears) * kT;
    e.flower += (f.flower - e.flower) * kF;
    e.darkFace += (f.darkFace - e.darkFace) * kB;
    // g64 修正：死区只在「目标已经是 0」时才收干。原先写成 e.tears<0.02 就归零，
    // 而眼泪每帧只长 0.018，永远迈不过这个门坎 → tears 恒为 0（离线测试抓到的）。
    if (f.blush === 0 && e.blush < 0.02) e.blush = 0;
    if (f.tears === 0 && e.tears < 0.02) e.tears = 0;
    if (f.flower === 0 && e.flower < 0.02) e.flower = 0;
    if (f.darkFace === 0 && e.darkFace < 0.02) e.darkFace = 0;
    if (Math.abs(e.smile2) < 0.05) e.smile2 = 0;
    if (Math.abs(e.mouthX) < 0.02) e.mouthX = 0;
    if (Math.abs(e.mk) < 0.02) e.mk = 0;
    if (Math.abs(e.teeth) < 0.004) e.teeth = 0;
    if (Math.abs(e.asym) < 0.004) e.asym = 0;
    if (e.browSyl < 0.004) e.browSyl = 0;
  }

  // ---- g69：桌面唱歌编排化（读 show.json 的 desk 档） ----
  // 「为她而唱」唱出来的歌有一份离线编排好的 show.json（舞台和桌面共用）。有它时，
  // 桌面上的各层不再各自看「此刻的音频」，而是读同一份演出脚本：
  //   · 手势：不再按 2.6~4.8 秒定时器随机挑，只在脚本事件上触发（desk 白名单里的
  //     nod/tilt/lean/look_away/smile_cam/wave，加上长音/高音/副歌入口/关键词）；
  //     动作曲线仍是 SDK 的三段关键帧（gestPeak/gestEval 不变）；
  //   · 情绪：歌词证据换成逐句 emotion/valence/arousal/intensity，权重高于关键词表；
  //   · 幅度：× 逐句 motion.amp × desk.amp_scale × 句内包络（起句展开、句尾回收）；
  //   · 眨眼：仍然是主程序自己的眨眼器在眨（眼睑参数我们一个都不写），只是把它的
  //     随机定时按住，改在 phrase_end / blink 事件上起一次；长音和对视前 0.5 秒不眨；
  //   · 视线：眼球照旧每帧回默认值（poseOverride），「看哪」用头的朝向表达；
  //     副歌入口和关键词句看镜头；
  //   · 仲裁器：同一时刻只有一个主动作（长音仰头 > 高音 > 副歌入口 > 关键词点头 >
  //     脚本手势 > 律动），主动作在播时律动/身体慢摆/情绪头偏这些次要层降到 30%。
  // 没有 show.json（普通放歌、编排还没算完、拉取失败）时 desk.on=false，所有乘数
  // 恒为 1、偏移恒为 0、随机手势照旧——和 g68 逐帧一致。
  var DESK_REF = 0.36;     // motion.amp × amp_scale 的「常态」：两者乘积等于它时幅度 ×1
  var DESK_PRI = { long: 5, high: 4, chorus: 3, kw: 2, gesture: 1.5 };
  var DESK_FAM = { nod: 'quick-nod', tilt: 'curious-tilt', lean: 'small-lean-in',
                   look_away: 'side-glance', smile_cam: 'warm-sway', wave: 'bright-pop' };
  // 逐句情绪 → 掌控（D）：契约里只有 V/A，D 按情绪类别给
  var DESK_DOM = { determined: 0.40, proud: 0.35, joy: 0.20, playful: 0.10,
                   tender: -0.10, longing: -0.20, sad: -0.35 };
  var desk = {
    on: false, track: '', want: '', fetching: false, fails: 0, failTrack: '',
    lines: null, ev: null, holds: null, ends: null, dur: 0, ampScale: 0.5, wl: null,
    ei: 0, li: 0, t: -1, line: null, lineEnd: 0,
    k: 1, kS: 1, arb: 1, gGain: 1, gx: 0, gy: 0, gaze: '', camUntil: -1,
    blinkHeld: false, lastBlink: -9, vetoLast: null,
    stat: null
  };
  function deskStatNew() {
    return { random: 0, script: 0, sup: 0, latN: 0, latSum: 0, latMax: 0,
             blinks: 0, blinkEnd: 0, blinkVeto: 0, blinkSafe: 0, blinkLog: [],
             fired: [], moodRows: 0 };
  }
  desk.stat = deskStatNew();
  function deskHash(x) {   // 0..1，确定性（同一事件每次都一样）
    var s = Math.sin(x * 12.9898 + 78.233) * 43758.5453;
    return s - Math.floor(s);
  }
  function deskFirstAt(arr, t) {   // 第一个 arr[i].t >= t
    var lo = 0, hi = arr.length;
    while (lo < hi) { var mid = (lo + hi) >> 1; if (arr[mid].t < t) lo = mid + 1; else hi = mid; }
    return lo;
  }
  function deskOff() {
    desk.on = false; desk.track = ''; desk.lines = null; desk.ev = null; desk.holds = null;
    desk.ends = null; desk.line = null; desk.k = 1; desk.kS = 1; desk.arb = 1; desk.gGain = 1;
    desk.gx = 0; desk.gy = 0; desk.gaze = ''; desk.camUntil = -1;
    deskBlinkRelease();
  }
  function deskLoad(show, tid) {
    if (!show || show.schema !== 'neko.show/1' || !(Number(show.duration) > 0)) return false;
    if (!Array.isArray(show.lines) || !Array.isArray(show.events)) return false;
    var d = show.desk || {};
    var wl = Array.isArray(d.gesture_whitelist) ? d.gesture_whitelist : [];
    var lines = [], ends = [], holds = [], ev = [], i;
    for (i = 0; i < show.lines.length; i++) {
      var L = show.lines[i] || {};
      var t0 = Number(L.t0), t1 = Number(L.t1);
      if (!isFinite(t0) || !isFinite(t1) || t1 <= t0) continue;
      var kw = false, ch = L.chars || [];
      for (var c = 0; c < ch.length; c++) { if (ch[c] && ch[c][4]) { kw = true; break; } }
      lines.push({ t0: t0, t1: t1, i: lines.length, emotion: String(L.emotion || ''),
                   v: clamp(Number(L.valence) || 0, -1, 1), a: clamp(Number(L.arousal) || 0, -1, 1),
                   inten: clamp(Number(L.intensity) || 0, 0, 1),
                   amp: clamp(Number(L.motion && L.motion.amp) || 0.5, 0, 2),
                   gaze: kw ? 'camera' : String(L.gaze || 'camera'), kw: kw });
      if (L.kind !== 'title' && L.kind !== 'flash') ends.push(t1);
    }
    lines.sort(function (x, y) { return x.t0 - y.t0; });
    var lastKw = -9;
    var src = show.events.slice().sort(function (x, y) { return (Number(x.t) || 0) - (Number(y.t) || 0); });
    for (i = 0; i < src.length; i++) {
      var e = src[i] || {}, t = Number(e.t), ty = String(e.type || ''), du = Number(e.dur) || 0;
      if (!isFinite(t)) continue;
      var amp = clamp(Number(e.amp) || 0.5, 0, 1);
      if (ty === 'gesture') {
        var fam = DESK_FAM[String(e.arg || '')];
        if (fam && wl.indexOf(String(e.arg)) >= 0) {
          ev.push({ t: t, kind: 'gesture', pri: DESK_PRI.gesture, fam: fam, amp: amp, dur: du, arg: String(e.arg) });
        }
      } else if (ty === 'long_note' && du >= 0.6) {
        ev.push({ t: t, kind: 'long', pri: DESK_PRI.long, fam: 'long-lift', amp: amp, dur: du });
        holds.push([t - 0.5, t + du]);
      } else if (ty === 'high_note') {
        ev.push({ t: t, kind: 'high', pri: DESK_PRI.high, fam: 'high-lift', amp: amp, dur: du });
      } else if (ty === 'chorus_in') {
        ev.push({ t: t, kind: 'chorus', pri: DESK_PRI.chorus, fam: 'bright-pop', amp: amp, dur: 0 });
        holds.push([t - 0.5, t + 0.5]);
      } else if (ty === 'keyword') {
        if (t - lastKw < 0.6) continue;     // 同一个词的几个字只点一次头
        lastKw = t;
        ev.push({ t: t, kind: 'kw', pri: DESK_PRI.kw, fam: 'kw-nod', amp: amp, dur: 0 });
      } else if (ty === 'phrase_end' || ty === 'blink') {
        ev.push({ t: t, kind: ty === 'blink' ? 'blink' : 'pend', pri: 0, amp: amp, dur: 0 });
      }
    }
    desk.track = tid; desk.lines = lines; desk.ev = ev; desk.holds = holds;
    desk.ends = ends; desk.dur = Number(show.duration);
    desk.ampScale = clamp(Number(d.amp_scale) || 0.5, 0, 1); desk.wl = wl;
    desk.ei = 0; desk.li = 0; desk.t = -1; desk.line = null;
    desk.stat = deskStatNew(); desk.lastBlink = -9; desk.vetoLast = null;
    desk.on = true;
    return true;
  }
  // 由 lipsPoll 调：这首歌有 show.json 就拉一次（只走 GET），换歌/停唱就退回实时模式
  function deskSync(st) {
    var tid = String(st.track_id || '');
    var want = !!(lips.active && st.show && tid && LIPS_BASE)
               && !((Number(lips.t) || 0) > (Number(st.duration) || 1e9) + 1.0);
    // 全屏舞台在演时人物归舞台运行时（P3），桌面档让开，行为与 g68 一致；
    // 用户点 ✕ 退出舞台而歌还在唱时，桌面档接手。
    var stageOn = false;
    try { stageOn = !!(document.body && document.body.classList.contains('neko-solo-stage')); } catch (e) { stageOn = false; }
    if (!want || stageOn) { if (desk.on || desk.track) deskOff(); desk.want = ''; return; }
    desk.want = tid;
    if (desk.track === tid) return;
    if (desk.track) deskOff();
    if (desk.fetching || (desk.failTrack === tid && desk.fails >= 2)) return;
    desk.fetching = true;
    fetch(LIPS_BASE + '/show.json?track=' + encodeURIComponent(tid), { cache: 'no-store' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (show) {
        desk.fetching = false;
        if (desk.want !== tid) return;           // 拉的时候已经换歌/停唱
        if (!deskLoad(show, tid)) throw new Error('bad show');
        desk.fails = 0;
      })
      .catch(function () {
        desk.fetching = false;
        if (desk.failTrack !== tid) { desk.failTrack = tid; desk.fails = 0; }
        desk.fails += 1;                         // 同一首最多拉两次，失败就保持实时模式
      });
  }
  function deskLineAt(t) {
    var ls = desk.lines;
    if (!ls || !ls.length) return null;
    var i = desk.li;
    if (i >= ls.length || ls[i].t0 > t) i = 0;
    while (i + 1 < ls.length && ls[i + 1].t0 <= t) i++;
    desk.li = i;
    var L = ls[i];
    return (L.t0 <= t && t <= L.t1) ? L : null;
  }
  // 句内包络：起句 0.22 展开到满，最后 18% 回收；句与句之间 0.6
  function deskEnv(L, t) {
    if (!L) return 0.6;
    var u = clamp((t - L.t0) / Math.max(0.2, L.t1 - L.t0), 0, 1);
    return 0.6 + 0.4 * smoothstep(0, 0.22, u) - 0.3 * smoothstep(0.82, 1, u);
  }
  function deskVeto(t) {
    var h = desk.holds;
    if (!h) return false;
    for (var i = 0; i < h.length; i++) { if (t >= h[i][0] && t <= h[i][1]) return true; }
    return false;
  }
  // 命中的禁眨窗口起点（没命中返回 null）：用来让同一窗口只计一次
  function deskVetoWin(t) {
    var h = desk.holds;
    if (!h) return null;
    for (var i = 0; i < h.length; i++) { if (t >= h[i][0] && t <= h[i][1]) return h[i][0]; }
    return null;
  }
  function deskNearEnd(t) {
    var e = desk.ends;
    if (!e) return false;
    for (var i = 0; i < e.length; i++) { if (t >= e[i] - 0.35 && t <= e[i] + 0.7) return true; }
    return false;
  }
  // 眨眼：借主程序的眨眼器（live2d-model.js _updateEyeBlink 的 0/1/2 状态机），
  // 只改「什么时候眨」，眼睑参数仍由它写。被别的模块接管（suspend）时不碰。
  function deskBlinkMgr() {
    var mg = eyeManager();
    if (!mg || !mg._autoEyeBlinkEnabled || mg._suspendEyeBlinkOverride) return null;
    if (typeof mg._eyeBlinkState !== 'number') return null;
    return mg;
  }
  function deskBlinkHold() {
    var mg = deskBlinkMgr();
    if (!mg) return;
    desk.blinkHeld = true;
    if (mg._eyeBlinkState === 0 && mg._eyeBlinkNextTime < 1e8) mg._eyeBlinkNextTime = 1e9;
  }
  function deskBlinkRelease() {
    if (!desk.blinkHeld) return;
    desk.blinkHeld = false;
    var mg = eyeManager();
    // 交还给主程序：用它自己的间隔公式（2~6 秒）重新起算
    if (mg && mg._eyeBlinkNextTime >= 1e8) mg._eyeBlinkNextTime = 2 + Math.random() * 4;
  }
  function deskBlink(t, why) {
    var mg = deskBlinkMgr();
    if (!mg || mg._eyeBlinkState !== 0) return false;
    if (t - desk.lastBlink < (why === 'idle' ? 2.5 : 1.2)) return false;
    if (deskVeto(t)) {
      // 同一段禁眨窗口每帧都会再进来一次（兜底分支），只计第一次，不改行为
      var vw = deskVetoWin(t);
      if (vw !== desk.vetoLast) { desk.vetoLast = vw; desk.stat.blinkVeto += 1; }
      return false;
    }
    mg._eyeBlinkState = 1;
    mg._eyeBlinkTimer = 0;
    desk.lastBlink = t;
    var s = desk.stat, end = deskNearEnd(t);
    s.blinks += 1;
    if (end) s.blinkEnd += 1;
    if (why === 'safety') s.blinkSafe += 1;
    if (s.blinkLog.length < 400) s.blinkLog.push([Math.round(t * 100) / 100, why, end ? 1 : 0]);
    return true;
  }
  // 主动作：仲裁 + 用 SDK 的三段关键帧起一段（确定性，不走 gestRnd）
  function deskFire(e, t) {
    var cur = gestSt.cur, s = desk.stat;
    if (cur && gestSt.t - cur.at0 < cur.dur) {
      var cp = cur.pri || 0;
      if (e.pri < cp || (e.pri === cp && gestSt.t - cur.at0 < cur.dur * 0.58)) { s.sup += 1; return; }
    }
    var h = deskHash(e.t * 7.31 + e.pri);
    var a = clamp((0.30 + e.amp * 0.55) * desk.k, 0.15, 0.95);
    var hold = (e.kind === 'long' || e.kind === 'high') ? Math.min(e.dur, 2.6) : 0;
    var dur = hold > 0 ? Math.max(0.9, hold + 0.45) : 0.96 + h * 0.58 + a * 0.45;
    var attack = 0.20 + deskHash(h * 3.7) * 0.16;
    var settleAt = attack * (0.78 + h * 0.24);
    var retAt = hold > 0 ? Math.max(attack + 0.12, hold) : dur * (0.52 + h * 0.16);
    var side = ((desk.line ? desk.line.i : Math.floor(e.t)) % 2) ? 1 : -1;
    var peak = gestPeak(e.fam, a, side), settle = {}, rest = {}, from = null, k;
    for (k in peak) {
      settle[k] = peak[k] * (hold > 0 ? 0.86 : 0.48 + h * 0.24);
      rest[k] = 0;
    }
    if (cur) { from = {}; for (k in gestSt.out) from[k] = gestSt.out[k]; }
    gestSt.cur = {
      label: e.fam, at0: gestSt.t, dur: dur, pri: e.pri, from: from,
      frames: [
        { time: 0, dur: attack, out: true, v: peak },
        { time: settleAt, dur: Math.max(0.16, (hold > 0 ? retAt - settleAt : dur * 0.28)), out: false, v: settle },
        { time: retAt, dur: Math.max(0.22, dur - retAt), out: true, v: rest }
      ]
    };
    gestSt.label = e.fam;
    gestSt.n += 1;
    s.script += 1;
    var lat = Math.abs(t - e.t);
    s.latN += 1; s.latSum += lat; if (lat > s.latMax) s.latMax = lat;
    if (s.fired.length < 400) s.fired.push([Math.round(e.t * 1000) / 1000, e.kind, Math.round(lat * 1000)]);
  }
  // 每帧一次（updateEmotion 开头）：当前句、幅度、视线、事件、眨眼
  function deskTick(dt) {
    if (!desk.on) return;
    if (!lips.active) return;
    var t = Number(lips.t) || 0;
    // 口型服务断线时 lips.active 不会自己变假：唱过曲尾就主动退回实时模式，
    // 把眨眼器还给主程序
    if (t > desk.dur + 1.0) { deskOff(); return; }
    var ev = desk.ev;
    // 跳播/回放：重新定位，不补放跳过去的事件
    if (desk.t < 0 || t < desk.t - 0.3 || t > desk.t + 1.0) desk.ei = deskFirstAt(ev, t - 0.02);
    desk.t = t;
    var L = deskLineAt(t);
    desk.line = L;
    if (L) desk.lineEnd = L.t1;
    var base = L ? L.amp : 0.5;
    desk.k = clamp(base * desk.ampScale * deskEnv(L, t) / DESK_REF, 0.35, 1.3);
    while (desk.ei < ev.length && ev[desk.ei].t <= t) {
      var e = ev[desk.ei++];
      if (t - e.t > 0.35) continue;            // 太旧（卡顿后）就不补了
      if (e.kind === 'pend') { deskBlink(t, 'phrase_end'); continue; }
      if (e.kind === 'blink') { deskBlink(t, deskNearEnd(t) ? 'blink' : 'idle'); continue; }
      if (e.kind === 'chorus') desk.camUntil = e.t + 1.6;
      deskFire(e, t);
    }
    deskBlinkHold();
    // 兜底：太久没眨（句尾都被长音/对视禁掉了）也要眨——先等下一个句尾附近，
    // 10 秒还没等到才在任意不禁眨的时刻补一次
    var since = t - desk.lastBlink;
    if ((since > 7 && deskNearEnd(t)) || since > 10) deskBlink(t, 'safety');
    // 仲裁：主动作在播 → 次要层压到 30%
    var cur = gestSt.cur;
    var prim = !!(cur && gestSt.t - cur.at0 < cur.dur && (cur.pri || 0) > 0);
    desk.arb += ((prim ? 0.3 : 1) - desk.arb) * (1 - Math.exp(-dt / 0.18));
    desk.kS = clamp(desk.k, 0.35, 1.3) * desk.arb;
    desk.gGain = Math.min(1, desk.kS);
    // 视线 → 头的朝向（度）。眼球不写。
    var gz = L ? L.gaze : '';
    if (t < desk.camUntil) gz = 'camera';
    var side = (L && L.i % 2) ? 1 : -1, gxT = 0, gyT = 0;
    if (gz === 'far') { gyT = 3; gxT = 4 * side; }
    else if (gz === 'down') { gyT = -5; }
    else if (gz === 'side') { gxT = 7 * side; }
    else if (gz === 'closed') { gyT = -2; }
    var gs = clamp(desk.ampScale / 0.55, 0.6, 1.2);
    var kg = 1 - Math.exp(-dt / 0.6);
    desk.gx += (gxT * gs - desk.gx) * kg;
    desk.gy += (gyT * gs - desk.gy) * kg;
    desk.gaze = gz;
  }
  // 逐句情绪（代替关键词表）：句子唱完后 0.9 秒内淡出，和旧通道同一个手感
  function deskMood(t) {
    if (!desk.on || !desk.lines) return null;
    var L = desk.line, fade = 1;
    if (!L) {
      var ls = desk.lines, P = ls[desk.li];
      if (!P || t < P.t0 || t - P.t1 > 2.5) return null;
      L = P; fade = Math.exp(-(t - P.t1) / 0.9);
    }
    var conf = (0.70 + 0.25 * L.inten) * fade;
    if (conf < 0.12) return null;
    var d = DESK_DOM[L.emotion];
    return { v: L.v, a: L.a, d: d == null ? 0 : d, conf: conf, show: true };
  }
  function deskTag() {
    if (!desk.on && !desk.track) return '';
    var s = desk.stat;
    return (desk.on ? 1 : 0) + ',' + String(desk.track).slice(0, 6) + ',' + desk.k.toFixed(2)
      + ',' + desk.arb.toFixed(2) + ',' + (gestSt.label || '-') + ',' + s.random + ',' + s.script
      + ',' + s.sup + ',' + Math.round(s.latMax * 1000) + ',' + s.blinks + '/' + s.blinkEnd
      + '/' + s.blinkVeto + '/' + s.blinkSafe + ',' + (desk.gaze || '-');
  }

  // ---- 情绪 → 动作（g58） ----
  // 移植 Soullink Emotion SDK 的 VADGestureController：情绪一变化就挑一段
  // 「关键帧小动作」播一遍（起 → 收 → 回），段内按 easeOut / easeInOut
  // 混合，播完回 0。原版输出全脸 FACS；这里只留看得见、又不跟主程序抢的
  // 通道：头（AngleX/Y/Z）、上身（BodyAngleX/Y/Z）、眉、嘴形。
  // **眼球（EyeBallX/Y）和眼睑（EyeL/R Open/Smile）一概不写**——那条线不动。
  //
  // 触发条件按唱歌重配：原版靠「新一段文字情绪」造成大的目标跳变；歌里没有
  // 文字切句，目标每帧只动一点点。所以三条任一成立就播：
  //   1) 表情原型真的换了档（moodSt.id 变了）；
  //   2) 目标在近 0.7 秒里累计移动够大（dAcc）；
  //   3) 隔了 2.6~4.8 秒的「到点」——一直在唱就总有小动作。
  var GEST = {
    head: 16.0,    // 头的峰值（度）：g62 13 → 16（模型 ±30，律动自己也会到 20 上下）
    body: 5.5,     // 上身的峰值（度）：g62 4 → 5.5（身体区间只有 ±10，留出律动的份）
    brow: 0.75,    // 眉的峰值（参数单位）：g62 0.55 → 0.75
    mouth: 0.45,   // 嘴形的峰值（参数单位）：g62 0.30 → 0.45
    freq: 1.0,     // 频率倍率
    avoid: 3       // 最近几个动作不重复
  };
  var gestSt = {
    t: 0, cur: null, nextAt: 0, recent: [], seed: 977,
    label: '', n: 0, dAcc: 0, moodId: -1,
    pv: 0, pa: 0, pd: 0,
    out: { headX: 0, headY: 0, headZ: 0, bodyX: 0, bodyY: 0, bodyZ: 0,
           browIn: 0, browOut: 0, browDn: 0, mouth: 0 }
  };
  function gestRnd() {   // mulberry32：同一种子可复现，和 SDK 的 seededRandom 同型
    var t = gestSt.seed = (gestSt.seed + 0x6D2B79F5) >>> 0;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  function gestClear() {
    var o = gestSt.out;
    for (var k in o) o[k] = 0;
  }
  function gestMag(v, a, d) {
    return clamp((Math.abs(v) + Math.abs(a) * 0.82 + Math.abs(d) * 0.64) / 2.46, 0, 1);
  }
  function gestEmotion() {
    // 和 SDK 的 normalizeGestureEmotion 同一套判定顺序（它先认文字标签，
    // 这里只有 VAD）。
    var V = emo.v, A = emo.a, D = emo.d;
    if (V > 0.22 && A > 0.35) return 'excited';
    if (V > 0.18 && D < -0.18) return 'shy';
    if (V > 0.20 && A > 0.24) return 'happy';
    if (V > 0.08) return 'affectionate';
    if (V < -0.24 && A > 0.18) return D > 0.1 ? 'anger' : 'anxiety';
    if (V < -0.18) return 'sad';
    if (V < -0.05) return 'concerned';
    if (A > 0.18) return 'curious';
    if (A < -0.18) return 'calm';
    return 'neutral';
  }
  function gestPick(e) {   // 各情绪的动作权重照抄 SDK
    var c;
    if (e === 'shy' || e === 'anxiety') {
      c = [['shy-dip', 0.44], ['side-glance', 0.31], ['soft-sink', 0.15], ['warm-sway', 0.1]];
    } else if (e === 'curious' || e === 'confused') {
      c = [['curious-tilt', 0.44], ['small-lean-in', 0.3], ['side-glance', 0.15], ['quick-nod', 0.11]];
    } else if (e === 'excited') {
      c = [['bright-pop', 0.42], ['quick-nod', 0.34], ['curious-tilt', 0.14], ['warm-sway', 0.1]];
    } else if (e === 'anger' || e === 'annoyed') {
      c = [['firm-lean', 0.42], ['side-set', 0.34], ['slow-glance-down', 0.13], ['small-lean-in', 0.11]];
    } else if (e === 'sad' || e === 'concerned') {
      c = [['soft-sink', 0.42], ['slow-glance-down', 0.32], ['side-glance', 0.15], ['warm-sway', 0.11]];
    } else if (e === 'happy' || e === 'affectionate') {
      c = [['quick-nod', 0.38], ['warm-sway', 0.36], ['small-lean-in', 0.15], ['shy-dip', 0.11]];
    } else if (e === 'calm') {
      c = [['warm-sway', 0.44], ['small-lean-in', 0.25], ['side-glance', 0.17], ['soft-sink', 0.14]];
    } else {
      c = [['warm-sway', 0.34], ['curious-tilt', 0.3], ['side-glance', 0.2], ['quick-nod', 0.16]];
    }
    var fresh = [], i;
    for (i = 0; i < c.length; i++) {
      if (gestSt.recent.indexOf(c[i][0]) < 0) fresh.push(c[i]);
    }
    var pool = fresh.length ? fresh : c, total = 0;
    for (i = 0; i < pool.length; i++) total += pool[i][1];
    var roll = gestRnd() * total;
    for (i = 0; i < pool.length; i++) {
      roll -= pool[i][1];
      if (roll <= 0) return pool[i][0];
    }
    return pool[0][0];
  }
  // 动作峰值：单位是「满幅的比例」，写入时再乘 GEST 的度数/参数增益。
  // SDK 里的 gazeX/gazeY 换成头的朝向（眼球不写），blush/sweat/eyeSquint
  // 这些没有对应参数，直接去掉。
  function gestPeak(fam, a, side) {
    var positive = Math.max(0, emo.v), negative = Math.max(0, -emo.v);
    var aroused = Math.max(0, emo.a), dominant = Math.max(0, emo.d);
    var p = {};
    if (fam === 'quick-nod') {
      p.headY = a * (0.42 + aroused * 0.18); p.headZ = side * a * 0.18;
      p.bodyY = a * 0.28; p.bodyZ = side * a * 0.16;
      p.mouth = a * (0.42 + positive * 0.22); p.browOut = a * 0.14;
    } else if (fam === 'warm-sway') {
      p.headX = side * a * 0.22; p.headZ = side * a * 0.36;
      p.bodyX = side * a * 0.24; p.bodyZ = side * a * 0.30;
      p.mouth = a * 0.34; p.browOut = a * 0.06;
    } else if (fam === 'shy-dip') {
      p.headY = -a * 0.48; p.headZ = -side * a * 0.42;
      p.bodyX = side * a * 0.12; p.bodyY = -a * 0.18; p.bodyZ = -side * a * 0.28;
      p.browIn = a * 0.24; p.mouth = a * 0.18;
    } else if (fam === 'side-glance') {
      p.headX = side * a * 0.18; p.headY = -a * 0.05; p.headZ = -side * a * 0.28;
      p.bodyZ = -side * a * 0.18; p.browIn = a * 0.20;
    } else if (fam === 'curious-tilt') {
      p.headX = side * a * 0.20; p.headY = a * 0.12; p.headZ = side * a * 0.62;
      p.bodyX = side * a * 0.10; p.bodyY = a * 0.16; p.browOut = a * 0.28;
    } else if (fam === 'small-lean-in') {
      p.headY = a * 0.20; p.headZ = side * a * 0.22;
      p.bodyY = a * 0.38; p.bodyZ = side * a * 0.14;
      p.browOut = a * 0.18; p.mouth = positive * a * 0.24;
    } else if (fam === 'bright-pop') {
      p.headY = a * 0.34; p.headZ = side * a * 0.22;
      p.bodyY = a * 0.36; p.bodyZ = side * a * 0.20;
      p.browOut = a * 0.42; p.mouth = positive * a * 0.36;
    } else if (fam === 'firm-lean') {
      p.headY = a * 0.22; p.headZ = side * a * 0.20;
      p.bodyY = a * 0.42; p.bodyZ = side * a * 0.16;
      p.browDn = a * (0.32 + dominant * 0.28); p.mouth = -a * 0.22;
    } else if (fam === 'side-set') {
      p.headX = -side * a * 0.18; p.headZ = side * a * 0.34;
      p.bodyZ = side * a * 0.24;
      p.browDn = a * 0.24; p.mouth = -a * 0.16;
    } else if (fam === 'soft-sink') {
      p.headY = -a * 0.34; p.headZ = side * a * 0.16; p.bodyY = -a * 0.24;
      p.browIn = a * 0.28; p.mouth = -a * (0.22 + negative * 0.16);
    } else if (fam === 'long-lift') {     // g69 桌面档：长音仰头（撑住整个长音再收）
      p.headY = a * 0.50; p.headZ = side * a * 0.14; p.bodyY = a * 0.22;
      p.browOut = a * 0.32; p.mouth = positive * a * 0.18;
    } else if (fam === 'high-lift') {     // g69 桌面档：高音抬头挑眉
      p.headY = a * 0.36; p.headZ = side * a * 0.26; p.bodyY = a * 0.16;
      p.browOut = a * 0.40;
    } else if (fam === 'kw-nod') {        // g69 桌面档：关键词往下点一下（和舞台同向）
      p.headY = -a * 0.40; p.headZ = side * a * 0.10; p.bodyY = -a * 0.12;
      p.browOut = a * 0.12; p.mouth = positive * a * 0.16;
    } else {   // slow-glance-down：原版的视线下移换成低头
      p.headY = -a * 0.22; p.bodyY = -a * 0.18;
      p.browIn = a * 0.22; p.mouth = -a * 0.14;
    }
    return p;
  }
  function gestStart(trigger, changed) {
    var t = gestSt.t;
    var amp = clamp((0.30 + (changed ? 0.42 : 0) + trigger * 0.60)
                    * (0.90 + gestRnd() * 0.46), 0.22, 0.95);
    var dur = 0.96 + gestRnd() * 0.58 + amp * 0.45;
    var attack = 0.20 + gestRnd() * 0.16;
    var settleAt = attack * (0.78 + gestRnd() * 0.24);
    var retAt = dur * (0.52 + gestRnd() * 0.16);
    var side = gestRnd() < 0.5 ? -1 : 1;
    var fam = gestPick(gestEmotion());
    var peak = gestPeak(fam, amp, side), settle = {}, rest = {}, i;
    for (i in peak) {
      settle[i] = peak[i] * (0.48 + gestRnd() * 0.24);
      rest[i] = 0;                              // 播完回 0，交还给律动和待机
    }
    gestSt.cur = {
      label: fam, at0: t, dur: dur,
      frames: [
        { time: 0, dur: attack, out: true, v: peak },
        { time: settleAt, dur: Math.max(0.16, dur * 0.28), out: false, v: settle },
        { time: retAt, dur: Math.max(0.22, dur - retAt), out: true, v: rest }
      ]
    };
    if (GEST.avoid > 0) {
      gestSt.recent.push(fam);
      gestSt.recent = gestSt.recent.slice(-GEST.avoid);
    } else {
      gestSt.recent = [];
    }
    var fs = 1 / Math.sqrt(Math.max(0.35, GEST.freq));
    gestSt.nextAt = t + dur + (1.4 + gestRnd() * 2.2) * fs;
    gestSt.dAcc = 0;
    gestSt.label = fam;
    gestSt.n += 1;
    if (desk.on) desk.stat.random += 1;   // 桌面档下应该一直是 0（验收指标）
  }
  // 关键帧求值：和 SDK 的 evaluateExpressionTimeline 一样，把已经开始的帧
  // 按顺序往当前值上 blend；最后一帧的目标是 0，所以动作一定收得回来。
  function gestEval(cur, el) {
    var o = gestSt.out, fr = cur.frames, i, key;
    gestClear();
    // g69：桌面档的高优先级动作会打断正在播的动作，从被打断时的姿态接着走，不先跳回 0
    if (cur.from) for (key in o) o[key] = cur.from[key] || 0;
    for (i = 0; i < fr.length; i++) {
      var f = fr[i];
      if (el < f.time) continue;
      var x = f.dur <= 0 ? 1 : clamp((el - f.time) / f.dur, 0, 1);
      x = f.out ? 1 - (1 - x) * (1 - x)
                : (x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2);
      for (key in o) o[key] += ((f.v[key] || 0) - o[key]) * x;
    }
  }
  function gestTick(dt) {
    // 只在「在唱 / 有口型 / 律动在跑」时做动作；停唱立刻归零，整段交回待机。
    var playing = emo.pres > 0.02 || lips.active || blend > 0.10;
    if (!playing) {
      gestSt.cur = null; gestSt.dAcc = 0; gestSt.label = '';
      gestSt.pv = 0; gestSt.pa = 0; gestSt.pd = 0;
      gestClear();
      return;
    }
    gestSt.t += dt;
    var tv = emo.tV || 0, ta = emo.tA || 0, td = emo.tD || 0;
    var dMag = gestMag(tv - gestSt.pv, ta - gestSt.pa, td - gestSt.pd);
    gestSt.pv = tv; gestSt.pa = ta; gestSt.pd = td;
    // 目标累计移动：0.7 秒窗口的积分，长时间的小起伏也能攒够一次
    gestSt.dAcc = gestSt.dAcc * Math.exp(-dt / 0.7) + dMag;

    var cur = gestSt.cur;
    if (cur && gestSt.t - cur.at0 >= cur.dur) {
      gestSt.cur = null; cur = null; gestClear(); gestSt.label = '';
    }
    var changed = moodSt.id !== gestSt.moodId;
    if (changed) gestSt.moodId = moodSt.id;

    // 上一个动作至少播到 58% 才允许换（SDK 同款），否则动作会互相打断成抖
    // g69：桌面档下手势只由脚本事件触发（deskTick → deskFire），这里不再随机起动作
    var canStart = !desk.on && gestSt.t >= gestSt.nextAt
                   && (!cur || gestSt.t - cur.at0 >= cur.dur * 0.58);
    if (canStart && (changed || gestSt.dAcc >= 0.22 || gestSt.t >= gestSt.nextAt)) {
      gestStart(clamp(gestSt.dAcc * 0.9, 0.08, 0.55), changed);
    }

    cur = gestSt.cur;
    if (!cur) { gestClear(); gestSt.label = ''; return; }
    gestEval(cur, gestSt.t - cur.at0);
    gestSt.label = cur.label;
  }

  // ---- 眼睑开合 + 腮部变形（g59；g61 起眼睑停写）----
  // g61：下面这套倍率写入**已经整条关掉**（见 eyeApply 里的说明）。原因是它在
  // 开心的歌上会把开合长期钉在 1.0，主程序自己的眨眼/松弛被盖住 → 「眼白大、
  // 眼珠小」。现在这层只留腮部鼓动（applyFace 的 ParamCheek），眼睑归主程序。
  // 用户要的是「眼睛开闭、脸颊鼓/塌」这种参数级变形，不是换表情。
  // 难点只有一个：主程序在 coreModel.update 的包装里用 _updateEyeBlink 每帧写
  // **绝对值**（live2d-model.js: `_updateEyeBlink` → setParameterValueByIndex），
  // 而那个包装在 beforeModelUpdate 之后才跑，所以我们以前在这里加的偏移会被
  // 整段盖掉——「眼睛抽动」就是这么来的（见 applyEmo 里那段注释）。
  // g59 改走主程序自己留的「渲染前最后姿态覆盖」注入点：
  //   live2d-model.js: _applyTemporaryPoseOverride（origCoreModelUpdate 前后各调一次），
  // 在眨眼算完之后按**倍率**缩放已经写好的开合值：
  //   · 眨眼仍归主程序（闭到 0 × 倍率还是 0），不会被我们顶开；
  //   · 只写 ParamEyeLOpen / ParamEyeROpen，眼球 ParamEyeBallX/Y 一律不碰；
  //   · 只在唱歌时登记，停唱立刻 clear，待机一帧都不碰。
  var EYE = {
    openGain: 0.45,   // f.eyeOpen（-1..1）→ 倍率
    sadLid: 0.38,     // 难过的歌：眼皮往下沉（V=-0.6 时约 0.77 倍，副歌能到 0.7）
    effort: 0.16,     // 激动/高音：用力眯一点
    relax: 0.05,      // 平和的正向：眼睛松开（最多 1.05 倍）
    squint: 0.38,     // 最多眯到 0.62 倍
    wide: 0.06,       // 最多 1.06 倍（这个模型开到 1 就是全睁，再大就假）
    blinkDepth: 0.85, // 换气眨眼的闭合深度
    blinkTau: 0.14,   // 换气眨眼回弹时间常数（秒）
    cool: 3.2,        // 换气眨眼冷却：一句一次，不连着眨
    cheekBeat: 0.42   // 拍点/句首让腮帮子鼓一下的幅度
  };
  var eyeSt = { k: 1, cur: 1, blink: 0, cool: 0, n: 0, baseTok: -1, baseAt: 0,
                baseL: 1, baseR: 1, on: 0, mask: '', cheek: 0 };
  var eyeCache = { core: null, idx: null };

  function eyeOn() {
    return emo.pres > 0.03 || blend > 0.15 || lips.active;
  }
  function eyeIds(core) {
    if (eyeCache.core === core) return eyeCache.idx;
    var idx = {};
    var ids = ['ParamEyeLOpen', 'ParamEyeROpen', 'ParamCheek'];
    for (var i = 0; i < ids.length; i++) {
      var j = -1;
      try { j = core.getParameterIndex ? core.getParameterIndex(ids[i]) : -1; } catch (e) { j = -1; }
      idx[ids[i]] = j;
    }
    eyeCache = { core: core, idx: idx };
    eyeSt.mask = (idx.ParamEyeLOpen >= 0 ? '1' : '0')
               + (idx.ParamEyeROpen >= 0 ? '1' : '0')
               + (idx.ParamCheek >= 0 ? '1' : '0');
    return idx;
  }
  function eyeBounds(core, id) {
    var i = eyeIds(core)[id];
    if (!(i >= 0)) return null;
    var lo = 0, hi = 1;
    try { lo = Number(core.getParameterMinimumValue(i)); } catch (e) { lo = 0; }
    try { hi = Number(core.getParameterMaximumValue(i)); } catch (e) { hi = 1; }
    if (!isFinite(lo) || !isFinite(hi) || hi <= lo) { lo = 0; hi = 1; }
    return { idx: i, lo: lo, hi: hi };
  }
  function eyeRead(core, id) {
    var b = eyeBounds(core, id);
    if (!b) return NaN;
    var v = NaN;
    try { v = Number(core.getParameterValueById(id)); } catch (e) { v = NaN; }
    return isFinite(v) ? v : b.hi;
  }
  function eyeWrite(core, id, base) {
    if (!isFinite(base)) return 0;
    var b = eyeBounds(core, id);
    if (!b) return 0;
    var v = base * eyeSt.cur;
    if (v < b.lo) v = b.lo;
    if (v > b.hi) v = b.hi;
    try { core.setParameterValueById(id, v); return 1; } catch (e) { return 0; }
  }

  // 每帧算一次：目标开合、换气眨眼、腮帮子脉冲
  function eyeTick(dt) {
    var V = Number(emo.v) || 0, A = Number(emo.a) || 0;
    var neg = Math.max(0, -V), ar = Math.max(0, A);
    // 目标开合：情绪层的眼形 + 三个「唱歌才有」的来源
    //   · 难过的歌：眼皮往下沉（这是最能看出「她听懂了」的一条）
    //   · 激动 / 高音：用力眯一点，像真的在使劲
    //   · 平和的正向：眼睛松开一点
    var d = (emo.f.eyeOpen || 0) * EYE.openGain
          - EYE.sadLid * neg
          - EYE.effort * ar * (0.35 + 0.65 * clamp(Number(emo.pHi) || 0, 0, 1))
          + EYE.relax * clamp(V, 0, 1) * Math.max(0, 1 - ar);
    var target = 1 + clamp(d, -EYE.squint, EYE.wide);
    if (eyeSt.cool > 0) eyeSt.cool = Math.max(0, eyeSt.cool - dt);
    // 换气眨眼：一句唱完（faceDyn.pres 掉下去）的瞬间轻轻闭一下，比定时眨眼自然
    if (faceDyn.end) {
      if (eyeSt.cool <= 0 && eyeOn()) { eyeSt.cool = EYE.cool; eyeSt.blink = 1; }
      faceDyn.end = 0;                       // 冷却中的也丢掉，不攒着补眨
    }
    eyeSt.blink *= Math.exp(-dt / EYE.blinkTau);
    if (eyeSt.blink < 0.01) eyeSt.blink = 0;
    var k = target * (1 - EYE.blinkDepth * eyeSt.blink);
    k = clamp(k, 0.30, 1.10);
    // 眯得快、睁开慢：情绪上来时眼皮先动，回去时慢慢松
    var tau = k < eyeSt.cur ? 0.10 : 0.22;
    eyeSt.cur += (k - eyeSt.cur) * (1 - Math.exp(-dt / tau));
    // 腮帮子：拍点和句首各鼓一下（快起慢落），叠在乐句通道之上
    var ct = clamp(0.65 * clamp(mus.beat || 0, 0, 1)
                   + 0.55 * clamp(faceDyn.onset || 0, 0, 1), 0, 1);
    var kc = 1 - Math.exp(-dt / (ct > eyeSt.cheek ? 0.055 : 0.30));
    eyeSt.cheek += (ct - eyeSt.cheek) * kc;
    eyeSt.on = eyeOn() ? 1 : 0;
  }

  // 主程序每帧调两次（origCoreModelUpdate 前后各一次）。基数只在「这一帧第一次」
  // 读：否则第二次会把我们自己写过的值当基数，越乘越小（眼睛越唱越小）。
  function eyeApply(core, ctx) {
    // g61：**这里一帧都不写眼了。**
    // g59 的倍率写法（base × 0.62~1.06）在开心/激动的歌上会把开合长期钉在 1.0，
    // 主程序自己的眨眼间隙、松弛全被盖住——用户看到的是「一直瞪大眼：眼白大、
    // 眼珠显小」（2026-11-10 截图）。这一层唯一的视觉毛病就是这条，而眼睑归主程序
    // 和模型本体，所以整条通道关掉。腮部（ParamCheek）在 applyFace 里，照旧。
    return;
  }

  function eyeManager() {
    try { return (typeof window !== 'undefined') ? window.live2dManager : null; } catch (e) { return null; }
  }
  // g67：眼球回默认值（用户 2026-11-11 要求「把眼球恢复正常值」）。
  // g51~g60 的限幅只能写在 applyFace（beforeModelUpdate）里，而主程序的「随机视线」
  // 在 origCoreModelUpdate 里、并且之后还会再写，所以待机时眼珠照样被推来推去
  // （实测待机 EyeBallX 在 -0.32 ~ +0.27 之间乱跳）。这里挂到主程序自己留的
  // 「渲染前最后姿态覆盖」（_applyTemporaryPoseOverride，origCoreModelUpdate 前后
  // 各调一次），每帧把 ParamEyeBallX/Y 写回**模型默认值**（Blue cat 是 0）。
  // 这是最后一道写，谁先写都会被盖住；待机和唱歌都生效。
  var GAZE_SOURCE = 'nekoMusicGaze';
  function gazeApply(core) {
    if (!core || typeof core.getParameterIndex !== 'function') return;
    // 舞台接管：演出中（body 带 neko-solo-stage）眼球由舞台写，这里不回默认值；退出后自动恢复
    var stageOwns = false;
    try { stageOwns = !!(document.body && document.body.classList.contains('neko-solo-stage')); } catch (e) { stageOwns = false; }
    if (stageOwns) return;
    // 按语义通道写回模型默认值；量程 ±30 的眼球（按度写的模型）同样适用。
    if (!paProfile(core)) return;
    pa().reset(core, 'eyeBallX');
    pa().reset(core, 'eyeBallY');
  }
  function ensureGazeHook() {
    var mgr = eyeManager();
    if (!mgr || typeof mgr.setTemporaryPoseOverride !== 'function') return false;
    var map = mgr._temporaryPoseOverrides;
    if (map && typeof map.has === 'function' && map.has(GAZE_SOURCE)) return true;
    try {
      // priority 35：排在眼睑覆盖（30）之后，数值越大越后写
      mgr.setTemporaryPoseOverride(GAZE_SOURCE, gazeApply, { priority: 35 });
      return true;
    } catch (e) { return false; }
  }
  function ensureEyeHook() {
    var mgr = eyeManager();
    if (!mgr || typeof mgr.setTemporaryPoseOverride !== 'function') return false;
    var map = mgr._temporaryPoseOverrides;
    if (map && typeof map.has === 'function' && map.has('nekoMusicEyes')) return true;
    try {
      // priority 大 = 排在别的覆盖之后（_applyTemporaryPoseOverride 按 priority 升序）
      mgr.setTemporaryPoseOverride('nekoMusicEyes', eyeApply, { priority: 30 });
      return true;
    } catch (e) { return false; }
  }
  function releaseEyeHook() {
    var mgr = eyeManager();
    if (!mgr || typeof mgr.clearTemporaryPoseOverride !== 'function') return;
    try { mgr.clearTemporaryPoseOverride('nekoMusicEyes'); } catch (e) { /* 没登记过 */ }
  }

  // 模型有没有这些参数（按模型缓存一次）。和上面 faceIds 分开探测：这一层
  // 还要用到眉头形、眼开合、眼球、头角度，不是每个模型都齐。
  var emoIds = ['ParamBrowLY', 'ParamBrowRY', 'ParamBrowLAngle', 'ParamBrowRAngle',
                'ParamBrowLForm', 'ParamBrowRForm', 'ParamEyeLSmile', 'ParamEyeRSmile',
                'ParamCheek', 'ParamEyeLOpen', 'ParamEyeROpen', 'ParamEyeBallX',
                'ParamEyeBallY', 'ParamAngleX', 'ParamAngleY', 'ParamAngleZ',
                'ParamMouthForm',
                // g62：第二套表情通道（主程序与口型轨都不写这几个）
                'ParamMouthForm2', 'ParamMouthForm3', 'ParamMouthOpenY23',
                // g63：Param202 已移出——它是「变小」（幼体形态），不许写。
                // g64：贴图特效（脸红/流泪/花花/脸黑）
                'Param57', 'Param64', 'Param58', 'Param56',
                'Param54'];
  var emoCache = { core: null, has: null };
  function emoSupport(core) {
    if (emoCache.core === core) return emoCache.has;
    var has = {}, mask = '';
    for (var i = 0; i < emoIds.length; i++) {
      var ok = false;
      try {
        ok = typeof core.getParameterIndex === 'function'
             && core.getParameterIndex(emoIds[i]) >= 0;
      } catch (e) { ok = false; }
      has[emoIds[i]] = ok;
      mask += ok ? '1' : '0';
    }
    emoCache = { core: core, has: has };
    faceStats.support2 = mask;
    return has;
  }

  // ---- P6：逐句情绪 → 唱歌态脸部参数（与 choreo/expression.py、stage-runtime.js 同一组数值）----
  // 眉/脸颊/嘴形按句驱动；强度分档 weak .6 / mid .85 / strong 1；句间平滑。
  // 不碰眨眼、口型、眼形（g60 禁写）、EYE_SMILE_CAP。
  var SUNG_EXPR = {
    joy:        { browY: .3, browF: .3, browA: 0, mouth: .6, cheek: .3 },
    playful:    { browY: .25, browF: .4, browA: 0, mouth: .55, cheek: .35 },
    tender:     { browY: .1, browF: .2, browA: 0, mouth: .35, cheek: .45 },
    longing:    { browY: .4, browF: -.2, browA: .5, mouth: .05, cheek: .1 },
    sad:        { browY: .5, browF: -.5, browA: .7, mouth: -.4, cheek: 0 },
    proud:      { browY: .2, browF: .4, browA: -.1, mouth: .4, cheek: 0 },
    determined: { browY: -.3, browF: -.3, browA: -.4, mouth: .05, cheek: 0 }
  };
  var sungSt = { on: false, cur: null, t: 0 };
  function sungTier(inten) {
    var x = Number(inten);
    if (!isFinite(x)) return 0.85;
    return x < 0.45 ? 0.6 : (x < 0.70 ? 0.85 : 1);
  }
  // 唱歌态每帧：当前句 → 目标值 → 平滑（变强 0.35s、变弱 0.9s）。无逐句数据时回落到中性 0。
  function sungFace() {
    var L = desk.on ? deskCurrentLine() : null;
    var target = { browY: 0, browF: 0, browA: 0, mouth: 0, cheek: 0 };
    if (L && SUNG_EXPR[L.emotion]) {
      var m = sungTier(L.inten), base = SUNG_EXPR[L.emotion];
      for (var k in target) target[k] = base[k] * m;
    }
    var dt = Math.max(0, (performance.now() - (sungSt.t || performance.now())) / 1000);
    sungSt.t = performance.now();
    if (!sungSt.cur) sungSt.cur = { browY: 0, browF: 0, browA: 0, mouth: 0, cheek: 0 };
    for (var key in target) {
      var rising = Math.abs(target[key]) > Math.abs(sungSt.cur[key]);
      var tau = rising ? 0.35 : 0.9;
      sungSt.cur[key] += (target[key] - sungSt.cur[key]) * (1 - Math.exp(-dt / tau));
    }
    var c = sungSt.cur;
    // 换算到 emo.f 的量纲：抬眉 browIn、压眉 browDn（正=压）、眉角 browOut、脸颊 cheek、嘴形 mouth
    var f = {
      browIn: clamp(c.browY + Math.max(0, c.browF) * 0.5, 0, 0.88),
      browOut: clamp(c.browA, 0, 0.88),
      browDn: clamp(-c.browF, -0.62, 0.62),
      cheek: clamp(c.cheek, 0, 0.9),
      mouth: clamp(c.mouth, -0.62, 0.62),
      eyeSmile: 0, eyeOpen: 0, gazeX: 0, gazeY: 0, headX: 0, headY: 0, headZ: 0,
      smile2: 0, mouthX: 0, angerMk: 0, blush: 0, tears: 0, flower: 0, darkFace: 0
    };
    return { f: f, cur: c };
  }
  function deskCurrentLine() {
    var L = desk.line;
    if (L) return L;
    var P = desk.lines ? desk.lines[desk.li] : null;
    return (P && desk.t >= P.t0 && desk.t - P.t1 <= 2.5) ? P : null;
  }

  function applyEmo(core) {
    if (typeof core.addParameterValueById !== 'function') return 0;
    // P6：唱歌态不再把眉/脸颊/嘴形归零，改按逐句情绪驱动（见 sungFace）。
    // 旧注释「唱歌态表情层一律归零」已被推翻（依据：待办 t_2a91bab4919c，用户要求
    // 人物脸上表情要跟着每句情绪变化，不能全程正常表情）。眨眼、口型、眼形仍不动。
    var sf = lips.active ? sungFace() : null;
    var f = sf ? sf.f : (lips.active ? ZERO_FACE : emo.f), w = 0, g = gestSt.out;
    // 唱歌态：细节表情层（乐句弧/咬字/颤音/特效）仍用 0，只留动作层 g。
    var ex = lips.active ? ZERO_EXTRA : extra;
    emoSupport(core);
    // 叠加：眉毛/笑眼/脸颊/嘴形主程序不强制占着，叠一层不会打架。
    // ch 是语义通道，v 是参考模型单位（见 paProfile 的说明）。
    function add(ch, v) {
      if (!v) return;
      w += chAdd(core, ch, v);
    }
    // 特效类通道：v 是 0..1 的「强度」，直接按模型自己的满量程展开。
    function fx(ch, v) {
      if (!v || !chHas(core, ch)) return;
      w += pa().add(core, ch, v);
    }
    // 读现值→加偏移→钳到模型区间→写回。眼开合/视线/头角度被主程序、物理和
    // 律动同时写，只有「基于当前值」才不会每帧累积把模型甩出去。
    function rel(ch, dv) {
      if (!dv) return;
      // g77：「当前值 + 偏移」只能在模型 update 周期里写（帧末 loadParameters 会还原）；
      // rAF 兜底路径（渲染停着时）写进去不会被还原，逐帧累加会把头推到 ±30 卡死。
      if (!faceInEvent && /^head/.test(ch)) return;
      // 适配层的叠加写就是「读现值 + 偏移 + 钳到模型区间」。
      w += chAdd(core, ch, dv);
    }
    // 眉和嘴形：本帧的情绪值 + 动作层的偏移。动作层的量在 gestTick 里按
    // 关键帧算好，这里只乘 gain，不单独改语义。
    var browIn  = f.browIn  + g.browIn  * GEST.brow;
    var browOut = f.browOut + g.browOut * GEST.brow;
    var browDn  = f.browDn  + g.browDn  * GEST.brow;
    // g62：左右眉差一点。两边永远一模一样才是「像面具」的主因；一侧略高
    // （跟着头转向 + 咬字那一下）就会读成「挑眉 / 疑惑 / 俏皮」。
    var asym = ex.asym + ex.browSyl * 0.55;
    add('browLY', browIn + ex.browSyl + asym);
    add('browRY', browIn + ex.browSyl * 0.6 - asym * 0.35);
    add('browLAngle', browOut);
    add('browRAngle', browOut);
    // 压眉：SDK 标准表里 browDown 的 scale 是 -0.85（负号 = 往下压）。
    add('browLForm', -browDn * 0.85);
    add('browRForm', -browDn * 0.85);
    // g60：眼形（ParamEyeLSmile / ParamEyeRSmile）**一律不再写**。
    // 实测这个模型（Blue cat）的 EyeSmile 会把下眼睑往下拉：眼白变大、虹膜
    // 显小（用户报的「眼珠子非常小」）。而且 addParameterValueById 是叠加写，
    // 我们停手时没人把它复位——上一首歌写进去的 0.13 会一直留在参数里
    // （/hello?ed= 实测：没在唱时 EyeLSmile 还是 0.13、face_writes=0）。
    // 眼睛形状归主程序和模型本体，这一层不再碰。
    // 脸颊：g62 起乘乐句弧（同一句里越到最后越红一点），上限 0.9×0.85。
    add('cheek', f.cheek * 0.9 * (0.76 + 0.30 * ex.arc));
    add('mouthForm', f.mouth + g.mouth * GEST.mouth);
    // ---- g62：第二套表情通道 ----
    // 嘴角（笑/撇嘴）：主程序和口型轨都不写 ParamMouthForm2，所以这条不会被
    // 口型绝对写压掉，是「看得出在笑」的那一条。
    // 参考模型（Blue cat）上这几条的量程是 ±30 / 0..1；换成「占满量程的比例」写，
    // 别的模型有同名效果就按它自己的量程展开，没有就跳过。
    fx('smile2', ex.smile2 / 30);
    // 嘴的左右位移：跟着头转。
    fx('mouthX', ex.mouthX / 30);
    // 露牙：只在嘴真的张开时（extraTick 里的 teethGate 已经把闭口压成 0）。
    fx('teeth', ex.teeth);
    // g63：Param202「变小」不写（见上面 EMO 通道表的说明）。
    // 生气符号特效：只有 anger 原型 + 情绪够重才长，0.55s 长出来，不会突然跳出。
    fx('angerMark', ex.mk / 30);
    // ---- g64：贴图特效（大幅度的「看得见」） ----
    // 这四个都是这个模型自带的画师参数，满量程时一眼就能认出来；形变类参数
    // （眉/嘴角/脸颊）在整身视角下只有几个像素，靠它们撑不起「大表情」。
    // 全部由 act 门控：间奏/停歌时 f.* 归零，extraTick 再把实际写入量收干，
    // 平滑收尾 0.7~0.9s，不会留在脸上。
    // Param57 脸红（0..30）：写 22 就是很红，再高会糊成一块。
    // 脸红贴图：参考模型满量程 30，写 22 ≈ 73%。没有专门贴图的模型落到 ParamCheek 上。
    if (chHas(core, 'blushFx')) fx('blushFx', ex.blush * 22 / 30);
    else fx('cheek', ex.blush * 0.5);
    // Param64 流泪（0..1）：模型自带的两行泪。
    fx('tears', ex.tears);
    // 「花花」（头发上的花变红）：只认显示名正好叫「花花」的参数；别的模型同编号是别的东西。
    fx('flower', ex.flower);
    // Param56 脸黑（0..30）：anger 专用，和生气符号（Param54）配套。
    fx('darkFace', ex.darkFace * 16 / 30);
    // 眼开合（ParamEyeLOpen/ROpen）这里**不写**。主程序在 coreModel.update 的
    // 拦截器里（libs/index.min.js：emit("beforeModelUpdate") → n.update()）用
    // _updateEyeBlink 每帧写绝对值，那个时机在我们之后；而它又会因为
    // _isEyeDrivenByMotion / _suspendEyeBlinkOverride 逐帧翻转而偶尔跳过。
    // 于是我们的偏移「大多数帧被盖掉、少数帧突然生效」，眼睛就会不受控地闭合又
    // 弹开——用户看到的「眼睛抽动」就是这么来的。眨眼归主程序，我们只做眼形
    // （EyeL/R Smile）。
    // g66：用户要求眼球回默认值，情绪层的视线偏移（原来写 ParamEyeBallX/Y）
    // 也一并停掉——眼珠只由 applyGazeLimit 每帧写回默认值，别的通道一律不碰。
    // 头的朝向：情绪层的常态偏移 + 动作层的摆幅（动作层在这里给「度」，
    // 直接加在 EMO.maxHead 那一项之外，幅度才够看见）。
    // g69：桌面档下情绪偏移是次要层（× kS，主动作在播时压到 30%），再加逐句视线
    var ks = desk.on ? desk.kS : 1;
    rel('headX', f.headX * EMO.maxHead * ks + g.headX * GEST.head + desk.gx);
    rel('headY', f.headY * EMO.maxHead * ks + g.headY * GEST.head + desk.gy);
    rel('headZ', f.headZ * EMO.maxHead * ks + g.headZ * GEST.head);
    if (w) emo.writes += 1;
    return w;
  }

  // ---- 情绪 → 身体（g47） ----
  // 唱歌时身体的角度由 MusicGroove 绝对驱动（下面那段）。这一层**不换驱动源**，
  // 只在它之后加一层偏移：律动给「怎么摆」，情绪给「摆得什么味道」。
  //   · 正负情绪（V）→ 上身立起 / 前倾下沉；
  //   · 唤醒（A）→ 慢摆的幅度、呼吸的深浅、头发的滞后；
  //   · 掌控（D）→ 前倾程度（dominance 低时塌一点）。
  // 写入点在 applyFace 里（beforeModelUpdate 队尾），也就是律动在
  // grooveReassertListener 里重写完、渲染之前，所以是「叠在律动上」，
  // 不会和它抢绝对值；同一帧只叠一次（事件与 rAF 兜底都调也只算一次）。
  // 不唱歌时一次都不写，整段交回待机。
  // 头发（ParamHairFront/Side/Back）没有任何别的写入者（物理会算，但那是本帧
  // 早些时候的独立结果），是这层唯一独占的参数；主程序只写 Angle*/BodyAngle*。
  var BODY = {
    lean: 3.8,     // 情绪造成的上身倾角（度）：g62 2.6 → 3.8
    sway: 3.0,     // 慢摆幅度（度）：g62 2.2 → 3.0（律动在 BodyAngleZ 上只用 ±1.6°，还有余量）
    breath: 0.18,  // 呼吸深度偏移（参数单位）：g62 0.10 → 0.18
    hair: 0.60,    // 头发滞后强度（乘在参数自身的量纲上）：g62 0.42 → 0.60
    tau: 0.32      // 情绪落到身体的时间常数（秒）
  };
  var bodySt = { x: 0, y: 0, z: 0, br: 0, phase: 0, px: 0, at: -2, t: 0, w: 0, on: 0 };

  var bodyIds = ['ParamBodyAngleX', 'ParamBodyAngleY', 'ParamBodyAngleZ',
                 'ParamBreath', 'ParamHairFront', 'ParamHairSide', 'ParamHairBack'];
  var bodyCache = { core: null, has: null };
  function bodySupport(core) {
    if (bodyCache.core === core) return bodyCache.has;
    var has = {}, mask = '';
    for (var i = 0; i < bodyIds.length; i++) {
      var ok = false;
      try {
        ok = typeof core.getParameterIndex === 'function'
             && core.getParameterIndex(bodyIds[i]) >= 0;
      } catch (e) { ok = false; }
      has[bodyIds[i]] = ok;
      mask += ok ? '1' : '0';
    }
    bodyCache = { core: core, has: has };
    faceStats.support3 = mask;
    return has;
  }

  function applyBody(core) {
    if (typeof core.addParameterValueById !== 'function') return 0;
    if (bodySt.at === frameSeq) return 0;      // 同一帧只叠一次
    bodySt.at = frameSeq;
    var now = performance.now();
    var dt = bodySt.t ? clamp((now - bodySt.t) / 1000, 0.002, 0.10) : 0.016;
    bodySt.t = now;
    bodySupport(core);

    var v = emo.v, a = emo.a, d = emo.d;
    var ar = Math.max(0, a), sub = Math.max(0, -d);
    // 在不在唱：干声响度/口型（emo.pres）、律动权重（blend）、口型轨（lips.active）
    // 三个任一个成立就叠。都不成立 → 不写，身体整段交回待机/律动自身。
    var p = 0;
    if (emo.pres > 0.03 || blend > 0.15 || lips.active) {
      p = clamp(0.35 + 0.65 * Math.max(emo.pres, blend, lips.active ? 0.5 : 0), 0, 1);
    }
    if (!p) {
      bodySt.x = bodySt.y = bodySt.z = bodySt.br = 0;
      bodySt.px = 0; bodySt.on = 0; bodySt.w = 0;
      return 0;
    }

    // 慢摆：频率跟着唤醒走（0.085~0.16Hz），不会和鼓点抢拍，只是让身体「活着」
    bodySt.phase += 2 * Math.PI * (0.085 + 0.075 * ar) * dt;
    var sw = Math.sin(bodySt.phase);
    var swD = Math.sin(bodySt.phase - 0.9);    // 同一个摆，滞后约 0.15s，给头发用

    var tx = sw * BODY.sway * (0.35 + 0.65 * ar) * p;
    var ty = (0.55 * v + 0.30 * (ar - 0.45)) * BODY.lean * p;
    var tz = Math.sin(bodySt.phase + 0.7) * BODY.sway * 0.75 * (0.30 + 0.70 * ar) * p;
    var tb = (0.30 * ar - 0.22 * sub + 0.06 * v) * BODY.breath * p;

    var k = 1 - Math.exp(-dt / BODY.tau);
    bodySt.x += (tx - bodySt.x) * k;
    bodySt.y += (ty - bodySt.y) * k;
    bodySt.z += (tz - bodySt.z) * k;
    bodySt.br += (tb - bodySt.br) * k;

    // 头发：身体角速度的滞后（甩出去）+ 慢摆的延迟版。角速度限在 ±260°/s，
    // 免得换段时一帧大跳把头发甩到极限。
    var vel = clamp((bodySt.x - bodySt.px) / Math.max(dt, 1 / 240), -260, 260);
    bodySt.px = bodySt.x;
    var lag = clamp(-vel * 0.0022, -1, 1);
    var hs = swD * (0.35 + 0.65 * ar) * p;
    var hf = (lag * 0.55 + hs * 0.45) * BODY.hair;
    var hsd = (lag * 0.85 + hs) * BODY.hair;
    var hb = (lag * 1.00 + hs * 1.10) * BODY.hair;

    var n = 0;
    var ks = desk.on ? desk.kS : 1;   // g69：身体慢摆/情绪倾角是次要层
    function put(ch, val) {
      if (!isFinite(val) || Math.abs(val) < 0.004) return;
      // 适配层按模型区间换算并钳位，不会把参数甩出去。
      n += chAdd(core, ch, val);
    }
    put('bodyX', bodySt.x * ks + gestSt.out.bodyX * GEST.body);
    put('bodyY', bodySt.y * ks + gestSt.out.bodyY * GEST.body);
    put('bodyZ', bodySt.z * ks + gestSt.out.bodyZ * GEST.body);
    put('breath', bodySt.br);
    // 头发：没有专门摇摆参数的模型（大多数），物理会跟着身体自己动，这里跳过即可。
    put('hairFront', hf);
    put('hairSide', hsd);
    put('hairBack', hb);
    bodySt.w = n;
    bodySt.on = n > 0 ? 1 : 0;
    return n;
  }

  // 诊断用：这一帧叠到身体上的偏移（度 / 参数单位）
  // ---- 眼珠归零（g51 → g66） ----
  // 主程序在不跟踪鼠标时（用户设置里 mouseTrackingEnabled=false）每帧跑
  // 「随机视线」：live2d-model.js:1984 写
  //   ParamEyeBallX = _lookAtCurrentX / 30      // _lookAtCurrentX ∈ ±15 度
  //   ParamEyeBallY = _lookAtCurrentY / 30      // ±10 度
  // 这个 /30 是按「眼球参数区间 ±30」写的，可这个模型（Blue cat）的
  // ParamEyeBallX/Y 是 **±1、默认 0**（/hello?ed= 实测）。于是同一份代码在
  // ±30 的模型上是 1.7%（几乎看不见的小瞟），在这里就是区间的 30%~50%——
  // 眼珠被推到眼白边上，看起来「眼神发直 / 眼珠位置不对」。
  //
  // g51~g60 一直是「限幅」（g60 待机收到 ±0.06），但主程序的随机视线每几秒换
  // 目标、幅度几乎都超上限，饱和后眼珠就**长期贴着边界**（/hello?ed= 实测待机
  // EyeBallX -0.06 / EyeBallY +0.06），仍然是「歪着不动」。
  // g66：用户要求「把眼球恢复正常值」——不再限幅，每帧直接写回模型自己的默认值
  // （读不到默认就写 0）。眼珠回到正中，这个模型的默认值就是 0。
  //
  // 只在「归一化区间」（|min|/|max| ≤ 2，也就是 ±1 这一类）上写：
  // ±30 的按度写的模型一概不碰。
  function applyGazeLimit(core) {
    var n = 0;
    // g75：舞台接管时（body 带 neko-solo-stage）眼球归舞台写，这里不回默认值；
    // 否则舞台写的视线每帧被拉回正中，眼珠永远不左右上下看。退出舞台自动恢复。
    try { if (document.body && document.body.classList.contains('neko-solo-stage')) return 0; } catch (eS) { }
    if (!paProfile(core)) return 0;
    var chs = ['eyeBallX', 'eyeBallY'];
    for (var i = 0; i < 2; i++) {
      var m = chMap(core, chs[i]);
      if (!m) continue;
      var cur = NaN;
      try { cur = Number(core.getParameterValueById(m.id)); } catch (e) { cur = NaN; }
      // 允许误差按量程算：±30 的眼球 0.0005 太严，会每帧都写
      if (!isFinite(cur) || Math.abs(cur - m.def) < 0.0005 * Math.max(1, m.hi - m.lo)) continue;
      n += pa().reset(core, chs[i]);
    }
    return n;
  }

  function bodyTag() {
    if (!bodySt.on) return '0';
    return bodySt.x.toFixed(1) + ',' + bodySt.y.toFixed(1) + ','
         + bodySt.z.toFixed(1) + ',' + bodySt.br.toFixed(2);
  }

  // ---- 眼睛探针（只读，g48 → g50：自愈 + 绝对值 + 主程序眨眼配置） ----
  // 「人物抽动 / 眼睛数值不对」只能量在**渲染用的最终值**上：主程序的口型/眨眼/
  // 常驻表情是在 coreModel.update 的拦截器里写的（libs/index.min.js：
  // emit("beforeModelUpdate") → n.update()），也就是在我们 beforeModelUpdate
  // 的写入之后。所以这里包一层 coreModel.update：原样调用、原样返回，返回后只读
  // 一遍参数。全程不写任何一个参数。
  //
  // g48 的版本有两个毛病，g50 修掉：
  //   1) 主程序会自己重装 coreModel.update（live2d-model.js:3330 新建闭包，
  //      2946/590 还原成 _origCoreModelUpdate），我们的包会被挤掉，而
  //      probe.core === core 的判断让 probeInstall 之后再也不重装——
  //      于是 hooked 一直是 1、却一帧样本都没有（/hello 里就是 "1:"）。
  //      现在每帧比对 core.update === probe.wrap，被换掉就重新包上去。
  //   2) 只有「相邻帧最大跳变」，看不出当前值本身是不是正常。现在同时上报
  //      眼开合/笑眼/眼球的 当前值[区间]默认值，以及主程序眨眼器的
  //      open/closed/default（heavy 眼皮是不是它把 openValue 当成默认值写的）
  //      和常驻表情名单。
  var probe = { core: null, wrap: null, prev: null, acc: null, n: 0, at: 0, out: '',
                hooked: 0, calls: 0, last: null, dump: '', at2: 0, frame: -1,
                lo: null, hi: null, mm: '' };
  var PROBE_IDS = ['ParamEyeLOpen', 'ParamEyeROpen', 'ParamEyeLSmile', 'ParamEyeBallX',
                   'ParamAngleX', 'ParamAngleY', 'ParamAngleZ', 'ParamBodyAngleX',
                   'ParamBodyAngleY', 'ParamBreath',
                   // g55：嘴的两个参数。抽动如果出在嘴上（包络毛刺、时钟跳），
                   // 这里会直接看到相邻帧的跳变；前 10 项的顺序不动，
                   // mm 极值仍然只报前 6 项。
                   'ParamMouthOpenY', 'ParamMouthForm',
                   // g63：眼球 Y 单独补进来。前面 10 项里有 EyeBallX 却没有 Y，
                   // 而「虹膜被上眼睑压住、看着变小」最可能的凶手就是 Y 挂边。
                   // 追加在末尾，mm 只看前 6 项，不受影响。
                   'ParamEyeBallY'];
  function probeInstall(core) {
    if (!core || typeof core.update !== 'function') return;
    // 还在链上（主程序没动它）就什么都不做，开销就是一次比较。
    if (probe.core === core && core.update === probe.wrap) return;
    var orig = core.update;
    probe.core = core;
    probe.prev = null;
    probe.acc = null;
    var wrap = function () {
      var r = orig.apply(this, arguments);
      probe.calls += 1;
      probeSample(core);
      return r;
    };
    wrap.__mgProbe = 1;
    try {
      core.update = wrap;
      probe.wrap = wrap;
      probe.hooked = 1;
    } catch (e) { probe.hooked = 0; }
  }
  function probeSample(core) {
    // 同一个渲染帧只采一次（主程序把我们包在它里面时，外层还会再调一次）。
    if (probe.frame === frameSeq) return;
    probe.frame = frameSeq;
    var cur = [], i, v;
    for (i = 0; i < PROBE_IDS.length; i++) {
      try { v = Number(core.getParameterValueById(PROBE_IDS[i])); } catch (e) { v = NaN; }
      cur.push(isFinite(v) ? v : 0);
    }
    probe.last = cur;
    if (probe.prev) {
      if (!probe.acc) {
        probe.acc = [];
        for (i = 0; i < cur.length; i++) probe.acc.push(0);
      }
      for (i = 0; i < cur.length; i++) {
        var d = Math.abs(cur[i] - probe.prev[i]);
        if (d > probe.acc[i]) probe.acc[i] = d;
      }
      probe.n += 1;
    }
    // g52：窗口内的最小值/最大值。只看「最大跳变」分不清「一直在动」和
    // 「大多数帧不动、偶尔弹一下」；有了极值就能看出眼开合是不是在 0/1 之间抽。
    if (!probe.lo) {
      probe.lo = cur.slice(0);
      probe.hi = cur.slice(0);
    } else {
      for (i = 0; i < cur.length; i++) {
        if (cur[i] < probe.lo[i]) probe.lo[i] = cur[i];
        if (cur[i] > probe.hi[i]) probe.hi[i] = cur[i];
      }
    }
    probe.prev = cur;
    var now = performance.now();
    if (now - probe.at >= 2000) {
      // 每 2 秒输出一次「这两秒内相邻帧的最大跳变」：眼开合、眼开合、笑眼、
      // 眼球X、头X/Y/Z、身体X/Y、呼吸（顺序同 PROBE_IDS），@ 后面是帧数。
      if (probe.acc) {
        probe.out = probe.acc.map(function (x) { return x.toFixed(2); }).join(',')
                  + '@' + probe.n;
      }
      probe.acc = null; probe.n = 0; probe.at = now;
      try {
        // g53：多带眼球X和头的Y轴。眼珠「变小」时，如果 EyeBallY 一直挂在
        // 边界上就是视线跑到下眼睑下面去了；如果 EyeBallY 老实、只有
        // AngleY 在大幅摆，那就是点头把头压扁的。
        var mmParts = [];
        for (var mi = 0; mi < 6; mi++) {
          mmParts.push(fx(probe.lo[mi]) + '-' + fx(probe.hi[mi]));
        }
        probe.mm = '|mm' + mmParts.join(',');
      } catch (e6) { probe.mm = ''; }
      probe.lo = null; probe.hi = null;
      probe.dump = eyeDump(core);
    }
  }
  function fx(v) { return isFinite(v) ? String(Math.round(v * 100) / 100) : '?'; }
  // 当前值[最小,最大]默认值 —— 用来看「这个数本身正不正常」，不只看跳变。
  function eyeDump(core) {
    var ids = ['ParamEyeLOpen', 'ParamEyeROpen', 'ParamEyeLSmile', 'ParamEyeBallX',
               'ParamEyeBallY',
               // g55：嘴和头的绝对值。抽动时「当前值本身是多少」比跳变更能说明
               // 是哪个环节在写（比如嘴角一直被压在某一端）。
               'ParamMouthOpenY', 'ParamMouthForm',
               'ParamAngleX', 'ParamAngleY', 'ParamAngleZ'];
    var parts = [];
    for (var i = 0; i < ids.length; i++) {
      try {
        var idx = core.getParameterIndex(ids[i]);
        if (idx < 0) { parts.push(ids[i].slice(5) + ':-'); continue; }
        var cur = Number(core.getParameterValueById(ids[i]));
        var lo = Number(core.getParameterMinimumValue(idx));
        var hi = Number(core.getParameterMaximumValue(idx));
        var def = NaN;
        try { def = Number(core.getParameterDefaultValue(idx)); } catch (e1) {}
        parts.push(ids[i].slice(5) + ':' + fx(cur) + '[' + fx(lo) + ',' + fx(hi) + ']d'
                   + fx(def));
      } catch (e2) { parts.push(ids[i].slice(6) + ':!'); }
    }
    var mg = null;
    try { mg = (typeof window !== 'undefined') ? window.live2dManager : null; } catch (e3) {}
    if (!mg) return parts.join(',') + '|no-mg';
    var flags = 'ab' + (mg._autoEyeBlinkEnabled ? 1 : 0)
              + 'sb' + (mg._suspendEyeBlinkOverride ? 1 : 0)
              + 'em' + (mg._isEyeDrivenByMotion ? 1 : 0)
              + 'st' + (Number(mg._eyeBlinkState) || 0);
    // 主程序的眨眼器：open 是它每帧写的「睁眼值」，closed 是闭眼值。
    // 如果 open 本身就远小于 max，那「眼皮很重」是它按默认值写的，不是我们。
    var bk = '';
    try {
      var bp = mg._eyeBlinkParams;
      if (bp && bp.length) {
        var b0 = bp[0];
        bk = '|bk' + String(b0.id || '').slice(5) + 'o' + fx(b0.openValue)
           + 'c' + fx(b0.closedValue) + 'd' + fx(b0.defaultValue);
      } else {
        bk = '|bk-none';
      }
    } catch (e4) {}
    var px = '';
    try {
      var pe = mg.persistentExpressionParamsByName;
      var names = pe ? Object.keys(pe) : [];
      px = '|px' + (names.length ? names.join('+').slice(0, 40) : '-');
    } catch (e5) {}
    // g63：只读诊断——Param202..206 在 cdi3 里全叫「变小」（整只角色切幼体形态），
    // 我们 g62 误当成猫耳写过，已删。这里把这五个的**当前值**读出来：
    // 如果下次重启后它们不是 0，说明是主程序的表情/动作（exp3/motion3）在写，
    // 那才是「眼球看着变小」的真凶——本插件一个字都不写它们。
    var bd = '';
    try {
      var bi = ['Param202', 'Param203', 'Param204', 'Param205', 'Param206'];
      var bt = [];
      for (var bi2 = 0; bi2 < bi.length; bi2++) {
        var bidx = core.getParameterIndex(bi[bi2]);
        if (bidx < 0) { bt.push('-'); continue; }
        bt.push(fx(Number(core.getParameterValueById(bi[bi2]))));
      }
      bd = '|bd' + bt.join(',');
    } catch (e7) { bd = '|bd!'; }
    return parts.join(',') + '|' + flags + bk + px + bd + (probe.mm || '');
  }
  function probeTag() {
    // g55：口型的统计一起带上——时钟前跳/回跳次数、张口/嘴形被限速的次数、
    // 当前响度和唱到的秒数。「抽动」到底是时钟跳、包络毛刺还是动作层，看这一串。
    var lp = '';
    try {
      lp = '|lp' + (lips.clockFwd || 0) + '/' + (lips.clockBack || 0)
         + '/' + (lips.openClamp || 0) + '/' + (lips.formClamp || 0)
         + '/' + (lips.loudF == null ? '-' : Number(lips.loudF).toFixed(2))
         + '/' + Number(lips.t || 0).toFixed(1);
    } catch (e) { lp = ''; }
    // g59：眼睑开合/腮帮子。格式 eo<倍率>/<覆盖写入次数>/<在不在管>/<眨眼>/<鼓脸>/<支持掩码>。
    // 次数一直是 0 = 宿主没有那个「渲染前最后姿态覆盖」注入点，覆盖没装上；
    // 倍率一直 1.00 = 没在唱（停唱时我们一帧都不写）。
    var eo = '';
    try {
      eo = '|eo' + Number(eyeSt.cur).toFixed(2) + '/' + (eyeSt.n || 0)
         + '/' + (eyeSt.on || 0) + '/' + Number(eyeSt.blink).toFixed(2)
         + '/' + Number(eyeSt.cheek).toFixed(2) + '/' + (eyeSt.mask || '----');
    } catch (e) { eo = ''; }
    // g64：特效参数的**实际当前值**（脸红57/流泪64/花花58/脸黑56）。
    // 和 s.fx（我们的意图值）对得上，才能证明写入真的活过了 loadParameters；
    // 如果 fe 一直是 0 而 fx 不是 0，就说明写入被主程序的参数重置吃掉了。
    var fe = '';
    try {
      var c0 = probe.core;
      if (c0 && typeof c0.getParameterIndex === 'function') {
        var fi = ['Param57', 'Param64', 'Param58', 'Param56'], fo = [];
        for (var fj = 0; fj < fi.length; fj++) {
          var fidx = c0.getParameterIndex(fi[fj]);
          fo.push(fidx < 0 ? '-' : fx(Number(c0.getParameterValueById(fi[fj]))));
        }
        fe = '|fe' + fo.join(',');
      }
    } catch (e9) { fe = ''; }
    return (probe.hooked ? '1' : '0') + (probe.calls ? '+' : '-') + ':'
         + (probe.out || '') + lp + eo + fe;
  }
  function probeTag2() { return probe.dump || ''; }

  // ---- 唱歌时的「全身」：交给「为她所听」的律动引擎 ----
  // g39~g41 自己写了一套「干净人声驱动的全身」：慢摆 + 每字点头 + 歪头换边。
  // 不管怎么加过渡，它和 MusicGroove 是**两个驱动源**——只要 blend 在阈值上下
  // 抖一下，或者两条链交替接手，ParamBodyAngleY 就会一上一下地跳（用户看到的
  // 「身体上下抽动」）。而且人声包络本来就带字的起伏，直接驱动身体必然一顿一顿。
  //
  // g42 按「为她所听」的分法把信号源拆开：
  //   · 身体（头/上身/呼吸）只由 MusicGroove 驱动，喂的是**成品混音**的频谱
  //     （主窗口自己播的那条 <audio>，带伴奏）——鼓点、段落、力度都在里面；
  //   · 嘴仍然只由干净人声的口型轨驱动（上面那条链），和伴奏无关。
  // 两个驱动源各管各的参数，物理上不可能再互相拽。没有律动数据时身体一次
  // 都不写，整段交回待机；绝不退回「人声直接摆身体」。
  // g47 在这一层之上加了「情绪偏移」（见上面的 applyBody）：它不产生新的驱动源，
  // 只在律动写完的角度上加一个几度以内的偏移，信号来自情绪层而不是人声包络。

  // ---- 连音（legato）与收口 ----
  // 以前每个字都在自己的时长里做一次「张开→收回 0.35」，字与字之间像在嚼东西。
  // 唱歌时一口气里的字是连着唱的：嘴一直开着，只在元音之间换形状。
  //   · 两个字之间的空隙 ≤ LIPS_LEGATO：连音，嘴一直张着；
  //   · 空隙 ≤ LIPS_BREATH：换气口，嘴松到 45%，不整个合上；
  //   · 再长（间奏、下一句隔很久）：才真正收口。
  var LIPS_LEGATO = 0.28;
  var LIPS_BREATH = 0.9;
  var LIPS_BLEND = 0.09;    // 元音之间换形状的过渡窗口（秒）
  var LIPS_ATTACK = 0.030;  // 张口时间常数
  var LIPS_RELEASE = 0.110; // 合口时间常数（比张口慢，嘴不会「啪」地闭上）
  var LIPS_FORM_TAU = 0.070; // 嘴形（圆/扁）的平滑
  // 句尾换气：LRC 行与行常常首尾相接，时间轴上根本没有空隙。
  // 口型轨会在每行最后一个字上打标记，桥按标记做一次短暂收口。
  var LIPS_BREATH_BACK = 0.18;  // 句尾前多久开始收
  var LIPS_BREATH_FWD = 0.20;   // 句尾后多久重新张开
  var LIPS_BREATH_HOLD = 0.05;  // 完全合上的保持段（句尾前后各 50ms）
  // g55 防抽动：真唱出来的嘴不会在一帧里跳半个量程。正常吐字最快也就
  // 20 上下每秒，只有时钟跳变 / 包络毛刺会超；超了就按这个速度限住。
  var LIPS_OPEN_RATE = 22;      // 张口度最大变化速度（/秒）
  var LIPS_FORM_RATE = 14;      // 嘴形最大变化速度（/秒）
  // 干声响度先过一个起音快、收音慢的跟随器。包络是 50ms 一帧的，
  // 直接拿原值乘张口度，伴奏漏进干声的鼓点会让嘴跟着一下一下地弹。
  var LIPS_LOUD_ATTACK = 0.045;
  var LIPS_LOUD_RELEASE = 0.120;

  function smoothstep(lo, hi, v) {
    var t = clamp((v - lo) / Math.max(hi - lo, 1e-9), 0, 1);
    return t * t * (3 - 2 * t);
  }

  function lipsFindPort(from) {
    var i = from || 0;
    if (i >= LIPS_PORTS.length) return;
    var port = LIPS_PORTS[i];
    fetch('http://127.0.0.1:' + port + '/hello', { cache: 'no-store' })
      .then(function (r) { return r.json(); })
      .then(function (st) {
        if (!st || !st.ok) { lipsFindPort(i + 1); return; }
        LIPS_BASE = 'http://127.0.0.1:' + port;
        lips.port = port;
        lips.seq = -1;              // 换端口 = 换实例，重新拉整条口型轨
        lipsPoll();
      })
      .catch(function () { lipsFindPort(i + 1); });
  }

  function lipsCueIndex(cues, t) {
    var lo = 0, hi = cues.length - 1, idx = -1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (cues[mid][2] <= t) { idx = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    return idx;
  }

  /**
   * 这一个 cue 该用哪个韵母。rest 是「占时值但不发音」的字——长音符 ー、
   * 促音 っ、重复记号 々、以及拼音表里没有的汉字。旧版口型轨里它们会直接
   * 闭嘴；这里改成沿用最近一个真元音（行首就借后面第一个），
   * 真正的停顿由 cue 之间的空隙来表达，而不是靠这些字。
   */
  function lipsVisemeAt(cues, idx) {
    for (var k = idx; k >= 0 && k > idx - 8; k--) {
      if (cues[k][1] !== 'rest') return cues[k][1];
    }
    for (var j = idx + 1; j < cues.length && j < idx + 9; j++) {
      if (cues[j][1] !== 'rest') return cues[j][1];
    }
    return 'A';
  }

  /**
   * 句尾换气系数（1 = 正常张开，0 = 完全合上）。
   * 口型轨 v4 起每行最后一个 cue 带标记（第 5 位 = 1），在它的结束时刻
   * 做一个约 0.48 秒的收口：前后各 50 毫秒完全合上，中间是平滑的斜坡。
   * 没有标记的旧口型轨返回 1，行为不变。
   */
  function lipsBreathV(d) {
    var span = d < 0 ? LIPS_BREATH_BACK : LIPS_BREATH_FWD;
    var ad = Math.abs(d);
    if (ad <= LIPS_BREATH_HOLD) return 0;
    var u = clamp((ad - LIPS_BREATH_HOLD) / Math.max(span - LIPS_BREATH_HOLD, 1e-6),
                  0, 1);
    return u * u * (3 - 2 * u);
  }

  function lipsBreathFactor(cues, t) {
    var idx = lipsCueIndex(cues, t);
    if (idx < 0) return 1;
    var best = 1;
    // 从当前 cue 往回找：句尾后面可能还跟着很短的 cue（一个字不到 0.1 秒），
    // 只查 idx-1 会漏掉真正带标记的那个。往回扫到「已经过去超过 FWD」为止。
    for (var k = idx; k >= 0 && k > idx - 10; k--) {
      if (cues[k][3] < t - LIPS_BREATH_FWD) break;
      if (!cues[k][4]) continue;
      var d = t - cues[k][3];
      if (d <= -LIPS_BREATH_BACK || d >= LIPS_BREATH_FWD) continue;
      var f = lipsBreathV(d);
      if (f < best) best = f;
    }
    // 收口从句尾前 0.18 秒就开始了，那时带标记的那个字可能还没开始唱
    // （短句的最后一个字甚至不到 0.18 秒），所以也要往后看几个 cue。
    for (var j = idx + 1; j < cues.length && j <= idx + 8; j++) {
      if (cues[j][2] > t + LIPS_BREATH_BACK) break;
      if (!cues[j][4]) continue;
      var d2 = t - cues[j][3];
      if (d2 <= -LIPS_BREATH_BACK || d2 >= LIPS_BREATH_FWD) continue;
      var f2 = lipsBreathV(d2);
      if (f2 < best) best = f2;
    }
    return best;
  }

  /**
   * 每个字唱得多快：前后各三个字里「字与字开始时间的间隔」的中位数。
   * 0 = 慢（≥0.45s 一个字），1 = 快（≤0.18s 一个字）。
   * 歌换了才重算一次——唱快了嘴要跟得上，唱慢了可以更从容。
   */
  function lipsRates(cues) {
    var n = cues ? cues.length : 0;
    var out = new Array(n);
    for (var i = 0; i < n; i++) {
      var gaps = [];
      for (var j = Math.max(1, i - 3); j <= Math.min(n - 1, i + 3); j++) {
        var d = cues[j][2] - cues[j - 1][2];
        if (d > 0.02 && d < 1.5) gaps.push(d);
      }
      gaps.sort(function (a, b) { return a - b; });
      var ioi = gaps.length ? gaps[gaps.length >> 1] : 0.35;
      out[i] = clamp((0.45 - ioi) / 0.27, 0, 1);
    }
    return out;
  }

  /**
   * v12：cue 第 7 位的音内相位串（``声母:占比:韵尾:占比``，缺的写 ``-``）。
   * 「妈」的 b/p/m 闭唇、「天」的 n 收口，都靠这两段放进同一个字里。
   * 解析结果挂在 cue 自己身上（一首歌只解析一次）。
   */
  function cuePhases(cue) {
    if (cue.__ph !== undefined) return cue.__ph;
    var ph = null;
    try {
      if (cue.length > 6 && typeof cue[6] === 'string' && cue[6]) {
        var p = cue[6].split(':');
        if (p.length === 4) {
          var of = Number(p[1]), cf = Number(p[3]);
          ph = {
            o: p[0] === '-' ? '' : p[0],
            of: isFinite(of) ? clamp(of, 0, 0.35) : 0,
            c: p[2] === '-' ? '' : p[2],
            cf: isFinite(cf) ? clamp(cf, 0, 0.35) : 0
          };
          if (!ph.o || LIPS_OPEN[ph.o] == null || ph.o === 'rest') { ph.o = ''; ph.of = 0; }
          if (!ph.c || LIPS_OPEN[ph.c] == null || ph.c === 'rest') { ph.c = ''; ph.cf = 0; }
          if (!ph.o && !ph.c) ph = null;
        }
      }
    } catch (e) { ph = null; }
    cue.__ph = ph;
    return ph;
  }

  /**
   * 唱到 t 时嘴的目标状态。核心是「一口气里的字不闭嘴」：
   * 字内保持这个元音该有的开口度（只有 ±6% 的轻微起伏，长音不会像定格表情），
   * 字尾前 75ms 才开始换到下一个元音的嘴形；只有两个字之间真的隔了一口气，
   * 才把嘴松开——而且换气口只松到 45%，不回到 0。
   */
  function lipsCuesAt(cues, t) {
    var idx = lipsCueIndex(cues, t);
    if (idx < 0) return { open: 0, form: 0, viseme: 'rest', stress: 0.5, rate: 0.5, hold: false };
    var cue = cues[idx];
    var next = cues[idx + 1] || null;
    // 长音：一个字撑 0.38s 以上（v10 的干声切段之后长音就是「一个字一直唱」，
    // 不再被均分给后面的字）。长音不是一串重复的起伏，而是「撑住、微微收尾」。
    var dur = Math.max(0, Number(cue[3]) - Number(cue[2]));
    var holdNote = dur >= 0.38 || String(cue[0] || '') === 'ー';
    var vis = lipsVisemeAt(cues, idx);
    var a = LIPS_OPEN[vis] == null ? 0.8 : LIPS_OPEN[vis];
    var f = LIPS_FORM[vis] == null ? 0 : LIPS_FORM[vis];
    var nVis = next ? lipsVisemeAt(cues, idx + 1) : 'A';
    var na = LIPS_OPEN[nVis] == null ? 0.8 : LIPS_OPEN[nVis];
    var nf = LIPS_FORM[nVis] == null ? 0 : LIPS_FORM[nVis];
    var gap = next ? next[2] - cue[3] : Infinity;
    var cont = !!next && gap <= 0.02;   // 紧挨着唱（只是取整误差），可以提前换口型
    // 重音（v7 起 cue[5]）：重音字的换形更快，轻音慢一点，看起来才有语气。
    var st = cue.length > 5 ? Number(cue[5]) : 0.5;
    if (!isFinite(st)) st = 0.5;
    // 这一段的唱速（0 慢 / 1 快）：换形窗口跟着速度走，快歌不吃字。
    var rt = lips.rate && lips.rate[idx] != null ? lips.rate[idx] : 0.5;
    var blend = LIPS_BLEND * (1.55 - 0.95 * st) * (1.15 - 0.45 * rt);
    // v12 音内相位：这个字有没有声母/韵尾，以及它们各自的张口/嘴形。
    // 收势口型同时决定「这个字唱完时嘴多大」——s0 必须先按它算，
    // 否则字尾收口、下一秒又按元音重新张开，字与字之间会跳一格。
    var ph = cuePhases(cue);
    var hasCoda = !!(ph && ph.c && ph.cf > 0);
    // 短音来不及真的「闭唇→满开→收口」：真实 LRC 里偶尔一整句被挤进半秒
    // （比如 12 个字 0.64s，每个字 53ms），硬做相位就是每秒抖十几次的抽动，
    // 画面上正是「嘴在抽」。按音长把相位幅度收小——短音只做半程口型，
    // ≥0.12s 的正常音 depth=1，行为和以前完全一样。
    var depth = clamp(dur / 0.20, 0.30, 1);
    var codaA0 = hasCoda ? (LIPS_OPEN[ph.c] == null ? a : LIPS_OPEN[ph.c]) : a;
    var codaF0 = hasCoda ? (LIPS_FORM[ph.c] == null ? f : LIPS_FORM[ph.c]) : f;
    var codaA = a + (codaA0 - a) * depth;
    var codaF = f + (codaF0 - f) * depth;
    var endA = hasCoda ? codaA : (cont ? na : a);
    var endF = hasCoda ? codaF : (cont ? nf : f);
    // 下一个字真正给出的第一个嘴形：它要是带起手声母（b/p/m 闭唇…），
    // 交接就该落到那个口型上，而不是直接跳到元音——否则「天」收口之后
    // 「妈」先满开、再闭唇，接缝上就是一帧闭合的抽动。
    // 目标形状要按「下一个字自己的音长」收同样的 depth，两个字的接缝才对得上。
    var nHandA = na, nHandF = nf;
    if (next) {
      var nPh = cuePhases(next);
      if (nPh && nPh.o && nPh.of > 0) {
        var ndur = Math.max(0, Number(next[3]) - Number(next[2]));
        var ndepth = clamp(ndur / 0.20, 0.30, 1);
        var nOa = LIPS_OPEN[nPh.o] == null ? na : LIPS_OPEN[nPh.o];
        var nOf = LIPS_FORM[nPh.o] == null ? nf : LIPS_FORM[nPh.o];
        nHandA = na + (nOa - na) * ndepth;
        nHandF = nf + (nOf - nf) * ndepth;
      }
    }
    // 入场嘴形：上一个字紧贴着唱且带韵尾时从它的收口形状进来；隔着整口气时
    // 从闭口进来（上一个字那一刻已经合上了）。有间隙、又没有韵尾的情况不滑——
    // 上一字在空隙里已经滑到下一个字的口型了（gap 分支负责）。
    var pA = null, pF = 0;
    if (idx > 0) {
      var prevCue = cues[idx - 1];
      var pg = Number(cue[2]) - Number(prevCue[3]);
      var pPh = cuePhases(prevCue);
      if (pg <= 0.02) {
        if (pPh && pPh.c && pPh.cf > 0) {
          // 上一个字结束时停在什么形状：它也要按上一个字自己的音长收 depth，
          // 否则短音收尾（半程口型）和这里按满量程算的入场值差着半个量程。
          var pVis = lipsVisemeAt(cues, idx - 1);
          var pa = LIPS_OPEN[pVis] == null ? 0.8 : LIPS_OPEN[pVis];
          var pf = LIPS_FORM[pVis] == null ? 0 : LIPS_FORM[pVis];
          var pdur = Math.max(0, Number(prevCue[3]) - Number(prevCue[2]));
          var pdepth = clamp(pdur / 0.20, 0.30, 1);
          var pca = LIPS_OPEN[pPh.c] == null ? pa : LIPS_OPEN[pPh.c];
          var pcf = LIPS_FORM[pPh.c] == null ? pf : LIPS_FORM[pPh.c];
          pA = pa + (pca - pa) * pdepth;
          pF = pf + (pcf - pf) * pdepth;
        }
      } else if (pg > LIPS_BREATH) {
        pA = 0; pF = 0;
      }
    }
    // 一个字唱完时的张口度（字内起伏在两端都是 0.94 倍，用它做各段之间的接缝，
    // 字与字、字与换气口之间才不会有一帧的跳变）。
    var s0 = endA * 0.94;
    var s1 = nHandA * 0.94;

    if (t <= cue[3]) {
      var local = clamp((t - cue[2]) / Math.max(cue[3] - cue[2], 1e-6), 0, 1);
      // 字内起伏：长音时嘴不是一张定格的图片；重音字的颤音/收放更明显，
      // 轻音几乎是平的（0.03 的幅度在画面上就是「纹丝不动」）。
      // 长音另走一条：几乎平着撑住，只在最后一小段轻轻收一点。以前长音和
      // 普通字一样每个字一个 sin 包，一个字拖 1 秒就会看到「重复张嘴」，
      // 听感上是把长音唱成了好几个短音。
      var arc;
      if (holdNote) {
        var tail = clamp((local - 0.85) / 0.15, 0, 1);
        arc = 0.97 + 0.015 * Math.sin(Math.PI * Math.min(1, local / 0.25)) - 0.035 * tail;
      } else {
        arc = 0.94 + (0.03 + 0.05 * st) * Math.sin(Math.PI * local);
      }
      // 咬字：重音开头 60~80ms 先冲出去一点再落回，轻音没有这一下——
      // 这是「重音听起来更用劲」在画面上最容易被看见的部分。长音只在起头
      // 咬一下，后面不再每 0.4 秒重咬一次。
      var punch = holdNote
        ? 1 + 0.05 * st * st * Math.exp(-local * 12)
        : 1 + 0.10 * st * st * Math.exp(-local * 7);
      // v12：字内相位。前 of 占比是声母（b/p/m 闭唇、f 唇齿、w 圆唇），
      // 后 cf 占比是韵尾（n 收口、ai/ao 往滑音收）。中间仍由元音口型主导，
      // 两端各用一个 smoothstep 滑进去，不会出现台阶。
      var phw = 0;
      // 相位窗口按「比例」和「绝对时长」取大：0.18s 的短字按比例只有 20ms，
      // 声母还没闭住就过去了；至少给 55ms（声母）/ 50ms（韵尾），最多占这个字
      // 的一半。窗口太短时嘴是在一帧里从闭到开的——那本身就是一次抽动。
      var ow = ph && ph.o ? clamp(ph.of * dur, 0.055, dur * 0.45) : 0;
      var cw = hasCoda ? clamp(ph.cf * dur, 0.05, dur * 0.45) : 0;
      if (ph && ph.o && ow > 0 && local * dur < ow) {
        var uo = smoothstep(0.20, 1, local * dur / ow);
        var oa = LIPS_OPEN[ph.o] == null ? a : LIPS_OPEN[ph.o];
        var ofo = LIPS_FORM[ph.o] == null ? f : LIPS_FORM[ph.o];
        oa = a + (oa - a) * depth;
        ofo = f + (ofo - f) * depth;
        a = oa + (a - oa) * uo;
        f = ofo + (f - ofo) * uo;
      }
      if (hasCoda && cw > 0 && (dur - local * dur) < cw) {
        phw = smoothstep(0, 1, (cw - (dur - local * dur)) / cw);
        a = a + (codaA - a) * phw;
        f = f + (codaF - f) * phw;
      }
      // g55：入场。上一个字紧贴着唱、而且是收口结尾（n/ng 韵尾）时，
      // 这个字开头的嘴形和上一帧差着大半个量程；隔着整口气时也一样
      // （上一字已经把嘴合上了）。先算出这个字「此刻想要」的脸，再从
      // 上一个字结束的嘴形滑过来——不滑这一下，接缝上就是一帧的跳，
      // 那正是用户说的「唱歌的时候抽一下」。
      var uw0 = 1;
      if (pA != null) {
        var ew = clamp(dur * 0.55, 0.035, 0.085);
        if (local * dur < ew) {
          uw0 = smoothstep(0, 1, local * dur / ew);
          a = pA + (a - pA) * uw0;
          f = pF + (f - pF) * uw0;
        }
      }
      // 协同发音：字尾提前滑向下一个字的口型（只对紧挨着的字）。
      // 有韵尾时这一下要让位——先收口再起下一个字，才是真的咬字；
      // 交接目标是下一个字「第一个给出的嘴形」（有起手声母就是那个口型）。
      // 短字（相位窗口和协同窗口会重叠）还要再让一步：入场没走完、或者还没唱
      // 到字尾，就别被下一个字提前拉走——否则 53ms 的字一开场就被拽向下一字，
      // 每一帧都在两个形状之间弹，那就是「抽动」。
      if (next && cont && cue[3] - t < blend) {
        var u = smoothstep(0, 1, 1 - (cue[3] - t) / blend) * (1 - phw)
                * uw0 * smoothstep(0, 0.45, local);
        a += (nHandA - a) * u;
        f += (nHandF - f) * u;
      }
      return { open: a * arc * punch, form: f, viseme: vis, stress: st, rate: rt,
               hold: holdNote };
    }

    // 这个字已经唱完，看和下一个字之间有多长空隙。
    if (next) {
      if (gap <= LIPS_LEGATO) {
        // 连音：一直张着，在两个元音之间平滑地换过去。
        var g = Math.max(gap, 1e-6);
        var u2 = smoothstep(0, 1, clamp((t - cue[3]) / g, 0, 1));
        return { open: s0 + (s1 - s0) * u2, form: endF + (nHandF - endF) * u2,
                 viseme: vis, stress: st, rate: rt, hold: holdNote };
      }
      if (gap <= LIPS_BREATH) {
        // 换气口：嘴真的合一下再张开。这里不是「松一点」——只松到 45%
        // 在画面上几乎看不出来，用户看到的会是「一直张着嘴」。
        // 末尾留一段提前起手，句与句之间不会有一帧的顿挫。
        var k2 = smoothstep(0, 1, clamp((t - cue[3]) / 0.10, 0, 1));
        var low = Math.min(s0, s1) * 0.10;
        var o0 = s0 + (low - s0) * k2;
        var f0 = endF * (1 - k2);
        var hold = Math.min(0.14, gap * 0.4);
        var back = smoothstep(0, 1,
          clamp((t - (next[2] - hold)) / Math.max(hold, 1e-3), 0, 1));
        return { open: o0 + (s1 - o0) * back,
                 form: f0 + (nHandF - f0) * back, viseme: nVis, stress: st, rate: rt,
                 hold: holdNote };
      }
    }
    // 这一句唱完，后面是长间隙（间奏、下一句隔了 0.9s 以上）或干脆是曲终。
    // 以前这里直接返回 0：最后一帧还张着 0.8，下一帧就归零——画面上就是
    // 「唱完猛地抽一下嘴」。现在用 90ms 顺着合上，数值从字尾的真实收口 s0
    // 出发，和 t=cue[3] 那一帧接得上。
    var since = Math.max(0, t - cue[3]);
    var k3 = smoothstep(0, 1, clamp(since / 0.09, 0, 1));
    return { open: s0 * (1 - k3), form: endF * (1 - k3),
             viseme: 'rest', stress: st, rate: rt, hold: holdNote };
  }

  // 自测入口：不启动任何轮询，也能在 node 里验证嘴型曲线（不影响运行时）。
  if (typeof window !== 'undefined') {
    window.__nekoLipSample = function (cues, t) { return lipsCuesAt(cues, t); };
    window.__nekoLipBreath = function (cues, t) { return lipsBreathFactor(cues, t); };
  }

  // 主窗口自己播这首歌：<audio> 边下边播，嘴读 audio.currentTime。
  // 这是最短的一条链——声音和嘴在同一个时钟、同一个渲染循环里，中间
  // 没有「另一个进程解码 → 解析进度 → HTTP 轮询」这几跳。
  function lipsReport(route, seq) {
    if (!LIPS_BASE) return;
    fetch(LIPS_BASE + '/' + route + '?seq=' + seq + '&v=' + VERSION,
          { cache: 'no-store' }).catch(function () {});
  }

  function lipsStopAudio() {
    var el = lips.audioEl;
    lips.audioSeq = 0;
    if (!el) return;
    try { el.pause(); el.removeAttribute('src'); el.load(); } catch (e) { /* 无 audio */ }
  }

  function lipsEnsureAudio(seq, path, at) {
    if (!path) { lipsStopAudio(); return; }
    if (lips.audioSeq === seq && lips.audioEl) return;
    if (!lips.audioEl) {
      try {
        lips.audioEl = new Audio();
        lips.audioEl.preload = 'auto';
        // 必须赶在第一次 src 之前设好：这首歌是从 127.0.0.1 的另一个端口发过来的，
        // 不声明 CORS，AnalyserNode 拿到的会是「被污染」的静音流——歌就哑了。
        lips.audioEl.crossOrigin = 'anonymous';
        // 真正出声 / 出错都立刻回报：插件那头按这个决定要不要退回 ffplay，
        // 不用干等 1.8 秒，也不会两路声音一起响太久。
        lips.audioEl.addEventListener('playing', function () {
          if (lips.audioSeq) lipsReport('audio_play', lips.audioSeq);
        });
        lips.audioEl.addEventListener('error', function () {
          if (lips.audioSeq) lipsReport('audio_fail', lips.audioSeq);
        });
      } catch (e) { lips.audioEl = null; }
      if (!lips.audioEl) { lipsReport('audio_fail', seq); return; }
    }
    lips.audioSeq = seq;
    var el = lips.audioEl;
    try {
      el.src = LIPS_BASE + path + '?seq=' + seq;
      // 页面刷新/中途接管时对到当前进度，不会从 0 重新唱一遍。
      if (at > 1) el.currentTime = at;
    } catch (e) {
      lipsReport('audio_fail', seq);
      return;
    }
    var p = el.play();
    if (p && typeof p.then === 'function') {
      p.then(function () { lipsReport('audio_play', seq); })
       .catch(function () { lipsReport('audio_fail', seq); });
    }
  }

  // ---- 原声节奏分析（成品混音 = 这首歌的背景音乐） ----
  // 「为她所听」的律动引擎只认这几样：现在多响、低频多沉、拍点在哪、音色亮暗。
  // 这些量从主窗口自己播的那条 <audio>（成品，人声+伴奏都在里面）直接分析，
  // 和 stage.html 的本地音频模式是同一套取特征方法：
  //   AnalyserNode(fftSize 2048, smoothing 0.72) → 低/中/高三段 + 总能量 +
  //   谱重心 + 起音包络。
  // 这样身体听的是「整首歌」——鼓点、贝斯、副歌的力度都在频谱里；嘴听的是
  // 另一条干净人声轨。两条链各管各的参数，不会再抢同一个参数。
  var mixAudio = {
    el: null, ctx: null, an: null, data: null,
    ok: false, failed: false, retryAt: 0,
    energy: 0, low: 0, mid: 0, high: 0, pitch: 0.5,
    beatEnv: 0, lastBeat: -9, bpm: 100,
    frames: 0
  };

  // 音频图只建一次，而且必须等 AudioContext 真的 running：
  // 把 <audio> 接到一个 suspended 的 context 上，等于把声音送进一个不输出
  // 任何东西的节点——歌会整个没声。没起来就下次再试，这期间歌照常播。
  function ensureMixAudio() {
    var el = lips.audioEl;
    if (!el || !lips.audioSeq || mixAudio.failed) return false;
    if (mixAudio.el !== el) {
      mixAudio.el = el;
      mixAudio.an = null;
      mixAudio.ok = false;
    }
    if (mixAudio.ok) return true;
    var now = performance.now();
    if (now < mixAudio.retryAt) return false;
    mixAudio.retryAt = now + 1000;
    var AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) { mixAudio.failed = true; return false; }
    try {
      if (!mixAudio.ctx) mixAudio.ctx = new AC();
      var ctx = mixAudio.ctx;
      if (ctx.state !== 'running') {
        try {
          var p = ctx.resume();
          if (p && typeof p.then === 'function') p.catch(function () {});
        } catch (e0) { /* 忽略 */ }
        if (ctx.state !== 'running') return false;
      }
      var an = ctx.createAnalyser();
      an.fftSize = 2048;
      an.smoothingTimeConstant = 0.72;
      var src = ctx.createMediaElementSource(el);
      src.connect(an);
      an.connect(ctx.destination);   // 分析之外还要出声：别把歌吞掉
      mixAudio.an = an;
      mixAudio.data = new Uint8Array(an.frequencyBinCount);
      mixAudio.ok = true;
      dbg.mix = 1;
      return true;
    } catch (e) {
      // 建不起来（老内核 / 元素已被别的图占用）就永久退回 48765 的监听链路，
      // 不再每帧重试；歌的声音不受影响。
      mixAudio.failed = true;
      dbg.mix = -1;
      return false;
    }
  }

  // 本地混音 → 引擎特征。归一化系数直接照抄 stage.html，保证和「为她所听」
  // 的观感一致；`beat` 是连续衰减的起音包络（引擎自己再做 swell 起手）。
  function mixFeatures(dt) {
    var an = mixAudio.an, data = mixAudio.data;
    if (!an || !data) return null;
    try { an.getByteFrequencyData(data); } catch (e) { return null; }
    var n = data.length;
    if (!n) return null;
    var low = 0, mid = 0, high = 0, total = 0, weighted = 0;
    for (var i = 0; i < n; i++) {
      var v = data[i] / 255;
      total += v;
      weighted += v * i;
      if (i < n * 0.12) low += v;
      else if (i < n * 0.45) mid += v;
      else high += v;
    }
    var energy = Math.min(1, (total / n) * 2.6);
    var now = performance.now();
    var flux = energy - mixAudio.beatEnv;
    if (flux > 0.06 && now - mixAudio.lastBeat > 220) {
      var gap = (now - mixAudio.lastBeat) / 1000;
      mixAudio.lastBeat = now;
      mixAudio.beatEnv = 1;
      // BPM 只做很慢的估计（4 秒时间常数）：它只影响摇摆快慢，估错也只是
      // 动作和鼓点慢慢错开，不会跳。丢拍时按倍数折回 60~180 再平滑。
      if (gap > 0.30 && gap < 1.6) {
        var bpmNew = 60 / gap;
        while (bpmNew < 60) bpmNew *= 2;
        while (bpmNew > 180) bpmNew /= 2;
        mixAudio.bpm += (bpmNew - mixAudio.bpm) * (1 - Math.exp(-dt / 4.0));
      }
    } else {
      mixAudio.beatEnv *= Math.exp(-dt / 0.23);
      if (mixAudio.beatEnv < 0.01) mixAudio.beatEnv = 0;
    }
    // 频谱本身已经过 AnalyserNode 的 0.72 平滑，这里只再抹一层 0.08s，
    // 抹掉逐帧量化噪声；太大就会「慢半拍」。
    var k = 1 - Math.exp(-dt / 0.08);
    mixAudio.energy += (energy - mixAudio.energy) * k;
    mixAudio.low += (Math.min(1, low / (n * 0.12) * 1.8) - mixAudio.low) * k;
    mixAudio.mid += (Math.min(1, mid / (n * 0.33) * 1.8) - mixAudio.mid) * k;
    mixAudio.high += (Math.min(1, high / (n * 0.55) * 2.2) - mixAudio.high) * k;
    mixAudio.pitch += ((total > 0 ? Math.min(1, weighted / total / (n * 0.6)) : 0.5)
                       - mixAudio.pitch) * (1 - Math.exp(-dt / 0.5));
    mixAudio.frames += 1;
    return {
      energy: mixAudio.energy,
      low: mixAudio.low,
      mid: mixAudio.mid,
      high: mixAudio.high,
      pitch: mixAudio.pitch,
      // 打击感：成品混音的总能量就是鼓点的力度；给个中等下限，安静的段不塌。
      punch: Math.max(0.35, mixAudio.energy),
      bpm: mixAudio.bpm,
      tempoTrust: 1,
      regularity: 0.6,
      beat: mixAudio.beatEnv,
      hit: 0,
      beatPos: 0,
      beatConf: 0,
      silent: mixAudio.energy < 0.02
    };
  }

  // 主窗口里有没有「正在出声的成品」可以分析。
  function mixAudioPlaying() {
    var el = lips.audioEl;
    return !!(el && lips.audioSeq && !el.paused && !el.ended
              && el.readyState >= 2 && el.currentTime > 0);
  }

  // ---- 干声实时分析（音高 / 音色） ----
  // 插件通过 /stem.wav 把这首歌的「干净人声」也发给主窗口。它只接到
  // AnalyserNode（再经一个 0 增益节点收尾），不会和成品混音叠成两路声音。
  // 两个用途：
  //   · 长音：用真实 F0 和颤音给张口度做细微起伏，代替按字均分时那种
  //     「每 0.45 秒一次」的假起伏；
  //   · 表情：人声的亮暗、力度和颤音是「唱得投入」最直接的信号，和背景
  //     音乐的能量/段落/拍点一起决定眉、笑眼、脸颊。
  // 没有干声文件（旧曲目 / 预览没挂 stem）时整块自动停用，行为退回 v42。
  var vox = {
    el: null, ctx: null, an: null, srcNode: null, time: null, freq: null,
    sr: 44100, ok: false, failed: false, url: '', seq: 0,
    energy: 0, bright: 0.4, low: 0.0, centroid: 1200, f0: 0, f0Ref: 0,
    clarity: 0, voiced: false, vib: 0, midi: 0, tickAt: 0, pitchAt: 0,
    hist: []
  };

  function voxPause() {
    if (vox.el && !vox.el.paused) { try { vox.el.pause(); } catch (e) { /* 忽略 */ } }
    vox.voiced = false;
  }

  // 音频图只建一次：同一个 <audio> 元素只能 createMediaElementSource 一次，
  // 换歌只换 src，不重建图。
  function voxEnsureGraph() {
    if (vox.failed) return false;
    if (!vox.el) {
      try {
        vox.el = new Audio();
        vox.el.preload = 'auto';
        // 注意：不能设 muted/volume=0。Chrome 里元素被静音后，送进
        // MediaElementSource 的数据也会变成静音，analyser 读到的全是 0。
        // 静音靠「这条源只接 analyser、不接 destination」——声音本来就走
        // 成品混音那条 <audio> 出，干声这一路不出声。
        vox.el.crossOrigin = 'anonymous';   // 不声明 CORS 会被污染成静音
      } catch (e) { vox.failed = true; return false; }
    }
    var AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) { vox.failed = true; return false; }
    try {
      if (!vox.ctx) vox.ctx = mixAudio.ctx || new AC();
      if (vox.ctx.state !== 'running') {
        try {
          var p = vox.ctx.resume();
          if (p && typeof p.then === 'function') p.catch(function () {});
        } catch (e0) { /* 忽略 */ }
        if (vox.ctx.state !== 'running') return false;
      }
      if (!vox.an) {
        var an = vox.ctx.createAnalyser();
        an.fftSize = 2048;
        an.smoothingTimeConstant = 0.35;
        var src = vox.ctx.createMediaElementSource(vox.el);
        src.connect(an);
        // 再用一个 0 增益接到 destination：既保证这条分析支路一定被渲染，
        // 又不会真的发出声音（干声只做分析，声音仍由成品混音那条 <audio> 出）。
        var sink = vox.ctx.createGain();
        sink.gain.value = 0;
        an.connect(sink);
        sink.connect(vox.ctx.destination);
        vox.an = an;
        vox.time = new Float32Array(an.fftSize);
        vox.freq = new Uint8Array(an.frequencyBinCount);
        vox.sr = vox.ctx.sampleRate || 44100;
      }
      vox.ok = true;
      return true;
    } catch (e) {
      vox.failed = true;
      return false;
    }
  }

  // 和成品混音同一个时间线：差得多就拉回，差得少让它自己走（避免每帧回跳）。
  function voxSync(t, playing) {
    if (!lips.active || !lips.stem || !lips.seq) { voxPause(); return; }
    if (!voxEnsureGraph()) return;
    var url = LIPS_BASE + lips.stem + '?seq=' + lips.seq;
    if (vox.url !== url) {
      vox.url = url;
      vox.seq = lips.seq;
      vox.hist.length = 0;
      vox.energy = 0; vox.voiced = false; vox.f0 = 0; vox.f0Ref = 0;
      vox.vib = 0; vox.clarity = 0;
      try { vox.el.src = url; vox.el.load(); } catch (e) { vox.failed = true; return; }
    }
    if (!playing) { voxPause(); return; }
    if (vox.el.readyState < 2) return;
    try {
      if (vox.el.paused) {
        vox.el.currentTime = Math.max(0, t || 0);
        var pr = vox.el.play();
        if (pr && typeof pr.then === 'function') {
          pr.catch(function () { vox.failed = true; });
        }
      } else if (Math.abs(vox.el.currentTime - (t || 0)) > 0.12) {
        vox.el.currentTime = Math.max(0, t || 0);
      }
    } catch (e2) { vox.failed = true; }
  }

  /** 自相关基频（70~1000Hz）。只取最后 1024 点，够一个低音周期。 */
  function voxPitch() {
    var a = vox.time, sr = vox.sr, n = a.length;
    if (!a || n < 512) return { f0: 0, clarity: 0 };
    var start = n - 1024, len = 1024;
    var mean = 0, i;
    for (i = start; i < n; i++) mean += a[i];
    mean /= len;
    var rms = 0;
    for (i = start; i < n; i++) { var d0 = a[i] - mean; rms += d0 * d0; }
    rms = Math.sqrt(rms / len);
    if (rms < 0.008) return { f0: 0, clarity: 0 };
    var minLag = Math.max(2, Math.floor(sr / 1000));
    var maxLag = Math.min(len - 2, Math.floor(sr / 70));
    var best = 0, bestLag = 0;
    for (var lag = minLag; lag <= maxLag; lag++) {
      var corr = 0, e1 = 0, e2 = 0;
      for (i = start; i < n - lag; i++) {
        var x = a[i] - mean, y = a[i + lag] - mean;
        corr += x * y; e1 += x * x; e2 += y * y;
      }
      var c = corr / Math.sqrt(e1 * e2 + 1e-12);
      if (c > best) { best = c; bestLag = lag; }
    }
    if (best < 0.32 || !bestLag) return { f0: 0, clarity: best };
    // 抛物线插值：整数 lag 的 ±1 档精度不够，颤音会被量化成台阶。
    var y0 = 0, y1 = best, y2 = 0;
    try {
      var calc = function (lg) {
        var cc = 0, q1 = 0, q2 = 0;
        for (var j = start; j < n - lg; j++) {
          var u = a[j] - mean, v = a[j + lg] - mean;
          cc += u * v; q1 += u * u; q2 += v * v;
        }
        return cc / Math.sqrt(q1 * q2 + 1e-12);
      };
      y0 = calc(bestLag - 1); y2 = calc(bestLag + 1);
    } catch (e) { y0 = y1; y2 = y1; }
    var den = y0 - 2 * y1 + y2;
    var off = Math.abs(den) > 1e-9 ? 0.5 * (y0 - y2) / den : 0;
    off = clamp(off, -0.5, 0.5);
    return { f0: sr / (bestLag + off), clarity: best };
  }

  function voxSample(dt, t) {
    var now = performance.now();
    // applyNow 可能被事件和 rAF 各调一次：同一帧只分析一次。
    if (now - vox.tickAt < 12) return;
    vox.tickAt = now;
    if (!vox.ok || !vox.an || !vox.el || vox.el.paused || vox.el.ended) {
      vox.energy *= Math.exp(-dt / 0.20);
      vox.vib *= Math.exp(-dt / 0.30);
      vox.voiced = false;
      vox.bright += (0.4 - vox.bright) * (1 - Math.exp(-dt / 0.6));
      return;
    }
    try { vox.an.getByteFrequencyData(vox.freq); } catch (e) { return; }
    try { vox.an.getFloatTimeDomainData(vox.time); } catch (e2) { return; }
    var n = vox.freq.length, total = 0, weighted = 0, low = 0;
    for (var i = 0; i < n; i++) {
      var v = vox.freq[i] / 255;
      total += v;
      weighted += v * i;
      if (i < n * 0.12) low += v;
    }
    // 时域 RMS 比频谱均值更接近「唱得多响」，不受高频噪声摊薄。
    var a = vox.time, rms = 0;
    for (var j = 0; j < a.length; j++) rms += a[j] * a[j];
    rms = Math.sqrt(rms / Math.max(1, a.length));
    var energy = clamp(rms * 5.5, 0, 1);
    var centroid = total > 0 ? (weighted / total) * (vox.sr * 0.5) / n : 0;
    var bright = clamp((centroid - 500) / 2500, 0, 1);
    var k = 1 - Math.exp(-dt / 0.06);
    vox.energy += (energy - vox.energy) * k;
    vox.low += (Math.min(1, low / (n * 0.12) * 1.8) - vox.low) * k;
    vox.centroid += (centroid - vox.centroid) * (1 - Math.exp(-dt / 0.12));
    vox.bright += (bright - vox.bright) * (1 - Math.exp(-dt / 0.12));

    // 音高：20Hz 就够跟上颤音（5~7Hz），再快只是白烧 CPU。
    if (now - vox.pitchAt >= 45) {
      vox.pitchAt = now;
      var got = voxPitch();
      vox.clarity += (got.clarity - vox.clarity) * 0.55;
      if (got.f0 > 60 && got.f0 < 1200 && got.clarity > 0.33) {
        vox.f0 = got.f0;
        vox.voiced = true;
        if (!vox.f0Ref) vox.f0Ref = got.f0;
        // 慢参考音高（1.6s）：颤音相对它摆动；长音上行时它慢慢跟上。
        vox.f0Ref += (got.f0 - vox.f0Ref) * (1 - Math.exp(-dt / 1.6));
        var dev = Math.abs(got.f0 - vox.f0Ref) / Math.max(vox.f0Ref, 60);
        vox.vib += (clamp(dev / 0.020, 0, 1) - vox.vib) * 0.18;
        vox.midi = 69 + 12 * Math.log(got.f0 / 440) / Math.LN2;
      } else {
        vox.voiced = false;
        vox.vib *= Math.exp(-dt / 0.35);
      }
    }
  }

  // ---- 表情的音乐侧：背景音乐的能量 / 亮度 / 段落 ----
  // 表情不能只看人声：副歌鼓点起来、段落推进，脸上的「兴奋度」也该跟着走。
  // mixAudio 是成品混音（BGM+人声），engine.frame.section 是「为她所听」
  // 已经判好的段落（intro/build/verse/chorus/outro）。
  var mus = { energy: 0, low: 0, bright: 0.5, beat: 0, sect: 0, sectName: '' };
  function musicSample(dt) {
    var e = 0, low = 0, bright = 0.5, beat = 0;
    if (mixAudio.ok) {
      e = mixAudio.energy; low = mixAudio.low;
      bright = mixAudio.pitch; beat = mixAudio.beatEnv;
    } else if (latest) {
      e = Number(latest.energy) || 0;
      low = Number(latest.low) || 0;
      bright = Number(latest.pitch) || 0.5;
      beat = Number(latest.onset_peak) || 0;
    }
    var f = engine && engine.frame;
    var sec = f && f.section ? String(f.section) : '';
    var w = sec === 'chorus' ? 1
          : sec === 'build' ? 0.55
          : sec === 'verse' ? 0.42
          : 0.12;
    var k = 1 - Math.exp(-dt / 0.8);
    mus.energy += (e - mus.energy) * k;
    mus.low += (low - mus.low) * k;
    mus.bright += (bright - mus.bright) * (1 - Math.exp(-dt / 1.1));
    mus.beat = Math.max(beat, mus.beat * Math.exp(-dt / 0.16));
    mus.sect += (w - mus.sect) * (1 - Math.exp(-dt / 1.0));
    if (sec) mus.sectName = sec;
  }

  // ---- 独唱舞台（solo-stage.js）联动：她真在「唱」的时候自动进舞台 ----
  // 演唱入口（为她而唱 ms_sing / ms_play，audio=True）会把成品音频一起挂到口型
  // 服务上；这里看到「口型轨 active + 有 audio」就派发进入事件，画面交给舞台脚本。
  // 面板的口型试听（ms_lips_preview）没有 audio，普通放歌没有口型轨，都不触发。
  // 换歌（seq 变）时重新派发，让舞台跟着换一首重来；唱完（active 变假）派发结束。
  var soloStage = { on: false, seq: 0 };
  function soloStageSync(st) {
    var on = !!(lips.active && st.audio && lips.duration > 3);
    var seq = Number(st.seq) || 0;
    if (on === soloStage.on && (!on || seq === soloStage.seq)) { return; }
    soloStage.on = on;
    soloStage.seq = on ? seq : 0;
    try {
      if (on) {
        window.dispatchEvent(new CustomEvent('neko-solo-stage', {
          detail: {
            title: lips.song || '',
            duration: lips.duration,
            lips: true,
            // 舞台加载器从这两个地址 GET 演出脚本：show_url 给当前曲目的编排，
            // stage_base 是舞台静态资源的根（跨端口只走 GET）。
            show_url: LIPS_BASE + '/show.json?track=' + encodeURIComponent(String(st.track_id || '')),
            stage_base: LIPS_BASE + '/stage/'
          }
        }));
      } else {
        window.dispatchEvent(new CustomEvent('neko-solo-stage-end'));
      }
    } catch (e) { /* 没加载舞台脚本时无事发生 */ }
  }

  function lipsPoll() {
    // 插件没开时别每 200ms 去连 5 个端口；找不到就歇 3 秒再来。
    if (!LIPS_BASE) {
      var now = performance.now();
      if (now < lips.nextFind) return;
      lips.nextFind = now + 3000;
      lipsFindPort(0);
      return;
    }
    fetch(LIPS_BASE + '/lips.json?since=' + lips.seq + '&v=' + VERSION
          + '&vx=' + (vox.ok ? 1 : 0) + '&ve=' + (vox.energy || 0).toFixed(2)
          + '&vf0=' + Math.round(vox.f0 || 0) + '&msc=' + (mus.sectName || '-')
          // 表情诊断：面部分支支持表（face）和情绪层支持表+当前值（fv）。
          // 服务端把它们记进 ms_lips_status，用来判断是参数缺失还是逻辑没跑。
          + '&face=' + (faceStats.support || '') + '&fv=' + (faceStats.support2 || '')
          + '|' + (faceStats.last || '') + '@' + (faceStats.writes || 0)
          // g64：/hello 的 face_hooked / face_writes 一直显示 -1 / 0，查下来是这条
          // 200ms 的轮询没带 fh/fw（每 5s 的 ping 带，但它的 fv 会被这条盖掉）。
          // 现在两条都带；fx 是贴图特效的实时写入量。
          + '&fh=' + (faceHooked ? 1 : 0) + '&fw=' + (faceStats.writes || 0)
          + '&fx=' + extra.blush.toFixed(2) + ',' + extra.tears.toFixed(2)
          + ',' + extra.flower.toFixed(2) + ',' + extra.darkFace.toFixed(2)
          // 情绪→身体（g47）的诊断也走这条通道：lips_server 把它记进
          // /hello 的 body_support / body_values，刷新页面后就能看到有没有在叠。
          + '&bs=' + (faceStats.support3 || '') + '&bd=' + bodyTag()
          + '&ey=' + probeTag() + '&ed=' + probeTag2(),
          { cache: 'no-store' })
      .then(function (r) { return r.json(); })
      .then(function (st) {
        if (!st || !st.ok) return;
        // track=true 表示这是换歌后的整条轨：歌词口型和干声包络二选一，
        // 换过来就要把另一种清掉，否则会接着上一首的嘴形。
        if (st.track) {
          lips.cues = st.cues && st.cues.length ? st.cues : null;
          lips.env = st.env && st.env.length ? st.env : null;
          lips.hop = Number(st.hop) || 0.05;
          // 换歌：口型时钟的连续性守卫和响度跟随器都从新轨重新起步。
          lips.tClock = null;
          lips.loudF = null;
          // 每行歌词的情绪（v/a/d + 置信度）：桥把它掺进表情目标值。
          lips.moods = st.moods && st.moods.length ? st.moods : null;
          // 换歌才重算一次「每一句唱多快」：字距越密，嘴的平滑时间常数越短。
          lips.rate = lips.cues ? lipsRates(lips.cues) : null;
        }
        lips.seq = Number(st.seq) || 0;
        lips.active = !!st.active;
        lips.mode = st.mode || '';
        lips.song = st.song || '';
        lips.stem = st.stem || '';
        lips.duration = Number(st.duration) || 0;
        // 每次轮询都硬换时间原点，会让嘴每 200ms 轻微回跳一次——看起来一卡一卡。
        // 只有差得够多（换歌 / 跳播 / 掉线后追回）才重新对齐；平时沿用本地
        // 时钟往前推，画面是连续的。
        var nowMs = performance.now();
        var nextElapsed = Number(st.elapsed) || 0;
        var predicted = (lips.baseElapsed || 0) + (nowMs - (lips.at || nowMs)) / 1000;
        if (!lips.at || Math.abs(nextElapsed - predicted) > 0.08) {
          lips.baseElapsed = nextElapsed;
          lips.at = nowMs;
        }
        // 插件把成品挂上来了就让主窗口自己播（链路最短）；没有就保持 ffplay。
        if (!lips.active) {
          if (lips.audioSeq) lipsStopAudio();
        } else if (st.audio) {
          lipsEnsureAudio(lips.seq, st.audio, nextElapsed);
        } else if (lips.audioSeq) {
          lipsStopAudio();            // 插件退回 ffplay 了，别两路一起响
        }
        lips.misses = 0;
        window.__nekoMusicLips = lips;
        soloStageSync(st);
        deskSync(st);
      })
      .catch(function () {
        lips.misses += 1;
        // 插件偶尔会忙到连续几十次响应不过来（分离/变声跑在同一个 Python 里）。
        // 以前 10 次（2 秒）就放弃端口，嘴会在歌中间停好几秒。现在保持口型轨
        // 和本地时钟继续往前走，只把端口重新找一遍；唱到 duration 才收嘴。
        if (lips.misses > 40) {
          lips.misses = 0;
          LIPS_BASE = '';
          lips.seq = -1;
        }
      });
  }

  // 主程序自己在说话（TTS）时嘴归它管，别两条轨抢同一个参数。
  function speechBusy() {
    var st = window.NekoSpeechPlaybackState;
    return !!(st && st.active && (Date.now() - (Number(st.updatedAt) || 0) < 1200));
  }

  // 嘴必须走主程序自己的口型通道（LanLan1 / live2dManager）：
  // live2d-model.js 的 installMouthOverride 会在 coreModel.update 之前按
  // mouthValue 强制写 ParamMouthOpenY，绕过它直接写参数会被每帧盖掉。
  function lipsRelease() {
    var o = lips.owner;
    lips.owner = null;
    if (o && window.LanLan1 && typeof window.LanLan1.endLipSync === 'function') {
      try { window.LanLan1.endLipSync(o); } catch (e) { /* 模型换了 */ }
    }
  }

  // 唱歌开始/停唱的一次性复位：表情层用的是叠加写（addParameterValueById），
  // 停手后旧值会留在参数里（历史上 EyeSmile 残留、MouthForm=0.75 都是这么来的）。
  // 这里在「进唱歌」和「出唱歌」两个边沿把本层写过的参数显式写回默认 0，
  // 并把平滑状态清零；不碰眼睑开合（归主程序 blink 器）与身体/动作层。
  // 复位按语义通道走：写回模型自己的默认值（不是 0——有的模型默认不在 0），
  // 也不会把同编号、别的含义的参数（另一个模型的 Param57 等）误写成 0。
  var FACE_RESET_CHANNELS = [
    'browLY', 'browRY', 'browLAngle', 'browRAngle', 'browLForm', 'browRForm', 'cheek',
    'mouthForm', 'smile2', 'mouthX', 'mouthOpen', 'teeth', 'eyeLSmile', 'eyeRSmile',
    'angerMark', 'darkFace', 'blushFx', 'tears', 'flower'
  ];
  function faceResetToDefault(core) {
    if (core && paProfile(core)) {
      for (var i = 0; i < FACE_RESET_CHANNELS.length; i++) pa().reset(core, FACE_RESET_CHANNELS[i]);
    }
    face.brow = face.smile = face.cheek = face.angle = 0;
    extra.smile2 = extra.mouthX = extra.teeth = extra.mk = 0;
    extra.asym = extra.browSyl = extra.blush = extra.tears = 0;
    extra.flower = extra.darkFace = extra.arc = 0;
    for (var k in emo.f) emo.f[k] = 0;
    // 停唱边沿把情绪状态也归零：solveEmoFACS 每帧会从 emo.v/a/d 重新算 f，
    // 只清 f 不清源值的话，停唱后几秒仍会继续往眉毛/嘴形叠加「残余情绪」，
    // 直到 VAD 自己衰减（实测停唱 3 秒后仍 excited、MouthForm≈0.5）。
    emo.v = emo.a = emo.d = 0;
    emo.tV = emo.tA = emo.tD = 0;
    emo.pHi = emo.pLo = 0;
    emo.pres = 0; emo.hold = 0; emo.blink = 0;
    emo.tag = 'neutral'; emo.arch = 'neutral';
    lips.lastOpen = 0;
    lips.lastForm = 0;
  }

  function applyLips(core) {
    var bridge = window.LanLan1;
    var bridged = !!(bridge && typeof bridge.beginLipSync === 'function'
                      && typeof bridge.setMouth === 'function');
    var hasCues = !!(lips.cues && lips.cues.length);
    var hasEnv = !!(lips.env && lips.env.length > 1);
    var now = performance.now();
    // 平滑按真实时间推进：applyNow 一帧可能被事件和 rAF 各调一次，
    // 用 tickAt 保证只有第一次真正推进，后面几次 dt≈0，不会重复平滑。
    var dt = lips.tickAt ? Math.min(0.1, Math.max(0.001, (now - lips.tickAt) / 1000)) : 1 / 60;
    lips.tickAt = now;
    // 进唱歌 / 出唱歌的边沿：表情层显式复位一次（唱歌中由 applyFace/applyEmo
    // 按住为中性，停唱后不再有叠加残留）。眼睑开合与动作层不受影响。
    var faceActiveNow = !!lips.active;
    if (faceActiveNow !== lips.faceActive) {
      lips.faceActive = faceActiveNow;
      faceResetToDefault(core);
    }
    if (!lips.active || (!hasCues && !hasEnv) || speechBusy()) {
      lipsRelease();
      lips.lastOpen = 0;
      lips.lastForm = 0;
      voxPause();
      // 停唱之后脸上的表情要自己收回去，否则歌停了眉还挑着。
      faceSmooth(dt, 0, 0, 0, 0);
      return;
    }
    // 声音就在本窗口里放的时候，嘴直接读 audio.currentTime——这是最短的一跳，
    // 不用「另一个进程解码 → 解析进度 → HTTP 轮询 → JS 推算」绕一圈。
    var audioEl = lips.audioEl;
    var fromAudio = !!(audioEl && lips.audioSeq && !audioEl.paused
                       && !audioEl.ended && audioEl.readyState >= 2);
    var t;
    if (fromAudio) {
      t = Number(audioEl.currentTime) || 0;
    } else {
      t = (lips.baseElapsed || 0) + (now - (lips.at || now)) / 1000;
    }
    // g55：时钟连续性。音频元素卡一下、或者本地钟被轮询对齐拉回，t 会一步
    // 跳几十上百毫秒；嘴跟着跳就是「唱歌的时候抽一下」。往前最多按真实时间
    // 多追一帧，往后一律不追（原地等音频自己的时间回来）。真正的跳播
    // （换歌/重播）在 lipsPoll 里把 tClock 清掉，不走这条。
    var prevT = lips.tClock;
    if (prevT != null) {
      if (t > prevT + 0.25) {
        t = prevT + Math.max(dt, 0.016);
        lips.clockFwd = (lips.clockFwd || 0) + 1;
      } else if (t < prevT - 0.04 && t > prevT - 1.5) {
        t = prevT;
        lips.clockBack = (lips.clockBack || 0) + 1;
      }
    }
    lips.tClock = t;
    if (fromAudio) {
      lips.baseElapsed = t;
      lips.at = now;
    }
    lips.t = t;   // 情绪层要用同一个歌词时钟去查「这一句的词」
    // 口型服务暂时断线也要把这首唱完：本地时钟继续走，唱到曲尾才收嘴。
    if (lips.duration > 0 && t > lips.duration + 0.5) {
      lipsRelease();
      lips.lastOpen = 0;
      lips.lastForm = 0;
      voxPause();
      faceSmooth(dt, 0, 0, 0, 0);   // 曲终：脸上的表情也要落下来
      return;
    }
    // 干声实时分析：和成品同一个时钟（stem 是静音播的，不出声）。
    // 长音用它做「真人撑着唱」的细微起伏；表情用它拿人声亮度和颤音。
    // 没有 stem（旧曲目 / 预览没带干声）时整块自动失效，行为退回 v42。
    voxSync(t, true);
    voxSample(dt, t);
    var open = 0, form = 0, breath = 1, stress = 0.5, rate = 0.5;
    // 干声响度（0..1，50ms 一帧）：歌词轨用它调制张口度。
    var amp = null;
    if (hasEnv) {
      var pos = t / (lips.hop || 0.05);
      var i0 = Math.floor(pos);
      if (i0 >= 0 && i0 < lips.env.length) {
        var v0 = Number(lips.env[i0]) || 0;
        var v1 = i0 + 1 < lips.env.length ? (Number(lips.env[i0 + 1]) || 0) : v0;
        amp = v0 + (v1 - v0) * (pos - i0);
      }
    }
    if (hasCues) {
      var got = lipsCuesAt(lips.cues, t);
      open = got.open;
      form = got.form;
      breath = lipsBreathFactor(lips.cues, t);
      stress = got.stress == null ? 0.5 : got.stress;
      // 这一段唱得多快：快歌（字距 0.18s）嘴要更跟得上，慢歌可以更从容。
      rate = got.rate == null ? 0.5 : got.rate;
      // v6/v7：先按真实响度调制张口度（渐强、渐弱、颤音都进得来），
      // 再按这个字的轻重加一档力度——重音咬得更开、轻音收着唱。
      // 以前每个元音只有一个固定张度，字与字之间是"定格"的，很木头。
      if (amp != null) {
        // g55：响度先过跟随器再乘张口度。包络是 50ms 一帧的，干声里漏进来的
        // 鼓点/混响尾巴会让原值在相邻帧之间弹一下，嘴就跟着一抽一抽。
        var lraw = clamp(amp, 0, 1);
        if (lips.loudF == null) lips.loudF = lraw;
        var kL = 1 - Math.exp(-dt / (lraw > lips.loudF
          ? LIPS_LOUD_ATTACK : LIPS_LOUD_RELEASE));
        lips.loudF += (lraw - lips.loudF) * kL;
        var loud = lips.loudF;
        var gain = 0.40 + 0.85 * loud;
        // 干净人声包络 0.06 以下就是「没在唱」——分离残留的鼓点、混响尾巴、
        // 房间底噪都落在这个区间。以前只是线性轻压（0.1 还能把嘴撑开 ~30%），
        // 间奏看着就像在跟伴奏张嘴（误张）。现在是一条 0.06→0.18 的硬斜坡：
        // 下面完全闭嘴，上面才正常开合；0.18 以上真实唱句一拍不少。
        gain *= smoothstep(0.06, 0.18, loud);
        open *= gain;
        // 轻音 0.60 倍、重音 1.18 倍：同一个元音，重字比轻字多张嘴近一倍。
        // 轻音要真的「收着唱」——以前下限是 0.76，轻音看起来和重音差不多。
        open *= 0.60 + 0.58 * stress;
        // 长音：用干声的真实音高与颤音给张口度一点「人气」。
        // 上行长音越唱越高、嘴略张；颤音让口型轻轻呼吸。只有 ±10%，
        // 不会唱错字，但一个字拖一秒的时候不再是定格画面。
        if (got.hold && vox.voiced && vox.clarity > 0.45) {
          var rel = clamp((vox.f0 - vox.f0Ref) / Math.max(vox.f0Ref, 60), -0.35, 0.35);
          open *= clamp(1 + 0.16 * rel + 0.05 * vox.vib, 0.90, 1.10);
        }
        if (open > 1) open = 1;
      }
    } else if (hasEnv) {
      // 没歌词（日文/冷门翻唱常见）：跟着分离出的干声响度开合，至少是真在唱的地方张嘴。
      open = amp || 0;
    }
    // ---- 表情目标值：干声（用力/亮度/颤音）+ 背景音乐（能量/段落/拍点） ----
    if (hasCues) {
      var idxNow = lipsCueIndex(lips.cues, t);
      var visNow = idxNow >= 0 ? lipsVisemeAt(lips.cues, idxNow) : 'rest';
      var bright = LIPS_BRIGHT[visNow] == null ? 0.4 : LIPS_BRIGHT[visNow];
      var presence = amp == null ? 1 : smoothstep(0.06, 0.20, amp);
      // 乐句响度：0.35s 平滑。表情跟乐句走，不在 50ms 的包络上抖。
      var kl = 1 - Math.exp(-dt / 0.35);
      faceDyn.loud += ((amp == null ? 0 : amp) - faceDyn.loud) * kl;
      // 句首：从「没在唱」进入到「在唱」的一刻挑一下眉，像换气后开口。
      if (presence > 0.25 && faceDyn.pres <= 0.10) faceDyn.onset = 1;
      else faceDyn.onset *= Math.exp(-dt / 0.30);
      // 句尾：从「在唱」掉到「没在唱」——换气眨眼的触发（eyeTick 消费完清零）
      if (presence < 0.10 && faceDyn.pres >= 0.25) faceDyn.end = 1;
      faceDyn.pres = presence;
      var onset = clamp(faceDyn.onset, 0, 1);

      // 「投入度」的两个来源：她唱得多响（干声）+ 伴奏多满（BGM）。
      // 没有干声文件时整段只靠 BGM，不会因为缺 stem 就变成面无表情。
      var stemOn = !!(vox.ok && lips.stem);
      var energyMix = stemOn
        ? clamp(0.55 * mus.energy + 0.45 * vox.energy, 0, 1)
        : clamp(mus.energy, 0, 1);
      // 音色亮度：元音本身（a/i 的明亮度）比音量更贴近「表情」，再加上
      // 干声的频谱重心和 BGM 的高频亮度——副歌的镲片/高音区也参与。
      var brightMix = stemOn
        ? clamp(0.40 * bright + 0.30 * vox.bright + 0.30 * mus.bright, 0, 1)
        : clamp(0.55 * bright + 0.45 * mus.bright, 0, 1);
      // 段落权重（副歌 1 / build .55 / verse .42 / 前奏尾奏 .12）平滑过，
      // 再乘能量：安静的主歌不该和副歌一样兴奋。
      var sect = clamp(mus.sect, 0, 1);
      var beatPulse = clamp(mus.beat, 0, 1);
      var vib = clamp(vox.vib, 0, 1);
      faceSmooth(dt,
        // 眉毛：重音抬得最明显，明亮音色再挑一点；句首、副歌、拍点额外抬。
        clamp(presence * (0.08 + 0.40 * stress + 0.16 * brightMix)
              + 0.16 * onset
              + 0.18 * sect * (0.30 + 0.70 * energyMix)
              + 0.07 * beatPulse, 0, 0.55),
        // 笑眼：明亮音色 + 投入度；副歌更明显，颤音让它「唱得动情」。
        // g52：原来上限 0.90，配情绪层能到 0.8 以上，等于整首都在动眼睑。
        // g54：这个模型的 ParamEyeLSmile 不是「眯眼」，而是把**下眼睑拉开**——
        // 数值一上去眼睛就变大一圈，虹膜相对就显得小（用户实测截图对比：
        // 0.05 时虹膜几乎撑满眼白，0.40 时上下都露出大片眼白）。用户要求
        // 「不要再动眼珠子大小」，所以整条通道压到只够看出一点笑意。
        clamp((presence * (0.12 + 0.46 * brightMix) * (0.35 + 0.65 * presence)
              + 0.22 * sect * energyMix
              + 0.10 * vib * energyMix) * 0.16, 0, 0.14),
        // 脸颊：唱得用力时鼓起来；BGM 低频砸下来时跟着一鼓。
        clamp(presence * (0.06 + 0.34 * stress)
              + 0.10 * sect * energyMix
              + 0.05 * mus.low * beatPulse, 0, 0.60),
        // 眉角：抬眉的眉形（唱得越投入越整条抬起）。
        clamp(presence * (0.06 + 0.24 * stress) + 0.08 * onset
              + 0.10 * sect * energyMix, 0, 0.60));
    } else if (hasEnv) {
      var e2 = smoothstep(0.06, 0.20, amp || 0);
      var eMix = clamp(0.6 * mus.energy + 0.4 * (vox.energy || 0), 0, 1);
      faceSmooth(dt, 0.03 * e2 + 0.08 * mus.sect * eMix,
                 // g54：没有歌词轨时也不许把眼睑拉开（同一条「不动眼珠大小」的约束）
                 0.03 * e2 + 0.03 * mus.sect * eMix,
                 0.07 * e2, 0.02 * e2 + 0.04 * mus.sect * eMix);
    } else {
      faceSmooth(dt, 0, 0, 0, 0);
    }
    // 起手快、收口慢。轻重之别主要就靠这一条：
    //   轻音 80ms 才张开、175ms 才收回，听起来（看起来）是「飘过去」的；
    //   重音 16ms 就咬下去、65ms 收口，是一下子的爆发。
    // 整首再按唱速缩放：快段（rate→1）×0.72 更跟手，慢段（→0）×1.22 更从容。
    // 长音例外：要跟得上干声的音高/颤音，平滑常数压到 50ms，嘴才会「呼吸」。
    var speed = 1.22 - 0.50 * rate;
    var holdNow = !!(hasCues && got && got.hold);
    var tau = holdNow ? 0.05
      : hasCues
      ? (open > lips.lastOpen ? (0.080 - 0.064 * stress)
                              : (0.175 - 0.110 * stress)) * speed
      : (open > lips.lastOpen ? LIPS_ATTACK : LIPS_RELEASE);
    var prevOpen = lips.lastOpen;
    lips.lastOpen += (open - prevOpen) * (1 - Math.exp(-dt / tau));
    // g55：硬限速。正常的张口/收口远到不了 22/s（重音那一冲也只有 ~15/s），
    // 超过它的只可能是时钟或包络的毛刺——直接压成斜坡，画面就不抽了。
    var dOpen = LIPS_OPEN_RATE * dt;
    var stepOpen = lips.lastOpen - prevOpen;
    if (stepOpen > dOpen || stepOpen < -dOpen) {
      lips.lastOpen = prevOpen + (stepOpen > 0 ? dOpen : -dOpen);
      lips.openClamp = (lips.openClamp || 0) + 1;
    }
    if (lips.lastOpen < 0.001) lips.lastOpen = 0;
    // 嘴形也跟着轻重和唱速走：重音换形快（咬得利落），轻音慢慢滑过去。
    var formTau = hasCues ? (0.115 - 0.060 * stress) * speed : LIPS_FORM_TAU;
    var prevForm = lips.lastForm;
    lips.lastForm += (form - prevForm) * (1 - Math.exp(-dt / formTau));
    var dForm = LIPS_FORM_RATE * dt;
    var stepForm = lips.lastForm - prevForm;
    if (stepForm > dForm || stepForm < -dForm) {
      lips.lastForm = prevForm + (stepForm > 0 ? dForm : -dForm);
      lips.formClamp = (lips.formClamp || 0) + 1;
    }
    // 句尾换气作用在「显示值」上，不写回平滑状态：否则乘性衰减会逐帧叠加，
    // 嘴会比设计更早、更久地闭上。breath 连续，所以不会跳变。
    var shown = lips.lastOpen;
    if (breath < 0.999) {
      shown *= breath;
      if (shown < 0.01) shown = 0;
    }

    if (bridged) {
      if (!lips.owner) {
        // 已经有别人（语音 TTS）占着嘴就别抢，等它说完再接管。
        var mgr = window.live2dManager;
        if (mgr && typeof mgr.hasActiveLipSync === 'function' && mgr.hasActiveLipSync()) return;
        var owner = {};
        try {
          if (!bridge.beginLipSync(owner)) return;  // 不是 live2d，或 TTS 正占着
        } catch (e) { return; }
        lips.owner = owner;
      }
      try {
        if (bridge.setMouth(shown, lips.owner) === false) lipsRelease();
        else lips.writes = (lips.writes || 0) + 1;
      } catch (e2) { lipsRelease(); }
    } else if (core) {
      // 没有主程序口型通道时自己写：嘴的开合通道（ParamMouthOpenY / LipSync 组 / ParamA）。
      if (paProfile(core)) pa().set(core, 'mouthOpen', shown);
    }

    // 嘴形（圆/扁）单独写：主程序只强制开合，不管 ParamMouthForm。
    if (core) {
      try {
        // 嘴形区间不一定是 -1..1（有的模型 0..1、有的 ±0.8），适配层按真实区间展开，
        // 免得噘嘴全被钳成同一个值。写平滑后的值：每 0.2 秒跳一次的嘴形同样显得
        // 「一个一个字地嚼」。没有嘴形参数、只有嘴角上扬（mao_pro）的模型写到那条上。
        var unit = clamp(lips.lastForm || 0, -1, 1);
        if (chHas(core, 'mouthForm')) pa().set(core, 'mouthForm', unit);
        else if (chHas(core, 'smile2')) pa().set(core, 'smile2', Math.max(0, unit) * 0.6);
      } catch (e4) { /* 模型没有嘴形参数 */ }
    }
  }

  // ---- 表情的平滑与写入 ----
  // 四条通道用不同的时间常数：眉毛/句首反应快（0.14s），笑眼最慢（0.34s），
  // 脸颊和眉角居中。一句唱完，笑眼会比眉毛晚一点落下来，像真的在收情绪。
  function faceSmooth(dt, brow, smile, cheek, angle) {
    // g62：起和落分开。眉毛起手 0.09s（重音一咬就看得到），落 0.16s；
    // 脸颊起手 0.20s、落 0.34s。以前两个方向同一个常数，表情永远像被
    // 一根橡皮筋匀速拉着——起得慢、落得也慢。
    var kb = 1 - Math.exp(-dt / (brow > face.brow ? 0.09 : 0.16));
    var ks = 1 - Math.exp(-dt / 0.65);   // g52：笑眼慢一些，不在句与句之间抽
    var kc = 1 - Math.exp(-dt / (cheek > face.cheek ? 0.20 : 0.34));
    var ka = 1 - Math.exp(-dt / 0.20);
    face.brow += (brow - face.brow) * kb;
    face.smile += (smile - face.smile) * ks;
    face.cheek += (cheek - face.cheek) * kc;
    face.angle += ((angle || 0) - face.angle) * ka;
    if (Math.abs(face.brow) < 0.002) face.brow = 0;
    if (Math.abs(face.smile) < 0.002) face.smile = 0;
    if (Math.abs(face.cheek) < 0.002) face.cheek = 0;
    if (Math.abs(face.angle) < 0.002) face.angle = 0;
  }

  // 模型有没有这些参数（按模型缓存一次）。不是每个模型都齐全：
  // yui 系列有笑眼+脸颊，kemomimi 有笑眼没脸颊，ziraitikuwa 两样都没有（还有眉毛）。
  // 眼开合（ParamEyeLOpen/ROpen）不在这里：它归主程序的眨眼/视线所有，
  // 每帧都会被绝对值覆盖，叠加只会和眨眼抢同一个参数。
  var faceIds = ['ParamBrowLY', 'ParamBrowRY', 'ParamBrowLAngle', 'ParamBrowRAngle',
                 'ParamEyeLSmile', 'ParamEyeRSmile', 'ParamCheek'];
  var faceCache = { core: null, has: null };
  function faceSupport(core) {
    if (faceCache.core === core) return faceCache.has;
    var has = {};
    for (var fi = 0; fi < faceIds.length; fi++) {
      var ok = false;
      try {
        ok = typeof core.getParameterIndex === 'function'
             && core.getParameterIndex(faceIds[fi]) >= 0;
      } catch (e) { ok = false; }
      has[faceIds[fi]] = ok;
    }
    faceCache = { core: core, has: has };
    var mask = '';
    for (var fj = 0; fj < faceIds.length; fj++) mask += has[faceIds[fj]] ? '1' : '0';
    faceStats.support = mask;
    return has;
  }

  /**
   * 在 beforeModelUpdate 里叠加面部表情。
   * 事件在「动作/表情/物理都算完、model.update 之前」触发；用 addParameterValueById
   * 加一层，不覆盖她本来的表情，也不会累积到下一帧（事件之后内部会 loadParameters
   * 还原参数）。这是唯一能「叠加且不打架」的写入点——写绝对值和待机动作抢同一个
   * 参数，写头/身体角度又会和物理、律动抢，都不行。
   *
   * 主程序自己也会在这个事件上挂常驻表情/手动表情的监听（它们写绝对值）。
   * EventEmitter 按注册顺序调用，所以每帧在 afterMotionUpdate 里把自己挪到队尾，
   * 保证本帧的 emit 里我们最后写——否则主程序换一次表情就会把我们的叠加整段盖掉，
   * 脸上就是「有表情 / 没表情」来回跳。
   *
   * ``fromEvent`` 为 true 说明确实是内部事件调到的；tick 用它判断事件到底有没有
   * 触发，1 秒没触发就改走 rAF 兜底（有的运行时 .on 存在但根本不 emit）。
   */
  var faceInEvent = false;
  function applyFace(core, fromEvent) {
    faceInEvent = !!fromEvent;
    if (fromEvent) faceStats.eventAt = performance.now();
    if (!core) {
      if (!hookedModel) return;
      var im0 = hookedModel.internalModel;
      core = im0 && im0.coreModel;
      if (!core) return;
    }
    // 唱歌态：脸部表情回默认基线（用户要求）。口型轨仍独立驱动嘴，动作层的
    // 手势仍由 gest 通道负责；这里只把「乐句表情」四条通道按住。
    var singing = !!lips.active;
    var b = singing ? 0 : face.brow, s = singing ? 0 : face.smile;
    var c = singing ? 0 : face.cheek, a = singing ? 0 : face.angle;
    // g59：拍点/句首的腮帮子脉冲，叠在乐句平均值上（乐句管「唱得多投入」，
    // 脉冲管「这一下鼓起来」）。
    var ck = singing ? 0 : eyeSt.cheek * EYE.cheekBeat;
    // 情绪层给这四条通道一个增益：高兴时笑眼更弯，低落/紧张时眉毛压下去。
    // 只调幅度、不新增通道，所以换模型时行为和 v43 一致，只是更「有情绪」。
    if (b) b = clamp(b * (1 + 0.30 * Math.max(0, emo.a) - 0.22 * Math.max(0, -emo.v)), 0, 1);
    if (s) s = clamp(s * (1 + 0.35 * Math.max(0, emo.v) - 0.18 * Math.max(0, -emo.a)), 0, 1);
    if (typeof core.addParameterValueById !== 'function') return;  // 老核心不支持叠加
    // g76：舞台接管时（body 带 neko-solo-stage），头身动作和单动作只由舞台排程；
    // 桥自己的手势/情绪层在舞台之后再叠一次，会和舞台的动作交叠（上一个没收完下一个就出来）。
    var stageOn76 = false;
    try { stageOn76 = !!(document.body && document.body.classList.contains('neko-solo-stage')); } catch (eS) { stageOn76 = false; }
    var emoWrote = stageOn76 ? 0 : applyEmo(core);   // 情绪层：眉头/眼形/视线/头的朝向
    var bodyWrote = stageOn76 ? 0 : applyBody(core); // 情绪→身体：在律动之上叠一层（含头发/呼吸）
    // 眼珠限幅必须排在 applyEmo 的视线之后：它读的是「这一帧所有写入者」
    // 叠加后的最终值，再决定要不要压回边界。
    var gazeWrote = applyGazeLimit(core);
    if (!b && !s && !c && !ck && !a && !emoWrote && !bodyWrote && !gazeWrote) return;
    faceSupport(core);
    var wrote = 0;
    function add(ch, value) {
      if (!value) return;
      wrote += chAdd(core, ch, value);
    }
    // 笑眼是唯一被两条通道同时加的参数：乐句通道（face.smile，上限 0.90）和
    // 情绪层（f.eyeSmile，上限 0.45）。两边各自看都合理，加起来能顶到参数
    // 上限——眼睛「笑没了」，而且只要一边被主程序的常驻表情压掉，总和就在
    // 上限和中间值之间来回跳（眼睛一抽一抽）。这里按模型区间给总和封顶。
    function addCapped(id, value, capFrac) {
      if (!value || !has[id]) return;
      try {
        var idx = core.getParameterIndex(id);
        var cur = Number(core.getParameterValueById(id)) || 0;
        var lo = Number(core.getParameterMinimumValue(idx));
        var hi = Number(core.getParameterMaximumValue(idx));
        if (!isFinite(lo) || !isFinite(hi) || hi <= lo) {
          core.addParameterValueById(id, value); wrote += 1; return;
        }
        var nv = Math.min(lo + capFrac * (hi - lo), cur + value);
        if (nv - cur <= 0.001) return;
        core.setParameterValueById(id, nv);
        wrote += 1;
      } catch (e) { /* 模型没有这个参数 */ }
    }
    add('browLY', b);
    add('browRY', b);
    add('browLAngle', a);
    add('browRAngle', a);
    // g60：笑眼不再写（见 applyEmo 里的说明）。乐句通道的 s 仍然算，用来
    // 判断这一帧要不要往下走，但不再落到 ParamEyeLSmile/EyeRSmile 上。
    add('cheek', c + ck);
    if (wrote || emoWrote || bodyWrote || gazeWrote) {
      faceStats.writes += 1;
      faceStats.last = b.toFixed(2) + ',' + s.toFixed(2) + ','
                     + c.toFixed(2) + ',' + a.toFixed(2) + '|' + emo.tag
                     // g62：第二套通道的实时值（/hello?ed= 里看得见）。g63 去掉 ear。
                     + '|x' + extra.smile2.toFixed(1)
                     + ',' + extra.mk.toFixed(1) + ',' + extra.teeth.toFixed(2);
    }
  }

  var frameSeq = 0;      // tick 每帧 +1
  var appliedSeq = -1;
  var faceHooked = false; // internalModel 事件可用时，表情在 beforeModelUpdate 写
  var faceListener = null;

  // 主程序也会在 beforeModelUpdate 上挂常驻表情/手动表情的监听，而且它们写的是
  // 绝对值。EventEmitter 按注册顺序调用，我们必须在它们之后写才不会被整段盖掉。
  // 主程序每次换表情都会重新注册自己的监听（于是又跑到我们后面），所以这里
  // 只负责「挂上」；挪队尾交给 moveFaceListenerToTail，每帧一次。
  function ensureFaceListener(im) {
    if (!im || typeof im.on !== 'function') { faceHooked = false; return false; }
    if (!faceListener) {
      faceListener = function () { applyFace(null, true); };
    }
    if (im.__nekoFaceOwner === faceListener) { faceHooked = true; return true; }
    try {
      im.on('beforeModelUpdate', faceListener);
      im.__nekoFaceOwner = faceListener;
      faceHooked = true;
      return true;
    } catch (e) {
      faceHooked = false;
      return false;
    }
  }

  // 把自己挪到 beforeModelUpdate 监听队列的队尾：主程序换表情时可能把它的
  // 绝对写入挂在我们后面，那一帧我们的叠加就被抹掉（脸上「有表情/没表情」跳）。
  // 每帧 afterMotionUpdate 里做一次 remove + push，保证本帧 emit 我们最后写。
  function moveFaceListenerToTail(im) {
    if (!im || im.__nekoFaceOwner !== faceListener) return;
    if (typeof im.off !== 'function') return;   // 老运行时：只挂在最后注册的位置
    try {
      im.off('beforeModelUpdate', faceListener);
      im.on('beforeModelUpdate', faceListener);
    } catch (e) { /* 忽略 */ }
  }

  // ---- 身体的唯一驱动源：MusicGroove ----
  // 所有写入都走引擎自己的 applyToModel（内部已做 0.06s 低通、softLim 平滑
  // 限幅、换段/换档的定点 smoothstep）。bridge 这边只管「什么时候交给它」
  // 和「从待机姿态怎么过渡进来」。
  // 同一帧里 afterMotionUpdate / rAF 兜底可能各调一次：只有第一次做 blend
  // （从模型当前值接到律动值），后面按目标值原样重写，过渡不会叠加。
  function grooveActive() {
    return !!(engine && blend > 0.15 && typeof window.MusicGroove !== 'undefined');
  }

  function grooveApply(core, forceBlend) {
    if (!core || !engine || !window.MusicGroove) return false;
    var first = appliedSeq !== frameSeq;
    if (first) appliedSeq = frameSeq;
    var useBlend = typeof forceBlend === 'number' ? forceBlend
      : (first ? blend : 1);
    paProfile(core);
    window.MusicGroove.applyToModel(core, engine.frame,
      // g69：桌面档下律动是最低一档，× 逐句幅度 × 仲裁（主动作在播时 30%）
      { gain: desk.on ? desk.gGain : 1, breathBase: 0.45, blend: useBlend });
    dbg.applied += 1;
    return true;
  }

  // 渲染前再写一遍。SDK 在 afterMotionUpdate 和 beforeModelUpdate 之间还会跑
  // updateFocus（视线）和 updateNaturalMovements（呼吸）的加法，不重写就会被
  // 拽偏一点、下一帧再弹回来——头/身「一抽一抽」就是这个。
  // 眼球参数单独保存/还原：律动会写 ParamEyeBallX/Y，但在这里写会盖掉主程序
  // 的视线跟随，那不该归唱歌管（stage.html 里没有待机视线，所以不用管）。
  function grooveReassertListener() {
    var im = hookedModel && hookedModel.internalModel;
    var core = im && im.coreModel;
    if (!core || !grooveActive()) return;
    var mx = chMap(core, 'eyeBallX'), my = chMap(core, 'eyeBallY');
    var ex = null, ey = null;
    try { ex = mx ? core.getParameterValueById(mx.id) : null; } catch (e) { ex = null; }
    try { ey = my ? core.getParameterValueById(my.id) : null; } catch (e2) { ey = null; }
    if (!grooveApply(core)) return;
    try {
      if (mx && typeof ex === 'number' && isFinite(ex)) core.setParameterValueById(mx.id, ex);
    } catch (e3) { /* 没有这个参数 */ }
    try {
      if (my && typeof ey === 'number' && isFinite(ey)) core.setParameterValueById(my.id, ey);
    } catch (e4) { /* 没有这个参数 */ }
  }

  function applyNow() {
    // g67：保证「眼球回默认值」的渲染前覆盖一直挂着（换模型/管理器重建后会重挂）
    ensureGazeHook();
    if (!hookedModel) return;
    var im = hookedModel.internalModel;
    var core = im && im.coreModel;
    if (!core) return;
    moveFaceListenerToTail(im);
    // 口型独立于律动：唱歌时即使没有频谱/律动也要动嘴。
    applyLips(core);
    // 身体只有一个驱动源。有律动数据就全交给 MusicGroove；没有就一次都不写，
    // 整段交回待机。绝不再退回「人声直接摆身体」——那正是身体上下抽动的源头。
    if (grooveActive()) grooveApply(core);
  }

  function hook(model) {
    if (!model || !model.internalModel || model === hookedModel) return;
    hookedModel = model;
    // 换模型：后台量一次这个模型的真实参数幅度（NekoL2DAdapt，约几十毫秒，量完还原），
    // 量好后适配层自动换上实测增益和替身参数；量好之前按区间换算，不阻塞第一帧。
    try {
      var C = window.NekoL2DAdapt;
      if (C && typeof C.calibrate === 'function') {
        C.calibrate(model).then(function () {
          try { dbg.adapter = pa() ? pa().describe(model.internalModel.coreModel) : null; } catch (e1) { /* 只是诊断 */ }
        }).catch(function () { /* 量不出就按区间换算 */ });
      }
    } catch (e0) { /* 没有校准器 */ }
    dbg.model = (model.internalModel.originalWidth || 0) + 'x' + (model.internalModel.originalHeight || 0);
    try {
      model.internalModel.on('afterMotionUpdate', applyNow);
    } catch (e) { /* 老运行时没有事件：交给 rAF 兜底 */ }
    // 渲染前最后一道：把这一帧算好的姿态原样再写一次。SDK 在这两个事件之间
    // 还会跑视线/呼吸的「加法」，任何晚到的写入都会让姿态偏一下、下一帧再弹回来
    // —— 那就是「一抽一抽」。再写一次之后渲染用的就是同一个值。
    try {
      if (model.internalModel.__nekoGrooveReassert !== grooveReassertListener) {
        model.internalModel.on('beforeModelUpdate', grooveReassertListener);
        model.internalModel.__nekoGrooveReassert = grooveReassertListener;
      }
    } catch (e2) { /* 老运行时：只靠 afterMotionUpdate */ }
    // 表情单独挂 beforeModelUpdate：那一刻她的表情/眨眼/视线都已经算完，
    // 叠加进去不会被覆盖。applyNow 还会再跑一次，但它只做绝对赋值，重复无妨。
    ensureFaceListener(model.internalModel);
  }

  // 监听器每 2 秒才刷新一次能量/频段，直接喂给引擎会让动作幅度每 2 秒「跳一格」，
  // 看起来就是一卡一卡的。先做 0.7s 低通，幅度变化变成连续的。
  var sm = { energy: 0, low: 0, mid: 0, high: 0, pitch: 0.5 };
  // 音乐停下时的收势包络：不是立刻交回待机动画，而是让动作自己慢慢停下来。
  var release = 0;
  // 最近一次的打击感（起音包络峰值）。监听器每 2 秒更新一次，这里原样转发，
  // 由引擎那边做 2 秒慢包络；播放停下时跟着 release 一起收。
  var punchNow = 0.45;
  // 重新开始播放时的缓入计时（暂停→继续 用更长的淡入，免得一上来就「啪」地进入律动）
  var resumeT = 1e9;
  var lastTarget = 0;

  function features(dt) {
    var active = isActive();
    if (active) {
      // 重新活跃时 release 不能一步跳回 1：能量是乘着 release 送进引擎的，
      // 直接置 1 会让幅度在一帧里回到满幅 —— 听感上就是「动一会儿咔一下」。
      // 用 0.25s 升回来，和停顿时 0.5s 的收势基本对称。
      if (release < 1) {
        release += (1 - release) * (1 - Math.exp(-dt / 0.25));
        if (release > 0.995) release = 1;
      }
      var e = Number(latest.energy) || 0;
      var k = 1 - Math.exp(-dt / 0.7);
      sm.energy += (e - sm.energy) * k;
      sm.low += ((Number(latest.low) || 0) - sm.low) * k;
      sm.mid += ((Number(latest.mid) || 0) - sm.mid) * k;
      sm.high += ((Number(latest.high) || 0) - sm.high) * k;
      sm.pitch += ((Number(latest.pitch) || 0.5) - sm.pitch) * k;
      punchNow = clamp(Number(latest.onset_peak) || 0, 0, 1);

      var clocked = beatPulse(dt);
      var beat;
      if (clocked >= 0) {
        beat = clocked;                       // 跟着歌里真实鼓点的时钟
      } else {
        beat = e - lastEnergy > 0.06 ? 1 : 0; // 没拿到包络时退回能量跳变
      }
      lastEnergy = e;
      // 突发重音包络：0.16s 衰减，和普通拍子互不干扰
      hitPulse *= Math.exp(-dt / 0.16);
      if (hitPulse < 0.01) hitPulse = 0;
      // 「这个速度/拍点有多可信」= 相位集中度。散拍、噪声、没鼓点的铺底
      // 都可能量出一个 BPM，但相位是散的；引擎靠这个把速度门控回慢速。
      var trust = clock.enabled ? clamp((clock.phaseConf - 0.20) / 0.30, 0, 1) : 0;
      return {
        // 乘 release：重新活跃时幅度从「刚才收到哪儿」接着涨回来，
        // 而不是一帧跳回满幅（release=1 时结果完全一样）。
        energy: sm.energy * release,
        low: sm.low * release,
        mid: sm.mid * release,
        high: sm.high * release,
        pitch: sm.pitch,
        punch: punchNow * release,
        bpm: clock.enabled ? clock.bpm : (Number(latest.bpm) || 100),
        tempoTrust: trust,
        regularity: clock.enabled ? clock.regularity : 0.5,
        beat: beat,
        hit: hitAmp * hitPulse,
        beatPos: clock.beatPos,
        beatConf: clock.phaseConf,
        silent: false
      };
    }

    // 不活跃（歌停了 / 关了开关 / 暂停）：按 0.5s 时间常数收势，约 1.6 秒衰减到 0，
    // 期间动作连贯地变小变慢，最后交回待机动画，而不是硬切。
    // 节拍脉冲也一起清掉：否则暂停时它停在半衰减，恢复播放的第一帧会冒出一个假重拍。
    clock.pulse = 0;
    hitPulse = 0;
    lastEnergy = 0;
    release *= Math.exp(-dt / 0.5);
    if (release < 0.03) release = 0;
    if (release === 0) {
      sm.energy = sm.low = sm.mid = sm.high = 0;
      return { energy: 0, beat: 0, hit: 0, beatPos: 0, beatConf: 0, silent: true, pitch: sm.pitch, punch: 0, bpm: clock.enabled ? clock.bpm : 100, tempoTrust: 0 };
    }
    return {
      energy: sm.energy * release,
      low: sm.low * release,
      mid: sm.mid * release,
      high: sm.high * release,
      pitch: sm.pitch,
      punch: punchNow * release,
      bpm: clock.enabled ? clock.bpm : 100,
      tempoTrust: 0,
      regularity: clock.regularity,
      beat: 0,
      hit: 0,
      beatPos: clock.beatPos,
      beatConf: 0,
      silent: false
    };
  }

  function tick() {
    requestAnimationFrame(tick);
    var now = performance.now();
    var dt = Math.min(0.1, (now - lastT) / 1000 || 0.016);
    lastT = now;
    frameSeq += 1;
    hook(currentModel());
    // 探针挂在 coreModel.update 上；主程序清理时会还原，所以每帧确认一次（有身份判断，开销为零）。
    if (hookedModel && hookedModel.internalModel) probeInstall(hookedModel.internalModel.coreModel);
    if (!engine) return;
    // 频谱优先取「主窗口自己播的成品」——这首歌的伴奏、鼓点、副歌力度都在里面，
    // 口型轨里的干净人声只用来做嘴，不再参与身体。
    // 拿不到（走 ffplay / 建不起 AudioContext / CORS 被拒）才退回 48765 的
    // 系统回环监听，行为和以前一样。
    var feat = null;
    var local = false;
    if (mixAudioPlaying() && ensureMixAudio()) {
      feat = mixFeatures(dt);
      if (feat) local = true;
    }
    var target;
    if (feat) {
      target = 1;                 // 歌在放：律动全程接管，静音段由引擎自己收势
    } else {
      feat = features(dt);
      target = isActive() ? 1 : 0;
    }
    dbg.src = local ? 1 : 0;
    if (target > 0 && lastTarget <= 0) resumeT = 0;   // 从静止重新开始
    lastTarget = target;
    resumeT += dt;
    // 暂停→继续（或切歌）时用 0.55s 的缓入从待机姿态接进来；正常入场 0.35s，退场 0.5s
    var tauIn = resumeT < 1.2 ? 0.55 : 0.35;
    blend += (target - blend) * (1 - Math.exp(-dt / (target > blend ? tauIn : 0.5)));
    if (blend < 0.002) blend = 0;
    dbg.active = blend > 0;
    dbg.blend = blend;
    // g76：舞台接管时，律动引擎自己的「强拍随机大甩头」关掉（冷却拉满），只留跟拍律动；
    // 大动作由舞台的单动作统一排程，避免两套大动作交叠。
    try { if (document.body && document.body.classList.contains('neko-solo-stage')) { engine._accentCooldown = 99; engine._accent = null; } } catch (eS2) { }
    engine.update(feat, dt);
    // 表情用的伴奏侧特征（能量 / 段落 / 拍点）：和身体/嘴分开采样，
    // 这样即使某帧没在唱，表情也能跟着音乐收放。
    musicSample(dt);
    // 情绪层：先把这一帧的 VAD（正负/唤醒/掌控）和 FACS 解出来，applyFace
    // 里再统一写；放在 musicSample 之后，用的就是本帧刚采到的能量和段落。
    updateEmotion(dt);
    // 事件可用时这里是重复的绝对赋值（结果相同）；事件不可用时它就是兜底。
    applyNow();
    // 表情：监听在 applyNow（afterMotionUpdate）里已经挪到队尾；如果 1 秒内
    // 事件一次都没触发（运行时 .on 存在但根本不 emit），就在 rAF 里补写一次。
    var im = hookedModel && hookedModel.internalModel;
    if (im) ensureFaceListener(im);
    // 眼睑 + 腮帮子（g59）：只在唱歌时登记「渲染前最后姿态覆盖」，
    // 停唱立刻撤掉并把状态归零，待机一帧都不碰眼睛。
    if (eyeSt.on) {
      ensureEyeHook();
    } else if (eyeSt.n) {
      releaseEyeHook();
      eyeSt.n = 0; eyeSt.blink = 0; eyeSt.cheek = 0;
      eyeSt.cur = 1; eyeSt.k = 1; eyeSt.baseTok = -1;
    }
    if (now - (faceStats.eventAt || 0) > 1000) {
      applyFace(im && im.coreModel);
    }
  }

  // g74：48765 回环服务没开时（为她所听插件停着），fetch 会一直挂在连接上；
  // 以前每 100ms 发一条、从不等上一条结束，几分钟就把渲染进程的连接配额耗光
  // （net::ERR_INSUFFICIENT_RESOURCES），连 /lips.json、/song.wav 也发不出去。
  // 现在同一时间只留一条请求，连续失败就按 0.5→8 秒退避。
  var pollBusy = false, pollFails = 0, pollNextAt = 0;
  function poll() {
    if (pollBusy) return;
    if (pollFails && performance.now() < pollNextAt) return;
    pollBusy = true;
    var ctl = typeof AbortController === 'function' ? new AbortController() : null;
    var killer = ctl ? setTimeout(function () { ctl.abort(); }, 1500) : null;
    fetch(BASE + '/state.json', ctl ? { cache: 'no-store', signal: ctl.signal } : { cache: 'no-store' })
      .then(function (r) { return r.json(); })
      .then(function (st) {
        pollBusy = false; pollFails = 0; if (killer) clearTimeout(killer);
        latest = st;
        var at = (st && Number(st.onset_at)) || 0;
        if (at <= 0) return;
        // 每轮都先看有没有「突然的一声」——它不等节拍，越快越好。
        detectHits(st);
        // 每次都校相位（BPM 在自己里面限成 0.8 秒一次）：
        // 相位修得越勤，拍点和鼓点贴得越紧。
        if (at !== clock.envAt || at < clock.envAt - 0.5) {
          clock.envAt = at;
          analyzeEnvelope(st);
        }
      })
      .catch(function () {
        pollBusy = false; if (killer) clearTimeout(killer);
        latest = null;
        pollFails += 1;
        pollNextAt = performance.now() + Math.min(8000, 500 * Math.pow(2, Math.min(4, pollFails - 1)));
      });
  }

  function start() {
    if (window.__nekoMusicGroove) return;
    window.__nekoMusicGroove = true;
    window.__nekoMusicGrooveDbg = dbg;
    window.__nekoMusicFace = faceStats;
    window.__nekoMusicMix = mixAudio;
    window.__nekoMusicVox = vox;
    window.__nekoMusicMus = mus;
    window.__nekoMusicDesk = desk;     // g69：桌面档状态（只读诊断，CDP 取）
    // 参数适配层：当前模型每个语义通道落到了哪个参数（只读诊断，CDP 取）
    window.__nekoMusicAdapter = function () {
      var c = hookedModel && hookedModel.internalModel && hookedModel.internalModel.coreModel;
      return c && pa() ? (paProfile(c), pa().describe(c)) : null;
    };
    engine = new window.MusicGroove({ style: 'auto', intensity: 1, variation: 0.8, seed: 'neko' });
    setInterval(poll, 100);
    poll();
    setInterval(lipsPoll, 200);
    lipsPoll();
    requestAnimationFrame(tick);
  }

  function ping() {
    var q = BASE + '/hello?v=' + VERSION + '&model=' + (hookedModel ? 1 : 0)
      + '&applied=' + dbg.applied + '&groove=' + (typeof window.MusicGroove === 'function' ? 1 : 0)
      + '&bpm=' + (dbg.bpm || 0) + '&onsets=' + (dbg.onsets || 0) + '&lag=' + (dbg.lag || 0)
      + '&lips=' + (lips.active ? 1 : 0) + '&lseq=' + (lips.seq || 0)
      + '&lport=' + (lips.port || 0) + '&lcues=' + (lips.cues ? lips.cues.length : 0)
      + '&lenv=' + (lips.env ? lips.env.length : 0) + '&lmode=' + (lips.mode || '')
      + '&lw=' + (lips.writes || 0) + '&lown=' + (lips.owner ? 1 : 0)
      // 表情诊断：support 是 7 位掩码（眉/眉/眉角/眉角/笑眼/笑眼/脸颊 哪些存在），
      // fw 是写入帧数，fv 是当前叠加值（眉,笑眼,脸颊,眉角）。
      + '&face=' + (faceStats.support || '') + '&fw=' + (faceStats.writes || 0)
      + '&fh=' + (faceHooked ? 1 : 0) + '&fv=' + (faceStats.last || '')
      // g64：贴图特效的实时值（脸红,眼泪,花花,脸黑的平滑后写入量）。
      + '&fx=' + extra.blush.toFixed(2) + ',' + extra.tears.toFixed(2)
      + ',' + extra.flower.toFixed(2) + ',' + extra.darkFace.toFixed(2)
      // 情绪→身体（g47）：bs 是 7 位「身体参数支持表」（BodyX/Y/Z、呼吸、三束头发），
      // bd 是本帧实际叠上去的偏移 x,y,z,呼吸,前发,侧发,后发；on=1 表示在叠。
      + '&bs=' + (faceStats.support3 || '') + '&bd=' + bodyTag()
      // 情绪→动作（g58）：gs=当前动作名,次数；gv=本帧动作层的头 x,y,z 和上身 y
      // （单位：度）。n 一直不涨说明触发条件没满足，=- 说明当前没在播动作。
      + '&gs=' + (gestSt.label || '-') + ',' + (gestSt.n || 0)
      + '&gv=' + (gestSt.out.headX * GEST.head).toFixed(2) + ','
      + (gestSt.out.headY * GEST.head).toFixed(2) + ','
      + (gestSt.out.headZ * GEST.head).toFixed(2) + ','
      + (gestSt.out.bodyY * GEST.body).toFixed(2)
      + '&ey=' + probeTag() + '&ed=' + probeTag2()
      // 全身：mix=1 表示身体在跟「本地成品混音」的分析（0=回退 48765 回环），
      // me=混音总能量，mb=估出来的 BPM，bl=律动权重（0=待机全权），mf=分析帧数。
      // 身体只由 MusicGroove 一个源驱动，所以这里不再有「人声全身」的权重。
      + '&mix=' + (dbg.src || 0) + '&me=' + (mixAudio.energy || 0).toFixed(2)
      + '&mb=' + Math.round(mixAudio.bpm || 100) + '&mf=' + (mixAudio.frames || 0)
      + '&bl=' + (blend || 0).toFixed(2)
      // 干声分析：vx=可用(1/0) ve=响度 vf0=基频 vv=颤音 vc=清晰度；
      // 伴奏侧：msec=段落权重 msn=段落名 mmb=拍点。都是给排查用的。
      + '&vx=' + (vox.ok ? 1 : 0) + '&ve=' + (vox.energy || 0).toFixed(2)
      + '&vf0=' + Math.round(vox.f0 || 0) + '&vv=' + (vox.vib || 0).toFixed(2)
      + '&vc=' + (vox.clarity || 0).toFixed(2)
      + '&msec=' + (mus.sect || 0).toFixed(2) + '&msn=' + (mus.sectName || '-')
      + '&mmb=' + (mus.beat || 0).toFixed(2)
      // g69 桌面档：on,曲目,幅度k,仲裁,当前动作,随机手势数,脚本动作数,被压数,
      // 最大对拍误差ms,眨眼总数/句尾/被禁/兜底,视线
      + '&dk=' + encodeURIComponent(deskTag());
    fetch(q, { cache: 'no-store' }).catch(function () {});
  }

  // 带版本号，避免主窗口复用上一轮的脚本缓存。
  load(STATIC + 'live2d_adapt.js?v=' + VERSION)
    .catch(function () { return load(BASE + '/live2d_adapt.js?v=' + VERSION); })
    .catch(function () { /* 实测校准是可选的：没有它就只按区间换算 */ })
    .then(function () {
      return load(STATIC + 'param_adapter.js?v=' + VERSION)
        .catch(function () { return load(BASE + '/param_adapter.js?v=' + VERSION); });
    })
    .then(function () {
      return load(STATIC + 'music_groove.js?v=' + VERSION)
        .catch(function () { return load(BASE + '/music_groove.js?v=' + VERSION); });
    })
    .then(function () {
      return load(STATIC + 'music_visemes.js?v=' + VERSION)
        .catch(function () { return load(BASE + '/music_visemes.js?v=' + VERSION); });
    })
    .then(function () { start(); setInterval(ping, 5000); ping(); })
    .catch(function () {
      try {
        fetch(BASE + '/hello?v=' + VERSION + '&stage=loadfail', { cache: 'no-store' }).catch(function () {});
      } catch (e) { /* 忽略 */ }
    });
})();
