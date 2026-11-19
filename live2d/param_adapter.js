/**
 * 模型参数适配层（为她所听 · Live2D）
 *
 * 动作 / 表情 / 嘴型代码只说「意图」（语义通道 + 归一化数值），这里负责把它
 * 落到当前模型上：
 *   1. 找参数：标准 ID 优先；没有就按 model3.json 的 LipSync/EyeBlink 分组、
 *      cdi3 里的中英日名字、模型自带表情文件去认。认不出就跳过并记下来，
 *      绝不按编号（Param57 之类）去猜——编号在每个模型里含义都不一样。
 *   2. 换算数值：通道统一用归一化量（双向 -1..1，单向 0..1），写入时按模型
 *      自己的最小/最大/默认值展开。±30 的眼球和 ±1 的眼球同样写 0.5，
 *      看起来转的幅度一样。
 *
 * 一个 coreModel 只解析一次（结果缓存在 WeakMap 里），每帧只做查表和乘法。
 * 只依赖 Cubism Core 的公开 API：getParameterIndex / getParameterCount /
 * getParameterMinimumValue / getParameterMaximumValue / getParameterDefaultValue /
 * getParameterValueById / setParameterValueById / addParameterValueById。
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.NekoParamAdapter = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ---- 语义通道 ----
  // kind: 'bi' 双向（以默认值为中心，-1..1 映射到 默认→最小 / 默认→最大）
  //       'uni' 单向（0..1 映射到 默认值→离默认最远的那一端）
  // ids: 标准 ID（大多数模型遵循 Cubism 标准参数表）
  // names: cdi3 显示名的正则（中文 / 日文 / 英文），标准 ID 缺失时才用
  // exclude: 名字命中但不该用的（例如「眼泪的摇晃」不是「流泪」本身）
  var CHANNELS = {
    headX:      { kind: 'bi', ids: ['ParamAngleX'], names: /^角度\s*X$|angle\s*x$/i },
    headY:      { kind: 'bi', ids: ['ParamAngleY'], names: /^角度\s*Y$|angle\s*y$/i },
    headZ:      { kind: 'bi', ids: ['ParamAngleZ'], names: /^角度\s*Z$|angle\s*z$/i },
    bodyX:      { kind: 'bi', ids: ['ParamBodyAngleX'], names: /身体.*(旋转|角度).*X|体.*X|body.*x/i },
    bodyY:      { kind: 'bi', ids: ['ParamBodyAngleY'], names: /身体.*(旋转|角度).*Y|体.*Y|body.*y/i },
    bodyZ:      { kind: 'bi', ids: ['ParamBodyAngleZ'], names: /身体.*(旋转|角度).*Z|体.*Z|body.*z/i },
    breath:     { kind: 'uni', ids: ['ParamBreath'], names: /呼吸|breath/i },
    eyeBallX:   { kind: 'bi', ids: ['ParamEyeBallX'], names: /眼球\s*X|eye\s*ball\s*x/i },
    eyeBallY:   { kind: 'bi', ids: ['ParamEyeBallY'], names: /眼球\s*Y|eye\s*ball\s*y/i },
    eyeLOpen:   { kind: 'uni', ids: ['ParamEyeLOpen'], group: 'EyeBlink', side: 'L' },
    eyeROpen:   { kind: 'uni', ids: ['ParamEyeROpen'], group: 'EyeBlink', side: 'R' },
    eyeLSmile:  { kind: 'uni', ids: ['ParamEyeLSmile'], names: /左眼.*(微笑|笑)|eye\s*l.*smile/i },
    eyeRSmile:  { kind: 'uni', ids: ['ParamEyeRSmile'], names: /右眼.*(微笑|笑)|eye\s*r.*smile/i },
    browLY:     { kind: 'bi', ids: ['ParamBrowLY'], names: /左眉.*(上下|Y)|brow\s*l.*y/i },
    browRY:     { kind: 'bi', ids: ['ParamBrowRY'], names: /右眉.*(上下|Y)|brow\s*r.*y/i },
    browLAngle: { kind: 'bi', ids: ['ParamBrowLAngle'], names: /左眉.*角度|brow\s*l.*angle/i },
    browRAngle: { kind: 'bi', ids: ['ParamBrowRAngle'], names: /右眉.*角度|brow\s*r.*angle/i },
    browLForm:  { kind: 'bi', ids: ['ParamBrowLForm'], names: /左眉.*(变形|形)|brow\s*l.*form/i },
    browRForm:  { kind: 'bi', ids: ['ParamBrowRForm'], names: /右眉.*(变形|形)|brow\s*r.*form/i },
    mouthOpen:  { kind: 'uni', ids: ['ParamMouthOpenY', 'ParamMouthOpen'], group: 'LipSync',
                  names: /嘴.*(张开|開閉|开合)|口.*開閉|mouth\s*open/i, alt: ['ParamA'] },
    mouthForm:  { kind: 'bi', ids: ['ParamMouthForm'], names: /嘴.*变形|口.*変形|mouth\s*form/i },
    cheek:      { kind: 'uni', ids: ['ParamCheek'], names: /^(脸红|臉紅|照れ|赤面|blush)|脸颊.*泛红|頬/i,
                  exclude: /呼吸|鼓|puff/i },
    hairFront:  { kind: 'bi', ids: ['ParamHairFront'], names: /前发.*(摇|摆)|hair\s*front/i },
    hairSide:   { kind: 'bi', ids: ['ParamHairSide'], names: /侧发.*(摇|摆)|hair\s*side/i },
    hairBack:   { kind: 'bi', ids: ['ParamHairBack'], names: /后发.*(摇|摆)|hair\s*back/i },
    // ---- 以下是模型私有的效果，只认名字，不认编号 ----
    smile2:     { kind: 'bi', names: /^(笑|嘴角)$|mouth\s*corner\s*up/i, exclude: /眼|eye/i },
    mouthX:     { kind: 'bi', names: /^嘴\s*X$|^mouth\s*x$/i, prefer: /^嘴/ },
    teeth:      { kind: 'uni', names: /^(牙|牙齿|歯)$|teeth|tooth/i },
    blushFx:    { kind: 'uni', names: /^(脸红|臉紅|照れ|赤面)$/ },
    tears:      { kind: 'uni', names: /^(流泪|泪|淚|涙)$|^tears?$/i, exclude: /摇|揺|眼泪[LR]\d/ },
    tearfulEye: { kind: 'uni', names: /^(涙目|泪眼|泪目)$|tearful/i },
    darkFace:   { kind: 'uni', names: /^(脸黑|黒顔|阴影脸)$/ },
    angerMark:  { kind: 'uni', names: /^(生气|怒|怒り|青筋|anger)$/i },
    sweat:      { kind: 'uni', names: /^(汗|冷汗)$|^sweat$/i },
    pout:       { kind: 'uni', names: /嘟嘴|pout/i },
    flower:     { kind: 'uni', names: /^花花$/ }
  };

  // 情绪 → 模型自带表情文件的名字关键词（没有对应参数时，用表情文件兜底）
  var EXPRESSION_HINTS = {
    happy:    /happy|joy|smile|笑|喜|开心|高兴|嬉|xxy|wy/i,
    sad:      /sad|cry|悲|哭|伤心|泣|涙/i,
    angry:    /angry|anger|怒|生气|气/i,
    surprised:/surpris|惊|驚|吃惊/i,
    shy:      /shy|blush|害羞|照れ|脸红/i
  };

  var cache = (typeof WeakMap === 'function') ? new WeakMap() : null;
  var lastProfile = null;

  function num(v, d) { v = Number(v); return isFinite(v) ? v : d; }
  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  function paramCount(core) {
    try { return num(core.getParameterCount(), 0); } catch (e) { return 0; }
  }

  function paramIds(core) {
    // Cubism Web Framework 的 CubismModel 有 _parameterIds；底层 Live2DCubismCore
    // 是 _model.parameters.ids。都拿不到时只能按标准 ID 逐个探测。
    try {
      if (core._parameterIds && core._parameterIds.length) {
        return core._parameterIds.map(function (x) {
          return typeof x === 'string' ? x : (x && typeof x.getString === 'function'
            ? String(x.getString().s || x.getString()) : String(x && (x.s || x.id || x)));
        });
      }
    } catch (e) { /* 换下一种 */ }
    try {
      var ids = core._model && core._model.parameters && core._model.parameters.ids;
      if (ids && ids.length) return Array.prototype.slice.call(ids).map(String);
    } catch (e2) { /* 换下一种 */ }
    return null;
  }

  function rangeOf(core, idx) {
    var lo = 0, hi = 1, def = 0;
    try { lo = num(core.getParameterMinimumValue(idx), 0); } catch (e) { lo = 0; }
    try { hi = num(core.getParameterMaximumValue(idx), 1); } catch (e) { hi = 1; }
    try { def = num(core.getParameterDefaultValue(idx), NaN); } catch (e) { def = NaN; }
    if (!(hi > lo)) return null;              // 锁死的参数当作不存在
    if (!isFinite(def)) def = (lo <= 0 && hi >= 0) ? 0 : lo;
    def = clamp(def, lo, hi);
    return { lo: lo, hi: hi, def: def };
  }

  // 双向通道的「中心」：通常是默认值；但默认值贴在一端的模型（例如嘴形 0..1、
  // 默认 0）没有另一侧可用，这时改用区间中点，-1..1 铺满整个区间。
  function biCenter(m) {
    var span = m.hi - m.lo;
    if (m.def - m.lo < span * 0.1 || m.hi - m.def < span * 0.1) return (m.lo + m.hi) / 2;
    return m.def;
  }

  function exists(core, id, count) {
    var idx = -1;
    try { idx = core.getParameterIndex(id); } catch (e) { idx = -1; }
    // Cubism 对不存在的 ID 会分配一个虚拟下标（>= 参数总数），要排除掉
    if (!(idx >= 0) || (count > 0 && idx >= count)) return -1;
    return idx;
  }

  /**
   * 为一个 coreModel 建立映射。info（可选）：
   *   { settings: model3.json 对象, displayInfo: cdi3.json 对象, name: 模型名 }
   */
  function profile(core, info) {
    if (!core) return null;
    if (cache) {
      var hit = cache.get(core);
      if (hit && (!info || hit.infoKey === infoKey(info))) return hit;
    }
    info = info || {};
    var count = paramCount(core);
    var ids = paramIds(core);
    var names = {};
    var dParams = info.displayInfo && info.displayInfo.Parameters;
    if (dParams && dParams.length) {
      for (var i = 0; i < dParams.length; i++) {
        if (dParams[i] && dParams[i].Id) names[dParams[i].Id] = String(dParams[i].Name || '').trim();
      }
      if (!ids) ids = dParams.map(function (p) { return p.Id; });
    }
    var groups = {};
    var gs = info.settings && info.settings.Groups;
    if (gs && gs.length) {
      for (var g = 0; g < gs.length; g++) {
        if (gs[g] && gs[g].Target === 'Parameter' && gs[g].Ids && gs[g].Ids.length) groups[gs[g].Name] = gs[g].Ids.slice();
      }
    }

    var map = {}, used = {}, missing = [], via = {};
    function take(ch, id, how) {
      var idx = exists(core, id, count);
      if (idx < 0 || used[id]) return false;
      var r = rangeOf(core, idx);
      if (!r) return false;
      map[ch] = { id: id, idx: idx, lo: r.lo, hi: r.hi, def: r.def, kind: CHANNELS[ch].kind };
      used[id] = ch;
      via[ch] = how;
      return true;
    }
    Object.keys(CHANNELS).forEach(function (ch) {
      var spec = CHANNELS[ch], k;
      if (spec.ids) for (k = 0; k < spec.ids.length; k++) if (take(ch, spec.ids[k], 'std')) return;
      if (spec.group && groups[spec.group]) {
        var gIds = groups[spec.group];
        if (spec.side) {
          // EyeBlink 组里按 L/R（或 左/右）挑一只眼
          var pick = gIds.filter(function (id) {
            var n = (names[id] || '') + ' ' + id;
            return spec.side === 'L' ? /L(?![a-z])|左|left/i.test(n) : /R(?![a-z])|右|right/i.test(n);
          });
          for (k = 0; k < pick.length; k++) if (take(ch, pick[k], 'group')) return;
        } else {
          for (k = 0; k < gIds.length; k++) if (take(ch, gIds[k], 'group')) return;
        }
      }
      if (spec.names && ids && spec.prefer) {
        // 同时有多个名字命中时，优先画师起的中文名（例如「嘴X」优于面捕输入「MOUTHX」）
        for (k = 0; k < ids.length; k++) {
          var pid = ids[k], pn = names[pid] || '';
          if (pn && !used[pid] && spec.prefer.test(pn) && spec.names.test(pn) && take(ch, pid, 'name')) return;
        }
      }
      if (spec.names && ids) {
        for (k = 0; k < ids.length; k++) {
          var id = ids[k], n = names[id] || '';
          if (!n || used[id]) continue;
          if (spec.exclude && spec.exclude.test(n)) continue;
          if (spec.names.test(n) && take(ch, id, 'name')) return;
        }
      }
      if (spec.alt) for (k = 0; k < spec.alt.length; k++) if (take(ch, spec.alt[k], 'alt')) return;
      missing.push(ch);
    });

    var exps = [];
    var fr = info.settings && info.settings.FileReferences;
    if (fr && fr.Expressions && fr.Expressions.length) {
      exps = fr.Expressions.map(function (e) { return { name: String(e.Name || ''), file: String(e.File || '') }; });
    }

    var p = {
      core: core, name: info.name || '', infoKey: infoKey(info), count: count,
      map: map, via: via, missing: missing, expressions: exps, calibrated: false,
      has: function (ch) { return !!map[ch]; }
    };
    applyCalibration(p);
    if (cache) cache.set(core, p);
    lastProfile = p;
    return p;
  }

  function infoKey(info) {
    if (!info) return '';
    return (info.name || '') + '|' + (info.settings ? 1 : 0) + '|' + (info.displayInfo ? 1 : 0);
  }

  // ---- 实测校准（live2d_adapt.js / NekoL2DAdapt）----
  // 那边在模型加载时用 Cubism Core 逐个参数推一下、量网格真实位移：
  //   · 标准参数存在但不动（白毛宝宝的 ParamAngleX 只喂物理）→ 换成真正干活的替身；
  //   · 每个通道一个增益，让同样的动作在各模型上看起来一样大（以 Blue cat 为 1 倍）。
  // 有校准结果就叠在本层的通道上；没有（还在量 / 量不出）就只按区间换算。
  var CAL_STD = { headX: 'ParamAngleX', headY: 'ParamAngleY', headZ: 'ParamAngleZ',
                  bodyX: 'ParamBodyAngleX', bodyY: 'ParamBodyAngleY', bodyZ: 'ParamBodyAngleZ',
                  eyeBallX: 'ParamEyeBallX', eyeBallY: 'ParamEyeBallY',
                  browLY: 'ParamBrowLY', browRY: 'ParamBrowRY', browLForm: 'ParamBrowLForm',
                  browRForm: 'ParamBrowRForm', browLAngle: 'ParamBrowLAngle', browRAngle: 'ParamBrowRAngle',
                  mouthOpen: 'ParamMouthOpenY', mouthForm: 'ParamMouthForm' };
  function calibrator() {
    var g = (typeof self !== 'undefined') ? self : (typeof globalThis !== 'undefined' ? globalThis : null);
    return g && g.NekoL2DAdapt || null;
  }
  function calProfile(core) {
    var C = calibrator();
    if (!C || !core) return null;
    // NekoL2DAdapt 按 model（internalModel.coreModel）缓存；这里只有 core，借它的 profileOf 需要 model 外壳
    try { var cp = C.profileOf({ internalModel: { coreModel: core } }); return cp && cp.ok ? cp : null; }
    catch (e) { return null; }
  }
  function applyCalibration(p) {
    var cp = calProfile(p.core);
    if (!cp) return false;
    Object.keys(CAL_STD).forEach(function (ch) {
      var cm = cp.map[CAL_STD[ch]];
      if (!cm) return;
      if (!cm.id) {                               // 实测：这个模型做不出这个通道
        if (p.map[ch]) { delete p.map[ch]; if (p.missing.indexOf(ch) < 0) p.missing.push(ch); p.via[ch] = 'dead'; }
        return;
      }
      var idx = exists(p.core, cm.id, p.count);
      var r = idx >= 0 ? rangeOf(p.core, idx) : null;
      if (!r) return;
      var m = { id: cm.id, idx: idx, lo: r.lo, hi: r.hi, def: r.def, kind: CHANNELS[ch].kind,
                calStd: CAL_STD[ch] };
      p.map[ch] = m;
      var mi = p.missing.indexOf(ch); if (mi >= 0) p.missing.splice(mi, 1);
      p.via[ch] = 'cal:' + (cm.why || 'std');
    });
    p.cal = cp;
    p.calibrated = true;
    return true;
  }

  // 归一化量 → 模型数值
  function toRaw(m, v) {
    if (m.kind === 'uni') {
      v = clamp(v, 0, 1);
      // 单向：从默认值往离它最远的一端展开（0..1 / 0..30 / -1..0 都适用）
      var far = (m.hi - m.def) >= (m.def - m.lo) ? m.hi : m.lo;
      return m.def + (far - m.def) * v;
    }
    v = clamp(v, -1, 1);
    var c = biCenter(m);
    return v >= 0 ? c + (m.hi - c) * v : c + (c - m.lo) * v;
  }

  // 归一化增量 → 模型增量（叠加写用）。按区间半宽换算，对称模型与旧行为一致。
  function toRawDelta(m, dv) {
    if (m.kind === 'uni') {
      var far = (m.hi - m.def) >= (m.def - m.lo) ? m.hi : m.lo;
      return (far - m.def) * dv;
    }
    var c = biCenter(m);
    return dv >= 0 ? (m.hi - c) * dv : (c - m.lo) * dv;
  }

  function toNorm(m, raw) {
    if (m.kind === 'uni') {
      var far = (m.hi - m.def) >= (m.def - m.lo) ? m.hi : m.lo;
      return far === m.def ? 0 : clamp((raw - m.def) / (far - m.def), 0, 1);
    }
    var c = biCenter(m);
    if (raw >= c) return m.hi === c ? 0 : clamp((raw - c) / (m.hi - c), 0, 1);
    return m.lo === c ? 0 : -clamp((c - raw) / (c - m.lo), 0, 1);
  }

  // 校准过的通道：调用方的归一化量先换回「参考模型单位」，再交给 NekoL2DAdapt 换算
  // （它的增益 / 张嘴曲线都以参考单位为输入）。
  var CAL_HALF = { ParamAngleX: 30, ParamAngleY: 30, ParamAngleZ: 30,
                   ParamBodyAngleX: 10, ParamBodyAngleY: 10, ParamBodyAngleZ: 10 };
  function calHalf(m) { return CAL_HALF[m.calStd] || 1; }
  function refreshCal(p) {
    // 校准是异步算完的：profile 建好之后才有结果，这里补一次（只做一次查表）
    if (!p.calibrated && calProfile(p.core)) applyCalibration(p);
  }

  /** 绝对写：v 是归一化量。返回 1 = 写了，0 = 模型没有这个通道。 */
  function set(core, ch, v) {
    var p = profile(core);
    if (p) refreshCal(p);
    var m = p && p.map[ch];
    if (!m || !isFinite(v)) return 0;
    try {
      if (m.calStd && p.cal) {
        var r = calibrator().resolve(p.cal, m.calStd, v * calHalf(m));
        if (!r) return 0;
        core.setParameterValueById(r[0], r[1]);
        return 1;
      }
      core.setParameterValueById(m.id, toRaw(m, v));
      return 1;
    } catch (e) { return 0; }
  }

  /**
   * 归一化量 → [真实参数 ID, 写入值]，不写。校准过的通道走实测增益（按水水对齐），
   * 没校准的按区间换算。律动引擎要自己做平滑 / 限速再写，用这个拿目标值。
   */
  function rawOf(core, ch, v) {
    var p = profile(core);
    if (p) refreshCal(p);
    var m = p && p.map[ch];
    if (!m || !isFinite(v)) return null;
    try {
      if (m.calStd && p.cal) {
        var r = calibrator().resolve(p.cal, m.calStd, clamp(v, -1, 1) * calHalf(m));
        return r ? [r[0], r[1]] : null;
      }
      return [m.id, toRaw(m, v)];
    } catch (e) { return null; }
  }

  /** 叠加写：dv 是归一化增量，叠到模型当前值上（结果钳在区间内）。 */
  function add(core, ch, dv) {
    var p = profile(core);
    if (p) refreshCal(p);
    var m = p && p.map[ch];
    if (!m || !isFinite(dv) || !dv) return 0;
    try {
      var cur = num(core.getParameterValueById(m.id), m.def);
      var delta;
      if (m.calStd && p.cal) {
        var r = calibrator().resolveDelta(p.cal, m.calStd, dv * calHalf(m));
        if (!r) return 0;
        delta = r[1];
      } else {
        delta = toRawDelta(m, dv);
      }
      core.setParameterValueById(m.id, clamp(cur + delta, m.lo, m.hi));
      return 1;
    } catch (e) { return 0; }
  }

  /** 读：返回归一化量；模型没有这个通道返回 NaN。 */
  function get(core, ch) {
    var p = profile(core), m = p && p.map[ch];
    if (!m) return NaN;
    try { return toNorm(m, num(core.getParameterValueById(m.id), m.def)); } catch (e) { return NaN; }
  }

  /** 把通道放回模型默认值（唱完复位用）。 */
  function reset(core, ch) {
    var p = profile(core), m = p && p.map[ch];
    if (!m) return 0;
    try { core.setParameterValueById(m.id, m.def); return 1; } catch (e) { return 0; }
  }

  // ---- 参考模型单位 ----
  // 现有的动作 / 表情数值是在「标准量程」模型上调出来的（头 ±30°、身体 ±10°、
  // 眼球/眉/嘴形 ±1、开合/脸颊 0..1）。这里按那套量程把旧数值换成归一化量，
  // 标准量程的模型行为和以前一模一样，量程不同的模型按比例缩放。
  var REF_HALF = { headX: 30, headY: 30, headZ: 30, bodyX: 10, bodyY: 10, bodyZ: 10 };

  function refHalf(ch) { return REF_HALF[ch] || 1; }

  /** 叠加写，dv 用「参考模型单位」（度 / 参数原值）。 */
  function addRef(core, ch, dv) { return add(core, ch, dv / refHalf(ch)); }

  /** 绝对写，v 用「参考模型单位」。 */
  function setRef(core, ch, v) { return set(core, ch, v / refHalf(ch)); }

  /** 情绪 → 模型自带表情名（第一个名字匹配的），没有返回 null。 */
  function expressionFor(core, emotion) {
    var p = profile(core), re = EXPRESSION_HINTS[emotion];
    if (!p || !re) return null;
    // 宿主约定的命名：<情绪>_<名字>（happy_xxx / sad_xxx），最可靠
    var prefix = new RegExp('^' + emotion + '[_-]', 'i');
    for (var j = 0; j < p.expressions.length; j++) {
      if (prefix.test(p.expressions[j].name)) return p.expressions[j].name;
    }
    for (var i = 0; i < p.expressions.length; i++) {
      var e = p.expressions[i];
      if (/^neutral[_-]/i.test(e.name)) continue;   // 宿主的「回到中性」表情，不是情绪本身
      if (re.test(e.name) || re.test(e.file)) return e.name;
    }
    return null;
  }

  /** 诊断：每个通道落到了哪个参数、怎么认出来的、哪些没有。 */
  function describe(core) {
    var p = profile(core);
    if (!p) return null;
    var out = { name: p.name, count: p.count, channels: {}, missing: p.missing.slice(),
                expressions: p.expressions.length };
    Object.keys(p.map).forEach(function (ch) {
      var m = p.map[ch];
      out.channels[ch] = m.id + '[' + m.lo + '..' + m.hi + ' d' + m.def + '](' + p.via[ch] + ')';
    });
    return out;
  }

  /** 宿主拿到 model3.json / cdi3.json 后调用，用名字补全映射。 */
  function attachInfo(core, info) {
    if (cache && core) cache.delete(core);
    return profile(core, info);
  }

  return {
    CHANNELS: CHANNELS,
    profile: profile, attachInfo: attachInfo,
    set: set, add: add, get: get, reset: reset, addRef: addRef, setRef: setRef, refHalf: refHalf,
    toRaw: toRaw, toRawDelta: toRawDelta, toNorm: toNorm, rawOf: rawOf,
    expressionFor: expressionFor, describe: describe,
    last: function () { return lastProfile; },
    calibrated: function (core) { var p = profile(core); if (p) refreshCal(p); return !!(p && p.calibrated); },
    REV: 'a2'
  };
});
