/* =============================================================================
 * NekoL2DAdapt —— 让同一套动作 / 表情 / 口型在不同 Live2D 模型上「看起来一样」
 *
 * 为什么需要它（2026-11-19 用 Cubism Core 实测 8 个模型，同一参数推满量程的可见位移）：
 *   - 参数名一样、区间一样（头 ±30、身 ±10），画师绑定的幅度却能差 3~6 倍：
 *     同样写 AngleY=10，「小侦探」点头很明显，mao_pro 只有它的 17%，几乎看不出。
 *   - 有的模型标准参数「存在但不动」：白毛宝宝的 ParamAngleX/Y 叫「面捕x/y」，只喂物理，
 *     真正转头的是 Param42/43（显示名「角度X / 角度Y」）。写进去 = 落空，人就僵着。
 *   - 有的模型干脆没有：mao_pro 没有 ParamMouthOpenY / ParamMouthForm，嘴型走 LipSync 组里的 ParamA。
 *
 * 做法：模型一加载就在页面里量一次（用模型自己的 coreModel，逐个参数推一下看网格动多少），
 *   1) 有标准参数但不动 → 在同模型里找「显示名 / 物理连线 / 命名」对应的替身参数；
 *   2) 每个通道算一个增益 = 参考幅度 / 本模型幅度（夹在 0.5~2.5，避免极端模型被拉变形）；
 *   3) 调用方照旧按「标准单位」写（头 ±30、身 ±10、眼球 ±1、嘴 0..1），本模块换算后写进真实参数。
 * 量不出来（老运行时 / 拿不到网格）就原样写，行为与之前完全一样，不会更差。
 * ========================================================================== */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.NekoL2DAdapt = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  // 「语义特效」：bridge 原来按 Blue cat 的参数编号写（Param57 = 脸红、Param64 = 流泪……），
  // 可同一个编号在别的模型上是别的东西：小侦探的 Param57 是「脸黑」、白毛宝宝的是「掏耳勺」、
  // 梦梦的是「裙子物理」。所以这些按「显示名」在当前模型里找，找不到就不写（宁可不做，也不做错）。
  // fullScale：在 Blue cat 上写多少算「满量程」（脸红 0..30 的模型和 0..1 的模型要按比例换算）。
  const FX = {
    blush: { names: /脸红|臉紅|頬染|赤面|害羞|blush/i, blue: 'Param57', blueHi: 30 },
    tears: { names: /^流泪$|^泪$|^涙$|眼泪$|tears?$/i, blue: 'Param64', blueHi: 1 },
    dark: { names: /^脸黑$|黑脸|^陰$|face.*dark|dark.*face/i, blue: 'Param56', blueHi: 30 },
    anger: { names: /^生气$|愤怒|^怒り?$|anger|angry/i, blue: 'Param54', blueHi: 30 },
    flower: { names: /^花花$/, blue: 'Param58', blueHi: 1 },
    smile2: { names: /^笑$|^微笑$|嘴角.*笑|mouth.*smile/i, blue: 'ParamMouthForm2', blueHi: 30, signed: true },
    mouthX: { names: /^嘴\s*x$|嘴.*(变形|位移)\s*x|^mouth\s*x$/i, blue: 'ParamMouthForm3', blueHi: 30, signed: true },
    teeth: { names: /^牙$|^牙齿$|^歯$|teeth/i, blue: 'ParamMouthOpenY23', blueHi: 1 },
  };
  // 编号 → 语义（bridge 里写死的 Blue cat 编号）
  const FX_BY_BLUE = {};
  for (const k in FX) FX_BY_BLUE[FX[k].blue] = k;

  function resolveFx(amps, names) {
    const out = {};
    for (const key in FX) {
      const f = FX[key];
      let best = null;
      for (const id in names) {
        if (!f.names.test(String(names[id]).trim())) continue;
        const p = amps[id];
        // 必须真的有效果：网格动了，或者透明度变了（脸红 / 泪多半是透明度）
        if (!p || (p.amp <= DEAD && !(p.op > 0.05))) continue;
        if (!best || (p.amp + p.op) > (best.p.amp + best.p.op)) best = { id, p };
      }
      out[key] = best ? { id: best.id, lo: best.p.lo, hi: best.p.hi, def: best.p.def, signed: !!f.signed,
        k: f.signed ? Math.max(Math.abs(best.p.hi - best.p.def), Math.abs(best.p.lo - best.p.def)) / f.blueHi
          : (best.p.hi - best.p.lo) / f.blueHi } : null;
    }
    return out;
  }

  // 默认原样（没量过时）。量过以后：
  //   嘴张开（ParamMouthOpenY）是曲线而不是一条直线：有的模型 0.3 就张满了（neko），有的要到 0.9（小侦探），
  //   单个增益对不上，所以张嘴用「曲线反查」——先量参考模型和本模型在 0..1 上的张开曲线，
  //   写入 v 时找出参考模型 v 对应的可见张开量，再在本模型曲线上反查出要写的值。
  const MOUTH_STEPS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
  // 参考模型（Blue cat）的张嘴曲线（可见位移，对应 MOUTH_STEPS）
  const REF_MOUTH = [0, 0.156, 0.313, 0.474, 0.640, 0.805, 0.828, 0.850, 0.872, 0.895, 0.917];
  // 量增益用的「典型动作量」：不用满量程，很多模型的绑定是非线性的（AngleY 只在 ±10 以内有形变），
  // 按满量程算增益会让日常的小动作差很多，所以在实际会用到的幅度上量。
  const PROBE = {
    ParamAngleX: 12, ParamAngleY: 10, ParamAngleZ: 8,
    ParamBodyAngleX: 4, ParamBodyAngleY: 4, ParamBodyAngleZ: 3,
    ParamEyeBallX: 0.5, ParamEyeBallY: 0.5,
    ParamBrowLY: 0.4, ParamBrowRY: 0.4, ParamBrowLForm: 0.4, ParamBrowRForm: 0.4,
    ParamBrowLAngle: 0.4, ParamBrowRAngle: 0.4,
    ParamMouthOpenY: 0.7, ParamMouthForm: 0.5,
  };
  // 参考：Blue cat（水水）在上面典型动作量下的可见位移，正向 / 反向各一个。
  // bridge.js / stage-runtime.js 里的动作和表情数值都是在这个模型上实测调出来的（代码注释里多处「实测这个模型（Blue cat）」），
  // 以它为「一倍」，水水自己的效果不变，别的模型向它看齐。
  // 水水右眉不动（BrowRY / BrowRAngle 为 0），右眉参考取左眉镜像。
  const REF = {
    ParamAngleX: [0.967, 0.966], ParamAngleY: [0.465, 0.697], ParamAngleZ: [1.802, 1.770],
    ParamBodyAngleX: [1.340, 1.338], ParamBodyAngleY: [0.949, 2.691], ParamBodyAngleZ: [0.931, 0.931],
    ParamEyeBallX: [0.202, 0.202], ParamEyeBallY: [0.073, 0.073],
    ParamBrowLY: [0.073, 0.081], ParamBrowRY: [0.073, 0.081], ParamBrowLForm: [0.098, 0.155],
    ParamBrowRForm: [0.098, 0.155], ParamBrowLAngle: [0.104, 0.231], ParamBrowRAngle: [0.231, 0.104],
    ParamMouthOpenY: [0.850, 0], ParamMouthForm: [0.145, 0.681],
  };
  // 标准半量程：头 ±30、身 ±10、眼球 / 眉 / 嘴形 ±1、张嘴 0..1
  const STD_HALF = {
    ParamAngleX: 30, ParamAngleY: 30, ParamAngleZ: 30,
    ParamBodyAngleX: 10, ParamBodyAngleY: 10, ParamBodyAngleZ: 10,
    ParamEyeBallX: 1, ParamEyeBallY: 1, ParamBrowLY: 1, ParamBrowRY: 1, ParamBrowLForm: 1, ParamBrowRForm: 1,
    ParamBrowLAngle: 1, ParamBrowRAngle: 1, ParamMouthOpenY: 1, ParamMouthForm: 1,
  };
  // 增益只在这些通道上用：它们是「动作幅度」，换模型后要看起来一样大。
  // 睁眼、眼笑、脸红、呼吸这类是「状态量」，0..1 的语义本身就对，不做幅度换算。
  // 上限 2.5：再大就会把小幅度绑定的模型拉出破面；下限 0.2：缩小不会破面，可以放得低。
  const GAIN_MIN = 0.2;
  const GAIN_MAX = 2.5;
  // 位移小于这个百分比就算「不动」（实测「不动」的参数都是 0，会动的最小也有 0.08）
  const DEAD = 0.02;
  // 替身：标准参数落空时，按这些线索在同模型里找真正干活的参数
  const ALIAS_HINTS = {
    ParamAngleX: { names: [/角度\s*x/i, /angle\s*x/i, /頭.*x/i], ids: [/^ParamAngleX\d+$/] },
    ParamAngleY: { names: [/角度\s*y/i, /angle\s*y/i, /頭.*y/i], ids: [/^ParamAngleY\d+$/] },
    ParamAngleZ: { names: [/角度\s*z/i, /angle\s*z/i, /頭.*z/i], ids: [/^ParamAngleZ\d+$/] },
    ParamBodyAngleX: { names: [/身体.*x/i, /body.*x/i, /体.*x/i], ids: [/^ParamBodyAngleX\d+$/] },
    ParamBodyAngleY: { names: [/身体.*y/i, /body.*y/i, /体.*y/i], ids: [/^ParamBodyAngleY\d+$/] },
    ParamBodyAngleZ: { names: [/身体.*z/i, /body.*z/i, /体.*z/i], ids: [/^ParamBodyAngleZ\d+$/] },
    ParamMouthOpenY: { names: [/张开|開閉|开合|口.*開|mouth.*open/i], ids: [/^ParamMouthOpen/, /^ParamA$/], lipSync: true },
    ParamMouthForm: { names: [/嘴.*变形$|口.*変形$|mouth.*form/i], ids: [/^ParamMouthForm\d*$/] },
  };

  const CACHE = typeof WeakMap === 'function' ? new WeakMap() : null;
  const LOG = [];

  // ---- 五官尺子（第六十九轮）----
  // 角度通道按「脸」量：x = 五官中心横移、y = 纵移（眼距 %）、rot = 五官连线转了几度。
  const FACE_AXIS = {
    ParamAngleX: 'x', ParamAngleY: 'y', ParamAngleZ: 'rot',
    ParamBodyAngleX: 'x', ParamBodyAngleY: 'y', ParamBodyAngleZ: 'rot',
  };
  // 水水在 PROBE 典型量下的五官读数（正 / 反向，眼距 % 或度）。_tools/l2d_adapt/groove_bench.js 同一把尺子实测。
  // 水水身体左右转（BodyAngleX）脸不跟着动（读数 0），这一条没有刻度，保持原来的量法。
  const FACE_CURVE = { ParamAngleX: 1, ParamAngleY: 1, ParamAngleZ: 1 };
  // 水水的五官曲线：标准值 |v|（0..30）→ 五官读数，正 / 反向各一条。face_probe.js 实测。
  // 水水这三条几乎是直线（误差 < 1%），所以只记 ±30 时的读数，按直线展开。
  const FACE_REF_END = { ParamAngleX: [28.171, 30.93], ParamAngleY: [13.938, 27.858], ParamAngleZ: [16.676, 16.946] };
  const lineCurve = end => ({ xs: [0, 30], ys: [0, end] });
  let FACE_REF_CURVE = {};
  for (const k in FACE_REF_END) FACE_REF_CURVE[k] = { pos: lineCurve(FACE_REF_END[k][0]), neg: lineCurve(FACE_REF_END[k][1]) };
  let FACE_REF = {
    ParamAngleX: [11.213, 12.368], ParamAngleY: [4.620, 9.279], ParamAngleZ: [4.814, 4.754],
    ParamBodyAngleY: [9.304, 42.249], ParamBodyAngleZ: [1.558, 1.625],
  };
  function boxOf(v) {
    let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    for (let j = 0; j < v.length; j += 2) {
      if (v[j] < x0) x0 = v[j]; if (v[j] > x1) x1 = v[j];
      if (v[j + 1] < y0) y0 = v[j + 1]; if (v[j + 1] > y1) y1 = v[j + 1];
    }
    return [x0, x1, y0, y1];
  }
  function maxMove(a, b) { let d = 0; for (let j = 0; j < a.length; j++) { const x = Math.abs(a[j] - b[j]); if (x > d) d = x; } return d; }
  function findFace(native, P, D, ids, amps, chosen, reset, grab) {
    try {
      reset();
      const rest = grab();
      const shot = (i, v) => { reset(); P.values[i] = v; return grab(); };
      // 眼珠：眼球 X 推满时动得最多的那几块，左右各取一块
      const eb = chosen.ParamEyeBallX && amps[chosen.ParamEyeBallX];
      if (!eb) return null;
      const se = shot(eb.i, eb.hi);
      const d = rest.map((b, i) => maxMove(se[i], b));
      const mx = Math.max(...d);
      if (!(mx > 0)) return null;
      const eyes = d.map((x, i) => (x >= mx * 0.5 ? i : -1)).filter(i => i >= 0);
      const cx = i => { const b = boxOf(rest[i]); return (b[0] + b[1]) / 2; };
      const xs = eyes.map(cx).sort((p, q) => p - q);
      const mid = (xs[0] + xs[xs.length - 1]) / 2;
      let li = -1, ri = -1;
      for (const i of eyes) { if (cx(i) < mid) { if (li < 0 || d[i] > d[li]) li = i; } else if (ri < 0 || d[i] > d[ri]) ri = i; }
      if (li < 0 || ri < 0) return null;
      let eyeW = Math.abs(cx(ri) - cx(li));
      // 头宽：歪头推满时会动的网格的横向跨度；眼距不到头宽 6% 说明眼球参数只带高光，退回头宽 × 0.4
      const az = chosen.ParamAngleZ && amps[chosen.ParamAngleZ];
      if (az) {
        const sz = shot(az.i, az.hi);
        let x0 = 1e9, x1 = -1e9;
        for (let i = 0; i < D.count; i++) if (maxMove(sz[i], rest[i]) > 1e-4) { const b = boxOf(rest[i]); x0 = Math.min(x0, b[0]); x1 = Math.max(x1, b[1]); }
        const headW = x1 > x0 ? x1 - x0 : 0;
        if (headW > 0 && !(eyeW > headW * 0.06)) eyeW = headW * 0.4;
      }
      if (!(eyeW > 0)) return null;
      const face = [li, ri];
      // 嘴：张嘴推满时高度增加最多、看得见、静止时不大的那一块
      const mo = chosen.ParamMouthOpenY && amps[chosen.ParamMouthOpenY];
      if (mo) {
        const sm = shot(mo.i, mo.hi);
        let best = 0, mi = -1;
        for (let i = 0; i < D.count; i++) {
          const b0 = boxOf(rest[i]), b1 = boxOf(sm[i]);
          const h0 = b0[3] - b0[2], dh = (b1[3] - b1[2]) - h0;
          if (sm.op[i] < 0.5 || h0 > eyeW * 0.6) continue;
          if (dh > best) { best = dh; mi = i; }
        }
        if (mi >= 0) face.push(mi);
      }
      const cen = s => { let x = 0, y = 0, n = 0; for (const i of face) { const v = s[i]; for (let j = 0; j < v.length; j += 2) { x += v[j]; y += v[j + 1]; n++; } } return [x / n, y / n]; };
      // 转角只看两颗眼珠的连线（嘴张合会让三点形状变，干扰旋转）
      const eyeAng = s => { const a = boxOf(s[li]), b = boxOf(s[ri]); return Math.atan2((b[2] + b[3]) / 2 - (a[2] + a[3]) / 2, (b[0] + b[1]) / 2 - (a[0] + a[1]) / 2) * 180 / Math.PI; };
      const c0 = cen(rest), a0 = eyeAng(rest);
      return {
        info: { eyes: [li, ri], mouth: face[2] === undefined ? -1 : face[2], eyeW },
        read(i, v, axis) {
          const s = shot(i, v);
          if (axis === 'rot') { let r = eyeAng(s) - a0; if (r > 180) r -= 360; if (r < -180) r += 360; return Math.abs(r); }
          const c = cen(s);
          return Math.abs((axis === 'x' ? c[0] - c0[0] : c[1] - c0[1]) / eyeW * 100);
        },
      };
    } catch (e) { return null; }
  }

  function at(v, i) { return (v && typeof v.at === 'function') ? v.at(i) : v[i]; }
  function size(v) { return v ? (typeof v.getSize === 'function' ? v.getSize() : v.length) : 0; }

  function coreOf(model) {
    const im = model && model.internalModel;
    return im && im.coreModel;
  }

  // 拿到 Cubism 原生模型（有顶点数据的那一层）
  function nativeOf(core) {
    try {
      const m = core && typeof core.getModel === 'function' ? core.getModel() : (core && core._model);
      if (m && m.drawables && m.parameters) return m;
    } catch (e) { /* 老运行时 */ }
    return null;
  }

  // 量参数的可见位移（占人物包围盒的百分比）：
  //   amp：推到 max / min 的较大值，用来判断「动不动」；
  //   pick(amps) 选好每个标准通道用哪个真实参数后，再在典型动作量下量正 / 反向位移算增益。
  // 替身参数的典型量按区间比例换算（典型量 / 标准半量程 × 替身半量程）。
  function measure(native, pick) {
    const P = native.parameters;
    const D = native.drawables;
    const saved = Float32Array.from(P.values);
    const reset = () => { for (let i = 0; i < P.count; i++) P.values[i] = P.defaultValues[i]; };
    const grab = () => {
      native.update();
      const a = new Array(D.count);
      for (let i = 0; i < D.count; i++) a[i] = Float32Array.from(D.vertexPositions[i]);
      a.op = Float32Array.from(D.opacities);
      return a;
    };
    try {
      reset();
      const base = grab();
      let minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9;
      for (const v of base) for (let j = 0; j < v.length; j += 2) {
        if (v[j] < minx) minx = v[j]; if (v[j] > maxx) maxx = v[j];
        if (v[j + 1] < miny) miny = v[j + 1]; if (v[j + 1] > maxy) maxy = v[j + 1];
      }
      const box = Math.max(maxx - minx, maxy - miny) || 1;
      let lastOp = 0;
      const disp = (idx, val) => {
        reset();
        P.values[idx] = val;
        const s = grab();
        let mx = 0, op = 0;
        for (let i = 0; i < D.count; i++) {
          const a = s[i], b = base[i];
          for (let j = 0; j < a.length; j++) { const d = Math.abs(a[j] - b[j]); if (d > mx) mx = d; }
          const o = Math.abs(s.op[i] - base.op[i]); if (o > op) op = o;
        }
        lastOp = op;
        return mx / box * 100;
      };
      const ids = Array.from(P.ids);
      const amps = {};
      for (let i = 0; i < P.count; i++) {
        const lo = P.minimumValues[i], hi = P.maximumValues[i], def = P.defaultValues[i];
        const a = disp(i, hi), oa = lastOp, b = disp(i, lo), ob = lastOp;
        amps[ids[i]] = { i, lo, hi, def, amp: Math.max(a, b), op: Math.max(oa, ob) };
      }
      const chosen = pick(amps);
      const probes = {};
      // 第六十九轮：头 / 身体的角度通道改按「五官」量（两颗眼珠 + 嘴的网格中心，单位 = 眼距 %）。
      // 原来量的是「全身最大位移 / 包围盒」，头发、耳朵、饰品一被带动就把转头量大（梦梦、白毛宝宝律动转头 1.7~2.5 倍），
      // 观众看的是脸往哪转多少，所以只量脸。找不到五官的模型退回原来的量法。
      const face = findFace(native, P, D, ids, amps, chosen, reset, grab);
      for (const std in chosen) {
        const id = chosen[std];
        const p = id && amps[id];
        if (!p) continue;
        const stdHalf = STD_HALF[std] || 1;
        const half = Math.max(Math.abs(p.hi - p.def), Math.abs(p.lo - p.def)) || 1;
        const step = PROBE[std] / stdHalf * (id === std ? stdHalf : half);
        probes[std] = {
          pos: disp(p.i, Math.min(p.hi, p.def + step)),
          neg: std === 'ParamMouthOpenY' ? 0 : disp(p.i, Math.max(p.lo, p.def - step)),
          scale: id === std ? 1 : half / stdHalf,
        };
        if (face && FACE_AXIS[std]) {
          probes[std].fpos = face.read(p.i, Math.min(p.hi, p.def + step), FACE_AXIS[std]);
          probes[std].fneg = face.read(p.i, Math.max(p.lo, p.def - step), FACE_AXIS[std]);
          // 头的三个角度再量整条曲线（默认值往两头各 12 段）：很多模型的绑定不是直线
          // （AngleY 只在 ±10 以内有形变、过了就不动），单个增益在大动作上对不上。
          if (FACE_CURVE[std]) {
            const side = end => { const xs = [], ys = []; for (let k = 0; k <= 12; k++) { const v = p.def + (end - p.def) * k / 12; xs.push(Math.abs(v - p.def)); ys.push(k ? face.read(p.i, v, FACE_AXIS[std]) : 0); } return { xs, ys: monotone(ys) }; };
            probes[std].fc = { pos: side(p.hi), neg: side(p.lo) };
          }
        }
        if (std === 'ParamMouthOpenY') {
          // 张嘴曲线：在本参数自己的区间上取 11 个点
          probes[std].curve = MOUTH_STEPS.map(s => disp(p.i, p.lo + (p.hi - p.lo) * s));
        }
      }
      return { amps, chosen, probes, face: face ? face.info : null };
    } finally {
      for (let i = 0; i < P.count; i++) P.values[i] = saved[i];
      try { native.update(); } catch (e) { /* 还原失败也不影响下一帧 */ }
    }
  }

  // 物理连线：「谁只是被某个标准参数单独驱动的输出」——白毛宝宝的 ParamAngleX → Param42 就是这种
  function physicsFollowers(model) {
    const map = {};
    try {
      const rig = model.internalModel.physics && model.internalModel.physics._physicsRig;
      if (!rig) return map;
      const n = size(rig.settings);
      for (let k = 0; k < n; k++) {
        const st = at(rig.settings, k);
        if (st.inputCount !== 1) continue;
        const src = String((at(rig.inputs, st.baseInputIndex).source || {}).id || '');
        for (let j = 0; j < st.outputCount; j++) {
          const d = String((at(rig.outputs, st.baseOutputIndex + j).destination || {}).id || '');
          if (src && d) (map[src] = map[src] || []).push(d);
        }
      }
    } catch (e) { /* 没有物理 */ }
    return map;
  }

  function lipSyncIds(model) {
    try {
      const st = model.internalModel.settings;
      const g = (st && (st.groups || (st.json && st.json.Groups))) || [];
      for (const x of g) if ((x.Name || x.name) === 'LipSync') return (x.Ids || x.ids || []).map(String);
    } catch (e) { /* 没有分组 */ }
    return [];
  }

  async function displayNames(model) {
    const out = {};
    try {
      const st = model.internalModel.settings;
      const json = st && st.json;
      const rel = json && json.FileReferences && json.FileReferences.DisplayInfo;
      if (!rel || !st.url) return out;
      const r = await fetch(new URL(rel, new URL(st.url, location.href)).href, { cache: 'force-cache' });
      if (!r.ok) return out;
      const j = await r.json();
      for (const p of (j.Parameters || [])) if (p && p.Id) out[p.Id] = String(p.Name || '');
    } catch (e) { /* 没有 cdi3 */ }
    return out;
  }

  function pickAlias(std, amps, names, phys, lips) {
    const hint = ALIAS_HINTS[std];
    if (!hint) return null;
    const alive = id => amps[id] && amps[id].amp > DEAD && id !== std;
    // 1) 物理：标准参数只喂给了某个参数，而那个参数会动 → 就是它（白毛宝宝 Angle → Param42）
    for (const id of (phys[std] || [])) if (alive(id)) return { id, why: 'physics' };
    // 2) LipSync 分组（mao_pro 的 ParamA）
    if (hint.lipSync) for (const id of lips) if (alive(id)) return { id, why: 'lipsync' };
    // 3) 显示名
    for (const re of hint.names) for (const id in names) if (alive(id) && re.test(names[id])) return { id, why: 'name' };
    // 4) 参数 ID 规律（ParamAngleX2 之类）
    for (const re of hint.ids) for (const id in amps) if (alive(id) && re.test(id)) return { id, why: 'id' };
    return null;
  }

  /**
   * 量一个模型，返回 profile。结果按 coreModel 缓存，换模型自动重算。
   * profile.map[std] = { id, gainPos, gainNeg, scale, lo, hi, def, why }；id 为 null 表示这个模型做不出这个通道。
   */
  async function calibrate(model) {
    const core = coreOf(model);
    if (!core) return null;
    if (CACHE && CACHE.has(core)) return CACHE.get(core);
    const native = nativeOf(core);
    const prof = { ok: false, map: {}, missing: [], aliased: [], t: Date.now() };
    if (CACHE) CACHE.set(core, prof);
    if (!native) { prof.reason = 'no-native-model'; return prof; }
    const names = await displayNames(model);
    const phys = physicsFollowers(model);
    const lips = lipSyncIds(model);
    const why = {};
    let m;
    try {
      m = measure(native, amps => {
        const chosen = {};
        for (const std in REF) {
          if (amps[std] && amps[std].amp > DEAD) { chosen[std] = std; why[std] = 'std'; continue; }
          const a = pickAlias(std, amps, names, phys, lips);
          chosen[std] = a ? a.id : null;
          if (a) why[std] = a.why;
        }
        return chosen;
      });
    } catch (e) { prof.reason = 'measure-failed: ' + e.message; return prof; }
    const clampG = g => +Math.min(GAIN_MAX, Math.max(GAIN_MIN, g)).toFixed(3);
    for (const std in REF) {
      const id = m.chosen[std];
      if (!id) { prof.map[std] = { id: null }; prof.missing.push(std); continue; }
      if (id !== std) prof.aliased.push(std + '→' + id + '(' + why[std] + ')');
      const p = m.amps[id], pr = m.probes[std];
      const [rp, rn] = REF[std];
      // 某个方向在本模型上不动（只做了单边形变），就沿用另一边的增益
      let gp = pr.pos > 1e-3 ? rp / pr.pos : (pr.neg > 1e-3 ? rn / pr.neg : 1);
      let gn = pr.neg > 1e-3 ? rn / pr.neg : gp;
      // 角度通道：有五官读数、也有水水的五官刻度时，按脸算增益（替换上面按全身最大位移算的）
      const fr = FACE_REF && FACE_REF[std];
      if (fr && m.face && pr.fpos !== undefined) {
        const FD = 0.05;
        if ((pr.fpos > FD && fr[0] > FD) || (pr.fneg > FD && fr[1] > FD)) {
          gp = pr.fpos > FD && fr[0] > FD ? fr[0] / pr.fpos : fr[1] / pr.fneg;
          gn = pr.fneg > FD && fr[1] > FD ? fr[1] / pr.fneg : gp;
          why[std] = (why[std] || 'std') + '+face';
        }
      }
      prof.map[std] = { id, why: why[std], lo: p.lo, hi: p.hi, def: p.def, scale: pr.scale,
        gainPos: clampG(gp), gainNeg: std === 'ParamMouthOpenY' ? clampG(gp) : clampG(gn) };
      if (pr.curve) prof.map[std].curve = monotone(pr.curve);
      if (pr.fc && FACE_REF_CURVE && FACE_REF_CURVE[std]) prof.map[std].fc = pr.fc;
    }
    prof.face = m.face;
    prof.faceProbe = {}; prof.faceCurve = {};
    for (const std in FACE_CURVE) if (m.probes[std] && m.probes[std].fc) prof.faceCurve[std] = m.probes[std].fc;
    for (const std in FACE_AXIS) if (m.probes[std] && m.probes[std].fpos !== undefined) prof.faceProbe[std] = [+m.probes[std].fpos.toFixed(4), +m.probes[std].fneg.toFixed(4)];
    prof.fx = resolveFx(m.amps, names);
    prof.fxIds = {};
    for (const k in prof.fx) prof.fxIds[k] = prof.fx[k] ? prof.fx[k].id : null;
    prof.ok = true;
    LOG.push({ t: prof.t, aliased: prof.aliased, missing: prof.missing });
    if (LOG.length > 20) LOG.shift();
    return prof;
  }

  function profileOf(model) {
    const core = coreOf(model);
    return core && CACHE ? (CACHE.get(core) || null) : null;
  }

  // 曲线做成单调不减（有的模型 0.5 之后形变会回落一点，反查要求单调）
  function monotone(c) {
    const out = c.slice();
    for (let i = 1; i < out.length; i++) if (out[i] < out[i - 1]) out[i] = out[i - 1];
    return out;
  }
  // 在单调曲线 ys（对应 MOUTH_STEPS）上，按 x∈[0,1] 线性插值取 y
  function curveAt(ys, x) {
    const k = Math.min(ys.length - 2, Math.max(0, Math.floor(x * 10)));
    const t = Math.min(1, Math.max(0, x * 10 - k));
    return ys[k] + (ys[k + 1] - ys[k]) * t;
  }
  // 反查：找 x 使 curveAt(ys, x) = y；超过本模型最大张开就取 1
  function curveInv(ys, y) {
    if (y <= ys[0]) return 0;
    for (let k = 0; k < ys.length - 1; k++) {
      if (y <= ys[k + 1]) {
        const span = ys[k + 1] - ys[k];
        return (k + (span > 1e-6 ? (y - ys[k]) / span : 0)) / 10;
      }
    }
    return 1;
  }

  function lerp(xs, ys, x) {
    if (x <= xs[0]) return ys[0];
    for (let k = 0; k < xs.length - 1; k++) if (x <= xs[k + 1]) { const t = (x - xs[k]) / ((xs[k + 1] - xs[k]) || 1); return ys[k] + (ys[k + 1] - ys[k]) * t; }
    return ys[ys.length - 1];
  }
  // ys 单调不减：找 x 使 ys(x) = y；超过最大值取最后一个 x
  function invMono(xs, ys, y) {
    if (y <= ys[0]) return xs[0];
    for (let k = 0; k < ys.length - 1; k++) if (y <= ys[k + 1]) { const sp = ys[k + 1] - ys[k]; return xs[k] + (xs[k + 1] - xs[k]) * (sp > 1e-9 ? (y - ys[k]) / sp : 0); }
    return xs[xs.length - 1];
  }

  // 标准单位 → 本模型真实参数和值。std 是我们一直在写的标准参数名，v 是标准单位下的值（以 0 为中立）。
  // 返回 [真实参数 ID, 真实值]；这个通道在本模型上做不出来时返回 null（调用方跳过）。
  function resolve(prof, std, v) {
    if (!prof || !prof.ok) return [std, v];
    const m = prof.map[std];
    if (!m) return [std, v];
    if (!m.id) return null;
    if (m.curve) {
      // 张嘴：参考模型在 v 时张多开 → 本模型写多少才一样开
      const x = curveInv(m.curve, curveAt(REF_MOUTH, Math.min(1, Math.max(0, v))));
      return [m.id, m.lo + (m.hi - m.lo) * x];
    }
    if (m.fc) {
      // 头的角度：水水在 |v| 时脸动多少 → 本模型同方向要写多少才动一样多（够不到就停在区间端点）
      const sd = v >= 0 ? 'pos' : 'neg', rc = FACE_REF_CURVE[std][sd], mc = m.fc[sd];
      const want = lerp(rc.xs, rc.ys, Math.abs(v));
      const off = invMono(mc.xs, mc.ys, want);
      let out = m.def + (v >= 0 ? off : -off);
      if (out < m.lo) out = m.lo;
      if (out > m.hi) out = m.hi;
      return [m.id, out];
    }
    const off = v * (v >= 0 ? m.gainPos : m.gainNeg) * m.scale;
    let out = m.def + off;
    if (out < m.lo) out = m.lo;
    if (out > m.hi) out = m.hi;
    return [m.id, out];
  }

  // 一步写入：按标准单位写，内部换算到本模型；没量过 / 不是标准通道就原样写
  function write(core, prof, std, v) {
    const r = resolve(prof, std, v);
    if (!r) return false;
    try {
      if (core.getParameterIndex(r[0]) < 0) return false;
      core.setParameterValueById(r[0], r[1]);
      return true;
    } catch (e) { return false; }
  }

  // 叠加写（addParameterValueById）用：标准单位的偏移量 dv → [真实 ID, 真实偏移]
  function resolveDelta(prof, std, dv) {
    const m = prof && prof.ok ? prof.map[std] : null;
    if (!m) return [std, dv];
    if (!m.id) return null;
    return [m.id, dv * (dv >= 0 ? m.gainPos : m.gainNeg) * m.scale];
  }
  // 真实参数值 → 标准单位（读当前值再叠加时用）
  function toStd(prof, std, real) {
    const m = prof && prof.ok ? prof.map[std] : null;
    if (!m || !m.id) return real;
    if (m.curve) {
      const x = (real - m.lo) / ((m.hi - m.lo) || 1);
      return curveInv(REF_MOUTH, curveAt(m.curve, Math.min(1, Math.max(0, x))));
    }
    const off = real - m.def;
    if (m.fc) {
      const sd = off >= 0 ? 'pos' : 'neg', rc = FACE_REF_CURVE[std][sd], mc = m.fc[sd];
      const sv = invMono(rc.xs, rc.ys, lerp(mc.xs, mc.ys, Math.abs(off)));
      return off >= 0 ? sv : -sv;
    }
    const g = (off >= 0 ? m.gainPos : m.gainNeg) * m.scale;
    return g ? off / g : 0;
  }
  function idOf(prof, std) {
    const m = prof && prof.ok ? prof.map[std] : null;
    return m ? m.id : std;
  }

  /**
   * 给调用方一个「按标准单位读写」的 coreModel 外壳：
   *   set / add / get 标准参数名 → 自动换成本模型的真实参数和幅度；
   *   非标准参数（模型自带的 Param57 之类）原样透传；
   *   这个模型做不出来的通道（id 为 null）写入直接忽略、读出 0、getParameterIndex 返回 -1。
   * 没量过（profile 还在算 / 量不出）时完全等于原 coreModel，行为与改之前一样。
   */
  function wrapCore(core, prof) {
    if (!core || !prof || !prof.ok) return core;
    const P = prof.map;
    // Blue cat 的特效编号（Param57 等）→ 本模型同语义参数；本模型就是用这个编号表示同一件事时 k=1
    const fxOf = id => {
      const key = FX_BY_BLUE[id];
      if (!key) return undefined;
      return prof.fx ? prof.fx[key] : undefined;
    };
    const real = id => {
      const m = P[id];
      if (m) return m.id;
      const f = fxOf(id);
      if (f !== undefined) return f ? f.id : null;
      return id;
    };
    const fxVal = (f, v) => {
      let out = (f.signed ? f.def : f.lo) + v * f.k;
      if (out < f.lo) out = f.lo;
      if (out > f.hi) out = f.hi;
      return out;
    };
    // 调用方拿 index 去查 min / max 时，被换算的通道要报「标准区间」——它们按标准单位思考、按标准区间夹值；
    // 用很大的假索引区分（FAKE_BASE + k；调用方只判断 < 0 = 没有），真实参数照旧用真实索引。
    const STD_RANGE = {
      ParamAngleX: [-30, 30, 0], ParamAngleY: [-30, 30, 0], ParamAngleZ: [-30, 30, 0],
      ParamBodyAngleX: [-10, 10, 0], ParamBodyAngleY: [-10, 10, 0], ParamBodyAngleZ: [-10, 10, 0],
      ParamEyeBallX: [-1, 1, 0], ParamEyeBallY: [-1, 1, 0], ParamBrowLY: [-1, 1, 0], ParamBrowRY: [-1, 1, 0],
      ParamBrowLForm: [-1, 1, 0], ParamBrowRForm: [-1, 1, 0], ParamBrowLAngle: [-1, 1, 0], ParamBrowRAngle: [-1, 1, 0],
      ParamMouthOpenY: [0, 1, 0], ParamMouthForm: [-1, 1, 0],
    };
    const fakeIdx = [];
    const FAKE_BASE = 1000000;
    const rangeOf = (i, k) => {
      if (i >= FAKE_BASE) { const std = fakeIdx[i - FAKE_BASE]; const r = std && STD_RANGE[std]; if (r) return r[k]; }
      return k === 0 ? core.getParameterMinimumValue(i) : k === 1 ? core.getParameterMaximumValue(i) : core.getParameterDefaultValue(i);
    };
    return {
      __adapted: true,
      __raw: core,
      getParameterIndex: id => {
        const m = P[id];
        if (m) {
          if (!m.id) return -1;
          let k = fakeIdx.indexOf(id);
          if (k < 0) { fakeIdx.push(id); k = fakeIdx.length - 1; }
          return FAKE_BASE + k;
        }
        const r = real(id);
        return r ? core.getParameterIndex(r) : -1;
      },
      getParameterCount: () => core.getParameterCount(),
      getParameterMinimumValue: i => rangeOf(i, 0),
      getParameterMaximumValue: i => rangeOf(i, 1),
      getParameterDefaultValue: i => rangeOf(i, 2),
      getParameterValueById: id => {
        const m = P[id];
        if (m) return m.id ? toStd(prof, id, Number(core.getParameterValueById(m.id)) || 0) : 0;
        const f = fxOf(id);
        if (f === null) return 0;
        if (f) return (Number(core.getParameterValueById(f.id)) || 0) / (f.k || 1);
        return core.getParameterValueById(id);
      },
      setParameterValueById: (id, v, w) => {
        let r;
        if (P[id]) r = resolve(prof, id, v);
        else {
          const f = fxOf(id);
          r = f === null ? null : (f ? [f.id, fxVal(f, v)] : [id, v]);
        }
        if (!r) return;
        return w === undefined ? core.setParameterValueById(r[0], r[1]) : core.setParameterValueById(r[0], r[1], w);
      },
      addParameterValueById: (id, v, w) => {
        let r;
        if (P[id]) r = resolveDelta(prof, id, v);
        else {
          const f = fxOf(id);
          r = f === null ? null : (f ? [f.id, v * f.k] : [id, v]);
        }
        if (!r) return;
        return w === undefined ? core.addParameterValueById(r[0], r[1]) : core.addParameterValueById(r[0], r[1], w);
      },
    };
  }

  // 页面里常用的一步到位：拿模型的 profile（没有就后台开始量），返回包好的 core；量好之前返回原 core
  const PENDING = typeof WeakSet === 'function' ? new WeakSet() : null;
  function adapted(model) {
    const core = coreOf(model);
    if (!core) return core;
    const prof = CACHE ? CACHE.get(core) : null;
    if (prof) return prof.ok ? wrapCore(core, prof) : core;
    if (PENDING && !PENDING.has(core)) {
      PENDING.add(core);
      Promise.resolve().then(() => calibrate(model)).catch(() => {});
    }
    return core;
  }

  function setFaceRef(r, c) { if (r) FACE_REF = r; if (c) FACE_REF_CURVE = c; }
  return { REF, PROBE, FACE_AXIS, setFaceRef, calibrate, profileOf, resolve, resolveDelta, toStd, write, idOf, wrapCore, adapted, log: LOG };
});
