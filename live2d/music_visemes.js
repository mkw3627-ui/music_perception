/*!
 * music_visemes.js — 逐字口型（为她而唱）
 *
 * 把一行歌词拆成每个字的 A/I/U/E/O 口型，给出起止时间。
 * 中文没有拼音库时用字符哈希做一个稳定但近似的元音，日语/英语按罗马音元音处理。
 * 目标是“大概对得上”，不是音素级精确。
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.MusicVisemes = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var VOWELS = ['A', 'I', 'U', 'E', 'O'];
  var LATIN_MOUTH = {
    a: 'A', i: 'I', u: 'U', e: 'E', o: 'O', y: 'I',
    m: 'M', b: 'M', p: 'M', f: 'F', v: 'F',
    n: 'N', l: 'L', r: 'L', w: 'U'
  };
  var PUNCT = /[\s，。！？、,.!?;；:：'"“”‘’—\-…（）()\[\]{}<>《》]/;

  function hash(s) {
    var h = 2166136261;
    for (var i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }

  function isCjk(ch) {
    var c = ch.codePointAt(0);
    return (c >= 0x3040 && c <= 0x30ff) ||   // 日文假名
           (c >= 0x4e00 && c <= 0x9fff) ||   // 汉字
           (c >= 0xac00 && c <= 0xd7af);     // 韩文
  }

  /** 单字 → 口型（近似）。 */
  function visemeForChar(ch) {
    if (!ch) return 'rest';
    if (PUNCT.test(ch)) return 'rest';
    if (isCjk(ch)) {
      // 没有拼音数据时用稳定哈希；同一个字每次都得到同一个口型。
      return VOWELS[hash(ch) % VOWELS.length];
    }
    return LATIN_MOUTH[ch.toLowerCase()] || 'A';
  }

  /**
   * 把一行歌词按时间均分到每个字。
   * @param {string} text
   * @param {number} start seconds
   * @param {number} end   seconds
   */
  function lineCues(text, start, end) {
    var chars = Array.from(String(text || '')).filter(function (ch) { return !PUNCT.test(ch); });
    if (!chars.length || !(end > start)) return [];
    var span = (end - start) / chars.length;
    return chars.map(function (ch, i) {
      return {
        char: ch,
        viseme: visemeForChar(ch),
        start: start + i * span,
        end: start + (i + 1) * span
      };
    });
  }

  /**
   * @param {Array<{text:string,start:number,end:number}>} lines
   * @returns {Array} cues sorted by start
   */
  function fromLines(lines) {
    var out = [];
    (lines || []).forEach(function (line) {
      out = out.concat(lineCues(line.text, line.start, line.end));
    });
    out.sort(function (a, b) { return a.start - b.start; });
    return out;
  }

  /** 采样某一时刻的口型；使用最近 40ms 的轻微平滑。 */
  function sample(cues, t) {
    if (!cues || !cues.length) return { viseme: 'rest', open: 0, char: '' };
    // 二分找最后一个 start <= t
    var lo = 0, hi = cues.length - 1, idx = -1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (cues[mid].start <= t) { idx = mid; lo = mid + 1; } else hi = mid - 1;
    }
    if (idx < 0) return { viseme: 'rest', open: 0, char: '' };
    var cue = cues[idx];
    if (t > cue.end) {
      // 句尾 60ms 内收口，避免突然闭合。
      var fade = Math.max(0, 1 - (t - cue.end) / 0.06);
      return { viseme: 'rest', open: 0.15 * fade, char: '' };
    }
    var local = (t - cue.start) / Math.max(cue.end - cue.start, 1e-6);
    var env = Math.sin(Math.PI * Math.min(1, Math.max(0, local)));
    return { viseme: cue.viseme, open: 0.35 + 0.65 * env, char: cue.char };
  }

  /** 口型 → Cubism 4 嘴部参数。 */
  function applyToModel(coreModel, state) {
    var form = { A: 0.8, I: 1.0, U: 0.55, E: 0.9, O: 0.5, M: 1.0, N: 0.85, F: 0.75, L: 0.9, rest: 1.0 };
    var open = state && typeof state.open === 'number' ? state.open : 0;
    var v = (state && state.viseme) || 'rest';
    var openV = open * (v === 'rest' ? 0.15 : 1);
    var formV = form[v] == null ? 1 : form[v];
    // 有参数适配层就按语义通道写（开合可能是 ParamA、嘴形区间可能是 0..1），否则按标准 ID。
    var PA = (typeof self !== 'undefined' && self.NekoParamAdapter) || null;
    if (PA && PA.profile(coreModel)) {
      PA.set(coreModel, 'mouthOpen', openV);
      PA.set(coreModel, 'mouthForm', formV);
      return;
    }
    try {
      if (coreModel.getParameterIndex('ParamMouthOpenY') >= 0) {
        coreModel.setParameterValueById('ParamMouthOpenY', openV);
      }
      if (coreModel.getParameterIndex('ParamMouthForm') >= 0) {
        coreModel.setParameterValueById('ParamMouthForm', formV);
      }
    } catch (e) { /* 忽略缺失参数 */ }
  }

  return {
    VOWELS: VOWELS,
    visemeForChar: visemeForChar,
    lineCues: lineCues,
    fromLines: fromLines,
    sample: sample,
    applyToModel: applyToModel
  };
});
