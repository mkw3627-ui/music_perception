// 毛玻璃主题：只负责外观，不改变任何按钮或功能。
// 四个插件共用同一套材质，只有识别色不同。
export const GLASS_ACCENT = "#8b7cf6"
export const GLASS_ACCENT_2 = "#60a5fa"

export const GLASS_STYLES = `
:root {
  color-scheme: dark;
  --bg: #0b1120;
  --g-accent: ${GLASS_ACCENT};
  --g-accent-2: ${GLASS_ACCENT_2};
  --g-text: #eef2f8;
  --g-muted: #a9b5c7;
  --g-faint: #7f8ba0;
  --g-line: rgba(255, 255, 255, 0.13);
  --g-line-strong: rgba(255, 255, 255, 0.22);
  --g-glass: rgba(255, 255, 255, 0.075);
  --g-glass-2: rgba(255, 255, 255, 0.045);
  --g-glass-strong: rgba(22, 30, 48, 0.86);
  --g-blur: blur(20px) saturate(150%);
  --g-shadow: 0 12px 34px rgba(0, 0, 0, 0.30), inset 0 1px 0 rgba(255, 255, 255, 0.09);
  --surface: var(--g-glass);
  --surface-strong: var(--g-glass-strong);
  --text: var(--g-text);
  --muted: var(--g-muted);
  --border: var(--g-line);
  --primary: var(--g-accent);
  --info: var(--g-accent);
  --success: #5fd3a0;
  --warning: #f5c26b;
  --danger: #ff8a93;
  --radius-sm: 10px;
  --radius-md: 14px;
  --radius-lg: 18px;
  --radius-xl: 22px;
  --shadow-soft: var(--g-shadow);
}
html, body {
  background-color: #0b1120;
  background-image:
    radial-gradient(900px 520px at 8% -6%, color-mix(in srgb, var(--g-accent) 30%, transparent), transparent 62%),
    radial-gradient(760px 520px at 100% 8%, color-mix(in srgb, var(--g-accent-2) 24%, transparent), transparent 60%),
    radial-gradient(900px 640px at 50% 110%, rgba(56, 78, 140, 0.30), transparent 62%),
    linear-gradient(160deg, #0b1120 0%, #101a2e 52%, #0a0f1c 100%);
  background-attachment: fixed;
  color: var(--g-text);
  font-family: "HarmonyOS Sans SC", "MiSans", "PingFang SC", "Microsoft YaHei UI", "Microsoft YaHei", system-ui, sans-serif;
}
.neko-page { padding: 22px 22px 34px; gap: 16px; color: var(--g-text); }
.neko-page-title { font-size: 24px; font-weight: 800; letter-spacing: 0.5px; color: #fff; }
.neko-page-subtitle { color: var(--g-muted); font-size: 13px; }
.neko-text { color: var(--g-muted); line-height: 1.75; }
.neko-text strong { color: var(--g-text); }

/* 玻璃卡片 */
.neko-card, .neko-stat, .neko-accordion, .neko-status-strip, .neko-empty, .g-glass {
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.09), rgba(255, 255, 255, 0.04));
  border: 1px solid var(--g-line);
  border-radius: var(--radius-lg);
  -webkit-backdrop-filter: var(--g-blur);
  backdrop-filter: var(--g-blur);
  box-shadow: var(--g-shadow);
}
.neko-card { overflow: visible; }
.neko-card-header { padding: 16px 18px 2px; }
.neko-card-title { font-size: 15px; font-weight: 750; color: #fff; display: flex; align-items: center; gap: 8px; }
.neko-card-title::before {
  content: ""; width: 4px; height: 15px; border-radius: 4px; flex: 0 0 4px;
  background: linear-gradient(180deg, var(--g-accent), var(--g-accent-2));
}
.neko-card-body { padding: 12px 18px 18px; gap: 12px; }
.neko-stat { padding: 14px 16px; }
.neko-stat-label { color: var(--g-muted); font-weight: 600; }
.neko-stat-value { color: #fff; font-size: 22px; }
.neko-empty { color: var(--g-muted); border-style: dashed; box-shadow: none; }
.neko-empty-title { color: var(--g-text); }

/* 按钮 */
.neko-button {
  border-radius: 12px; padding: 8px 15px; font-weight: 650;
  color: var(--g-text);
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid var(--g-line);
  -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
}
.neko-button:hover:not(:disabled) {
  transform: translateY(-1px);
  background: color-mix(in srgb, var(--g-accent) 18%, rgba(255, 255, 255, 0.06));
  border-color: color-mix(in srgb, var(--g-accent) 60%, transparent);
  box-shadow: 0 8px 22px color-mix(in srgb, var(--g-accent) 22%, transparent);
}
.neko-button[data-tone="primary"], .neko-button[data-tone="info"] {
  color: #fff;
  background: linear-gradient(135deg, color-mix(in srgb, var(--g-accent) 52%, transparent), color-mix(in srgb, var(--g-accent-2) 40%, transparent));
  border-color: color-mix(in srgb, var(--g-accent) 70%, transparent);
}
.neko-button[data-tone="danger"] { color: #ffd5d9; background: rgba(255, 110, 125, 0.16); border-color: rgba(255, 138, 147, 0.45); }
.neko-button[data-tone="success"] { color: #d3ffe9; background: rgba(95, 211, 160, 0.16); border-color: rgba(95, 211, 160, 0.45); }
.neko-button[data-tone="warning"] { color: #ffeccc; background: rgba(245, 194, 107, 0.16); border-color: rgba(245, 194, 107, 0.45); }
.neko-button[data-tone="default"], .neko-button[data-tone="secondary"] { color: var(--g-text); }
.neko-button:disabled { opacity: 0.5; cursor: not-allowed; }
.neko-page :is(button, a, input, select, textarea, [tabindex]):focus-visible { outline: 2px solid var(--g-accent); outline-offset: 2px; }

/* 状态标签 */
.neko-badge {
  color: var(--g-text); background: rgba(255, 255, 255, 0.08); border: 1px solid var(--g-line);
  padding: 4px 10px; font-weight: 650;
}
.neko-badge::before { background: var(--g-accent); box-shadow: 0 0 8px var(--g-accent); }
.neko-badge[data-tone="success"] { color: #c9f7e2; border-color: rgba(95, 211, 160, 0.40); background: rgba(95, 211, 160, 0.12); }
.neko-badge[data-tone="warning"] { color: #ffe6b8; border-color: rgba(245, 194, 107, 0.40); background: rgba(245, 194, 107, 0.12); }
.neko-badge[data-tone="danger"] { color: #ffd0d5; border-color: rgba(255, 138, 147, 0.40); background: rgba(255, 138, 147, 0.12); }
.neko-badge[data-tone="success"]::before { box-shadow: 0 0 8px var(--success); }
.neko-badge[data-tone="warning"]::before { box-shadow: 0 0 8px var(--warning); }
.neko-badge[data-tone="danger"]::before { box-shadow: 0 0 8px var(--danger); }

/* 提示条 */
.neko-alert, .neko-tip {
  color: var(--g-text); border-radius: 14px; padding: 12px 15px;
  background: color-mix(in srgb, var(--g-accent) 11%, rgba(255, 255, 255, 0.04));
  border: 1px solid color-mix(in srgb, var(--g-accent) 32%, transparent);
  -webkit-backdrop-filter: blur(12px); backdrop-filter: blur(12px);
}
.neko-alert[data-tone="success"] { background: rgba(95, 211, 160, 0.11); border-color: rgba(95, 211, 160, 0.34); }
.neko-alert[data-tone="warning"], .neko-tip { background: rgba(245, 194, 107, 0.11); border-color: rgba(245, 194, 107, 0.34); }
.neko-alert[data-tone="danger"], .neko-warning { background: rgba(255, 138, 147, 0.11); border-color: rgba(255, 138, 147, 0.36); }

/* 表单 */
.neko-input, .neko-select, .neko-textarea {
  color: var(--g-text); background: rgba(8, 14, 28, 0.45); border: 1px solid var(--g-line); border-radius: 12px;
}
.neko-input::placeholder, .neko-textarea::placeholder { color: var(--g-faint); }
.neko-input:focus, .neko-select:focus, .neko-textarea:focus {
  border-color: var(--g-accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--g-accent) 22%, transparent);
}
.neko-select option { color: #e8edf5; background: #141c2e; }
.neko-checkbox, .neko-choice-input { accent-color: var(--g-accent); }
.neko-switch { color: var(--g-muted); }
.neko-slider-input { accent-color: var(--g-accent); }

/* 分段切换 / 页签 */
.neko-segmented, .neko-tab-list {
  background: rgba(255, 255, 255, 0.05); border: 1px solid var(--g-line); border-radius: 14px; padding: 4px;
  -webkit-backdrop-filter: blur(14px); backdrop-filter: blur(14px);
  flex-wrap: wrap; max-width: 100%;
}
.neko-segmented-button, .neko-tab-button { color: var(--g-muted); border-radius: 10px; border: 0; background: transparent; }
.neko-segmented-button:hover, .neko-tab-button:hover { color: var(--g-text); background: rgba(255, 255, 255, 0.06); }
.neko-segmented-button.is-active, .neko-tab-button.is-active {
  color: #fff;
  background: linear-gradient(135deg, color-mix(in srgb, var(--g-accent) 46%, transparent), color-mix(in srgb, var(--g-accent-2) 34%, transparent));
  box-shadow: 0 4px 14px color-mix(in srgb, var(--g-accent) 25%, transparent);
}

/* 其他组件 */
.neko-table { border-color: var(--g-line); }
.neko-table th { color: var(--g-muted); background: rgba(255, 255, 255, 0.05); }
.neko-table td, .neko-table th { border-color: var(--g-line); }
.neko-key-value-row, .neko-divider { border-color: var(--g-line); }
.neko-divider { background: var(--g-line); }
.neko-key-value-key { color: var(--g-muted); }
.neko-key-value-value { color: var(--g-text); }
.neko-progress-track { background: rgba(255, 255, 255, 0.10); }
.neko-progress-bar { background: linear-gradient(90deg, var(--g-accent), var(--g-accent-2)); }
.neko-accordion-trigger { color: var(--g-text); padding: 13px 16px; }
.neko-accordion-icon { color: var(--g-accent); }
.neko-modal-backdrop { background: rgba(4, 8, 18, 0.55); -webkit-backdrop-filter: blur(6px); backdrop-filter: blur(6px); }
.neko-modal, .neko-toast, .neko-tooltip-content {
  color: var(--g-text); background: var(--g-glass-strong); border: 1px solid var(--g-line-strong);
  -webkit-backdrop-filter: var(--g-blur); backdrop-filter: var(--g-blur);
}
.neko-code, .neko-page pre, .neko-page code { color: #dbe4f2; background: rgba(0, 0, 0, 0.28); border-radius: 10px; }
.neko-page a { color: color-mix(in srgb, var(--g-accent) 80%, #fff); }
.neko-page ::selection { background: color-mix(in srgb, var(--g-accent) 45%, transparent); }
.neko-page ::-webkit-scrollbar { width: 10px; height: 10px; }
.neko-page ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.16); border-radius: 10px; border: 2px solid transparent; background-clip: padding-box; }

/* 顶部“现在怎么样”总览卡 */
.g-hero {
  position: relative; overflow: hidden;
  padding: 20px 22px; border-radius: 22px;
  background:
    radial-gradient(420px 200px at 0% 0%, color-mix(in srgb, var(--g-accent) 30%, transparent), transparent 70%),
    linear-gradient(135deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.04));
  border: 1px solid var(--g-line-strong);
  -webkit-backdrop-filter: var(--g-blur); backdrop-filter: var(--g-blur);
  box-shadow: var(--g-shadow);
}
.g-hero-top { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
.g-hero-icon {
  width: 46px; height: 46px; border-radius: 15px; display: grid; place-items: center; flex: 0 0 46px;
  font-size: 22px; color: #fff;
  background: linear-gradient(135deg, var(--g-accent), var(--g-accent-2));
  box-shadow: 0 8px 22px color-mix(in srgb, var(--g-accent) 40%, transparent);
}
.g-hero-name { font-size: 22px; font-weight: 820; color: #fff; letter-spacing: 0.5px; }
.g-hero-tag { font-size: 13px; color: var(--g-muted); margin-top: 2px; }
.g-hero-status {
  margin-left: auto; display: inline-flex; align-items: center; gap: 8px;
  padding: 7px 13px; border-radius: 999px; font-size: 13px; font-weight: 700;
  background: rgba(255, 255, 255, 0.08); border: 1px solid var(--g-line);
}
.g-hero-status::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--g-dot, #5fd3a0); box-shadow: 0 0 10px var(--g-dot, #5fd3a0); }
.g-hero-status[data-tone="warning"] { --g-dot: #f5c26b; }
.g-hero-status[data-tone="danger"] { --g-dot: #ff8a93; }
.g-hero-status[data-tone="idle"] { --g-dot: #9aa6b8; }
.g-hero-line { margin-top: 14px; font-size: 15px; line-height: 1.7; color: var(--g-text); }
.g-hero-facts { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.g-fact {
  display: inline-flex; gap: 6px; align-items: baseline; padding: 6px 11px; border-radius: 11px;
  background: rgba(255, 255, 255, 0.06); border: 1px solid var(--g-line); font-size: 12px; color: var(--g-muted);
}
.g-fact b { color: #fff; font-size: 13px; }
.g-help { font-size: 12px; color: var(--g-faint); line-height: 1.6; margin-top: 3px; }
@media (max-width: 560px) {
  .neko-page { padding: 14px 12px 26px; }
  .g-hero-status { margin-left: 0; }
  .neko-grid { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
}
@media (prefers-reduced-motion: reduce) {
  .neko-button, .neko-card, .neko-page { transition: none !important; animation: none !important; }
}
`
