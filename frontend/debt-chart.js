/* The debt-trap chart, shared by the chat card and the standalone visualizer.
 *
 * Contains NO loan mathematics. It plots `analysis` exactly as the backend
 * returned it, which is what guarantees the chart, the summary strip and any
 * figure quoted in a grievance letter can never disagree.
 *
 * Two series over the loan term:
 *   - what the borrower actually pays (flat schedule, fees included)
 *   - what an honest reducing-balance loan at the same advertised rate costs
 * The shaded region between them IS `analysis.hidden_cost_gap`.
 */
"use strict";

function debtChartSeries(analysis) {
  return [
    {
      key: 'actual',
      label: `What you actually pay — ${pct(analysis.all_in_apr)} true APR`,
      colorVar: '--chart-actual',
      fill: true,
      dashed: false,
      points: analysis.flat_schedule.map(p => p.cumulative_paid),
    },
    {
      key: 'honest',
      label: `An honest ${pct(analysis.quoted_rate)} reducing-balance loan`,
      colorVar: '--chart-honest',
      fill: false,
      dashed: true,
      points: analysis.honest_schedule.map(p => p.cumulative_paid),
    },
  ];
}

function niceMax(maxVal) {
  if (maxVal <= 0) return 100;
  const mag = Math.pow(10, Math.floor(Math.log10(maxVal)));
  const res = maxVal / mag;
  if (res <= 1.2) return 1.2 * mag;
  if (res <= 1.5) return 1.5 * mag;
  if (res <= 2) return 2 * mag;
  if (res <= 3) return 3 * mag;
  if (res <= 5) return 5 * mag;
  if (res <= 7.5) return 7.5 * mag;
  return 10 * mag;
}

/**
 * Render into `svg`. `opts.compact` shrinks it for a chat bubble.
 * Returns the geometry the hover handler needs, or null if there is nothing
 * to draw.
 */
function renderDebtChart(svg, analysis, opts) {
  opts = opts || {};
  if (!analysis || !analysis.flat_schedule || !analysis.flat_schedule.length) return null;

  const narrow = opts.compact || window.innerWidth <= 600;
  const W = narrow ? 460 : 1000;
  const H = narrow ? 300 : 460;
  const PAD = narrow
    ? { top: 26, bottom: 44, left: 62, right: 14 }
    : { top: 44, bottom: 58, left: 84, right: 22 };
  const tickSize = narrow ? 13 : 17;
  const labelSize = narrow ? 14 : 19;

  const series = debtChartSeries(analysis);
  const months = analysis.flat_schedule.map(p => p.month);

  if (!svg.querySelector('.grid-layer')) {
    svg.innerHTML =
      '<g class="grid-layer"></g><g class="gap-layer"></g>' +
      '<g class="path-layer"></g><g class="hover-layer"></g>';
  }
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

  const yMax = niceMax(Math.max(...series.flatMap(s => s.points)));
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const xPos = i => PAD.left + (months.length > 1 ? (i / (months.length - 1)) * plotW : 0);
  const yPos = v => PAD.top + plotH - (v / yMax) * plotH;

  // --- grid and axes ---
  let grid = '';
  const yTicks = narrow ? 4 : 5;
  for (let i = 0; i <= yTicks; i++) {
    const val = (yMax / yTicks) * i;
    const y = yPos(val);
    if (i > 0) {
      grid += `<line x1="${PAD.left}" y1="${y}" x2="${W - PAD.right}" y2="${y}"
        stroke="var(--stroke-default)" stroke-dasharray="4 4" stroke-width="1"/>`;
    }
    grid += `<text x="${PAD.left - 10}" y="${y + 5}" text-anchor="end"
      class="axis-tick" font-size="${tickSize}">${inrAxis(val)}</text>`;
  }

  const step = Math.max(1, Math.ceil(months.length / (narrow ? 6 : 12)));
  months.forEach((m, i) => {
    if (i % step === 0 || i === months.length - 1) {
      const x = xPos(i);
      grid += `<text x="${x}" y="${PAD.top + plotH + 21}" text-anchor="middle"
        class="axis-tick" font-size="${tickSize}">${m}</text>`;
      grid += `<line x1="${x}" y1="${PAD.top + plotH}" x2="${x}" y2="${PAD.top + plotH + 5}"
        stroke="var(--outline-variant)" stroke-width="1"/>`;
    }
  });

  grid += `<text x="${PAD.left + plotW / 2}" y="${H - 6}" text-anchor="middle"
    class="axis-label" font-size="${labelSize}" font-weight="500">Month</text>`;
  grid += `<text x="0" y="${PAD.top - 12}" text-anchor="start"
    class="axis-label" font-size="${labelSize}" font-weight="500">Total repaid</text>`;
  grid += `<line x1="${PAD.left}" y1="${PAD.top}" x2="${PAD.left}" y2="${PAD.top + plotH}"
    stroke="var(--outline-variant)" stroke-width="1"/>`;
  grid += `<line x1="${PAD.left}" y1="${PAD.top + plotH}" x2="${W - PAD.right}" y2="${PAD.top + plotH}"
    stroke="var(--outline-variant)" stroke-width="1"/>`;
  svg.querySelector('.grid-layer').innerHTML = grid;

  // --- the gap: shade between the curves and bracket it at the end ---
  const actual = series[0].points;
  const honest = series[1].points;
  let gapHtml = '';
  if (analysis.hidden_cost_gap > 1) {
    let d = `M ${xPos(0)},${yPos(actual[0])}`;
    for (let i = 1; i < actual.length; i++) d += ` L ${xPos(i)},${yPos(actual[i])}`;
    for (let i = honest.length - 1; i >= 0; i--) d += ` L ${xPos(i)},${yPos(honest[i])}`;
    gapHtml += `<path d="${d} Z" fill="${cssVar('--chart-actual')}" opacity="0.14"/>`;

    const lastX = xPos(actual.length - 1);
    const yTop = yPos(actual[actual.length - 1]);
    const yBot = yPos(honest[honest.length - 1]);
    const labelX = lastX - (narrow ? 6 : 12);
    const big = narrow ? 14 : 18;
    const small = narrow ? 11 : 14;

    // Centre the caption inside the bracket when there is room; when the gap
    // is only a few pixels tall, lift it clear of the curves instead.
    const baseY = Math.abs(yBot - yTop) >= 46
      ? (yTop + yBot) / 2 - 4
      : Math.max(PAD.top + big, yTop - small - 8);

    gapHtml += `<line x1="${lastX}" y1="${yTop}" x2="${lastX}" y2="${yBot}"
      stroke="${cssVar('--chart-actual')}" stroke-width="2"/>`;
    gapHtml += `<text x="${labelX}" y="${baseY}" text-anchor="end"
      font-family="var(--ff-sans)" font-size="${big}" font-weight="500"
      fill="${cssVar('--chart-actual')}">+${inr(analysis.hidden_cost_gap)}</text>`;
    gapHtml += `<text x="${labelX}" y="${baseY + small + 3}" text-anchor="end"
      font-family="var(--ff-sans)" font-size="${small}"
      fill="var(--on-surface-de-emphasis)">hidden cost</text>`;
  }
  svg.querySelector('.gap-layer').innerHTML = gapHtml;

  // --- series ---
  const pathLayer = svg.querySelector('.path-layer');
  pathLayer.innerHTML = '';
  series.forEach(s => {
    const color = cssVar(s.colorVar);
    let line = `M ${xPos(0)},${yPos(s.points[0])}`;
    for (let i = 1; i < s.points.length; i++) line += ` L ${xPos(i)},${yPos(s.points[i])}`;
    if (s.fill) {
      pathLayer.insertAdjacentHTML('beforeend',
        `<path d="${line} L ${xPos(s.points.length - 1)},${yPos(0)} L ${xPos(0)},${yPos(0)} Z"
           fill="${color}" opacity="0.12"/>`);
    }
    pathLayer.insertAdjacentHTML('beforeend',
      `<path d="${line}" fill="none" stroke="${color}" stroke-width="2.5"
         stroke-linejoin="round" stroke-linecap="round"
         stroke-dasharray="${s.dashed ? '7 5' : 'none'}"/>`);
  });

  // --- hover targets ---
  const hoverLayer = svg.querySelector('.hover-layer');
  let hover = `<rect class="hover-area" x="${PAD.left}" y="${PAD.top}"
    width="${plotW}" height="${plotH}" fill="transparent"/>`;
  hover += `<line class="hover-line" x1="0" y1="${PAD.top}" x2="0" y2="${PAD.top + plotH}"
    stroke="var(--outline)" stroke-width="1" stroke-dasharray="4 4" opacity="0"/>`;
  series.forEach(s => {
    hover += `<circle class="hover-dot" data-key="${s.key}" cx="0" cy="0" r="5"
      fill="${cssVar(s.colorVar)}" stroke="var(--surface)" stroke-width="2" opacity="0"/>`;
  });
  hoverLayer.innerHTML = hover;

  return { W, H, PAD, plotW, plotH, months, series, xPos, yPos };
}

function renderDebtLegend(container, analysis) {
  container.innerHTML = debtChartSeries(analysis).map(s =>
    `<div class="legend-item">
       <div class="legend-dot" style="background:${cssVar(s.colorVar)}"></div>
       <span>${escapeHtml(s.label)}</span>
     </div>`).join('');
}

/** Wire pointer tracking. `geometry` is what renderDebtChart returned. */
function attachDebtChartHover(svg, container, tooltip, geometry) {
  if (!geometry) return;
  const area = svg.querySelector('.hover-area');
  const line = svg.querySelector('.hover-line');
  if (!area) return;

  const { months, series, xPos, yPos, W } = geometry;

  function moveTo(clientX, clientY) {
    const rect = svg.getBoundingClientRect();
    const svgX = ((clientX - rect.left) / rect.width) * W;
    const idx = Math.round(((svgX - geometry.PAD.left) / geometry.plotW) * (months.length - 1));
    const i = Math.max(0, Math.min(months.length - 1, idx));
    const x = xPos(i);

    line.setAttribute('x1', x);
    line.setAttribute('x2', x);
    line.setAttribute('opacity', '1');

    let html = `<div style="font-weight:500;margin-bottom:4px">Month ${months[i]}</div>`;
    series.forEach(s => {
      const value = s.points[i];
      const dot = svg.querySelector(`.hover-dot[data-key="${s.key}"]`);
      if (dot) {
        dot.setAttribute('cx', x);
        dot.setAttribute('cy', yPos(value));
        dot.setAttribute('opacity', '1');
      }
      html += `<div><span style="color:${cssVar(s.colorVar)}">&#9679;</span>
        ${s.key === 'actual' ? 'You pay' : 'Honest loan'}:
        <span class="tooltip-value">${inrExact(value)}</span></div>`;
    });
    const diff = series[0].points[i] - series[1].points[i];
    if (diff > 1) {
      html += `<div style="margin-top:4px;color:${cssVar('--chart-actual')}">
        Extra so far: <span class="tooltip-value">${inrExact(diff)}</span></div>`;
    }

    tooltip.innerHTML = html;
    tooltip.style.opacity = '1';

    const cRect = container.getBoundingClientRect();
    let left = clientX - cRect.left + 16;
    const tw = tooltip.offsetWidth || 220;
    if (left + tw > cRect.width) left = clientX - cRect.left - tw - 16;
    tooltip.style.left = Math.max(4, left) + 'px';
    tooltip.style.top = Math.max(4, clientY - cRect.top - 10) + 'px';
  }

  const clear = () => {
    line.setAttribute('opacity', '0');
    svg.querySelectorAll('.hover-dot').forEach(d => d.setAttribute('opacity', '0'));
    tooltip.style.opacity = '0';
  };

  area.onmousemove = e => moveTo(e.clientX, e.clientY);
  area.onmouseleave = clear;
  area.ontouchmove = e => {
    if (e.touches[0]) { moveTo(e.touches[0].clientX, e.touches[0].clientY); e.preventDefault(); }
  };
  area.ontouchend = clear;
}
