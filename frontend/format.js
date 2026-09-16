/* INR and percentage formatting.
 *
 * This is the only arithmetic the frontend is allowed to do. Every loan
 * figure comes from the backend already computed; these functions decide
 * how it is written, never what it is.
 *
 * Indian digit grouping via toLocaleString('en-IN'), with lakh/crore
 * abbreviation above 1,00,000 because that is how the amounts are spoken.
 */
"use strict";

function inr(n) {
  if (!isFinite(n)) return '—';
  const sign = n < 0 ? '-' : '';
  const abs = Math.abs(n);
  if (abs >= 1e7) return sign + '₹' + (abs / 1e7).toFixed(2).replace(/\.00$/, '') + ' Cr';
  if (abs >= 1e5) return sign + '₹' + (abs / 1e5).toFixed(2).replace(/\.00$/, '') + ' L';
  return sign + '₹' + Math.round(abs).toLocaleString('en-IN');
}

/** Full rupee figure, no abbreviation — for anything a borrower may quote. */
function inrExact(n) {
  if (!isFinite(n)) return '—';
  const sign = n < 0 ? '-' : '';
  return sign + '₹' + Math.round(Math.abs(n)).toLocaleString('en-IN');
}

function inrAxis(n) {
  const abs = Math.abs(n);
  if (abs >= 1e7) return '₹' + (n / 1e7).toFixed(1) + 'Cr';
  if (abs >= 1e5) return '₹' + (n / 1e5).toFixed(1) + 'L';
  if (abs >= 1e3) return '₹' + Math.round(n / 1e3) + 'K';
  return '₹' + Math.round(n);
}

function pct(n) {
  if (!isFinite(n)) return '—';
  return (Math.round(n * 100) / 100).toFixed(2) + '%';
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
