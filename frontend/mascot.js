/* शेरनी (Sherni) — the mascot.
 *
 * WHY A LIONESS, and why not something cuddly.
 *
 * The person holding this loan paper is frightened, often ashamed, and being
 * telephoned before dawn. A bouncing cartoon would trivialise that, and this
 * screen also gets read by people who assess lenders for a living. So the
 * brief was: dignified, warm, and never comic.
 *
 * A lioness because Indian microfinance is overwhelmingly lent to women
 * through self-help and joint-liability groups — statistically, the borrower
 * reading this IS a woman. Because "शेरनी" is pan-Indian shorthand for
 * standing your ground. And because the Ashoka lions sit at the head of every
 * Indian court document: she borrows the visual authority of the justice
 * system, which is precisely what the borrower does not feel they have.
 *
 * THE ANIMATION IS POSTURE. The product's emotional arc is frightened →
 * informed → upright, so that is what she does with her body. Idle motion is
 * deliberately small (breath, a blink, a slow tail) and there is exactly one
 * big beat — `standing` — which fires when the borrower is told their rights.
 * Spending it once is what makes it land.
 *
 * She rises until her head meets the shirorekha, the line that sits above
 * Devanagari letters and which this project already uses as its structural
 * device. The letters hang from that line; so does she.
 *
 * Self-contained: no dependencies, no build step, and it renders identically
 * on the landing page and inside the chat. Honours prefers-reduced-motion by
 * holding each pose instead of animating between them.
 */
"use strict";

const MASCOT_STATES = ["waiting", "listening", "reading", "standing", "cheer"];

const MASCOT_SVG = `
<svg viewBox="0 0 200 230" role="img" class="sherni-svg" aria-hidden="true">
  <!-- The shirorekha she stands up to. Hidden until she earns it. -->
  <line class="sh-rekha" x1="26" y1="22" x2="174" y2="22"/>

  <g class="sh-all">
    <!-- Tail sweeping out from behind the haunch. -->
    <path class="sh-tail" d="M150 186 C 176 182, 182 156, 172 138"/>
    <path class="sh-tuft" d="M172 140 C 164 130, 172 120, 180 124 C 186 128, 180 140, 172 140 Z"/>

    <!-- Seated body: haunches wide at the base, chest narrowing to the neck. -->
    <path class="sh-body"
          d="M58 204 C 52 166, 70 126, 100 126 C 130 126, 148 166, 142 204 Z"/>
    <!-- Paler chest, the way a lioness is lighter down the front. -->
    <path class="sh-chest"
          d="M100 150 C 112 150, 119 170, 118 200 L 82 200 C 81 170, 88 150, 100 150 Z"/>
    <!-- Forelegs planted, not dangling. -->
    <rect class="sh-leg" x="74" y="176" width="20" height="28" rx="10"/>
    <rect class="sh-leg" x="106" y="176" width="20" height="28" rx="10"/>

    <g class="sh-head">
      <!-- Ears: rounded, set wide. A lioness has no mane — the earlier
           version gave her one, and fourteen even petals read as a
           sunflower rather than a cat. -->
      <path class="sh-ear sh-ear-l" d="M70 66 C 62 44, 78 40, 86 56 Z"/>
      <path class="sh-ear sh-ear-r" d="M130 66 C 138 44, 122 40, 114 56 Z"/>

      <!-- Head: wider than tall, with a jaw. Not a circle. -->
      <path class="sh-face"
            d="M100 56 C 130 56, 142 74, 142 92 C 142 116, 124 130, 100 130
               C 76 130, 58 116, 58 92 C 58 74, 70 56, 100 56 Z"/>

      <g class="sh-eyes">
        <circle class="sh-eye" cx="86" cy="88" r="6"/>
        <circle class="sh-glint" cx="88" cy="86" r="2"/>
        <circle class="sh-eye" cx="114" cy="88" r="6"/>
        <circle class="sh-glint" cx="116" cy="86" r="2"/>
      </g>

      <!-- Muzzle -->
      <ellipse class="sh-muzzle" cx="100" cy="110" rx="22" ry="14"/>
      <path class="sh-nose" d="M100 100 L 93 107 L 107 107 Z"/>
      <path class="sh-mouth" d="M100 107 L 100 113 M100 113 C 94 120, 87 117, 87 112
                                 M100 113 C 106 120, 113 117, 113 112"/>
      <g class="sh-whiskers">
        <path d="M76 108 L 56 104 M76 114 L 56 116"/>
        <path d="M124 108 L 144 104 M124 114 L 144 116"/>
      </g>
    </g>
  </g>
</svg>`;

const MASCOT_CSS = `
.sherni { display: inline-block; line-height: 0; position: relative; }
.sherni-svg { width: 100%; height: auto; overflow: visible; display: block; }

/* One palette, inherited from whichever page hosts her, so she never fights
   the surrounding design. */
/* --sh-line is the OUTLINE and is deliberately separate from --sh-ink, the
   facial features. On the dark hero band the outline has to flip to cream or
   she is ink-on-ink and reads as loose floating shapes; her eyes and nose,
   sitting on a marigold face, must stay dark in both settings. */
.sherni-svg {
  --sh-ink: #101A3D; --sh-mane: #FFB000; --sh-coat: #F7F3E8;
  --sh-line: var(--sh-ink);
}

.sh-body, .sh-face, .sh-ear, .sh-leg {
  fill: var(--sh-mane); stroke: var(--sh-line); stroke-width: 5;
  stroke-linejoin: round;
}
.sh-chest, .sh-muzzle {
  fill: var(--sh-coat); stroke: var(--sh-line); stroke-width: 4;
}
.sh-leg { stroke-width: 4; }
.sh-tail  { fill: none; stroke: var(--sh-line); stroke-width: 5; stroke-linecap: round; }
.sh-tuft  { fill: var(--sh-line); stroke: var(--sh-line); stroke-width: 4; stroke-linejoin: round; }
.sh-eye   { fill: var(--sh-ink); }
.sh-glint { fill: var(--sh-coat); }
.sh-nose  { fill: var(--sh-ink); }
.sh-mouth, .sh-whiskers path {
  fill: none; stroke: var(--sh-ink); stroke-width: 4; stroke-linecap: round;
}
.sh-whiskers path { stroke-width: 3; opacity: .75; }

.sh-rekha {
  stroke: var(--sh-mane); stroke-width: 6; stroke-linecap: round;
  opacity: 0; stroke-dasharray: 148; stroke-dashoffset: 148;
}

/* Everything animates from the feet, so she grows upward like a person
   straightening their back rather than floating. */
.sh-all  { transform-box: fill-box; transform-origin: 50% 100%; }
.sh-head { transform-box: fill-box; transform-origin: 50% 90%; }
.sh-tail { transform-box: fill-box; transform-origin: 100% 100%; }
.sh-eyes { transform-box: fill-box; transform-origin: 50% 50%; }
.sh-ear-l, .sh-ear-r { transform-box: fill-box; transform-origin: 50% 100%; }

/* ---- idle: breath, a slow tail, an occasional blink ------------------- */
@keyframes sh-breathe { 0%,100% { transform: scaleY(1); }
                        50%     { transform: scaleY(1.025); } }
@keyframes sh-tail    { 0%,100% { transform: rotate(-7deg); }
                        50%     { transform: rotate(7deg); } }
@keyframes sh-blink   { 0%,92%,100% { transform: scaleY(1); }
                        96%         { transform: scaleY(.1); } }

.sherni .sh-all  { animation: sh-breathe 4s ease-in-out infinite; }
.sherni .sh-tail { animation: sh-tail 3.4s ease-in-out infinite; }
.sherni .sh-eyes { animation: sh-blink 6s ease-in-out infinite; }

/* ---- listening: ears forward, leans in ------------------------------- */
.sherni[data-state="listening"] .sh-ear-l { transform: rotate(-18deg); }
.sherni[data-state="listening"] .sh-ear-r { transform: rotate(18deg); }
.sherni[data-state="listening"] .sh-head  { transform: translateY(-3px) rotate(-3deg); }
.sherni[data-state="listening"] .sh-tail  { animation-duration: 1.6s; }

/* ---- reading: head down at the paper --------------------------------- */
.sherni[data-state="reading"] .sh-head { transform: translateY(6px) rotate(7deg); }
.sherni[data-state="reading"] .sh-eyes { animation-duration: 2.4s; }

/* ---- standing: the one beat worth spending --------------------------- */
@keyframes sh-rise  { 0%   { transform: translateY(10px) scaleY(.96); }
                      60%  { transform: translateY(-6px) scaleY(1.03); }
                      100% { transform: translateY(0) scaleY(1); } }
@keyframes sh-flare { 0%   { transform: scale(.92) rotate(0deg); }
                      100% { transform: scale(1.12) rotate(12deg); } }
@keyframes sh-draw  { to { stroke-dashoffset: 0; opacity: 1; } }

.sherni[data-state="standing"] .sh-all {
  animation: sh-rise .85s cubic-bezier(.2,.9,.25,1) both;
}
.sherni[data-state="standing"] .sh-head { transform: translateY(-8px) scale(1.04); }
.sherni[data-state="standing"] .sh-rekha {
  animation: sh-draw .7s ease-out .25s both;
}
.sherni[data-state="standing"] .sh-ear-l { transform: rotate(-10deg); }
.sherni[data-state="standing"] .sh-ear-r { transform: rotate(10deg); }

/* ---- cheer: the landing page's single invitation --------------------- */
@keyframes sh-cheer { 0%,100% { transform: translateY(0) rotate(0deg); }
                      45%     { transform: translateY(-7px) rotate(-2deg); } }
.sherni[data-state="cheer"] .sh-all { animation: sh-cheer 2.6s ease-in-out infinite; }

/* ---- speech ---------------------------------------------------------- */
.sh-say {
  /* --sh-shift nudges the bubble back inside the viewport on a narrow screen,
     and --sh-arrow keeps the tail pointing at her while it does. Centring on
     the mascot alone put half the bubble off the left edge of a 360px phone,
     with the text cut mid-word. */
  --sh-shift: 0px;
  --sh-arrow: 50%;
  position: absolute; left: 50%; bottom: calc(100% - 6px);
  transform: translateX(calc(-50% + var(--sh-shift)));
  background: var(--sh-ink, #101A3D); color: #F7F3E8;
  padding: 10px 14px; border-radius: 14px; font-size: 14px; line-height: 1.45;
  max-width: min(260px, calc(100vw - 24px)); width: max-content; text-align: center;
  opacity: 0; transition: opacity .3s ease, transform .3s ease;
  pointer-events: none;
}
.sh-say::after {
  content: ""; position: absolute; top: 100%; left: var(--sh-arrow);
  transform: translateX(-50%);
  border: 8px solid transparent; border-top-color: var(--sh-ink, #101A3D);
}
.sh-say[data-show="1"] {
  opacity: 1;
  transform: translateX(calc(-50% + var(--sh-shift))) translateY(-4px);
}

/* Below her instead, for when she sits at the top of the page: in the chat
   header there is no room above, and the bubble rendered off-screen — every
   line she said was invisible. Flipped at display time rather than fixed per
   page, so wherever she is placed the speech follows. */
.sh-say[data-place="below"] { bottom: auto; top: calc(100% - 6px); }
.sh-say[data-place="below"]::after {
  top: auto; bottom: 100%; left: var(--sh-arrow);
  border-top-color: transparent; border-bottom-color: var(--sh-ink, #101A3D);
}
.sh-say[data-place="below"][data-show="1"] {
  transform: translateX(calc(-50% + var(--sh-shift))) translateY(4px);
}

/* A tool people use while frightened should not also make them motion-sick.
   Each pose still reads; only the movement between them goes. */
@media (prefers-reduced-motion: reduce) {
  .sherni *, .sherni { animation: none !important; transition: none !important; }
  .sherni[data-state="standing"] .sh-rekha { opacity: 1; stroke-dashoffset: 0; }
}
`;

function installMascotStyles() {
  if (document.getElementById("sherni-styles")) return;
  const style = document.createElement("style");
  style.id = "sherni-styles";
  style.textContent = MASCOT_CSS;
  document.head.appendChild(style);
}

/**
 * Put Sherni in `host`.
 *
 * `label` is what a screen reader announces — she is decorative on the
 * landing page and meaningful in the chat, so the caller decides.
 */
function createMascot(host, { state = "waiting", label = "" } = {}) {
  installMascotStyles();

  const root = document.createElement("div");
  root.className = "sherni";
  root.innerHTML = MASCOT_SVG;

  const svg = root.querySelector("svg");
  if (label) {
    svg.setAttribute("aria-label", label);
    svg.removeAttribute("aria-hidden");
  }

  // Announced politely: an encouragement should never interrupt whatever the
  // borrower is being told about their rights.
  const bubble = document.createElement("div");
  bubble.className = "sh-say";
  bubble.setAttribute("aria-live", "polite");
  root.appendChild(bubble);

  host.appendChild(root);

  let hideTimer = null;

  const api = {
    el: root,

    setState(next) {
      if (!MASCOT_STATES.includes(next)) return api;
      // Re-trigger one-shot animations even when the state is unchanged.
      if (root.dataset.state === next && (next === "standing" || next === "cheer")) {
        root.removeAttribute("data-state");
        void root.offsetWidth;
      }
      root.dataset.state = next;
      return api;
    },

    say(text, ms = 5200) {
      clearTimeout(hideTimer);
      if (!text) { bubble.dataset.show = "0"; bubble.textContent = ""; return api; }
      bubble.textContent = text;
      // Measure before showing: above by default, below when it would clip.
      bubble.style.setProperty("--sh-shift", "0px");
      bubble.style.setProperty("--sh-arrow", "50%");
      bubble.dataset.place = "above";
      bubble.dataset.show = "1";
      if (bubble.getBoundingClientRect().top < 8) bubble.dataset.place = "below";

      // Then pull it back inside the screen if it hangs off either edge, and
      // move the tail by the same amount so it still points at her.
      const margin = 8;
      const box = bubble.getBoundingClientRect();
      let shift = 0;
      if (box.left < margin) shift = margin - box.left;
      else if (box.right > window.innerWidth - margin) {
        shift = window.innerWidth - margin - box.right;
      }
      if (shift) {
        bubble.style.setProperty("--sh-shift", `${Math.round(shift)}px`);
        // The tail is centred on the bubble; undo the shift to keep it on her.
        const half = box.width / 2;
        const arrow = Math.min(Math.max(half - shift, 14), box.width - 14);
        bubble.style.setProperty("--sh-arrow", `${Math.round(arrow)}px`);
      }
      if (ms > 0) hideTimer = setTimeout(() => { bubble.dataset.show = "0"; }, ms);
      return api;
    },

    /** The one moment she stands up, optionally with a line. */
    celebrate(text) {
      api.setState("standing");
      if (text) api.say(text, 6500);
      return api;
    },
  };

  return api.setState(state);
}

if (typeof window !== "undefined") {
  window.createMascot = createMascot;
  window.MASCOT_STATES = MASCOT_STATES;
}
