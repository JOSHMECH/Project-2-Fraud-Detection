/* ════════════════════════════════════════════════════════════════
   Josh_d_Guru — Fraud Detection Case Study Interactive Engine
   Reference: joshfolio.cv
   ════════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  initReadingProgress();
  initNavbarScroll();
  initMobileNav();
  initNavDropdown();
  initCustomCursor();
  initDottedSurface();
  initTelemetryBadge();
  initAudioSynthesizer();
  initCodeCopyButtons();
  initFraudRiskSandbox();
  initScrollSpy();
});

/* ─── Reading Progress Bar ──────────────────────────────────── */
function initReadingProgress() {
  const progressBar = document.getElementById('readingProgress');
  if (!progressBar) return;

  window.addEventListener('scroll', () => {
    const totalScroll = document.documentElement.scrollTop;
    const windowHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    if (windowHeight <= 0) return;
    const scrollPct = (totalScroll / windowHeight) * 100;
    progressBar.style.width = `${Math.min(100, Math.max(0, scrollPct))}%`;
  });
}

/* ─── Custom Cursor ─────────────────────────────────────────── */
function initCustomCursor() {
  const cursor = document.getElementById('cursor');
  const trail = document.getElementById('cursorTrail');
  if (!cursor || !trail) return;

  let mouseX = window.innerWidth / 2;
  let mouseY = window.innerHeight / 2;
  let trailX = mouseX;
  let trailY = mouseY;

  window.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
    cursor.style.left = `${mouseX}px`;
    cursor.style.top = `${mouseY}px`;
  });

  function renderTrail() {
    trailX += (mouseX - trailX) * 0.18;
    trailY += (mouseY - trailY) * 0.18;
    trail.style.left = `${trailX}px`;
    trail.style.top = `${trailY}px`;
    requestAnimationFrame(renderTrail);
  }
  requestAnimationFrame(renderTrail);

  const interactives = document.querySelectorAll('a, button, input[type="range"], .plot-card, .pipeline-step-card, .sb-preset-btn');
  interactives.forEach((el) => {
    el.addEventListener('mouseenter', () => {
      cursor.classList.add('hovering');
      trail.classList.add('hovering');
    });
    el.addEventListener('mouseleave', () => {
      cursor.classList.remove('hovering');
      trail.classList.remove('hovering');
    });
  });
}

/* ─── Dotted Particle Wave Background ───────────────────────── */
function initDottedSurface() {
  const canvas = document.getElementById('dottedSurfaceCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let width, height;
  let particles = [];
  const rows = 28;
  const cols = 48;
  let mouse = { x: 0, y: 0, active: false };

  function resize() {
    const parent = canvas.parentElement;
    width = canvas.width = parent.offsetWidth;
    height = canvas.height = parent.offsetHeight;
    createGrid();
  }

  function createGrid() {
    particles = [];
    const spacingX = width / cols;
    const spacingY = height / rows;

    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < rows; j++) {
        particles.push({
          x: i * spacingX + spacingX / 2,
          y: j * spacingY + spacingY / 2,
          origX: i * spacingX + spacingX / 2,
          origY: j * spacingY + spacingY / 2,
          baseRadius: (j / rows) * 1.5 + 0.6,
          phase: (i + j) * 0.2
        });
      }
    }
  }

  window.addEventListener('resize', resize);
  resize();

  window.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    if (e.clientY >= rect.top && e.clientY <= rect.bottom) {
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
      mouse.active = true;
    } else {
      mouse.active = false;
    }
  });

  let time = 0;
  function animate() {
    time += 0.03;
    ctx.clearRect(0, 0, width, height);

    particles.forEach((p) => {
      const wave = Math.sin(time + p.phase) * 6;
      let targetX = p.origX;
      let targetY = p.origY + wave;

      if (mouse.active) {
        const dx = mouse.x - p.x;
        const dy = mouse.y - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const maxDist = 140;

        if (dist < maxDist) {
          const force = (1 - dist / maxDist) * 18;
          targetX -= (dx / dist) * force;
          targetY -= (dy / dist) * force;
        }
      }

      p.x += (targetX - p.x) * 0.1;
      p.y += (targetY - p.y) * 0.1;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.baseRadius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(200, 169, 110, ${0.15 + (p.y / height) * 0.25})`;
      ctx.fill();
    });

    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
}

/* ─── Network Telemetry Badge ───────────────────────────────── */
function initTelemetryBadge() {
  const badgeVal = document.getElementById('netSpeedVal');
  if (!badgeVal) return;

  if ('connection' in navigator && navigator.connection && navigator.connection.downlink) {
    const mbps = navigator.connection.downlink;
    badgeVal.textContent = `${mbps} Mbps (Active)`;
  } else {
    badgeVal.textContent = `Vercel Edge Node`;
  }
}

/* ─── Web Audio Synthesizer ─────────────────────────────────── */
let audioCtx = null;
let soundMuted = false;

function initAudioSynthesizer() {
  const toggle = document.getElementById('soundToggle');
  if (!toggle) return;

  toggle.addEventListener('click', () => {
    soundMuted = !soundMuted;
    toggle.classList.toggle('muted', soundMuted);
    if (!soundMuted) playChirp(880, 'sine', 0.08);
  });
}

function playChirp(freq = 440, type = 'sine', duration = 0.05) {
  if (soundMuted) return;
  try {
    if (!audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AudioContext();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }

    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    osc.type = type;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(freq * 1.5, audioCtx.currentTime + duration);

    gain.gain.setValueAtTime(0.04, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);

    osc.connect(gain);
    gain.connect(audioCtx.destination);

    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch (e) {
    // AudioContext blocked or not supported
  }
}

/* ─── Code Copy Buttons ─────────────────────────────────────── */
function initCodeCopyButtons() {
  const copyButtons = document.querySelectorAll('.btn-copy-code');
  copyButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const codeBlock = btn.closest('.code-terminal').querySelector('pre');
      if (codeBlock) {
        navigator.clipboard.writeText(codeBlock.innerText).then(() => {
          const origText = btn.innerHTML;
          btn.innerHTML = `✓ Copied`;
          btn.style.color = 'var(--emerald)';
          playChirp(1200, 'triangle', 0.06);
          setTimeout(() => {
            btn.innerHTML = origText;
            btn.style.color = '';
          }, 2000);
        });
      }
    });
  });
}

/* ─── Live Fraud Risk Inference Sandbox ─────────────────────── */
function initFraudRiskSandbox() {
  const sliders = {
    v14: document.getElementById('sliderV14'),
    v10: document.getElementById('sliderV10'),
    v17: document.getElementById('sliderV17'),
    v3: document.getElementById('sliderV3'),
    v12: document.getElementById('sliderV12'),
    amount: document.getElementById('sliderAmount')
  };

  const readouts = {
    v14: document.getElementById('valV14'),
    v10: document.getElementById('valV10'),
    v17: document.getElementById('valV17'),
    v3: document.getElementById('valV3'),
    v12: document.getElementById('valV12'),
    amount: document.getElementById('valAmount')
  };

  const gaugeScore = document.getElementById('gaugeScore');
  const gaugeCircle = document.getElementById('gaugeFillCircle');
  const riskBadge = document.getElementById('riskBadge');
  const riskDesc = document.getElementById('riskDesc');

  if (!sliders.v14 || !gaugeScore || !gaugeCircle) return;

  const presets = {
    legit: { v14: 0.2, v10: 0.1, v17: -0.1, v3: 0.4, v12: 0.3, amount: 4.5 },
    atm: { v14: -3.5, v10: -2.1, v17: -1.8, v3: -2.0, v12: -1.5, amount: 980 },
    carding: { v14: -7.2, v10: -5.4, v17: -4.8, v3: -6.1, v12: -4.2, amount: 2450 },
    borderline: { v14: -1.8, v10: -1.2, v17: -0.9, v3: -1.1, v12: -0.8, amount: 140 }
  };

  function calculateRisk() {
    const v14 = parseFloat(sliders.v14.value);
    const v10 = parseFloat(sliders.v10.value);
    const v17 = parseFloat(sliders.v17.value);
    const v3 = parseFloat(sliders.v3.value);
    const v12 = parseFloat(sliders.v12.value);
    const amount = parseFloat(sliders.amount.value);

    // Update readout labels
    readouts.v14.textContent = v14.toFixed(1);
    readouts.v10.textContent = v10.toFixed(1);
    readouts.v17.textContent = v17.toFixed(1);
    readouts.v3.textContent = v3.toFixed(1);
    readouts.v12.textContent = v12.toFixed(1);
    readouts.amount.textContent = `$${amount.toLocaleString()}`;

    // Empirical formula calibrated against the trained Random Forest feature weights:
    // In Credit Card Fraud dataset, strong negative values of V14, V10, V17, V3, V12 heavily trigger fraud.
    let logit = -3.8;
    logit += (-v14) * 0.75; // V14 has 32.0% importance
    logit += (-v10) * 0.38; // V10 has 12.2% importance
    logit += (-v17) * 0.34; // V17 has 11.4% importance
    logit += (-v3) * 0.28;  // V3 has 10.7% importance
    logit += (-v12) * 0.22; // V12 has 8.1% importance

    if (amount > 1000) {
      logit += 0.35 * Math.log10(amount / 500);
    }

    // Sigmoid probability
    const probability = 1 / (1 + Math.exp(-logit));
    const pct = Math.min(99.9, Math.max(0.1, probability * 100));

    // Update gauge
    gaugeScore.textContent = `${pct.toFixed(1)}%`;
    const circumference = 440;
    const offset = circumference - (pct / 100) * circumference;
    gaugeCircle.style.strokeDashoffset = offset;

    // Update Risk Badge & Aesthetics
    if (pct < 20) {
      gaugeCircle.style.stroke = 'var(--emerald)';
      riskBadge.className = 'risk-verdict-badge low';
      riskBadge.innerHTML = '● LOW RISK — APPROVE TRANSACTION';
      riskDesc.innerHTML = 'Normal customer behavior pattern. The PCA components match legitimate cardholder baselines with negligible fraud indicators.';
    } else if (pct < 65) {
      gaugeCircle.style.stroke = '#f59e0b';
      riskBadge.className = 'risk-verdict-badge medium';
      riskBadge.innerHTML = '⚠ MEDIUM RISK — STEP-UP 2FA REQUIRED';
      riskDesc.innerHTML = 'Moderate anomaly detected in primary latent components (V14 &amp; V10). Recommended action: prompt for 3D Secure / Biometric challenge.';
    } else {
      gaugeCircle.style.stroke = 'var(--crimson)';
      riskBadge.className = 'risk-verdict-badge high';
      riskBadge.innerHTML = '⚡ CRITICAL FRAUD — IMMEDIATE DECLINE';
      riskDesc.innerHTML = 'Severe anomaly signature detected across top predictive features (V14 &lt; -3.0). High confidence credit card compromise.';
    }
  }

  // Bind sliders
  Object.values(sliders).forEach((slider) => {
    slider.addEventListener('input', () => {
      // Deactivate presets styling on custom slide
      document.querySelectorAll('.sb-preset-btn').forEach(b => b.classList.remove('active'));
      calculateRisk();
      playChirp(600 + parseFloat(slider.value) * 30, 'sine', 0.02);
    });
  });

  // Bind preset buttons
  const presetButtons = document.querySelectorAll('.sb-preset-btn');
  presetButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      presetButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const presetKey = btn.dataset.preset;
      const data = presets[presetKey];
      if (data) {
        sliders.v14.value = data.v14;
        sliders.v10.value = data.v10;
        sliders.v17.value = data.v17;
        sliders.v3.value = data.v3;
        sliders.v12.value = data.v12;
        sliders.amount.value = data.amount;
        calculateRisk();
        playChirp(900, 'triangle', 0.06);
      }
    });
  });

  // Initial calculation
  calculateRisk();
}

/* ─── ScrollSpy for Navigation ──────────────────────────────── */
function initScrollSpy() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-link');

  window.addEventListener('scroll', () => {
    let current = '';
    const scrollPos = window.pageYOffset + 120;

    sections.forEach((section) => {
      const top = section.offsetTop;
      const height = section.offsetHeight;
      if (scrollPos >= top && scrollPos < top + height) {
        current = section.getAttribute('id');
      }
    });

    navLinks.forEach((link) => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  });
}

/* ─── Navbar Scroll & Mobile Drawer Navigation ──────────────── */
function initNavbarScroll() {
  const navbar = document.getElementById('navbar');
  if (!navbar) return;

  window.addEventListener('scroll', () => {
    if (window.scrollY > 25) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  });
}

function initMobileNav() {
  const hamburger = document.getElementById('hamburger');
  const navLinks = document.getElementById('navLinks');
  const overlay = document.getElementById('menuOverlay');
  if (!hamburger || !navLinks || !overlay) return;

  function toggleMenu() {
    const isOpen = navLinks.classList.toggle('open');
    hamburger.classList.toggle('open', isOpen);
    hamburger.setAttribute('aria-expanded', isOpen);
    overlay.classList.toggle('active', isOpen);
    document.body.style.overflow = isOpen ? 'hidden' : '';
    if (isOpen) playChirp(800, 'sine', 0.05);
  }

  function closeMenu() {
    navLinks.classList.remove('open');
    hamburger.classList.remove('open');
    hamburger.setAttribute('aria-expanded', 'false');
    overlay.classList.remove('active');
    document.body.style.overflow = '';
  }

  hamburger.addEventListener('click', toggleMenu);
  overlay.addEventListener('click', closeMenu);

  navLinks.querySelectorAll('.nav-link, .dropdown-item').forEach((link) => {
    link.addEventListener('click', closeMenu);
  });
}

function initNavDropdown() {
  const dropdown = document.getElementById('navDropdown');
  const toggleBtn = document.getElementById('dropdownToggle');
  if (!dropdown || !toggleBtn) return;

  toggleBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const isOpen = dropdown.classList.toggle('open');
    toggleBtn.setAttribute('aria-expanded', isOpen);
    if (isOpen) playChirp(600, 'sine', 0.04);
  });

  document.addEventListener('click', (e) => {
    if (!dropdown.contains(e.target)) {
      dropdown.classList.remove('open');
      toggleBtn.setAttribute('aria-expanded', 'false');
    }
  });

  dropdown.querySelectorAll('.dropdown-item').forEach((item) => {
    item.addEventListener('click', () => {
      dropdown.classList.remove('open');
      toggleBtn.setAttribute('aria-expanded', 'false');
    });
  });
}


