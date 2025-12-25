import './style.css'
import { projects } from './project-data.js';

// --- Intro Animation Controller ---
const introOverlay = document.getElementById('intro-overlay');
let introStarted = false;

function startIntroSequence() {
  if (introStarted || !introOverlay) return;
  introStarted = true;

  // Remove click hint
  const clickHint = document.querySelector('.intro-click-hint');
  if (clickHint) clickHint.style.display = 'none';

  // Start the main animation
  const introContent = document.querySelector('.intro-content');
  if (introContent) introContent.classList.add('animating');

  // Prevent scrolling during intro
  document.body.style.overflow = 'hidden';

  // Update loader text dynamically
  const loaderText = document.querySelector('.loader-text');
  const loadingMessages = [
    'Initializing...',
    'Loading neural networks...',
    'Calibrating AI systems...',
    'Ready!'
  ];

  let messageIndex = 0;
  const messageInterval = setInterval(() => {
    messageIndex++;
    if (loaderText && messageIndex < loadingMessages.length) {
      loaderText.textContent = loadingMessages[messageIndex];
    }
  }, 600);

  // Start reveal after loading completes (~3 seconds total)
  setTimeout(() => {
    clearInterval(messageInterval);
    if (loaderText) loaderText.textContent = 'Welcome!';

    // Add reveal class to trigger split animation
    setTimeout(() => {
      introOverlay.classList.add('reveal');

      // Re-enable scrolling
      document.body.style.overflow = '';

      // Speak welcome message during reveal (now allowed after user click)
      speakWelcomeMessage();

      // Mark as complete after reveal animation
      setTimeout(() => {
        introOverlay.classList.add('complete');
        introOverlay.classList.add('hidden');
      }, 1200);
    }, 300);
  }, 2900);
}

// Wait for user interaction to enable audio, then auto-start
function initIntro() {
  if (!introOverlay) return;

  // Add click hint text
  const clickHint = document.createElement('div');
  clickHint.className = 'intro-click-hint';
  clickHint.innerHTML = '<span>Click anywhere to enter</span>';
  introOverlay.appendChild(clickHint);

  // Hide the main content until clicked
  const introContent = document.querySelector('.intro-content');
  if (introContent) introContent.style.opacity = '0';

  // Start on any click/touch
  const startHandler = () => {
    // Show and animate content
    if (introContent) {
      introContent.style.opacity = '1';
    }
    startIntroSequence();
    document.removeEventListener('click', startHandler);
    document.removeEventListener('touchstart', startHandler);
    document.removeEventListener('keydown', startHandler);
  };

  document.addEventListener('click', startHandler);
  document.addEventListener('touchstart', startHandler);
  document.addEventListener('keydown', startHandler);
}

// Initialize intro on page load
initIntro();

// --- Scroll Observer ---
const observerOptions = {
  root: null,
  rootMargin: '0px',
  threshold: 0.1
};

const observer = new IntersectionObserver((entries, observer) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      observer.unobserve(entry.target);
    }
  });
}, observerOptions);

document.querySelectorAll('.fade-on-scroll, .slide-up-reveal').forEach((el) => {
  observer.observe(el);
});

// --- Canvas Neural Network Animation ---
const canvas = document.getElementById('neural-canvas');
const ctx = canvas.getContext('2d');

let width, height;
let particles = [];
let target = { x: 0, y: 0 };

function resize() {
  width = window.innerWidth;
  height = window.innerHeight;
  canvas.width = width;
  canvas.height = height;
}

class Particle {
  constructor() {
    this.x = Math.random() * width;
    this.y = Math.random() * height;
    this.vx = (Math.random() - 0.5) * 0.5;
    this.vy = (Math.random() - 0.5) * 0.5;
    this.size = Math.random() * 2 + 1;
    this.color = `rgba(0, 243, 255, ${Math.random() * 0.5 + 0.1})`;
  }

  update() {
    this.x += this.vx;
    this.y += this.vy;

    if (this.x < 0 || this.x > width) this.vx *= -1;
    if (this.y < 0 || this.y > height) this.vy *= -1;

    // Mouse attraction
    const dx = target.x - this.x;
    const dy = target.y - this.y;
    const distance = Math.sqrt(dx * dx + dy * dy);

    if (distance < 200) {
      this.x += dx * 0.005;
      this.y += dy * 0.005;
    }
  }

  draw() {
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
    ctx.fillStyle = this.color;
    ctx.fill();
  }
}

function initParticles() {
  particles = [];
  const numParticles = Math.min(width * 0.1, 150);
  for (let i = 0; i < numParticles; i++) {
    particles.push(new Particle());
  }
}

function animate() {
  ctx.clearRect(0, 0, width, height);

  particles.forEach((p, index) => {
    p.update();
    p.draw();

    for (let j = index + 1; j < particles.length; j++) {
      const p2 = particles[j];
      const dx = p.x - p2.x;
      const dy = p.y - p2.y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance < 150) {
        ctx.beginPath();
        ctx.strokeStyle = `rgba(0, 243, 255, ${1 - distance / 150})`;
        ctx.lineWidth = 0.5;
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
    }
  });

  requestAnimationFrame(animate);
}

// --- Custom Cursor ---
const cursorDot = document.querySelector('[data-cursor-dot]');
const cursorOutline = document.querySelector('[data-cursor-outline]');

window.addEventListener('mousemove', (e) => {
  const posX = e.clientX;
  const posY = e.clientY;

  if (cursorDot) {
    cursorDot.style.left = `${posX}px`;
    cursorDot.style.top = `${posY}px`;
  }

  if (cursorOutline) {
    cursorOutline.animate({
      left: `${posX}px`,
      top: `${posY}px`
    }, { duration: 500, fill: "forwards" });
  }

  target.x = posX;
  target.y = posY;
});

// --- Dynamic Projects Rendering ---
const projectGrid = document.querySelector('.project-grid');

function renderProjects() {
  projectGrid.innerHTML = projects.map(project => `
        <div class="project-card fade-on-scroll" data-id="${project.id}">
            <div class="card-inner">
                <p class="project-type">${project.type}</p>
                <h3>${project.title}</h3>
                <div class="mini-tags">
                   ${project.tech.slice(0, 3).map(t => `<span>${t}</span>`).join('')}
                </div>
            </div>
        </div>
    `).join('');

  // Re-observe new elements
  document.querySelectorAll('.project-card').forEach(el => observer.observe(el));

  // Add Click Listeners
  document.querySelectorAll('.project-card').forEach(card => {
    card.addEventListener('click', () => {
      const pid = card.getAttribute('data-id');
      const project = projects.find(p => p.id === pid);
      openModal(project);
    });
  });
}

// --- Modal Logic ---
const modal = document.getElementById('project-modal');
const modalClose = document.querySelector('.close-modal');

function openModal(project) {
  if (!project) return;
  document.getElementById('modal-title').innerText = project.title;
  document.getElementById('modal-category').innerText = project.category;

  // New Fields
  document.getElementById('modal-problem').innerText = project.problem;
  document.getElementById('modal-solution').innerText = project.solution;
  document.getElementById('modal-impact').innerHTML = project.impact.map(i => `<li>${i}</li>`).join('');

  document.getElementById('modal-tags').innerHTML = project.tech.map(t => `<span>${t}</span>`).join('');

  // Conditional Link with Icon
  let linkHtml = '';
  if (project.link && project.link !== '#') {
    linkHtml = `<a href="${project.link}" target="_blank" class="btn btn-primary">View Live Project</a>`;
  } else {
    linkHtml = `<span class="btn btn-secondary" style="opacity:0.6; cursor:not-allowed">Private / Demonstration</span>`;
  }
  document.getElementById('modal-links').innerHTML = linkHtml;

  modal.style.display = "block";
}

if (modalClose) {
  modalClose.addEventListener('click', () => {
    modal.style.display = "none";
  });
}

window.addEventListener('click', (e) => {
  if (e.target == modal) {
    modal.style.display = "none";
  }
});

// --- EmailJS Contact Form ---
const contactForm = document.getElementById('contact-form');

if (contactForm) {
  contactForm.addEventListener('submit', function (event) {
    event.preventDefault();

    const btn = contactForm.querySelector('button');
    const originalText = btn.innerText;
    btn.innerText = 'Sending...';

    // Simulating 2 second delay for UX
    setTimeout(() => {
      // In production: emailjs.sendForm('contact_service', 'contact_form', this)

      // For demo:
      alert('Message Sent! (Note: Connect your EmailJS Service ID to make this live)');
      contactForm.reset();
      btn.innerText = originalText;
    }, 1500);
  });
}


// --- Init ---
window.addEventListener('resize', () => {
  resize();
  initParticles();
});

resize();
initParticles();
animate();
renderProjects();

// Mobile Nav
const hamburger = document.querySelector('.hamburger');
const navLinks = document.querySelector('.nav-links');
if (hamburger) {
  hamburger.addEventListener('click', () => {
    navLinks.style.display = navLinks.style.display === 'flex' ? 'none' : 'flex';
    // Add minimal styling toggle class instead of inline styles for better maintanence
    // But inline for speed here:
    if (navLinks.style.display === 'flex') {
      navLinks.style.flexDirection = 'column';
      navLinks.style.position = 'absolute';
      navLinks.style.top = '70px';
      navLinks.style.right = '0';
      navLinks.style.background = '#050510';
      navLinks.style.width = '100%';
      navLinks.style.padding = '2rem';
      navLinks.style.borderBottom = '1px solid #333';
    }
  });
}

// --- Text Scramble Effect ---
const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()";

function scrambleText(element) {
  let iteration = 0;
  const originalText = element.dataset.text;
  let interval = null;

  clearInterval(interval);

  interval = setInterval(() => {
    element.innerText = originalText
      .split("")
      .map((letter, index) => {
        if (index < iteration) {
          return originalText[index];
        }
        return letters[Math.floor(Math.random() * letters.length)];
      })
      .join("");

    if (iteration >= originalText.length) {
      clearInterval(interval);
    }

    iteration += 1 / 3;
  }, 40); // Speed
}

const glitchText = document.querySelector('.glitch');
if (glitchText) {
  // Run on load only
  scrambleText(glitchText);
}

// --- Audio Interaction Setup ---
const synth = window.speechSynthesis;
let voices = [];

function loadVoices() {
  voices = synth.getVoices();
  // Debug log to confirm voices loaded
  if (voices.length > 0) console.log("Voices loaded:", voices.length);
}

// Initial load attempt
loadVoices();

// Retry mechanism for Chrome/Safari which load voices async
if (voices.length === 0) {
  setTimeout(() => {
    loadVoices();
    if (voices.length === 0) {
      setTimeout(loadVoices, 100);
    }
  }, 50);
}

if (speechSynthesis.onvoiceschanged !== undefined) {
  speechSynthesis.onvoiceschanged = loadVoices;
}

function speakText(text, pitch = 1, rate = 1) {
  if (synth.speaking) {
    // console.error('speechSynthesis.speaking');
    // Instead of returning, we CANCEL the current speech to allow the new one
    synth.cancel();
  }

  // Tiny delay to ensure cancel takes effect
  setTimeout(() => {
    if (text !== '') {
      const utterThis = new SpeechSynthesisUtterance(text);
      utterThis.onend = function (event) {
        console.log('SpeechSynthesisUtterance.onend');
      }
      utterThis.onerror = function (event) {
        console.error('SpeechSynthesisUtterance.onerror');
      }

      // Voice Selection Logic (Prioritize MALE / BOY voices)
      const voice = voices.find(v => v.name.includes('Google US English Male')) ||
        voices.find(v => v.name.includes('Microsoft David')) ||
        voices.find(v => v.name.includes('Male')) ||
        voices.find(v => v.lang.includes('en-US') && !v.name.includes('Female'));

      if (voice) {
        utterThis.voice = voice;
      }

      utterThis.pitch = pitch;
      utterThis.rate = rate;
      synth.speak(utterThis);
    }
  }, 10);
}

// --- Welcome Speech for Intro ---
function speakWelcomeMessage() {
  // Small delay to ensure voices are loaded and animation is visible
  setTimeout(() => {
    if (synth.paused) synth.resume();
    speakText("Welcome to Imran Kha-lid's portfolio", 1.2, 1);
  }, 200);
}

// --- Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {

  // 2. Professional Skills Title Hover
  const skillsTitle = document.querySelector('.skills-section .section-title h2');
  if (skillsTitle) {
    let lastPlayedTitle = 0;
    skillsTitle.addEventListener('mouseenter', () => {
      const now = Date.now();
      if (now - lastPlayedTitle > 4000) {
        speakText("Imran's professional skills", 1.1, 1);
        lastPlayedTitle = now;
      }
    });
  }

  // 3. Skill Category HOVER (Whole Card - Changed from Header)
  const skillCards = document.querySelectorAll('.skill-card');
  skillCards.forEach(card => {
    let lastPlayedSkill = 0;
    card.addEventListener('mouseenter', () => {
      const now = Date.now();
      // 5s cooldown to prevent chaos
      if (now - lastPlayedSkill > 5000) {
        // Find the H3 inside this card to get the text
        const header = card.querySelector('h3');
        if (header) {
          const textToSpeak = header.textContent.trim();
          speakText(textToSpeak, 1.1, 1);
          lastPlayedSkill = now;
        }
      }
    });
  });
});

// --- Staggered Fade for Tagline ---
const tagline = document.querySelector('.stagger-text');
if (tagline) {
  const words = tagline.innerText.split(' ');
  tagline.innerHTML = '';
  words.forEach((word, index) => {
    const span = document.createElement('span');
    span.innerText = word + ' ';
    span.className = 'stagger-word';
    span.style.animationDelay = `${index * 0.1 + 2}s`; // Start after 2s, stagger by 0.1s
    tagline.appendChild(span);
  });
}
