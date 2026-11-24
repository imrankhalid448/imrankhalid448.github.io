// Generate branded SVG cover art for every project.
// These are used as card thumbnails and as the first gallery image on
// project detail pages, so no project ever renders a broken image.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { projects, domains } from './src/project-data.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const outDir = path.join(__dirname, 'projects', 'images', 'covers');

const esc = (value) => String(value)
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;');

function motif(domain, accent) {
  switch (domain) {
    case 'medical':
      return `
        <g fill="none" stroke="${accent}" stroke-width="2" opacity=".65">
          <circle cx="900" cy="330" r="150" opacity=".35"/>
          <circle cx="900" cy="330" r="105" opacity=".5"/>
          <circle cx="900" cy="330" r="60" opacity=".7"/>
          <path d="M720 330h70l22-58 34 116 26-72 18 14h70" stroke-width="3"/>
          <path d="M900 210v240M780 330h240" opacity=".25"/>
        </g>
        <g fill="${accent}" opacity=".8">
          <circle cx="790" cy="272" r="4"/><circle cx="1012" cy="390" r="4"/>
          <circle cx="955" cy="222" r="3"/><circle cx="845" cy="430" r="3"/>
        </g>`;
    case 'cv':
      return `
        <g fill="none" stroke="${accent}" stroke-width="2" opacity=".6">
          <path d="M760 210h-50v50M1040 210h50v50M760 450h-50v-50M1040 450h50v-50"/>
          <rect x="840" y="260" width="120" height="140" rx="4" stroke-width="2.5"/>
          <rect x="905" y="300" width="140" height="120" rx="4" opacity=".55"/>
          <circle cx="900" cy="330" r="175" stroke-dasharray="4 12" opacity=".25"/>
        </g>
        <g fill="${accent}" opacity=".75">
          <rect x="840" y="252" width="46" height="6" rx="3"/>
          <rect x="905" y="292" width="40" height="5" rx="2.5" opacity=".55"/>
        </g>`;
    case 'nlp':
      return `
        <g fill="none" stroke="${accent}" stroke-width="2" opacity=".55">
          <rect x="790" y="225" width="250" height="86" rx="20"/>
          <rect x="850" y="335" width="230" height="78" rx="20" opacity=".75"/>
          <path d="M822 311l-8 28 30-28" />
          <path d="M880 262h120M880 284h80M910 368h110M910 390h70" stroke-width="5" stroke-linecap="round" opacity=".55"/>
        </g>
        <g fill="${accent}" opacity=".6">
          <circle cx="1042" cy="210" r="5"/><circle cx="770" cy="365" r="4"/>
        </g>`;
    case 'speech':
      return `
        <g fill="${accent}" opacity=".7">
          ${Array.from({ length: 28 }, (_, i) => {
            const h = 18 + Math.abs(Math.sin(i * 0.9)) * 116;
            return `<rect x="${670 + i * 18}" y="${340 - h / 2}" width="7" height="${h}" rx="3.5" opacity="${0.28 + (i % 4) * 0.13}"/>`;
          }).join('')}
        </g>
        <g fill="none" stroke="${accent}" stroke-width="2" opacity=".35">
          <circle cx="900" cy="340" r="170" stroke-dasharray="2 10"/>
        </g>`;
    case 'predictive':
      return `
        <g fill="none" stroke="${accent}" stroke-width="2" opacity=".45">
          <path d="M690 450h360M690 450V190" opacity=".4"/>
          <path d="M715 410l60-55 55 22 55-80 60 32 60-92 55 38" stroke-width="3.5"/>
          <path d="M715 410l60-55 55 22 55-80 60 32 60-92 55 38V450H715Z" fill="${accent}" opacity=".08" stroke="none"/>
        </g>
        <g fill="${accent}">
          <circle cx="775" cy="355" r="5"/><circle cx="830" cy="377" r="5"/>
          <circle cx="885" cy="297" r="5"/><circle cx="945" cy="329" r="5"/>
          <circle cx="1005" cy="237" r="5"/><circle cx="1060" cy="275" r="5"/>
        </g>`;
    case 'agents':
      return `
        <g fill="none" stroke="${accent}" stroke-width="2" opacity=".55">
          <circle cx="800" cy="260" r="38"/><circle cx="1010" cy="230" r="30"/>
          <circle cx="940" cy="400" r="42"/><circle cx="760" cy="405" r="26"/>
          <path d="M833 280l142-30M825 285l88 92M962 383l-176 12M786 380l20-86"/>
          <path d="M800 222v-34M1010 200v-26M940 358v-30M760 431v26" opacity=".5"/>
        </g>
        <g fill="${accent}">
          <circle cx="800" cy="260" r="8"/><circle cx="1010" cy="230" r="7"/>
          <circle cx="940" cy="400" r="8"/><circle cx="760" cy="405" r="6"/>
        </g>`;
    case 'autonomous':
      return `
        <g fill="none" stroke="${accent}" stroke-width="2" opacity=".55">
          <path d="M700 420c70-150 140-190 220-150s120 40 190-70" stroke-dasharray="6 10"/>
          <path d="M720 250c90 30 130 100 210 110s120-40 170-100" stroke-dasharray="6 10" opacity=".5"/>
          <path d="M740 390l38-22 38 22-38 22zM930 260l30-18 30 18-30 18zM1000 210l26-15 26 15-26 15z"/>
        </g>
        <g fill="${accent}" opacity=".75">
          <circle cx="740" cy="390" r="4"/><circle cx="930" cy="260" r="4"/><circle cx="1000" cy="210" r="4"/>
        </g>`;
    default:
      return '';
  }
}

fs.mkdirSync(outDir, { recursive: true });

projects.forEach((project) => {
  const domain = domains[project.domain] || { label: project.domain, color: '#00f3ff' };
  const accent = domain.color || '#00f3ff';
  const title = esc(project.title);
  const label = esc(domain.label);
  const labelUpper = esc(domain.label.toUpperCase());
  const shortTitle = project.title.length > 22 ? project.title.slice(0, 21) + '…' : project.title;

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 750" role="img" aria-labelledby="title desc">
  <title id="title">${title} — ${label}</title>
  <desc id="desc">Cover art for the ${title} project by Imran Khalid.</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#050510"/>
      <stop offset=".55" stop-color="#08081a"/>
      <stop offset="1" stop-color="#0d0b24"/>
    </linearGradient>
    <radialGradient id="glow" cx=".5" cy=".5" r=".5">
      <stop offset="0" stop-color="${accent}" stop-opacity=".38"/>
      <stop offset="1" stop-color="${accent}" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="line" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="${accent}" stop-opacity="0"/>
      <stop offset=".5" stop-color="${accent}"/>
      <stop offset="1" stop-color="${accent}" stop-opacity="0"/>
    </linearGradient>
    <pattern id="grid" width="48" height="48" patternUnits="userSpaceOnUse">
      <path d="M48 0H0V48" fill="none" stroke="${accent}" stroke-opacity=".09" stroke-width="1"/>
    </pattern>
    <filter id="soft" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="18"/>
    </filter>
  </defs>

  <rect width="1200" height="750" fill="url(#bg)"/>
  <rect width="1200" height="750" fill="url(#grid)"/>
  <circle cx="930" cy="320" r="330" fill="url(#glow)"/>
  <circle cx="930" cy="320" r="330" fill="none" stroke="${accent}" stroke-opacity=".12" stroke-width="1"/>
  <path d="M0 610H1200" stroke="url(#line)" stroke-opacity=".45"/>

  <g opacity=".9">
    ${motif(project.domain, accent)}
  </g>

  <g transform="translate(70 62)">
    <rect width="132" height="36" rx="18" fill="${accent}" fill-opacity=".08" stroke="${accent}" stroke-opacity=".55"/>
    <circle cx="20" cy="18" r="4" fill="${accent}"/>
    <text x="34" y="23" font-family="Inter, Arial, sans-serif" font-size="13" letter-spacing="2.5" fill="${accent}">AI / ML</text>
  </g>

  <text x="70" y="470" font-family="Poppins, Inter, Arial, sans-serif" font-size="${shortTitle.length > 17 ? 54 : 66}" font-weight="700" letter-spacing="-1" fill="#ffffff">${esc(shortTitle)}</text>
  <text x="72" y="522" font-family="Inter, Arial, sans-serif" font-size="19" letter-spacing="3.4" fill="${accent}" opacity=".92">${labelUpper}</text>
  <text x="72" y="660" font-family="Inter, Arial, sans-serif" font-size="14" letter-spacing="3" fill="#9a9ab0">IMRAN KHALID · AI &amp; ML SPECIALIST</text>

  <g transform="translate(1005 618)">
    <circle r="38" fill="#050510" fill-opacity=".65" stroke="${accent}" stroke-opacity=".45"/>
    <path d="M0-18v36M-18 0h36" stroke="${accent}" stroke-width="2" stroke-opacity=".8"/>
    <circle r="7" fill="${accent}"/>
  </g>
</svg>`;

  fs.writeFileSync(path.join(outDir, `${project.id}.svg`), svg, 'utf8');
});

console.log(`Generated ${projects.length} project covers in projects/images/covers/`);
