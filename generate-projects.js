// Generate Project Detail Pages
// This script creates individual HTML pages for each project

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Import project data
const projectDataPath = path.join(__dirname, 'src', 'project-data.js');
const projectDataContent = fs.readFileSync(projectDataPath, 'utf8');

// Extract projects and domains (simple regex extraction)
const projectsMatch = projectDataContent.match(/export const projects = \[([\s\S]*?)\];/);
const domainsMatch = projectDataContent.match(/export const domains = \{([\s\S]*?)\};/);

// Load template
const templatePath = path.join(__dirname, 'projects', 'template.html');
const template = fs.readFileSync(templatePath, 'utf8');

// Parse projects (using eval in controlled environment - only for build script)
const projectsCode = `const domains = {${domainsMatch[1]}}; const projects = [${projectsMatch[1]}]; projects;`;
const projects = eval(projectsCode);
const domains = eval(`({${domainsMatch[1]}})`);

console.log(`Found ${projects.length} projects to generate...`);

// Generate HTML for each project
projects.forEach((project, index) => {
  console.log(`Generating ${index + 1}/${projects.length}: ${project.title}...`);

  const domainInfo = domains[project.domain];

  // Build project links HTML
  let linksHTML = '';
  if (project.links) {
    if (project.links.github && project.links.github !== '#') {
      linksHTML += `<a href="${project.links.github}" target="_blank" class="btn btn-primary"><i class="fab fa-github"></i> View on GitHub</a>`;
    }
    if (project.links.demo && project.links.demo !== '#') {
      linksHTML += `<a href="${project.links.demo}" target="_blank" class="btn btn-secondary"><i class="fas fa-external-link-alt"></i> Live Demo</a>`;
    }
    if (project.links.paper && project.links.paper !== '#') {
      linksHTML += `<a href="${project.links.paper}" target="_blank" class="btn btn-secondary"><i class="fas fa-file-pdf"></i> Read Paper</a>`;
    }
  }

  if (!linksHTML) {
    linksHTML = '<span class="btn btn-secondary" style="opacity:0.6; cursor:not-allowed;">Private / Under Development</span>';
  }

  // Build architecture section
  let architectureHTML = '';
  if (project.architecture && project.architecture.components) {
    architectureHTML = `
      <section class="content-section">
        <h2><i class="fas fa-project-diagram"></i> Technical Architecture</h2>
        <p>${project.architecture.description || ''}</p>
        <div class="architecture-grid">
          ${project.architecture.components.map(comp => `
            <div class="arch-component">
              <h3>${comp.name}</h3>
              <p>${comp.description}</p>
            </div>
          `).join('')}
        </div>
      </section>
    `;
  }

  // Build methodology section
  let methodologyHTML = '';
  if (project.methodology && project.methodology.length > 0) {
    methodologyHTML = `
      <section class="content-section">
        <h2><i class="fas fa-tasks"></i> Methodology</h2>
        <ol class="methodology-steps">
          ${project.methodology.map(step => `<li>${step}</li>`).join('')}
        </ol>
      </section>
    `;
  }

  // Build metrics section
  let metricsHTML = '';
  if (project.results && project.results.metrics) {
    metricsHTML = `
      <div class="metrics-grid">
        ${project.results.metrics.map(metric => `
          <div class="metric-card">
            <span class="metric-value">${metric.value}</span>
            <span class="metric-label">${metric.label}</span>
            ${metric.description ? `<span class="metric-description">${metric.description}</span>` : ''}
          </div>
        `).join('')}
      </div>
    `;
  }

  // Build impact list
  let impactHTML = '';
  if (project.results && project.results.impact) {
    impactHTML = project.results.impact.map(item => `<li>${item}</li>`).join('');
  }

  // Build challenges section
  let challengesHTML = '';
  if (project.challenges && project.challenges.length > 0) {
    challengesHTML = `
      <section class="content-section">
        <h2><i class="fas fa-puzzle-piece"></i> Challenges & Solutions</h2>
        <div class="challenges-grid">
          ${project.challenges.map(ch => `
            <div class="challenge-item">
              <h4>${ch.challenge}</h4>
              <p class="solution">${ch.solution}</p>
            </div>
          `).join('')}
        </div>
      </section>
    `;
  }

  // Build code section
  let codeHTML = '';
  if (project.codeSnippets && project.codeSnippets.length > 0) {
    codeHTML = `
      <section class="content-section">
        <h2><i class="fas fa-code"></i> Key Implementation</h2>
        ${project.codeSnippets.map(snippet => `
          <div class="code-section">
            <h3>${snippet.title}</h3>
            <pre><code>${escapeHtml(snippet.code)}</code></pre>
          </div>
        `).join('')}
      </section>
    `;
  }

  // Build image gallery section
  let imageGalleryHTML = '';
  if (project.images && project.images.length > 0) {
    imageGalleryHTML = `
      <section class="content-section">
        <h2><i class="fas fa-images"></i> Project Gallery</h2>
        <div class="image-gallery">
          ${project.images.map((img, idx) => `
            <div class="gallery-item">
              <img src="${img}" alt="${project.title} - Image ${idx + 1}" loading="lazy">
            </div>
          `).join('')}
        </div>
      </section>
    `;
  }

  // Build tech stack
  const techStackHTML = project.tech.map(tech => `<span class="tech-badge">${tech}</span>`).join('');

  // Replace placeholders in template
  let html = template
    .replace(/\{\{PROJECT_TITLE\}\}/g, project.title)
    .replace(/\{\{DOMAIN_ICON\}\}/g, domainInfo.icon)
    .replace(/\{\{DOMAIN_LABEL\}\}/g, domainInfo.label)
    .replace(/\{\{PROJECT_TAGLINE\}\}/g, project.tagline || project.overview)
    .replace(/\{\{TIMELINE\}\}/g, project.timeline || 'N/A')
    .replace(/\{\{TEAM\}\}/g, project.team || 'Solo Project')
    .replace(/\{\{STATUS\}\}/g, project.status || 'Completed')
    .replace(/\{\{PROJECT_LINKS\}\}/g, linksHTML)
    .replace(/\{\{OVERVIEW\}\}/g, project.overview)
    .replace(/\{\{PROBLEM\}\}/g, project.problem)
    .replace(/\{\{SOLUTION\}\}/g, project.solution)
    .replace(/\{\{IMAGE_GALLERY_SECTION\}\}/g, imageGalleryHTML)
    .replace(/\{\{ARCHITECTURE_SECTION\}\}/g, architectureHTML)
    .replace(/\{\{METHODOLOGY_SECTION\}\}/g, methodologyHTML)
    .replace(/\{\{METRICS_SECTION\}\}/g, metricsHTML)
    .replace(/\{\{IMPACT_LIST\}\}/g, impactHTML)
    .replace(/\{\{CHALLENGES_SECTION\}\}/g, challengesHTML)
    .replace(/\{\{CODE_SECTION\}\}/g, codeHTML)
    .replace(/\{\{TECH_STACK\}\}/g, techStackHTML);

  // Create project directory
  const projectDir = path.join(__dirname, 'projects', project.id);
  if (!fs.existsSync(projectDir)) {
    fs.mkdirSync(projectDir, { recursive: true });
  }

  // Write HTML file
  const htmlPath = path.join(projectDir, 'index.html');
  fs.writeFileSync(htmlPath, html, 'utf8');

  console.log(`✓ Generated: projects/${project.id}/index.html`);
});

console.log(`\n✅ Successfully generated ${projects.length} project pages!`);

// Helper function to escape HTML
function escapeHtml(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, m => map[m]);
}
