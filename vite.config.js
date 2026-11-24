import { defineConfig } from 'vite';
import fs from 'fs';
import path from 'path';

// Vite only copies the `public/` directory to `dist/`. This plugin also
// ships the generated project detail pages and root-level static files so the
// deployed site (GitHub Pages) actually contains them.
function copyStaticAssetsPlugin() {
  return {
    name: 'copy-static-assets',
    closeBundle() {
      const root = __dirname;
      const dist = path.resolve(root, 'dist');

      const copyDir = (src, dest) => {
        if (fs.existsSync(src)) {
          fs.cpSync(src, dest, { recursive: true, force: true });
        }
      };
      const copyFile = (src, dest) => {
        if (fs.existsSync(src)) {
          fs.copyFileSync(src, dest);
        }
      };

      copyDir(path.resolve(root, 'projects'), path.resolve(dist, 'projects'));
      copyFile(path.resolve(root, 'robots.txt'), path.resolve(dist, 'robots.txt'));
      copyFile(path.resolve(root, 'sitemap.xml'), path.resolve(dist, 'sitemap.xml'));
      copyFile(path.resolve(root, 'CNAME'), path.resolve(dist, 'CNAME'));
    }
  };
}

export default defineConfig({
  base: '/',
  plugins: [copyStaticAssetsPlugin()],
  server: {
    allowedHosts: [
      'craniometrically-pseudoconfessional-esmeralda.ngrok-free.dev'
    ],
    host: true,
    port: 5174
  }
});
