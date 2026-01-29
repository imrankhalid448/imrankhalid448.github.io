import { defineConfig
} from 'vite';

export default defineConfig({
  base: '/',
  server: {
    allowedHosts: [
      'craniometrically-pseudoconfessional-esmeralda.ngrok-free.dev'
        ],
    host: true,
    port: 5174
    }
});
