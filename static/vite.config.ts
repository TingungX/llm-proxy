import { defineConfig } from 'vite';
import preact from '@preact/preset-vite';

export default defineConfig({
  plugins: [preact()],

  // Output to static/dist/, served by FastAPI at /static/
  base: '/static/',
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          'chart.js': ['chart.js'],
          'preact': ['preact', '@preact/signals'],
        },
      },
    },
  },

  // Dev server: proxy /api/* to FastAPI
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:4001',
        changeOrigin: true,
      },
    },
  },

  // Preview built dist
  preview: {
    port: 4173,
  },
});
