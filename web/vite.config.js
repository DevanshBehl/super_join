import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The built bundle is served by FastAPI under /app, so asset URLs must be
// absolute against that prefix. In dev, Vite serves at / and proxies the API
// straight through to uvicorn, so no CORS configuration is needed on the
// backend.
export default defineConfig({
  base: '/app/',
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
