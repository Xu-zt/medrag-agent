import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // No proxy — set VITE_API_URL=http://localhost:8000 in .env.local for dev.
    // The backend exposes CORS for all origins so direct browser requests work.
  },
})
