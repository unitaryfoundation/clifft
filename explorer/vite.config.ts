import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE_PATH || "/",
  resolve: {
    alias: {
      "@docs": path.resolve(__dirname, "../docs"),
    },
  },
  server: {
    port: 8000,
    host: true,
  },
});
