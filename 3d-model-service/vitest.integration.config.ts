import { defineConfig } from 'vitest/config';
import { resolve } from 'path';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    setupFiles: ['./test/integration/setup.ts'],
    include: ['**/integration/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'],
    exclude: ['**/node_modules/**', '**/dist/**', '**/coverage/**'],
    testTimeout: 60000, // Longer timeout for integration tests
    hookTimeout: 60000,
    teardownTimeout: 60000,
    pool: 'forks', // Run integration tests in separate processes
    poolOptions: {
      forks: {
        singleFork: true, // Prevent parallel container conflicts
      },
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
      '@test': resolve(__dirname, './test'),
    },
  },
});