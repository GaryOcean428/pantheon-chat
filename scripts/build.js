import { execSync } from 'child_process';
import { existsSync, mkdirSync } from 'fs';

console.log('Starting build process...\n');

try {
  // Step 1: Build frontend with Vite
  console.log('Building frontend with Vite...');
  execSync('vite build', { stdio: 'inherit' });
  console.log('✓ Frontend build complete\n');

  // Step 2: Bundle backend with esbuild
  console.log('Bundling backend with esbuild...');
  execSync('esbuild server/index.ts --platform=node --packages=external --bundle --format=esm --outdir=dist', { stdio: 'inherit' });
  console.log('✓ Backend bundle complete\n');

  // Ensure dist directory exists
  if (!existsSync('dist')) {
    mkdirSync('dist', { recursive: true });
  }

  console.log('Build completed successfully!');
} catch (error) {
  console.error('Build failed:', error.message);
  process.exit(1);
}
