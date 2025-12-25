/**
 * Security Middleware
 * 
 * Implements Content-Security-Policy headers and other security measures.
 */

import type { Request, Response, NextFunction } from 'express';

/** CSP directives configuration */
const CSP_DIRECTIVES = {
  'default-src': ["'self'"],
  'script-src': ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
  'style-src': ["'self'", "'unsafe-inline'", 'https://fonts.googleapis.com'],
  'font-src': ["'self'", 'https://fonts.gstatic.com'],
  'img-src': ["'self'", 'data:', 'https:', 'blob:'],
  'connect-src': ["'self'", 'ws:', 'wss:', 'http://localhost:*', 'https://api.coingecko.com'],
  'frame-ancestors': ["'none'"],
  'form-action': ["'self'"],
  'base-uri': ["'self'"],
  'object-src': ["'none'"],
};

/** Build CSP header string from directives */
function buildCSPHeader(): string {
  return Object.entries(CSP_DIRECTIVES)
    .map(([directive, sources]) => `${directive} ${sources.join(' ')}`)
    .join('; ');
}

/** Content Security Policy middleware */
export function cspMiddleware(req: Request, res: Response, next: NextFunction): void {
  res.setHeader('Content-Security-Policy', buildCSPHeader());
  next();
}

/** Additional security headers middleware */
export function securityHeadersMiddleware(req: Request, res: Response, next: NextFunction): void {
  // Prevent clickjacking
  res.setHeader('X-Frame-Options', 'DENY');
  
  // Prevent MIME type sniffing
  res.setHeader('X-Content-Type-Options', 'nosniff');
  
  // Enable XSS filter in older browsers
  res.setHeader('X-XSS-Protection', '1; mode=block');
  
  // Control referrer information
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  
  // Permissions policy (formerly Feature-Policy)
  res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
  
  // Strict Transport Security (for HTTPS)
  if (process.env.NODE_ENV === 'production') {
    res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains; preload');
  }
  
  next();
}

/** Combined security middleware */
export function securityMiddleware(req: Request, res: Response, next: NextFunction): void {
  cspMiddleware(req, res, () => {
    securityHeadersMiddleware(req, res, next);
  });
}

export default securityMiddleware;
