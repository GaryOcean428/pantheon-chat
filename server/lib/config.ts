/**
 * Shared Configuration Module
 * 
 * Centralizes environment variable checks and configuration logic
 * to ensure consistency across the application.
 */

/**
 * Check if QIG curriculum-only mode is enabled
 * @returns true if QIG_CURRICULUM_ONLY environment variable is set to 'true'
 */
export function isCurriculumOnlyMode(): boolean {
  return process.env.QIG_CURRICULUM_ONLY === 'true';
}

/**
 * Check if QIG purity mode is enabled
 * @returns true if QIG_PURITY_MODE environment variable is set to 'true'
 */
export function isPurityMode(): boolean {
  return process.env.QIG_PURITY_MODE === 'true';
}

/**
 * Get the Python backend URL
 * @returns The Python backend URL from environment or default
 */
export function getPythonBackendUrl(): string {
  return process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
}

/**
 * Get configuration summary for debugging
 * @returns Object with current configuration values
 */
export function getConfigSummary() {
  return {
    curriculumOnly: isCurriculumOnlyMode(),
    purityMode: isPurityMode(),
    pythonBackendUrl: getPythonBackendUrl(),
  };
}
