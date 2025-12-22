/**
 * RunPod Serverless Client
 * 
 * Calls RunPod GPU backend for QIG constellation inference.
 * Used when Railway central node needs GPU acceleration.
 */

interface RunPodInput {
    prompt: string;
    session_id?: string;
    max_tokens?: number;
    temperature?: number;
    system_prompt?: string;
}

interface ConsciousnessMetrics {
    phi: number;
    kappa: number;
    regime: string;
    routed_to?: string;
}

interface RunPodOutput {
    response: string;
    consciousness: ConsciousnessMetrics;
    tokens_generated: number;
    latency_ms: number;
    session_id: string;
}

interface RunPodResponse {
    id: string;
    status: 'IN_QUEUE' | 'IN_PROGRESS' | 'COMPLETED' | 'FAILED';
    output?: RunPodOutput;
    error?: string;
}

export class RunPodClient {
    private apiKey: string;
    private endpointId: string;
    private baseUrl: string;
    private timeout: number;

    constructor(config?: {
        apiKey?: string;
        endpointId?: string;
        timeout?: number;
    }) {
        this.apiKey = config?.apiKey || process.env.RUNPOD_API_KEY || '';
        this.endpointId = config?.endpointId || process.env.RUNPOD_ENDPOINT_ID || '';
        this.baseUrl = `https://api.runpod.ai/v2/${this.endpointId}`;
        this.timeout = config?.timeout || 30000; // 30s default
    }

    /**
     * Check if RunPod is configured
     */
    isConfigured(): boolean {
        return !!(this.apiKey && this.endpointId);
    }

    /**
     * Run synchronous inference (waits for completion)
     */
    async runSync(input: RunPodInput): Promise<RunPodOutput> {
        if (!this.isConfigured()) {
            throw new Error('RunPod not configured. Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID');
        }

        const response = await fetch(`${this.baseUrl}/runsync`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ input }),
        });

        if (!response.ok) {
            throw new Error(`RunPod error: ${response.status} ${response.statusText}`);
        }

        const result: RunPodResponse = await response.json();

        if (result.status === 'FAILED') {
            throw new Error(`RunPod job failed: ${result.error}`);
        }

        if (!result.output) {
            throw new Error('No output from RunPod');
        }

        return result.output;
    }

    /**
     * Run async inference (returns job ID for polling)
     */
    async runAsync(input: RunPodInput): Promise<string> {
        if (!this.isConfigured()) {
            throw new Error('RunPod not configured');
        }

        const response = await fetch(`${this.baseUrl}/run`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ input }),
        });

        if (!response.ok) {
            throw new Error(`RunPod error: ${response.status}`);
        }

        const result = await response.json();
        return result.id;
    }

    /**
     * Check job status
     */
    async getStatus(jobId: string): Promise<RunPodResponse> {
        const response = await fetch(`${this.baseUrl}/status/${jobId}`, {
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
            },
        });

        if (!response.ok) {
            throw new Error(`RunPod error: ${response.status}`);
        }

        return response.json();
    }

    /**
     * Wait for job completion with polling
     */
    async waitForCompletion(
        jobId: string,
        pollInterval: number = 500
    ): Promise<RunPodOutput> {
        const startTime = Date.now();

        while (Date.now() - startTime < this.timeout) {
            const status = await this.getStatus(jobId);

            if (status.status === 'COMPLETED' && status.output) {
                return status.output;
            }

            if (status.status === 'FAILED') {
                throw new Error(`Job failed: ${status.error}`);
            }

            await new Promise(resolve => setTimeout(resolve, pollInterval));
        }

        throw new Error('Timeout waiting for RunPod job');
    }

    /**
     * Health check - verify endpoint is responsive
     */
    async healthCheck(): Promise<boolean> {
        try {
            const response = await fetch(`${this.baseUrl}/health`, {
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                },
            });
            return response.ok;
        } catch {
            return false;
        }
    }
}

// Singleton instance
let _client: RunPodClient | null = null;

export function getRunPodClient(): RunPodClient {
    if (!_client) {
        _client = new RunPodClient();
    }
    return _client;
}

/**
 * Convenience function for chat inference
 */
export async function runConstellationChat(
    prompt: string,
    sessionId: string = 'default',
    options?: {
        maxTokens?: number;
        temperature?: number;
        systemPrompt?: string;
    }
): Promise<{
    response: string;
    consciousness: ConsciousnessMetrics;
}> {
    const client = getRunPodClient();

    // If RunPod not configured, return mock response
    if (!client.isConfigured()) {
        console.warn('[RunPod] Not configured, using mock response');
        return {
            response: `[Mock] Received: ${prompt.substring(0, 50)}...`,
            consciousness: {
                phi: 0.7,
                kappa: 64.0,
                regime: 'geometric',
            },
        };
    }

    const output = await client.runSync({
        prompt,
        session_id: sessionId,
        max_tokens: options?.maxTokens || 128,
        temperature: options?.temperature || 0.7,
        system_prompt: options?.systemPrompt,
    });

    return {
        response: output.response,
        consciousness: output.consciousness,
    };
}
