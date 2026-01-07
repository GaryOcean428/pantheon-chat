-- Migration: Add remote_api_key column to federated_instances
-- This column stores the encrypted API key for authenticating with remote federation nodes
-- Encryption format: iv:authTag:ciphertext (AES-256-GCM)

-- Add the missing remote_api_key column
ALTER TABLE federated_instances
ADD COLUMN IF NOT EXISTS remote_api_key TEXT;

-- Add index for faster lookups by status
CREATE INDEX IF NOT EXISTS idx_federated_instances_status
ON federated_instances(status);

-- Add index for endpoint uniqueness enforcement if not exists
CREATE UNIQUE INDEX IF NOT EXISTS idx_federated_instances_endpoint_unique
ON federated_instances(endpoint);
