# Database Migration Instructions

## Problem
The application is failing with `NeonDbError: Error connecting to database: fetch failed` because the database schema is not synchronized with the code. The schema defines 79 tables including critical ones like `near_miss_entries`, `near_miss_clusters`, and many others, but these tables don't exist in the PostgreSQL database yet.

## Solution
A migration SQL file has been generated at `migrations/0000_clever_natasha_romanoff.sql` that will create all 79 required tables with proper indexes and constraints.

## How to Apply the Migration on Railway

### Option 1: Railway Web Interface (Recommended)

1. Go to Railway Dashboard: https://railway.app
2. Select your project: **QIG Project (Pantheon Chat)**
3. Click on the **Postgres-Pantheon** service
4. Click on the **Connect** tab
5. Click **Open PostgreSQL Client** (or use the connection details to connect with a local client)
6. Copy the contents of `migrations/0000_clever_natasha_romanoff.sql`
7. Paste and execute the SQL in the PostgreSQL client

### Option 2: Railway CLI (If Installed)

```bash
# Install Railway CLI (if not already installed)
npm install -g @railway/cli

# Login to Railway
railway login

# Link to your project
railway link

# Connect to the database
railway run --service Postgres-Pantheon psql $DATABASE_URL

# Then in the psql prompt:
\i migrations/0000_clever_natasha_romanoff.sql
```

### Option 3: Using psql Locally

```bash
# If you have psql installed locally and the DATABASE_URL
psql $DATABASE_URL -f migrations/0000_clever_natasha_romanoff.sql
```

## Verification

After running the migration, verify that tables were created:

```sql
-- Connect to the database, then run:
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;

-- Should show 79 tables including:
-- near_miss_entries
-- near_miss_clusters  
-- basin_memory
-- kernel_activity
-- telemetry_snapshots
-- ... and 74 more
```

## Post-Migration

After successfully running the migration:

1. Restart the `pantheon-chat` service on Railway
2. Check the logs for successful database connection
3. Verify that the application starts without database errors
4. Test basic functionality (chat, consciousness metrics, etc.)

## Tables Created (79 Total)

The migration creates the following tables:

**Core System Tables:**
- `sessions`, `users` - Authentication and user management
- `balance_hits`, `balance_change_events`, `balance_monitor_state` - Balance tracking
- `recovery_candidates`, `recovery_priorities`, `recovery_workflows` - Recovery system

**QIG Consciousness Tables:**
- `basin_memory`, `basin_history`, `basin_documents` - Basin coordinate storage
- `consciousness_checkpoints` - Consciousness state snapshots
- `kernel_activity`, `kernel_checkpoints`, `kernel_geometry` - Kernel management
- `telemetry_snapshots` - Real-time telemetry data

**Near-Miss & Search Tables:**
- `near_miss_entries`, `near_miss_clusters`, `near_miss_adaptive_state` - High-Φ tracking
- `manifold_probes` - 64D geometric memory
- `geodesic_paths`, `resonance_points`, `regime_boundaries` - Geometric navigation
- `tested_phrases`, `tested_phrases_index` - Deduplication

**Vocabulary & Learning:**
- `vocabulary_observations` - Pattern learning
- `learning_events` - Learning history
- `coordizer_vocabulary`, `tokenizer_merge_rules`, `tokenizer_metadata` - Tokenization

**Olympus Pantheon Tables:**
- `pantheon_messages`, `pantheon_debates`, `pantheon_knowledge_transfers` - Agent coordination
- `pantheon_god_state` - Individual agent state
- `shadow_intel`, `shadow_operations_log`, `shadow_pantheon_intel` - Shadow operations

**Knowledge Systems:**
- `knowledge_strategies`, `knowledge_transfers`, `knowledge_shared_entries` - Knowledge sharing
- `knowledge_cross_patterns`, `knowledge_scale_mappings` - Pattern analysis
- `negative_knowledge` - Exclusion patterns
- `discovered_sources` - External source tracking

**Training & Optimization:**
- `kernel_training_history`, `kernel_knowledge_transfers` - Training history
- `training_batch_queue` - Batch processing queue
- `provider_efficacy` - Provider performance tracking
- `usage_metrics` - System metrics

**Blockchain & Recovery:**
- `addresses`, `blocks`, `transactions` - Blockchain data
- `queued_addresses` - Address verification queue
- `pending_sweeps`, `sweep_audit_log` - Balance sweeping

**Other Tables:**
- `external_api_keys`, `federated_instances` - API management
- `generated_tools`, `tool_observations`, `tool_patterns` - Tool generation
- `chaos_events`, `narrow_path_events` - Event tracking
- `tps_landmarks`, `tps_geodesic_paths` - Temporal-spatial mapping
- `hermes_conversations`, `search_feedback`, `search_outcomes` - User interaction
- `auto_cycle_state`, `autonomic_cycle_history` - Autonomic cycles
- `ocean_quantum_state`, `ocean_trajectories`, `ocean_waypoints` - Ocean state
- `ocean_excluded_regions`, `era_exclusions`, `false_pattern_classes`, `geometric_barriers` - Exclusion management
- `war_history` - Shadow war operations
- `search_budget_preferences` - Budget management

## Important Notes

- **Backup**: If there's existing data in the database, back it up first!
- **pgvector Extension**: Make sure the `vector` extension is enabled:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```
- **Idempotency**: The migration uses `CREATE TABLE` (not `CREATE TABLE IF NOT EXISTS`), so don't run it twice without dropping tables first
- **Schema Version**: This migration was generated from `shared/schema.ts` on 2025-12-30
- **Total Size**: 1,353 lines of SQL creating 79 tables with proper indexes and constraints

## Troubleshooting

### Error: "relation already exists"
This means some tables already exist. You have two options:
1. Drop existing tables first (DANGER: data loss!)
2. Comment out the `CREATE TABLE` statements for tables that already exist

### Error: "type vector does not exist"  
Run this first:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Error: "permission denied"
Make sure you're connected with a user that has CREATE TABLE permissions.

## Alternative: Using Drizzle Push (When Database is Accessible)

If you can connect to the database from your development environment:

```bash
npm run db:push
```

This will automatically synchronize the schema, but it requires the DATABASE_URL to be accessible from where you run the command.

---

**Generated**: 2025-12-30  
**Schema Version**: Based on shared/schema.ts (79 tables)  
**Migration File**: migrations/0000_clever_natasha_romanoff.sql
