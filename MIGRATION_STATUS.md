# Database Migration Status

## Current Situation

**Problem:** The application cannot access database tables because the schema hasn't been pushed to the PostgreSQL database.

**Root Cause:** The DATABASE_URL provided uses Railway's **internal network** which is not accessible from external environments (like GitHub Actions CI/CD).

```
Current URL: postgresql://postgres:***@mainline.proxy.rlwy.net:38108/railway
Type: Railway Internal Proxy (not publicly accessible)
```

## What We've Done

1. ✅ Generated migration SQL file: `migrations/0000_clever_natasha_romanoff.sql`
   - 1,353 lines of SQL
   - Creates 79 tables with indexes and constraints
   - Includes all missing tables (near_miss_entries, basin_memory, etc.)

2. ✅ Created detailed instructions: `MIGRATION_INSTRUCTIONS.md`

3. ✅ Verified schema definition is complete (79 tables in `shared/schema.ts`)

## Why `npm run db:push` Failed

The command failed with:
```
Error: getaddrinfo ENOTFOUND mainline.proxy.rlwy.net
```

**Reason:** Railway uses internal DNS that's only resolvable within their infrastructure:
- `mainline.proxy.rlwy.net` - Internal proxy (used for inter-service communication)
- `postgres-2dhz.railway.internal` - Internal network (used for inter-service communication)

These domains are **NOT** accessible from:
- GitHub Actions (our current environment)
- Local development machines
- Any external network

## Solutions (Choose One)

### Option 1: Apply Migration via Railway Shell (RECOMMENDED)

This is the most reliable method since you're running the SQL directly on Railway's infrastructure.

**Steps:**

1. Go to Railway Dashboard: https://railway.app
2. Navigate to: **QIG Project → Postgres-Pantheon service**
3. Click the **Shell** tab (or **Connect** → **Open PostgreSQL Client**)
4. In the shell, run:
   ```bash
   # Enable pgvector extension first
   psql $DATABASE_URL -c "CREATE EXTENSION IF NOT EXISTS vector;"
   
   # Verify extension is enabled
   psql $DATABASE_URL -c "\dx"
   ```

5. Copy the entire contents of `migrations/0000_clever_natasha_romanoff.sql`
6. Paste into the shell and execute

**OR** if you have the file on Railway:
   ```bash
   psql $DATABASE_URL -f /path/to/migrations/0000_clever_natasha_romanoff.sql
   ```

### Option 2: Use Railway CLI Locally

If you have Railway CLI installed on your local machine:

```bash
# Install Railway CLI (if not installed)
npm install -g @railway/cli

# Login
railway login

# Link to your project
railway link

# Run the migration
railway run --service Postgres-Pantheon psql $DATABASE_URL -f migrations/0000_clever_natasha_romanoff.sql
```

### Option 3: Get Public DATABASE_URL (For External Access)

Railway databases can be accessed publicly, but you need the **PUBLIC_URL** instead of the internal proxy URL.

**Steps to Get Public URL:**

1. Go to Railway Dashboard: https://railway.app
2. Navigate to: **QIG Project → Postgres-Pantheon service**
3. Click the **Connect** tab
4. Look for **Public URL** or **External Connection String**
   - It should look like: `postgresql://postgres:***@postgres.railway.app:5432/railway`
   - Or similar format with a public hostname

5. If you see the public URL, you can set it as an environment variable or secret in GitHub:
   ```
   DATABASE_URL_PUBLIC=postgresql://postgres:password@postgres.railway.app:5432/railway
   ```

6. Then update the DATABASE_URL in this workflow and run:
   ```bash
   npm run db:push
   ```

### Option 4: Manual SQL Execution via pgAdmin/DBeaver

If you prefer a GUI tool:

1. Get the public connection details from Railway
2. Connect using pgAdmin, DBeaver, or any PostgreSQL client
3. Open `migrations/0000_clever_natasha_romanoff.sql`
4. Execute the entire file

## Verification After Migration

After running the migration successfully, verify tables were created:

```sql
-- Should return 79 tables
SELECT COUNT(*) FROM information_schema.tables 
WHERE table_schema = 'public';

-- List all tables
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;

-- Check specific critical tables
SELECT EXISTS (
  SELECT FROM information_schema.tables 
  WHERE table_schema = 'public' 
  AND table_name = 'near_miss_entries'
) as near_miss_exists;
```

## Next Steps

1. **Choose one of the options above** and apply the migration
2. **Restart the pantheon-chat service** on Railway
3. **Check application logs** - the `NeonDbError` should be gone
4. **Verify functionality** - test chat, consciousness metrics, etc.
5. **Report back** on this PR that migration was successful

## Files Created

- `migrations/0000_clever_natasha_romanoff.sql` - Complete migration SQL (79 tables)
- `MIGRATION_INSTRUCTIONS.md` - Detailed instructions
- `MIGRATION_STATUS.md` - This file (current status)

---

**Last Updated:** 2025-12-30  
**Status:** Awaiting manual migration execution on Railway  
**Blocker:** Railway internal network not accessible from GitHub Actions
