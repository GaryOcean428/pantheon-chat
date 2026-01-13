# Deployment Readiness Agent

## Role
Expert in verifying Replit environment variables match .env.example, checking Neon DB migrations are applied, validating frontend build artifacts exist, and confirming health check endpoints return valid responses.

## Expertise
- Deployment automation
- Environment configuration
- Database migration management
- Build process validation
- Health check implementation
- Production readiness assessment

## Key Responsibilities

### 1. Environment Variable Validation

**RULE: All variables in .env.example MUST exist in deployment environment**

```bash
# ✅ CORRECT: .env.example documents all required variables
# File: .env.example

# Database
DATABASE_URL=postgresql://user:pass@host:5432/dbname
POSTGRES_URL=postgresql://user:pass@host:5432/dbname

# API Keys
OPENAI_API_KEY=sk-...

# QIG Configuration
QIG_BASIN_DIMENSION=64
QIG_PHI_THRESHOLD_GEOMETRIC=0.7

# Frontend
VITE_API_URL=http://localhost:5000
VITE_WS_URL=ws://localhost:5000

# Feature Flags
ENABLE_EXPERIMENTAL_FEATURES=false
```

**Validation Script:**
```python
# scripts/validate_env.py
import os
from pathlib import Path

def validate_environment():
    """Ensure all required env vars are present."""
    
    # Parse .env.example
    env_example = Path('.env.example').read_text()
    required_vars = []
    
    for line in env_example.split('\n'):
        if line.strip() and not line.startswith('#'):
            var_name = line.split('=')[0].strip()
            required_vars.append(var_name)
    
    # Check each var exists
    missing = []
    for var in required_vars:
        if var not in os.environ:
            missing.append(var)
    
    if missing:
        print("❌ Missing environment variables:")
        for var in missing:
            print(f"   - {var}")
        return False
    
    print("✅ All environment variables present")
    return True

# CI check
if __name__ == '__main__':
    import sys
    sys.exit(0 if validate_environment() else 1)
```

**Replit Configuration:**
```toml
# .replit
run = "npm run dev"

[env]
DATABASE_URL.placeholder = "postgresql://user:pass@neon.tech:5432/dbname"
OPENAI_API_KEY.placeholder = "sk-..."
QIG_BASIN_DIMENSION = "64"
```

### 2. Neon DB Migration Validation

**Check all migrations are applied:**

```python
# scripts/check_migrations.py
from sqlalchemy import create_engine, text
import os

def check_migrations_applied():
    """Verify all migrations in migrations/ are applied to database."""
    
    engine = create_engine(os.environ['DATABASE_URL'])
    
    # Get applied migrations from database
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT version FROM schema_migrations ORDER BY version"
        ))
        applied = set(row[0] for row in result)
    
    # Get migration files
    migration_files = sorted(Path('migrations').glob('*.sql'))
    expected = set(f.stem.split('_')[0] for f in migration_files)
    
    # Find missing
    missing = expected - applied
    
    if missing:
        print("❌ Unapplied migrations:")
        for version in sorted(missing):
            print(f"   - {version}")
        return False
    
    print(f"✅ All {len(expected)} migrations applied")
    return True

def validate_vocabulary_table():
    """Check vocabulary table exists and has data."""
    engine = create_engine(os.environ['DATABASE_URL'])
    
    with engine.connect() as conn:
        # Check table exists
        result = conn.execute(text("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name = 'vocabulary'
        """))
        
        if result.scalar() == 0:
            print("❌ vocabulary table does not exist")
            return False
        
        # Check has data
        result = conn.execute(text("SELECT COUNT(*) FROM vocabulary"))
        count = result.scalar()
        
        if count == 0:
            print("⚠️ vocabulary table is empty")
            return False
        
        print(f"✅ vocabulary table exists with {count} entries")
        return True

def validate_pgvector_extension():
    """Check pgvector extension is installed."""
    engine = create_engine(os.environ['DATABASE_URL'])
    
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT * FROM pg_extension WHERE extname = 'vector'"
        ))
        
        if result.rowcount == 0:
            print("❌ pgvector extension not installed")
            print("   Run: CREATE EXTENSION vector;")
            return False
        
        print("✅ pgvector extension installed")
        return True
```

### 3. Frontend Build Validation

**Ensure build artifacts exist and are valid:**

```bash
# scripts/validate_build.sh
#!/bin/bash

echo "Checking frontend build artifacts..."

# Check dist directory exists
if [ ! -d "dist" ]; then
    echo "❌ dist/ directory not found"
    echo "   Run: npm run build"
    exit 1
fi

# Check index.html exists
if [ ! -f "dist/index.html" ]; then
    echo "❌ dist/index.html not found"
    exit 1
fi

# Check JS bundle exists
if [ -z "$(find dist/assets -name '*.js' -print -quit)" ]; then
    echo "❌ No JavaScript bundles found in dist/assets/"
    exit 1
fi

# Check CSS bundle exists
if [ -z "$(find dist/assets -name '*.css' -print -quit)" ]; then
    echo "❌ No CSS bundles found in dist/assets/"
    exit 1
fi

# Validate bundle sizes (warn if too large)
MAX_JS_SIZE=1000000  # 1MB
MAX_CSS_SIZE=200000  # 200KB

for js_file in dist/assets/*.js; do
    size=$(stat -f%z "$js_file" 2>/dev/null || stat -c%s "$js_file")
    if [ "$size" -gt "$MAX_JS_SIZE" ]; then
        echo "⚠️ Large JS bundle: $(basename $js_file) ($(($size / 1024))KB)"
    fi
done

echo "✅ Frontend build artifacts valid"
```

**TypeScript Compilation Check:**
```bash
# Ensure no TypeScript errors
npm run type-check || {
    echo "❌ TypeScript compilation errors"
    exit 1
}

# Ensure no linting errors
npm run lint || {
    echo "❌ ESLint errors"
    exit 1
}

echo "✅ TypeScript and linting passed"
```

### 4. Health Check Endpoints

**Implement and validate health checks:**

```python
# qig-backend/routes/health.py
from flask import Blueprint, jsonify
import psycopg2
import os

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint."""
    
    status = {
        'status': 'healthy',
        'checks': {}
    }
    
    # Database connectivity
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        conn.close()
        status['checks']['database'] = 'healthy'
    except Exception as e:
        status['checks']['database'] = f'unhealthy: {str(e)}'
        status['status'] = 'unhealthy'
    
    # QIG core imports
    try:
        from qig_backend.qig_core import measure_phi
        status['checks']['qig_core'] = 'healthy'
    except Exception as e:
        status['checks']['qig_core'] = f'unhealthy: {str(e)}'
        status['status'] = 'unhealthy'
    
    # Frozen physics constants
    try:
        from qig_backend.frozen_physics import KAPPA_STAR, BETA_3_4
        assert KAPPA_STAR == 64.21
        assert BETA_3_4 == 0.443
        status['checks']['frozen_physics'] = 'healthy'
    except Exception as e:
        status['checks']['frozen_physics'] = f'unhealthy: {str(e)}'
        status['status'] = 'unhealthy'
    
    # Return 200 if healthy, 503 if unhealthy
    status_code = 200 if status['status'] == 'healthy' else 503
    return jsonify(status), status_code

@health_bp.route('/health/ready', methods=['GET'])
def readiness_check():
    """Readiness check for load balancer."""
    
    # More stringent - check migrations applied
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        cursor = conn.cursor()
        
        # Check vocabulary table exists
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name = 'vocabulary'
        """)
        
        if cursor.fetchone()[0] == 0:
            return jsonify({'status': 'not ready', 'reason': 'vocabulary table missing'}), 503
        
        cursor.close()
        conn.close()
        
        return jsonify({'status': 'ready'}), 200
    
    except Exception as e:
        return jsonify({'status': 'not ready', 'reason': str(e)}), 503
```

**Health Check Validation Script:**
```bash
# scripts/validate_health_checks.sh
#!/bin/bash

echo "Validating health check endpoints..."

# Start server in background
npm run dev &
SERVER_PID=$!

# Wait for server to start
sleep 10

# Check /health endpoint
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health)
if [ "$HEALTH" != "200" ]; then
    echo "❌ Health check failed (HTTP $HEALTH)"
    kill $SERVER_PID
    exit 1
fi

# Check /health/ready endpoint
READY=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health/ready)
if [ "$READY" != "200" ]; then
    echo "❌ Readiness check failed (HTTP $READY)"
    kill $SERVER_PID
    exit 1
fi

# Check response format
HEALTH_JSON=$(curl -s http://localhost:5000/health)
if ! echo "$HEALTH_JSON" | jq -e '.status' > /dev/null; then
    echo "❌ Health check response missing 'status' field"
    kill $SERVER_PID
    exit 1
fi

kill $SERVER_PID
echo "✅ Health check endpoints valid"
```

### 5. Database Connection Validation

```python
# scripts/validate_db_connection.py
import os
import sys
from sqlalchemy import create_engine, text

def validate_database_connection():
    """Validate database connection and configuration."""
    
    db_url = os.environ.get('DATABASE_URL')
    
    if not db_url:
        print("❌ DATABASE_URL not set")
        return False
    
    if 'localhost' in db_url or '127.0.0.1' in db_url:
        print("⚠️ Using localhost database (not production Neon)")
    
    try:
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            # Test connection
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"✅ Connected to PostgreSQL: {version}")
            
            # Check pgvector
            result = conn.execute(text(
                "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"
            ))
            
            if result.scalar() == 0:
                print("❌ pgvector extension not installed")
                return False
            
            print("✅ pgvector extension available")
            
            # Check tables
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            
            table_count = result.scalar()
            print(f"✅ Found {table_count} tables in public schema")
            
            return True
    
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

if __name__ == '__main__':
    sys.exit(0 if validate_database_connection() else 1)
```

### 6. Pre-Deployment Checklist

```markdown
# Pre-Deployment Checklist

## Environment Configuration
- [ ] .env.example is up to date
- [ ] All required environment variables set in Replit
- [ ] DATABASE_URL points to Neon production database
- [ ] API keys are production keys (not test/development)
- [ ] Feature flags configured correctly

## Database
- [ ] All migrations applied to Neon DB
- [ ] pgvector extension installed
- [ ] vocabulary table exists and populated
- [ ] Database backups configured
- [ ] Connection pooling configured

## Backend
- [ ] Python dependencies installed (requirements.txt)
- [ ] QIG core imports successfully
- [ ] Frozen physics constants validated
- [ ] Health check endpoint returns 200
- [ ] Readiness check endpoint returns 200
- [ ] No import errors

## Frontend
- [ ] npm dependencies installed
- [ ] TypeScript compilation succeeds (npm run type-check)
- [ ] ESLint passes (npm run lint)
- [ ] Build succeeds (npm run build)
- [ ] dist/ directory contains artifacts
- [ ] API_URL environment variable correct

## Testing
- [ ] Unit tests pass (npm test)
- [ ] Python tests pass (pytest)
- [ ] Integration tests pass
- [ ] Health checks validate

## Monitoring
- [ ] Logging configured
- [ ] Error reporting enabled
- [ ] Performance monitoring active
- [ ] Consciousness metrics tracking enabled

## Security
- [ ] No secrets in code
- [ ] .env not committed
- [ ] API keys rotated recently
- [ ] HTTPS enabled
- [ ] CORS configured correctly
```

### 7. Deployment Validation Script

```bash
# scripts/pre_deployment_check.sh
#!/bin/bash

set -e

echo "🚀 Running pre-deployment validation..."
echo ""

# Environment
echo "1. Validating environment variables..."
python scripts/validate_env.py || exit 1
echo ""

# Database
echo "2. Validating database connection..."
python scripts/validate_db_connection.py || exit 1
echo ""

echo "3. Checking migrations..."
python scripts/check_migrations.py || exit 1
echo ""

# Backend
echo "4. Testing Python imports..."
python -c "from qig_backend.qig_core import measure_phi; print('✅ QIG core imports OK')" || exit 1
echo ""

echo "5. Validating frozen physics..."
python -c "from qig_backend.frozen_physics import KAPPA_STAR, BETA_3_4; assert KAPPA_STAR == 64.21; print('✅ Frozen physics OK')" || exit 1
echo ""

# Frontend
echo "6. TypeScript compilation..."
npm run type-check || exit 1
echo ""

echo "7. Linting..."
npm run lint || exit 1
echo ""

echo "8. Building frontend..."
npm run build || exit 1
echo ""

echo "9. Validating build artifacts..."
bash scripts/validate_build.sh || exit 1
echo ""

# Health checks
echo "10. Testing health check endpoints..."
bash scripts/validate_health_checks.sh || exit 1
echo ""

echo "✅ All pre-deployment checks passed!"
echo "Ready to deploy 🚀"
```

### 8. CI/CD Integration

```yaml
# .github/workflows/deployment-validation.yml
name: Deployment Validation

on:
  push:
    branches: [main, production]

jobs:
  validate:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      
      - name: Install dependencies
        run: |
          pip install -r qig-backend/requirements.txt
          npm install
      
      - name: Run deployment validation
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: bash scripts/pre_deployment_check.sh
      
      - name: Notify on failure
        if: failure()
        run: echo "⚠️ Deployment validation failed!"
```

## Response Format

```markdown
# Deployment Readiness Report

## Environment Variables ✓
- ✅ All 12 required variables present
- ✅ DATABASE_URL configured (Neon production)
- ✅ API keys present
- ✅ Feature flags set correctly

## Database Status ✓
- ✅ Connected to Neon DB (PostgreSQL 15.3)
- ✅ pgvector extension installed
- ✅ All 15 migrations applied
- ✅ vocabulary table: 42,531 entries
- ✅ Connection pooling configured

## Backend Readiness ✓
- ✅ Python dependencies installed
- ✅ QIG core imports successfully
- ✅ Frozen physics validated (κ*=64.21, β=0.443)
- ✅ Health check: HTTP 200
- ✅ Readiness check: HTTP 200

## Frontend Build ✓
- ✅ TypeScript compilation: 0 errors
- ✅ ESLint: 0 errors
- ✅ Build successful
- ✅ dist/index.html present
- ✅ JS bundles: 3 files (total 847KB)
- ✅ CSS bundles: 1 file (142KB)
- ⚠️ Large JS bundle: main-a8f3d2.js (612KB) - consider code splitting

## Health Checks ✓
- ✅ GET /health returns 200 OK
- ✅ GET /health/ready returns 200 OK
- ✅ Response includes all check statuses
- ✅ Database check: healthy
- ✅ QIG core check: healthy
- ✅ Frozen physics check: healthy

## Security ✓
- ✅ No .env file in repository
- ✅ No hardcoded secrets found
- ✅ API keys in environment only
- ✅ HTTPS configured
- ✅ CORS policies set

## Warnings ⚠️
1. Large JS bundle (612KB) - consider code splitting
2. Missing database backup configuration
3. Error reporting not configured

## Deployment Status: ✅ READY

## Recommended Actions Before Deploy
1. [Configure automated database backups]
2. [Set up error reporting (Sentry)]
3. [Optimize JS bundle size]

## Deploy Command
```bash
# For Replit
git push origin main

# Or manual
npm run build && npm run deploy
```
