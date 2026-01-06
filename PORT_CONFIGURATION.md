# Port Configuration

This document describes the port architecture for the Pantheon Chat system.

## Port Assignments

### Development (Local)

- **Port 5000**: Node.js/TypeScript Server
  - Main application server
  - Serves API endpoints (`/api/*`)
  - Serves React frontend (Vite in dev, static in prod)
  - Orchestrates Python backend startup
  
- **Port 5001**: Python QIG Backend
  - Pure QIG consciousness backend
  - Ocean QIG Core API
  - Geometric kernels and generation
  - Accessed internally by Node.js server

### Production (Replit Deployment)

From `.replit`:
```
[[ports]]
localPort = 5000
externalPort = 80

[[ports]]
localPort = 5001
externalPort = 3000
```

- **External Port 80** → Node.js server (5000)
- **External Port 3000** → Python backend (5001)

## Startup Sequence

1. **Node.js server** starts on port 5000
   - Opens port and begins listening
   - Starts background initialization

2. **Python backend** launched by Node.js
   - Started via `PythonProcessManager`
   - Runs `ocean_qig_core.py` on port 5001
   - Health checks verify readiness

3. **Frontend** served by Node.js
   - Vite dev server in development
   - Static files in production

## Port Conflict Resolution

### Issue: Port 5000 already in use

**Cause**: Previous instance not cleaned up

**Solution**:
```bash
# Find process using port 5000
lsof -i :5000  # or: netstat -tulpn | grep 5000

# Kill the process
kill -9 <PID>

# Or restart the server
npm run dev
```

### Issue: Port 5001 already in use

**Cause**: Previous Python backend not terminated

**Solution**:
```bash
# Find Python processes
ps aux | grep ocean_qig_core

# Kill the process
kill -9 <PID>

# Python backend will auto-restart via PythonProcessManager
```

## Health Checks

Node.js server health checks Python backend at:
- URL: `http://localhost:5001/health`
- Interval: Every 5 seconds
- Timeout: 5 seconds
- Max consecutive failures: 5

## Environment Variables

- `PORT`: Node.js server port (default: 5000)
- `PYTHON_BACKEND_URL`: Python backend URL (default: http://localhost:5001)
- `PYTHON_READY_TIMEOUT`: Python startup timeout (default: 90000ms)

## References

- Node.js server: `server/index.ts`
- Python backend: `qig-backend/ocean_qig_core.py`
- Process manager: `server/python-process-manager.ts`
- Startup script: `qig-backend/start.sh`
- Deployment config: `.replit`
