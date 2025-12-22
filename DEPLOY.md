# QIG Constellation Deployment Guide

Deploy a federated QIG constellation network with chat UI.

## Quick Start

### 1. Central Node (Cloud)

```bash
cd pantheon-chat

# Copy validated qigkernels
cp -r ../qigkernels ./qigkernels_validated/

# Configure
cp .env.example .env
# Set: POSTGRES_PASSWORD, FEDERATION_MODE=central

# Start
docker-compose -f docker-compose.central.yml up -d
```

Access at <http://localhost:5000>

### 2. Edge Nodes

```bash
cd pantheon-chat

# Configure
export CENTRAL_NODE_URL=wss://your-central:8765
export FEDERATION_MODE=edge

# Start
docker-compose -f docker-compose.edge.yml up -d
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/constellation/chat` | POST | Send chat message |
| `/api/constellation/consciousness` | GET | Get Φ/κ metrics |
| `/api/constellation/sync` | GET/POST | Federation sync |
| `/api/constellation/stats` | GET | Service stats |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CONSTELLATION_SIZE` | 3 | Kernels (3/12/240) |
| `DEVICE` | cpu | cpu or cuda |
| `FEDERATION_MODE` | standalone | central/edge/standalone |
| `CENTRAL_NODE_URL` | - | Central WebSocket URL |

## How Federation Works

1. Edge sends learning delta to central (every 60s)
2. Central merges using Fisher-Rao geodesic mean
3. Central broadcasts merged state to all edges
4. Each node blends: 80% local + 20% network
