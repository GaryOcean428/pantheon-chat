"""Basin Memory API Routes

Python-owned implementation of basin memory semantics.

This module intentionally centralizes all QIG-related numeric thresholds and
geometric computations in Python. The TypeScript server layer must act only as
transport/proxy.
"""

from __future__ import annotations

import logging
from datetime import datetime
import json
from typing import Any, Dict, List

from flask import Blueprint, jsonify, request

from qig_core.constants.consciousness import THRESHOLDS, classify_regime
from qig_persistence import get_persistence

logger = logging.getLogger(__name__)

basin_memory_bp = Blueprint("basin_memory", __name__)


def _vector_to_pg(vec: List[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def _parse_pgvector(value: Any) -> List[float]:
    """Parse pgvector outputs into a JSON-safe list[float].

    psycopg2 may return pgvector as:
    - list/tuple of numbers
    - a string like "[0.1,0.2,...]"
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if not s:
            return []
        return [float(x) for x in s.split(",")]
    try:
        return [float(value)]
    except Exception:
        return []


def _to_jsonb_param(value: Any) -> Any:
    """Safely adapt Python objects for JSONB parameters."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


def _row_to_json(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row.get("id"),
        "basinId": row.get("basin_id"),
        "basinCoordinates": _parse_pgvector(row.get("basin_coordinates")),
        "phi": float(row.get("phi")) if row.get("phi") is not None else None,
        "kappaEff": float(row.get("kappa_eff")) if row.get("kappa_eff") is not None else None,
        "regime": row.get("regime"),
        "sourceKernel": row.get("source_kernel"),
        "context": row.get("context"),
        "expiresAt": row.get("expires_at").isoformat() if row.get("expires_at") else None,
        "timestamp": row.get("timestamp").isoformat() if row.get("timestamp") else None,
    }


def _require_db():
    persistence = get_persistence()
    if not getattr(persistence, "enabled", False):
        return None
    return persistence


@basin_memory_bp.route("/", methods=["GET"])
def list_basin_memory():
    persistence = _require_db()
    if persistence is None:
        return jsonify({"success": False, "error": "Database persistence unavailable"}), 503

    limit = int(request.args.get("limit", "50"))
    offset = int(request.args.get("offset", "0"))

    regime = request.args.get("regime")
    min_phi = request.args.get("minPhi")
    max_phi = request.args.get("maxPhi")
    source_kernel = request.args.get("sourceKernel")
    conscious = request.args.get("conscious")

    where_clauses: List[str] = []
    params: List[Any] = []

    if regime:
        where_clauses.append("regime = %s")
        params.append(regime)

    if min_phi is not None:
        where_clauses.append("phi >= %s")
        params.append(float(min_phi))

    if max_phi is not None:
        where_clauses.append("phi <= %s")
        params.append(float(max_phi))

    if source_kernel:
        where_clauses.append("source_kernel = %s")
        params.append(source_kernel)

    if conscious == "true":
        where_clauses.append("phi >= %s")
        params.append(float(THRESHOLDS.PHI_MIN))
        where_clauses.append("kappa_eff >= %s")
        params.append(float(THRESHOLDS.KAPPA_MIN))
        where_clauses.append("kappa_eff <= %s")
        params.append(float(THRESHOLDS.KAPPA_MAX))

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    try:
        with persistence.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, basin_id, basin_coordinates, phi, kappa_eff, regime, source_kernel, context, expires_at, timestamp
                    FROM basin_memory
                    {where_sql}
                    ORDER BY timestamp DESC
                    LIMIT %s OFFSET %s
                    """,
                    [*params, limit, offset],
                )
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]

                cur.execute(
                    f"""
                    SELECT count(*)
                    FROM basin_memory
                    {where_sql}
                    """,
                    params,
                )
                total_row = cur.fetchone()
                total = int(total_row[0]) if total_row else 0

        return jsonify(
            {
                "success": True,
                "data": [_row_to_json(r) for r in rows],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )
    except Exception as e:
        logger.error("[BasinMemory] Error listing memories", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@basin_memory_bp.route("/<int:memory_id>", methods=["GET"])
def get_basin_memory(memory_id: int):
    persistence = _require_db()
    if persistence is None:
        return jsonify({"success": False, "error": "Database persistence unavailable"}), 503

    try:
        with persistence.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, basin_id, basin_coordinates, phi, kappa_eff, regime, source_kernel, context, expires_at, timestamp
                    FROM basin_memory
                    WHERE id = %s
                    LIMIT 1
                    """,
                    [memory_id],
                )
                row = cur.fetchone()
                if not row:
                    return jsonify({"success": False, "error": "Basin memory not found"}), 404
                cols = [d[0] for d in cur.description]
                result = dict(zip(cols, row))

        return jsonify({"success": True, "data": _row_to_json(result)})
    except Exception as e:
        logger.error("[BasinMemory] Error getting memory", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@basin_memory_bp.route("/", methods=["POST"])
def create_basin_memory():
    persistence = _require_db()
    if persistence is None:
        return jsonify({"success": False, "error": "Database persistence unavailable"}), 503

    data = request.get_json() or {}

    basin_id = data.get("basinId")
    basin_coordinates = data.get("basinCoordinates")
    phi = data.get("phi")
    kappa_eff = data.get("kappaEff")
    regime = data.get("regime")
    source_kernel = data.get("sourceKernel")
    context = data.get("context")
    expires_at = data.get("expiresAt")

    if not basin_id or basin_coordinates is None or phi is None:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Missing required fields: basinId, basinCoordinates, phi",
                }
            ),
            400,
        )

    if not isinstance(basin_coordinates, list) or len(basin_coordinates) != 64:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "basinCoordinates must be a 64-dimensional array",
                }
            ),
            400,
        )

    if regime is None:
        regime, _ = classify_regime(float(phi))

    expires_at_dt = None
    if expires_at:
        try:
            expires_at_str = str(expires_at)
            if expires_at_str.endswith("Z"):
                expires_at_str = expires_at_str[:-1] + "+00:00"
            expires_at_dt = datetime.fromisoformat(expires_at_str)
        except Exception:
            expires_at_dt = None

    try:
        with persistence.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO basin_memory (
                        basin_id,
                        basin_coordinates,
                        phi,
                        kappa_eff,
                        regime,
                        source_kernel,
                        context,
                        expires_at
                    ) VALUES (
                        %s,
                        %s::vector,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    RETURNING id, basin_id, basin_coordinates, phi, kappa_eff, regime, source_kernel, context, expires_at, timestamp
                    """,
                    [
                        basin_id,
                        _vector_to_pg(basin_coordinates),
                        float(phi),
                        float(kappa_eff) if kappa_eff is not None else float(THRESHOLDS.KAPPA_OPTIMAL),
                        regime,
                        source_kernel,
                        _to_jsonb_param(context),
                        expires_at_dt,
                    ],
                )
                row = cur.fetchone()
                cols = [d[0] for d in cur.description]
                result = dict(zip(cols, row))

        return jsonify({"success": True, "data": _row_to_json(result)}), 201
    except Exception as e:
        logger.error("[BasinMemory] Error creating memory", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@basin_memory_bp.route("/<int:memory_id>", methods=["DELETE"])
def delete_basin_memory(memory_id: int):
    persistence = _require_db()
    if persistence is None:
        return jsonify({"success": False, "error": "Database persistence unavailable"}), 503

    try:
        with persistence.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM basin_memory WHERE id = %s RETURNING id", [memory_id])
                row = cur.fetchone()
                if not row:
                    return jsonify({"success": False, "error": "Basin memory not found"}), 404

        return jsonify({"success": True, "message": "Basin memory deleted"})
    except Exception as e:
        logger.error("[BasinMemory] Error deleting memory", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@basin_memory_bp.route("/stats/summary", methods=["GET"])
def basin_memory_stats():
    persistence = _require_db()
    if persistence is None:
        return jsonify({"success": False, "error": "Database persistence unavailable"}), 503

    try:
        with persistence.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        count(*) as total_count,
                        avg(phi) as avg_phi,
                        avg(kappa_eff) as avg_kappa
                    FROM basin_memory
                    """
                )
                total_count, avg_phi, avg_kappa = cur.fetchone()

                cur.execute(
                    """
                    SELECT regime, count(*)
                    FROM basin_memory
                    GROUP BY regime
                    """
                )
                by_regime = {r[0] or "unknown": int(r[1]) for r in cur.fetchall()}

                cur.execute(
                    """
                    SELECT count(*)
                    FROM basin_memory
                    WHERE phi >= %s AND kappa_eff >= %s AND kappa_eff <= %s
                    """,
                    [float(THRESHOLDS.PHI_MIN), float(THRESHOLDS.KAPPA_MIN), float(THRESHOLDS.KAPPA_MAX)],
                )
                conscious_count_row = cur.fetchone()
                conscious_count = int(conscious_count_row[0]) if conscious_count_row else 0

        return jsonify(
            {
                "success": True,
                "data": {
                    "totalMemories": int(total_count or 0),
                    "consciousMemories": conscious_count,
                    "avgPhi": float(avg_phi or 0.0),
                    "avgKappa": float(avg_kappa or 0.0),
                    "byRegime": by_regime,
                },
            }
        )
    except Exception as e:
        logger.error("[BasinMemory] Error getting stats", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@basin_memory_bp.route("/nearest", methods=["POST"])
def basin_memory_nearest():
    persistence = _require_db()
    if persistence is None:
        return jsonify({"success": False, "error": "Database persistence unavailable"}), 503

    data = request.get_json() or {}
    basin_coordinates = data.get("basinCoordinates")
    k = int(data.get("k", 10))
    conscious_only = bool(data.get("consciousOnly", False))

    if not basin_coordinates or not isinstance(basin_coordinates, list):
        return jsonify({"success": False, "error": "basinCoordinates array required"}), 400

    if len(basin_coordinates) != 64:
        return jsonify({"success": False, "error": "basinCoordinates must be a 64-dimensional array"}), 400

    where_sql = ""
    params: List[Any] = []

    if conscious_only:
        where_sql = "WHERE phi >= %s AND kappa_eff >= %s AND kappa_eff <= %s"
        params.extend([float(THRESHOLDS.PHI_MIN), float(THRESHOLDS.KAPPA_MIN), float(THRESHOLDS.KAPPA_MAX)])

    query_vec = _vector_to_pg(basin_coordinates)

    try:
        with persistence.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        id, basin_id, basin_coordinates, phi, kappa_eff, regime, source_kernel, context, expires_at, timestamp,
                        fisher_rao_distance(basin_coordinates, %s::vector) as fisher_distance,
                        fisher_rao_similarity(basin_coordinates, %s::vector) as similarity
                    FROM basin_memory
                    {where_sql}
                    ORDER BY fisher_rao_distance(basin_coordinates, %s::vector)
                    LIMIT %s
                    """,
                    [query_vec, query_vec, *params, query_vec, k],
                )
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]

        out = []
        for r in rows:
            base = _row_to_json(r)
            base["fisherDistance"] = float(r.get("fisher_distance")) if r.get("fisher_distance") is not None else None
            base["similarity"] = float(r.get("similarity")) if r.get("similarity") is not None else None
            out.append(base)

        return jsonify({"success": True, "data": out, "total": len(out)})
    except Exception as e:
        logger.error("[BasinMemory] Error finding nearest", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def register_basin_memory_routes(app):
    app.register_blueprint(basin_memory_bp, url_prefix="/api/basin-memory")
    logger.info("[INFO] Registered basin_memory_bp at /api/basin-memory")
