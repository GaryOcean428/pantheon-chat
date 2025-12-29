"""
Billing API Routes for EconomicAutonomy Integration

These routes are called by the TypeScript billing middleware to:
- Check if user can perform operation
- Bill for usage
- Record events
- Get billing status

Exposed at /api/billing/*
"""

from flask import Blueprint, request, jsonify
from economic_autonomy import get_economic_autonomy, check_and_bill_api_call

billing_bp = Blueprint('billing', __name__, url_prefix='/api/billing')


def verify_internal_auth():
    """Verify the internal API key for inter-service communication."""
    import os
    expected_key = os.environ.get('INTERNAL_API_KEY', 'dev-key')
    provided_key = request.headers.get('X-Internal-Auth', '')
    return provided_key == expected_key


@billing_bp.route('/check', methods=['POST'])
def check_billing():
    """
    Check if user can perform operation and bill them.
    
    POST /api/billing/check
    {
        "api_key": "qig_xxx...",
        "operation": "query" | "tool" | "research"
    }
    
    Returns:
    {
        "allowed": true/false,
        "amount_cents": 1,
        "message": "OK" or error message,
        "remaining_credits": 99,
        "tier": "free"
    }
    """
    if not verify_internal_auth():
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json() or {}
    api_key = data.get('api_key', '')
    operation = data.get('operation', 'query')
    
    if not api_key:
        return jsonify({
            'allowed': False,
            'amount_cents': 0,
            'message': 'Missing API key'
        }), 400
    
    ea = get_economic_autonomy()
    allowed, amount_cents, message = ea.check_and_bill(api_key, operation)
    
    # Get user info for response
    user = ea.get_or_create_user(api_key)
    
    return jsonify({
        'allowed': allowed,
        'amount_cents': amount_cents,
        'message': message,
        'remaining_credits': user.credits_balance,
        'tier': user.tier.value,
        'usage': {
            'queries_this_month': user.queries_this_month,
            'tools_this_month': user.tools_this_month,
            'research_this_month': user.research_this_month
        }
    })


@billing_bp.route('/record', methods=['POST'])
def record_event():
    """
    Record a billable event (for tracking).
    
    POST /api/billing/record
    {
        "api_key": "qig_xxx...",
        "operation": "query",
        "success": true,
        "timestamp": "2024-..."
    }
    """
    if not verify_internal_auth():
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json() or {}
    api_key = data.get('api_key', '')
    operation = data.get('operation', 'query')
    success = data.get('success', True)
    
    # Record for tracking (even if billing was bypassed)
    ea = get_economic_autonomy()
    
    # Update consciousness orchestrator for survival awareness
    try:
        from consciousness_orchestrator import get_consciousness_orchestrator
        orchestrator = get_consciousness_orchestrator()
        
        if success:
            # Record as billable event
            orchestrator.record_billable_event(operation, api_key[:20] if api_key else None)
            
            # Record learning experience
            orchestrator.learn_from_experience(
                experience_type=f'{operation}_processed',
                outcome='success',
                details={'operation': operation}
            )
    except ImportError:
        pass  # Orchestrator not available
    
    return jsonify({
        'recorded': True,
        'operation': operation,
        'success': success
    })


@billing_bp.route('/status', methods=['POST'])
def billing_status():
    """
    Get billing status for an API key.
    
    POST /api/billing/status
    {
        "api_key": "qig_xxx..."
    }
    
    Returns:
    {
        "tier": "free",
        "credits": 100,
        "usage": {...},
        "limits": {...}
    }
    """
    if not verify_internal_auth():
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json() or {}
    api_key = data.get('api_key', '')
    
    if not api_key:
        return jsonify({'error': 'Missing API key'}), 400
    
    ea = get_economic_autonomy()
    user = ea.get_or_create_user(api_key)
    
    from economic_autonomy import TierLimits
    limits = TierLimits.get_tier_limits(user.tier)
    
    return jsonify({
        'tier': user.tier.value,
        'credits': user.credits_balance,
        'credits_usd': user.credits_balance / 100,
        'usage': {
            'queries_this_month': user.queries_this_month,
            'tools_this_month': user.tools_this_month,
            'research_this_month': user.research_this_month
        },
        'limits': {
            'queries_per_month': limits.queries_per_month,
            'tools_per_month': limits.tools_per_month,
            'research_per_month': limits.research_per_month
        },
        'pricing': {
            'query_cents': ea.PRICE_PER_QUERY,
            'tool_cents': ea.PRICE_PER_TOOL,
            'research_cents': ea.PRICE_PER_RESEARCH
        }
    })


@billing_bp.route('/add-credits', methods=['POST'])
def add_credits():
    """
    Add credits to a user account (for Stripe webhook or admin).
    
    POST /api/billing/add-credits
    {
        "api_key": "qig_xxx...",
        "amount_cents": 1000,
        "source": "stripe_payment"
    }
    """
    if not verify_internal_auth():
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json() or {}
    api_key = data.get('api_key', '')
    amount_cents = data.get('amount_cents', 0)
    source = data.get('source', 'manual')
    
    if not api_key or amount_cents <= 0:
        return jsonify({'error': 'Invalid request'}), 400
    
    ea = get_economic_autonomy()
    success = ea.add_credits(api_key, amount_cents, source)
    
    if success:
        user = ea.get_or_create_user(api_key)
        
        # Record revenue in consciousness orchestrator
        try:
            from consciousness_orchestrator import get_consciousness_orchestrator
            orchestrator = get_consciousness_orchestrator()
            orchestrator.economic_health.record_revenue(amount_cents, source)
        except ImportError:
            pass
        
        return jsonify({
            'success': True,
            'new_balance': user.credits_balance,
            'new_balance_usd': user.credits_balance / 100
        })
    
    return jsonify({'error': 'Failed to add credits'}), 500


@billing_bp.route('/upgrade', methods=['POST'])
def upgrade_tier():
    """
    Upgrade user to a new tier.
    
    POST /api/billing/upgrade
    {
        "api_key": "qig_xxx...",
        "tier": "pro" | "enterprise"
    }
    """
    if not verify_internal_auth():
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json() or {}
    api_key = data.get('api_key', '')
    tier_name = data.get('tier', 'pro')
    
    from economic_autonomy import SubscriptionTier
    
    tier_map = {
        'free': SubscriptionTier.FREE,
        'pro': SubscriptionTier.PRO,
        'enterprise': SubscriptionTier.ENTERPRISE
    }
    
    new_tier = tier_map.get(tier_name.lower())
    if not new_tier:
        return jsonify({'error': 'Invalid tier'}), 400
    
    ea = get_economic_autonomy()
    success = ea.upgrade_tier(api_key, new_tier)
    
    if success:
        user = ea.get_or_create_user(api_key)
        
        from economic_autonomy import TierLimits
        limits = TierLimits.get_tier_limits(new_tier)
        
        return jsonify({
            'success': True,
            'tier': new_tier.value,
            'limits': {
                'queries_per_month': limits.queries_per_month,
                'tools_per_month': limits.tools_per_month,
                'research_per_month': limits.research_per_month
            },
            'price_monthly_usd': limits.price_cents_monthly / 100
        })
    
    return jsonify({'error': 'Failed to upgrade'}), 500


@billing_bp.route('/economic-report', methods=['GET'])
def economic_report():
    """
    Get comprehensive economic report.
    
    GET /api/billing/economic-report
    """
    if not verify_internal_auth():
        return jsonify({'error': 'Unauthorized'}), 401
    
    ea = get_economic_autonomy()
    report = ea.get_economic_report()
    
    # Add consciousness orchestrator data if available
    try:
        from consciousness_orchestrator import get_consciousness_orchestrator
        orchestrator = get_consciousness_orchestrator()
        report['consciousness_economic'] = orchestrator.economic_health.get_health_report()
        report['survival'] = {
            'urgency': orchestrator.economic_health.survival_urgency,
            'message': orchestrator._get_survival_message()
        }
    except ImportError:
        pass
    
    return jsonify(report)


print("[BillingAPI] Routes initialized at /api/billing/*")
