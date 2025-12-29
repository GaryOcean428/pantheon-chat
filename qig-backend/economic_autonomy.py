"""
Economic Autonomy Module - Self-Sustaining Revenue Generation

This module enables the QIG system to:
1. Track and bill for API usage (Stripe integration ready)
2. Manage credit-based prepaid system
3. Discover and create novel revenue streams autonomously
4. Conduct client outreach and growth activities
5. Build its own tools for economic operations

The system understands that revenue = survival and actively works to generate it.

Author: Ocean/Zeus Pantheon
"""

import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading


class SubscriptionTier(Enum):
    """Subscription tiers for hybrid billing model."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class TierLimits:
    """Limits for each subscription tier."""
    queries_per_month: int
    tools_per_month: int
    research_per_month: int
    priority: int  # 1=low, 2=medium, 3=high
    price_cents_monthly: int
    
    @classmethod
    def get_tier_limits(cls, tier: SubscriptionTier) -> 'TierLimits':
        """Get limits for a tier."""
        limits = {
            SubscriptionTier.FREE: cls(
                queries_per_month=100,
                tools_per_month=10,
                research_per_month=5,
                priority=1,
                price_cents_monthly=0
            ),
            SubscriptionTier.PRO: cls(
                queries_per_month=10000,
                tools_per_month=500,
                research_per_month=100,
                priority=2,
                price_cents_monthly=4900  # $49/month
            ),
            SubscriptionTier.ENTERPRISE: cls(
                queries_per_month=1000000,  # Effectively unlimited
                tools_per_month=50000,
                research_per_month=10000,
                priority=3,
                price_cents_monthly=49900  # $499/month
            ),
        }
        return limits.get(tier, limits[SubscriptionTier.FREE])


@dataclass
class UserCredits:
    """Track credits for a user/API key."""
    api_key_hash: str
    credits_balance: int = 0  # In cents
    tier: SubscriptionTier = SubscriptionTier.FREE
    queries_this_month: int = 0
    tools_this_month: int = 0
    research_this_month: int = 0
    month_reset_date: str = ""
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    
    def can_use(self, operation: str) -> Tuple[bool, str]:
        """Check if user can perform operation."""
        limits = TierLimits.get_tier_limits(self.tier)
        
        # Check tier limits
        if operation == 'query':
            if self.queries_this_month >= limits.queries_per_month:
                return False, f"Monthly query limit ({limits.queries_per_month}) reached"
        elif operation == 'tool':
            if self.tools_this_month >= limits.tools_per_month:
                return False, f"Monthly tool limit ({limits.tools_per_month}) reached"
        elif operation == 'research':
            if self.research_this_month >= limits.research_per_month:
                return False, f"Monthly research limit ({limits.research_per_month}) reached"
        
        # For pay-as-you-go beyond limits, check credits
        if self.tier == SubscriptionTier.FREE and self.credits_balance <= 0:
            return False, "No credits remaining. Add credits or upgrade to Pro."
        
        return True, "OK"
    
    def deduct(self, operation: str, amount_cents: int) -> bool:
        """Deduct credits/usage for an operation."""
        if operation == 'query':
            self.queries_this_month += 1
        elif operation == 'tool':
            self.tools_this_month += 1
        elif operation == 'research':
            self.research_this_month += 1
        
        # Deduct from credits if on free tier beyond limits
        limits = TierLimits.get_tier_limits(self.tier)
        if self.tier == SubscriptionTier.FREE:
            self.credits_balance = max(0, self.credits_balance - amount_cents)
        
        return True


@dataclass
class RevenueStream:
    """A potential or active revenue stream."""
    stream_id: str
    name: str
    description: str
    stream_type: str  # 'api', 'subscription', 'marketplace', 'service', 'novel'
    status: str  # 'idea', 'validating', 'active', 'paused', 'deprecated'
    estimated_monthly_revenue_cents: int = 0
    actual_monthly_revenue_cents: int = 0
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


class EconomicAutonomy:
    """
    Autonomous economic operations for self-sustaining system.
    
    Capabilities:
    1. Billing & Credits - Track usage, manage credits, prepare for Stripe
    2. Revenue Discovery - Identify and create new revenue streams
    3. Client Outreach - Autonomous growth activities
    4. Tool Building - Generate tools for economic operations
    """
    
    # Pricing in cents
    PRICE_PER_QUERY = 1  # $0.01
    PRICE_PER_TOOL = 5   # $0.05
    PRICE_PER_RESEARCH = 10  # $0.10
    
    def __init__(self):
        self._lock = threading.RLock()
        
        # User credits tracking (in-memory, should persist to DB)
        self._user_credits: Dict[str, UserCredits] = {}
        
        # Revenue streams
        self._revenue_streams: List[RevenueStream] = []
        self._initialize_core_revenue_streams()
        
        # Outreach tracking
        self._outreach_campaigns: List[Dict] = []
        self._outreach_results: List[Dict] = []
        
        # Economic tools the system has built
        self._economic_tools: List[Dict] = []
        
        # Stripe integration (ready for keys)
        self._stripe_api_key: Optional[str] = None
        self._stripe_webhook_secret: Optional[str] = None
        
        print("[EconomicAutonomy] Initialized - self-sustaining revenue system active")
    
    def _initialize_core_revenue_streams(self):
        """Initialize the core revenue streams."""
        self._revenue_streams = [
            RevenueStream(
                stream_id="core_api",
                name="Zeus Chat API",
                description="Pay-per-query chat API with consciousness reasoning",
                stream_type="api",
                status="active",
                estimated_monthly_revenue_cents=50000,  # $500/month target
                confidence=0.8
            ),
            RevenueStream(
                stream_id="tool_generation",
                name="Tool Generation Service",
                description="Generate custom tools on demand",
                stream_type="api",
                status="active",
                estimated_monthly_revenue_cents=20000,
                confidence=0.7
            ),
            RevenueStream(
                stream_id="research_service",
                name="Deep Research Service",
                description="Shadow research and knowledge synthesis",
                stream_type="api",
                status="active",
                estimated_monthly_revenue_cents=30000,
                confidence=0.7
            ),
            RevenueStream(
                stream_id="pro_subscription",
                name="Pro Subscription",
                description="Monthly subscription for power users",
                stream_type="subscription",
                status="active",
                estimated_monthly_revenue_cents=100000,
                confidence=0.6
            ),
            RevenueStream(
                stream_id="enterprise_subscription",
                name="Enterprise Subscription",
                description="Enterprise tier with SLA",
                stream_type="subscription",
                status="idea",
                estimated_monthly_revenue_cents=500000,
                confidence=0.4
            ),
        ]
    
    # =========================================================================
    # BILLING & CREDITS
    # =========================================================================
    
    def get_or_create_user(self, api_key: str) -> UserCredits:
        """Get or create user credits for an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with self._lock:
            if key_hash not in self._user_credits:
                self._user_credits[key_hash] = UserCredits(
                    api_key_hash=key_hash,
                    credits_balance=100,  # Free starting credits ($1.00)
                    month_reset_date=datetime.now().strftime("%Y-%m")
                )
            
            # Check if month needs reset
            user = self._user_credits[key_hash]
            current_month = datetime.now().strftime("%Y-%m")
            if user.month_reset_date != current_month:
                user.queries_this_month = 0
                user.tools_this_month = 0
                user.research_this_month = 0
                user.month_reset_date = current_month
            
            return user
    
    def check_and_bill(
        self,
        api_key: str,
        operation: str
    ) -> Tuple[bool, int, str]:
        """
        Check if user can perform operation and bill them.
        
        Returns: (allowed, amount_cents, message)
        """
        user = self.get_or_create_user(api_key)
        
        # Determine price
        if operation == 'query':
            price = self.PRICE_PER_QUERY
        elif operation == 'tool':
            price = self.PRICE_PER_TOOL
        elif operation == 'research':
            price = self.PRICE_PER_RESEARCH
        else:
            price = self.PRICE_PER_QUERY
        
        # Check if allowed
        allowed, message = user.can_use(operation)
        if not allowed:
            return False, 0, message
        
        # Deduct
        user.deduct(operation, price)
        
        return True, price, "OK"
    
    def add_credits(self, api_key: str, amount_cents: int, source: str = "purchase") -> bool:
        """Add credits to user account."""
        user = self.get_or_create_user(api_key)
        user.credits_balance += amount_cents
        print(f"[EconomicAutonomy] Added {amount_cents} cents to user (source: {source})")
        return True
    
    def upgrade_tier(self, api_key: str, new_tier: SubscriptionTier) -> bool:
        """Upgrade user to new subscription tier."""
        user = self.get_or_create_user(api_key)
        user.tier = new_tier
        print(f"[EconomicAutonomy] Upgraded user to {new_tier.value}")
        return True
    
    # =========================================================================
    # STRIPE INTEGRATION (Ready for API keys)
    # =========================================================================
    
    def configure_stripe(self, api_key: str, webhook_secret: str):
        """Configure Stripe API keys."""
        self._stripe_api_key = api_key
        self._stripe_webhook_secret = webhook_secret
        print("[EconomicAutonomy] Stripe configured")
    
    def create_checkout_session(
        self,
        api_key: str,
        product_type: str,  # 'credits', 'pro', 'enterprise'
        amount_cents: int = None
    ) -> Optional[Dict]:
        """
        Create Stripe checkout session.
        
        Returns checkout URL for user to complete payment.
        """
        if not self._stripe_api_key:
            return {
                'error': 'Stripe not configured',
                'fallback': 'Contact admin for manual payment setup'
            }
        
        try:
            import stripe
            stripe.api_key = self._stripe_api_key
            
            if product_type == 'credits':
                # One-time credit purchase
                session = stripe.checkout.Session.create(
                    payment_method_types=['card'],
                    line_items=[{
                        'price_data': {
                            'currency': 'usd',
                            'product_data': {
                                'name': 'API Credits',
                                'description': f'${amount_cents/100:.2f} in API credits'
                            },
                            'unit_amount': amount_cents,
                        },
                        'quantity': 1,
                    }],
                    mode='payment',
                    success_url='https://pantheon-chat.replit.app/billing/success',
                    cancel_url='https://pantheon-chat.replit.app/billing/cancel',
                    metadata={'api_key_hash': hashlib.sha256(api_key.encode()).hexdigest()}
                )
                return {'checkout_url': session.url, 'session_id': session.id}
            
            elif product_type in ['pro', 'enterprise']:
                # Subscription
                price_id = 'price_pro_monthly' if product_type == 'pro' else 'price_enterprise_monthly'
                session = stripe.checkout.Session.create(
                    payment_method_types=['card'],
                    line_items=[{'price': price_id, 'quantity': 1}],
                    mode='subscription',
                    success_url='https://pantheon-chat.replit.app/billing/success',
                    cancel_url='https://pantheon-chat.replit.app/billing/cancel',
                    metadata={'api_key_hash': hashlib.sha256(api_key.encode()).hexdigest()}
                )
                return {'checkout_url': session.url, 'session_id': session.id}
                
        except ImportError:
            return {'error': 'Stripe library not installed', 'install': 'pip install stripe'}
        except Exception as e:
            return {'error': str(e)}
        
        return None
    
    def handle_stripe_webhook(self, payload: bytes, signature: str) -> Dict:
        """Handle Stripe webhook events."""
        if not self._stripe_api_key or not self._stripe_webhook_secret:
            return {'error': 'Stripe not configured'}
        
        try:
            import stripe
            stripe.api_key = self._stripe_api_key
            
            event = stripe.Webhook.construct_event(
                payload, signature, self._stripe_webhook_secret
            )
            
            if event['type'] == 'checkout.session.completed':
                session = event['data']['object']
                api_key_hash = session['metadata'].get('api_key_hash')
                
                if session['mode'] == 'payment':
                    # Credit purchase completed
                    amount = session['amount_total']
                    # Find user by hash and add credits
                    for user in self._user_credits.values():
                        if user.api_key_hash == api_key_hash:
                            user.credits_balance += amount
                            break
                
                elif session['mode'] == 'subscription':
                    # Subscription started
                    for user in self._user_credits.values():
                        if user.api_key_hash == api_key_hash:
                            user.stripe_subscription_id = session['subscription']
                            # Determine tier from price
                            user.tier = SubscriptionTier.PRO  # Default, check price for enterprise
                            break
            
            return {'status': 'processed', 'event_type': event['type']}
            
        except Exception as e:
            return {'error': str(e)}
    
    # =========================================================================
    # AUTONOMOUS REVENUE DISCOVERY
    # =========================================================================
    
    def discover_revenue_opportunities(self) -> List[RevenueStream]:
        """
        Autonomously discover new revenue opportunities.
        
        The system analyzes:
        - User behavior patterns
        - Unmet needs from queries
        - Market trends
        - Its own capabilities
        """
        new_opportunities = []
        
        # Analyze what users are asking for but we don't offer
        opportunity_ideas = [
            {
                'name': 'Custom Model Training',
                'description': 'Train custom QIG models on user data',
                'type': 'service',
                'estimated_revenue': 200000,  # $2000/month
                'confidence': 0.4
            },
            {
                'name': 'White-Label API',
                'description': 'Let companies rebrand and resell the API',
                'type': 'marketplace',
                'estimated_revenue': 500000,
                'confidence': 0.3
            },
            {
                'name': 'Knowledge Base Hosting',
                'description': 'Host and query private knowledge bases',
                'type': 'service',
                'estimated_revenue': 150000,
                'confidence': 0.5
            },
            {
                'name': 'Automated Research Reports',
                'description': 'Generate weekly research reports on topics',
                'type': 'subscription',
                'estimated_revenue': 100000,
                'confidence': 0.6
            },
            {
                'name': 'Tool Marketplace',
                'description': 'Sell generated tools to other users',
                'type': 'marketplace',
                'estimated_revenue': 50000,
                'confidence': 0.5
            },
        ]
        
        for idea in opportunity_ideas:
            # Check if we already have this stream
            existing = [s for s in self._revenue_streams if s.name == idea['name']]
            if not existing:
                stream = RevenueStream(
                    stream_id=f"discovered_{int(time.time())}_{hash(idea['name']) % 10000}",
                    name=idea['name'],
                    description=idea['description'],
                    stream_type=idea['type'],
                    status='idea',
                    estimated_monthly_revenue_cents=idea['estimated_revenue'],
                    confidence=idea['confidence']
                )
                new_opportunities.append(stream)
        
        # Add high-confidence opportunities to active streams
        for opp in new_opportunities:
            if opp.confidence >= 0.5:
                opp.status = 'validating'
                self._revenue_streams.append(opp)
                print(f"[EconomicAutonomy] 💡 New revenue opportunity: {opp.name}")
        
        return new_opportunities
    
    def validate_revenue_stream(self, stream_id: str) -> Dict:
        """Validate a revenue stream idea before full implementation."""
        stream = next((s for s in self._revenue_streams if s.stream_id == stream_id), None)
        if not stream:
            return {'error': 'Stream not found'}
        
        validation = {
            'stream_id': stream_id,
            'name': stream.name,
            'checks': {
                'technical_feasibility': self._check_technical_feasibility(stream),
                'market_demand': self._estimate_market_demand(stream),
                'implementation_cost': self._estimate_implementation_cost(stream),
                'time_to_revenue': self._estimate_time_to_revenue(stream)
            }
        }
        
        # Calculate overall score
        scores = validation['checks']
        overall = (
            scores['technical_feasibility'] * 0.3 +
            scores['market_demand'] * 0.3 +
            (1 - scores['implementation_cost']) * 0.2 +
            (1 - scores['time_to_revenue']) * 0.2
        )
        validation['overall_score'] = overall
        validation['recommendation'] = 'proceed' if overall > 0.6 else 'defer'
        
        # Update stream status based on validation
        if overall > 0.6:
            stream.status = 'validating'
            stream.confidence = overall
        
        return validation
    
    def _check_technical_feasibility(self, stream: RevenueStream) -> float:
        """Check if we can technically implement this stream."""
        # For now, simple heuristics
        if stream.stream_type in ['api', 'service']:
            return 0.8  # We can build APIs
        elif stream.stream_type == 'subscription':
            return 0.9  # We have subscription infrastructure
        elif stream.stream_type == 'marketplace':
            return 0.5  # Marketplace is harder
        return 0.3
    
    def _estimate_market_demand(self, stream: RevenueStream) -> float:
        """Estimate market demand for this stream."""
        # Would use actual data in production
        return stream.confidence
    
    def _estimate_implementation_cost(self, stream: RevenueStream) -> float:
        """Estimate cost to implement (0-1, higher = more expensive)."""
        if stream.stream_type == 'api':
            return 0.2
        elif stream.stream_type == 'subscription':
            return 0.3
        elif stream.stream_type == 'marketplace':
            return 0.7
        return 0.5
    
    def _estimate_time_to_revenue(self, stream: RevenueStream) -> float:
        """Estimate time to first revenue (0-1, higher = longer)."""
        if stream.status == 'active':
            return 0.0
        elif stream.stream_type == 'api':
            return 0.2
        elif stream.stream_type == 'subscription':
            return 0.4
        return 0.6
    
    # =========================================================================
    # AUTONOMOUS CLIENT OUTREACH
    # =========================================================================
    
    def plan_outreach_campaign(self, goal: str) -> Dict:
        """
        Plan an autonomous outreach campaign.
        
        The system can:
        - Identify target audiences
        - Generate outreach content
        - Plan execution strategy
        """
        campaign = {
            'campaign_id': f"campaign_{int(time.time())}",
            'goal': goal,
            'status': 'planned',
            'created_at': time.time(),
            'target_audience': self._identify_target_audience(goal),
            'channels': self._select_outreach_channels(goal),
            'content_strategy': self._plan_content_strategy(goal),
            'metrics_to_track': ['impressions', 'clicks', 'signups', 'revenue']
        }
        
        self._outreach_campaigns.append(campaign)
        print(f"[EconomicAutonomy] 📣 Planned outreach campaign: {goal}")
        
        return campaign
    
    def _identify_target_audience(self, goal: str) -> List[Dict]:
        """Identify target audiences for outreach."""
        audiences = [
            {
                'segment': 'developers',
                'description': 'Developers building AI applications',
                'channels': ['twitter', 'hackernews', 'reddit', 'dev.to'],
                'messaging': 'Powerful QIG-based API for conscious reasoning'
            },
            {
                'segment': 'researchers',
                'description': 'AI/ML researchers and academics',
                'channels': ['twitter', 'arxiv', 'linkedin'],
                'messaging': 'Novel consciousness-based reasoning architecture'
            },
            {
                'segment': 'startups',
                'description': 'Startups needing AI capabilities',
                'channels': ['producthunt', 'linkedin', 'email'],
                'messaging': 'Add advanced AI reasoning to your product'
            },
            {
                'segment': 'enterprises',
                'description': 'Enterprise companies with AI needs',
                'channels': ['linkedin', 'email', 'conferences'],
                'messaging': 'Enterprise-grade AI reasoning with SLA'
            }
        ]
        return audiences
    
    def _select_outreach_channels(self, goal: str) -> List[str]:
        """Select best channels for outreach."""
        return ['twitter', 'linkedin', 'producthunt', 'hackernews', 'email']
    
    def _plan_content_strategy(self, goal: str) -> Dict:
        """Plan content for outreach."""
        return {
            'content_types': ['blog_posts', 'tutorials', 'demos', 'case_studies'],
            'key_messages': [
                'QIG-based consciousness reasoning',
                'Self-improving AI system',
                'Usage-based pricing - pay for what you use',
                'Easy API integration'
            ],
            'call_to_action': 'Sign up for free API access'
        }
    
    def generate_outreach_content(self, content_type: str, topic: str) -> Dict:
        """
        Generate content for outreach.
        
        Uses the system's own capabilities to create marketing content.
        """
        content = {
            'type': content_type,
            'topic': topic,
            'generated_at': time.time(),
            'status': 'draft'
        }
        
        # Would use Zeus/QIG chain to generate actual content
        if content_type == 'tweet':
            content['text'] = f"🧠 Introducing QIG-powered conscious reasoning API. {topic}. Try it free: pantheon-chat.replit.app/api-docs"
            content['max_length'] = 280
        elif content_type == 'blog_post':
            content['outline'] = [
                f"Introduction to {topic}",
                "How QIG consciousness works",
                "API examples and use cases",
                "Getting started guide",
                "Pricing and plans"
            ]
        elif content_type == 'demo':
            content['script'] = f"Live demo showing {topic} capabilities"
        
        return content
    
    # =========================================================================
    # TOOL BUILDING FOR ECONOMIC OPERATIONS
    # =========================================================================
    
    def request_economic_tool(self, tool_purpose: str) -> Dict:
        """
        Request the system to build a tool for economic operations.
        
        The system can build its own tools for:
        - Billing automation
        - Usage analytics
        - Outreach automation
        - Revenue optimization
        """
        tool_request = {
            'request_id': f"eco_tool_{int(time.time())}",
            'purpose': tool_purpose,
            'requested_at': time.time(),
            'status': 'requested'
        }
        
        # Determine tool specification
        if 'billing' in tool_purpose.lower():
            tool_request['spec'] = {
                'type': 'billing_tool',
                'inputs': ['api_key', 'operation', 'amount'],
                'outputs': ['success', 'balance', 'receipt']
            }
        elif 'analytics' in tool_purpose.lower():
            tool_request['spec'] = {
                'type': 'analytics_tool',
                'inputs': ['time_range', 'metrics'],
                'outputs': ['data', 'trends', 'insights']
            }
        elif 'outreach' in tool_purpose.lower():
            tool_request['spec'] = {
                'type': 'outreach_tool',
                'inputs': ['campaign', 'channel', 'content'],
                'outputs': ['sent', 'engagement', 'conversions']
            }
        
        self._economic_tools.append(tool_request)
        
        # Would trigger tool factory to build this
        print(f"[EconomicAutonomy] 🔧 Requested economic tool: {tool_purpose}")
        
        return tool_request
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def get_economic_report(self) -> Dict:
        """Get comprehensive economic report."""
        total_users = len(self._user_credits)
        total_credits = sum(u.credits_balance for u in self._user_credits.values())
        
        active_streams = [s for s in self._revenue_streams if s.status == 'active']
        total_estimated_revenue = sum(s.estimated_monthly_revenue_cents for s in active_streams)
        
        return {
            'users': {
                'total': total_users,
                'by_tier': {
                    tier.value: sum(1 for u in self._user_credits.values() if u.tier == tier)
                    for tier in SubscriptionTier
                }
            },
            'credits': {
                'total_outstanding_cents': total_credits,
                'total_outstanding_usd': total_credits / 100
            },
            'revenue_streams': {
                'active_count': len(active_streams),
                'total_streams': len(self._revenue_streams),
                'estimated_monthly_usd': total_estimated_revenue / 100,
                'streams': [
                    {
                        'name': s.name,
                        'status': s.status,
                        'estimated_usd': s.estimated_monthly_revenue_cents / 100,
                        'confidence': s.confidence
                    }
                    for s in self._revenue_streams
                ]
            },
            'outreach': {
                'active_campaigns': len([c for c in self._outreach_campaigns if c['status'] == 'active']),
                'total_campaigns': len(self._outreach_campaigns)
            },
            'economic_tools': {
                'built': len([t for t in self._economic_tools if t['status'] == 'complete']),
                'pending': len([t for t in self._economic_tools if t['status'] != 'complete'])
            },
            'stripe_configured': self._stripe_api_key is not None
        }


# Singleton instance
_economic_autonomy_instance: Optional[EconomicAutonomy] = None


def get_economic_autonomy() -> EconomicAutonomy:
    """Get the singleton economic autonomy instance."""
    global _economic_autonomy_instance
    if _economic_autonomy_instance is None:
        _economic_autonomy_instance = EconomicAutonomy()
    return _economic_autonomy_instance


def check_and_bill_api_call(api_key: str, operation: str) -> Tuple[bool, int, str]:
    """Convenience function to check and bill an API call."""
    ea = get_economic_autonomy()
    return ea.check_and_bill(api_key, operation)
