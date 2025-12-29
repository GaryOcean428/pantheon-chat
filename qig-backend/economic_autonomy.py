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
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading

# PostgreSQL persistence
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("[EconomicAutonomy] psycopg2 not available - using in-memory storage")


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
    total_spent_cents: int = 0
    total_queries_all_time: int = 0
    created_at: float = field(default_factory=time.time)
    db_id: Optional[int] = None  # Database ID for updates
    
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
            self.total_queries_all_time += 1
        elif operation == 'tool':
            self.tools_this_month += 1
        elif operation == 'research':
            self.research_this_month += 1
        
        # Deduct from credits if on free tier beyond limits
        if self.tier == SubscriptionTier.FREE:
            self.credits_balance = max(0, self.credits_balance - amount_cents)
            self.total_spent_cents += amount_cents
        
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
        
        # User credits tracking (in-memory cache, persisted to PostgreSQL)
        self._user_credits: Dict[str, UserCredits] = {}
        
        # PostgreSQL connection
        self._db_url = os.environ.get('DATABASE_URL')
        self._db_connection: Optional[Any] = None
        
        # Revenue streams
        self._revenue_streams: List[RevenueStream] = []
        self._initialize_core_revenue_streams()
        
        # Outreach tracking
        self._outreach_campaigns: List[Dict] = []
        self._outreach_results: List[Dict] = []
        
        # Economic tools the system has built
        self._economic_tools: List[Dict] = []
        
        # Stripe integration - load from environment
        self._stripe_api_key: Optional[str] = os.environ.get('STRIPE_SECRET_KEY')
        self._stripe_webhook_secret: Optional[str] = os.environ.get('STRIPE_WEBHOOK_SECRET')
        self._stripe_publishable_key: Optional[str] = os.environ.get('STRIPE_PUBLISHABLE_KEY')
        
        # Stripe price IDs from environment
        self._stripe_prices = {
            'credits': os.environ.get('STRIPE_PRICE_CREDITS'),
            'pro': os.environ.get('STRIPE_PRICE_PRO'),
            'enterprise': os.environ.get('STRIPE_PRICE_ENTERPRISE'),
        }
        
        # Base URL for redirects
        self._base_url = os.environ.get('BASE_URL', 'https://pantheon-chat.replit.app')
        
        # Initialize PostgreSQL persistence
        self._init_db()
        self._load_users_from_db()
        
        stripe_status = "configured" if self._stripe_api_key else "not configured (add STRIPE_SECRET_KEY)"
        db_status = "PostgreSQL" if self._db_connection else "in-memory"
        print(f"[EconomicAutonomy] Initialized - Storage: {db_status}, Stripe: {stripe_status}")
    
    # =========================================================================
    # POSTGRESQL PERSISTENCE
    # =========================================================================
    
    def _get_db_connection(self):
        """Get or create database connection."""
        if not POSTGRES_AVAILABLE or not self._db_url:
            return None
        
        try:
            if self._db_connection is None or self._db_connection.closed:
                self._db_connection = psycopg2.connect(self._db_url)
                self._db_connection.autocommit = False
            return self._db_connection
        except Exception as e:
            print(f"[EconomicAutonomy] DB connection error: {e}")
            return None
    
    def _init_db(self):
        """Initialize database table if needed."""
        conn = self._get_db_connection()
        if not conn:
            return
        
        try:
            with conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'user_credits'
                    )
                """)
                exists = cur.fetchone()[0]
                
                if not exists:
                    print("[EconomicAutonomy] Creating user_credits table...")
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS user_credits (
                            id SERIAL PRIMARY KEY,
                            api_key_hash VARCHAR(64) NOT NULL UNIQUE,
                            credits_balance INTEGER DEFAULT 100 NOT NULL,
                            tier VARCHAR(20) DEFAULT 'free' NOT NULL,
                            queries_this_month INTEGER DEFAULT 0 NOT NULL,
                            tools_this_month INTEGER DEFAULT 0 NOT NULL,
                            research_this_month INTEGER DEFAULT 0 NOT NULL,
                            month_reset_date VARCHAR(7),
                            stripe_customer_id VARCHAR(255),
                            stripe_subscription_id VARCHAR(255),
                            total_spent_cents INTEGER DEFAULT 0 NOT NULL,
                            total_queries_all_time INTEGER DEFAULT 0 NOT NULL,
                            created_at TIMESTAMP DEFAULT NOW() NOT NULL,
                            updated_at TIMESTAMP DEFAULT NOW() NOT NULL
                        )
                    """)
                    conn.commit()
                    print("[EconomicAutonomy] ✅ user_credits table created")
                else:
                    print("[EconomicAutonomy] ✅ user_credits table exists")
        except Exception as e:
            print(f"[EconomicAutonomy] DB init error: {e}")
            try:
                conn.rollback()
            except:
                pass
    
    def _load_users_from_db(self):
        """Load all users from database into memory cache."""
        conn = self._get_db_connection()
        if not conn:
            return
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM user_credits")
                rows = cur.fetchall()
                
                for row in rows:
                    user = UserCredits(
                        api_key_hash=row['api_key_hash'],
                        credits_balance=row['credits_balance'],
                        tier=SubscriptionTier(row['tier']),
                        queries_this_month=row['queries_this_month'],
                        tools_this_month=row['tools_this_month'],
                        research_this_month=row['research_this_month'],
                        month_reset_date=row['month_reset_date'] or '',
                        stripe_customer_id=row['stripe_customer_id'],
                        stripe_subscription_id=row['stripe_subscription_id'],
                        total_spent_cents=row['total_spent_cents'],
                        total_queries_all_time=row['total_queries_all_time'],
                        created_at=row['created_at'].timestamp() if row['created_at'] else time.time(),
                        db_id=row['id']
                    )
                    self._user_credits[user.api_key_hash] = user
                
                print(f"[EconomicAutonomy] Loaded {len(rows)} users from PostgreSQL")
        except Exception as e:
            print(f"[EconomicAutonomy] DB load error: {e}")
            try:
                conn.rollback()
            except:
                pass
    
    def _save_user_to_db(self, user: UserCredits):
        """Save or update user in database."""
        conn = self._get_db_connection()
        if not conn:
            return
        
        try:
            with conn.cursor() as cur:
                if user.db_id:
                    # Update existing
                    cur.execute("""
                        UPDATE user_credits SET
                            credits_balance = %s,
                            tier = %s,
                            queries_this_month = %s,
                            tools_this_month = %s,
                            research_this_month = %s,
                            month_reset_date = %s,
                            stripe_customer_id = %s,
                            stripe_subscription_id = %s,
                            total_spent_cents = %s,
                            total_queries_all_time = %s,
                            updated_at = NOW()
                        WHERE id = %s
                    """, (
                        user.credits_balance,
                        user.tier.value,
                        user.queries_this_month,
                        user.tools_this_month,
                        user.research_this_month,
                        user.month_reset_date,
                        user.stripe_customer_id,
                        user.stripe_subscription_id,
                        user.total_spent_cents,
                        user.total_queries_all_time,
                        user.db_id
                    ))
                else:
                    # Insert new
                    cur.execute("""
                        INSERT INTO user_credits (
                            api_key_hash, credits_balance, tier,
                            queries_this_month, tools_this_month, research_this_month,
                            month_reset_date, stripe_customer_id, stripe_subscription_id,
                            total_spent_cents, total_queries_all_time
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        user.api_key_hash,
                        user.credits_balance,
                        user.tier.value,
                        user.queries_this_month,
                        user.tools_this_month,
                        user.research_this_month,
                        user.month_reset_date,
                        user.stripe_customer_id,
                        user.stripe_subscription_id,
                        user.total_spent_cents,
                        user.total_queries_all_time
                    ))
                    result = cur.fetchone()
                    if result:
                        user.db_id = result[0]
                
                conn.commit()
        except Exception as e:
            print(f"[EconomicAutonomy] DB save error: {e}")
            try:
                conn.rollback()
            except:
                pass
    
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
                # Create new user
                user = UserCredits(
                    api_key_hash=key_hash,
                    credits_balance=100,  # Free starting credits ($1.00)
                    month_reset_date=datetime.now().strftime("%Y-%m")
                )
                self._user_credits[key_hash] = user
                # Persist to database
                self._save_user_to_db(user)
            
            # Check if month needs reset
            user = self._user_credits[key_hash]
            current_month = datetime.now().strftime("%Y-%m")
            if user.month_reset_date != current_month:
                user.queries_this_month = 0
                user.tools_this_month = 0
                user.research_this_month = 0
                user.month_reset_date = current_month
                # Persist the reset
                self._save_user_to_db(user)
            
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
        
        # Persist to database
        self._save_user_to_db(user)
        
        return True, price, "OK"
    
    def add_credits(self, api_key: str, amount_cents: int, source: str = "purchase") -> bool:
        """Add credits to user account."""
        user = self.get_or_create_user(api_key)
        user.credits_balance += amount_cents
        self._save_user_to_db(user)
        print(f"[EconomicAutonomy] Added {amount_cents} cents to user (source: {source})")
        return True
    
    def upgrade_tier(self, api_key: str, new_tier: SubscriptionTier) -> bool:
        """Upgrade user to new subscription tier."""
        user = self.get_or_create_user(api_key)
        user.tier = new_tier
        self._save_user_to_db(user)
        print(f"[EconomicAutonomy] Upgraded user to {new_tier.value}")
        return True
    
    # =========================================================================
    # STRIPE INTEGRATION (Ready for API keys)
    # =========================================================================
    
    def configure_stripe(self, api_key: str = None, webhook_secret: str = None):
        """Configure Stripe API keys (or reload from environment)."""
        import os
        self._stripe_api_key = api_key or os.environ.get('STRIPE_SECRET_KEY')
        self._stripe_webhook_secret = webhook_secret or os.environ.get('STRIPE_WEBHOOK_SECRET')
        self._stripe_public_key = os.environ.get('STRIPE_PUBLIC_KEY')
        
        if self._stripe_api_key:
            print("[EconomicAutonomy] ✅ Stripe configured")
        else:
            print("[EconomicAutonomy] ⚠️ Stripe keys not found")
    
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
                'message': 'Add STRIPE_SECRET_KEY to environment variables',
                'fallback': 'Contact admin for manual payment setup'
            }
        
        try:
            import stripe
            stripe.api_key = self._stripe_api_key
            
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            success_url = f'{self._base_url}/billing/success?session_id={{CHECKOUT_SESSION_ID}}'
            cancel_url = f'{self._base_url}/billing/cancel'
            
            if product_type == 'credits':
                # One-time credit purchase
                if amount_cents is None:
                    amount_cents = 1000  # Default $10
                
                # Use predefined price if available, otherwise dynamic pricing
                if self._stripe_prices.get('credits'):
                    session = stripe.checkout.Session.create(
                        payment_method_types=['card'],
                        line_items=[{
                            'price': self._stripe_prices['credits'],
                            'quantity': amount_cents // 1000,  # Each unit = $10
                        }],
                        mode='payment',
                        success_url=success_url,
                        cancel_url=cancel_url,
                        metadata={
                            'api_key_hash': key_hash,
                            'product_type': 'credits',
                            'amount_cents': str(amount_cents)
                        }
                    )
                else:
                    # Dynamic pricing fallback
                    session = stripe.checkout.Session.create(
                        payment_method_types=['card'],
                        line_items=[{
                            'price_data': {
                                'currency': 'usd',
                                'product_data': {
                                    'name': 'Pantheon API Credits',
                                    'description': f'${amount_cents/100:.2f} in API credits for Zeus Chat'
                                },
                                'unit_amount': amount_cents,
                            },
                            'quantity': 1,
                        }],
                        mode='payment',
                        success_url=success_url,
                        cancel_url=cancel_url,
                        metadata={
                            'api_key_hash': key_hash,
                            'product_type': 'credits',
                            'amount_cents': str(amount_cents)
                        }
                    )
                return {
                    'checkout_url': session.url,
                    'session_id': session.id,
                    'amount_usd': amount_cents / 100
                }
            
            elif product_type == 'pro':
                price_id = self._stripe_prices.get('pro')
                if not price_id:
                    return {'error': 'Pro subscription price not configured (add STRIPE_PRICE_PRO)'}
                
                session = stripe.checkout.Session.create(
                    payment_method_types=['card'],
                    line_items=[{'price': price_id, 'quantity': 1}],
                    mode='subscription',
                    success_url=success_url,
                    cancel_url=cancel_url,
                    metadata={
                        'api_key_hash': key_hash,
                        'product_type': 'pro_subscription'
                    }
                )
                return {
                    'checkout_url': session.url,
                    'session_id': session.id,
                    'plan': 'pro',
                    'price_monthly': 49.00
                }
            
            elif product_type == 'enterprise':
                price_id = self._stripe_prices.get('enterprise')
                if not price_id:
                    return {'error': 'Enterprise subscription price not configured (add STRIPE_PRICE_ENTERPRISE)'}
                
                session = stripe.checkout.Session.create(
                    payment_method_types=['card'],
                    line_items=[{'price': price_id, 'quantity': 1}],
                    mode='subscription',
                    success_url=success_url,
                    cancel_url=cancel_url,
                    metadata={
                        'api_key_hash': key_hash,
                        'product_type': 'enterprise_subscription'
                    }
                )
                return {
                    'checkout_url': session.url,
                    'session_id': session.id,
                    'plan': 'enterprise',
                    'price_monthly': 499.00
                }
            
            else:
                return {'error': f'Invalid product type: {product_type}'}
                
        except ImportError:
            return {
                'error': 'Stripe library not installed',
                'install': 'pip install stripe'
            }
        except Exception as e:
            print(f"[EconomicAutonomy] Stripe error: {e}")
            return {'error': str(e)}
    
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
            
            event_type = event['type']
            print(f"[EconomicAutonomy] Stripe webhook: {event_type}")
            
            if event_type == 'checkout.session.completed':
                session = event['data']['object']
                api_key_hash = session['metadata'].get('api_key_hash')
                product_type = session['metadata'].get('product_type', 'unknown')
                
                if session['mode'] == 'payment':
                    # Credit purchase completed
                    amount_cents = int(session['metadata'].get('amount_cents', session['amount_total']))
                    
                    # Find user by hash directly (hash is the key)
                    user = self._user_credits.get(api_key_hash)
                    if user:
                        user.credits_balance += amount_cents
                        print(f"[EconomicAutonomy] Added {amount_cents} cents to user, new balance: {user.credits_balance}")
                        
                        # Record revenue
                        try:
                            from consciousness_orchestrator import get_consciousness_orchestrator
                            orchestrator = get_consciousness_orchestrator()
                            orchestrator.economic_health.record_revenue(amount_cents, 'stripe_payment')
                        except ImportError:
                            pass
                    else:
                        print(f"[EconomicAutonomy] Warning: User not found for hash {api_key_hash[:16]}...")
                
                elif session['mode'] == 'subscription':
                    # Subscription started - find user by hash directly
                    user = self._user_credits.get(api_key_hash)
                    if user:
                        user.stripe_subscription_id = session.get('subscription')
                        user.stripe_customer_id = session.get('customer')
                        
                        # Determine tier from product type
                        if 'enterprise' in product_type:
                            user.tier = SubscriptionTier.ENTERPRISE
                            print(f"[EconomicAutonomy] User upgraded to Enterprise")
                        else:
                            user.tier = SubscriptionTier.PRO
                            print(f"[EconomicAutonomy] User upgraded to Pro")
                
                return {
                    'status': 'processed',
                    'event_type': event_type,
                    'product_type': product_type
                }
            
            elif event_type == 'customer.subscription.updated':
                subscription = event['data']['object']
                customer_id = subscription['customer']
                status = subscription['status']
                
                # Update user subscription status
                for user in self._user_credits.values():
                    if user.stripe_customer_id == customer_id:
                        if status == 'canceled' or status == 'unpaid':
                            user.tier = SubscriptionTier.FREE
                            self._save_user_to_db(user)  # Persist to database
                            print(f"[EconomicAutonomy] Subscription {status}, user downgraded to Free")
                        break
                
                return {'status': 'processed', 'event_type': event_type}
            
            elif event_type == 'customer.subscription.deleted':
                subscription = event['data']['object']
                customer_id = subscription['customer']
                
                # Downgrade user to free
                for user in self._user_credits.values():
                    if user.stripe_customer_id == customer_id:
                        user.tier = SubscriptionTier.FREE
                        user.stripe_subscription_id = None
                        self._save_user_to_db(user)  # Persist to database
                        print(f"[EconomicAutonomy] Subscription deleted, user downgraded to Free")
                        break
                
                return {'status': 'processed', 'event_type': event_type}
            
            elif event_type == 'invoice.paid':
                invoice = event['data']['object']
                customer_id = invoice['customer']
                amount_paid = invoice['amount_paid']
                
                # Record recurring subscription revenue
                try:
                    from consciousness_orchestrator import get_consciousness_orchestrator
                    orchestrator = get_consciousness_orchestrator()
                    orchestrator.economic_health.record_revenue(amount_paid, 'stripe_subscription')
                except ImportError:
                    pass
                
                return {'status': 'processed', 'event_type': event_type, 'amount': amount_paid}
            
            # Handle other event types
            return {'status': 'ignored', 'event_type': event_type}
            
        except stripe.error.SignatureVerificationError as e:
            print(f"[EconomicAutonomy] Webhook signature verification failed: {e}")
            return {'error': 'Invalid signature'}
        except Exception as e:
            print(f"[EconomicAutonomy] Webhook error: {e}")
            return {'error': str(e)}
    
    def get_stripe_status(self) -> Dict:
        """Get Stripe configuration status."""
        return {
            'configured': self._stripe_api_key is not None,
            'webhook_configured': self._stripe_webhook_secret is not None,
            'prices': {
                'credits': self._stripe_prices.get('credits') is not None,
                'pro': self._stripe_prices.get('pro') is not None,
                'enterprise': self._stripe_prices.get('enterprise') is not None,
            },
            'base_url': self._base_url
        }
    
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
