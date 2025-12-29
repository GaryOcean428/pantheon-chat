/**
 * Billing Dashboard - View credits, usage, and upgrade subscription
 * 
 * Enables self-sustaining economic operation by allowing users to:
 * - View their current credit balance
 * - See usage statistics
 * - Compare pricing tiers
 * - Upgrade subscription or buy credits
 */

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { 
  CreditCard, 
  Zap, 
  TrendingUp, 
  Clock, 
  CheckCircle2, 
  AlertCircle,
  Sparkles,
  Building2,
  Rocket,
  RefreshCw
} from 'lucide-react';

interface BillingStatus {
  tier: 'free' | 'pro' | 'enterprise';
  credits: number;
  credits_usd: number;
  usage: {
    queries_this_month: number;
    tools_this_month: number;
    research_this_month: number;
  };
  limits: {
    queries_per_month: number;
    tools_per_month: number;
    research_per_month: number;
  };
  pricing: {
    query_cents: number;
    tool_cents: number;
    research_cents: number;
  };
}

const TIER_INFO = {
  free: {
    name: 'Free',
    price: '$0',
    priceNote: 'Pay as you go',
    icon: Zap,
    color: 'bg-gray-500',
    features: [
      '100 queries/month included',
      '10 tool generations/month',
      '5 research requests/month',
      'Pay-per-use beyond limits',
      'Community support'
    ]
  },
  pro: {
    name: 'Pro',
    price: '$49',
    priceNote: '/month',
    icon: Rocket,
    color: 'bg-blue-500',
    features: [
      '10,000 queries/month',
      '500 tool generations/month',
      '100 research requests/month',
      'Priority processing',
      'Email support'
    ]
  },
  enterprise: {
    name: 'Enterprise',
    price: '$499',
    priceNote: '/month',
    icon: Building2,
    color: 'bg-purple-500',
    features: [
      'Unlimited queries',
      'Unlimited tool generations',
      'Unlimited research',
      'Dedicated support',
      'SLA guarantee',
      'Custom integrations'
    ]
  }
};

export function BillingPage() {
  const [billingStatus, setBillingStatus] = useState<BillingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [upgrading, setUpgrading] = useState(false);

  const fetchBillingStatus = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Get current API key from session/storage
      const apiKey = localStorage.getItem('pantheon_api_key') || '';
      
      if (!apiKey) {
        // Show demo data for users without an API key
        setBillingStatus({
          tier: 'free',
          credits: 100,
          credits_usd: 1.00,
          usage: {
            queries_this_month: 0,
            tools_this_month: 0,
            research_this_month: 0
          },
          limits: {
            queries_per_month: 100,
            tools_per_month: 10,
            research_per_month: 5
          },
          pricing: {
            query_cents: 1,
            tool_cents: 5,
            research_cents: 10
          }
        });
        setLoading(false);
        return;
      }

      const response = await fetch('/api/billing/user-status', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({ api_key: apiKey })
      });

      if (!response.ok) {
        throw new Error('Failed to fetch billing status');
      }

      const data = await response.json();
      setBillingStatus(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load billing info');
      // Set default values on error
      setBillingStatus({
        tier: 'free',
        credits: 100,
        credits_usd: 1.00,
        usage: {
          queries_this_month: 0,
          tools_this_month: 0,
          research_this_month: 0
        },
        limits: {
          queries_per_month: 100,
          tools_per_month: 10,
          research_per_month: 5
        },
        pricing: {
          query_cents: 1,
          tool_cents: 5,
          research_cents: 10
        }
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBillingStatus();
  }, []);

  const handleUpgrade = async (tier: string) => {
    setUpgrading(true);
    try {
      const apiKey = localStorage.getItem('pantheon_api_key') || '';
      
      const response = await fetch('/api/billing/create-checkout', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({ 
          product_type: tier,
          api_key: apiKey
        })
      });

      const data = await response.json();
      
      if (data.checkout_url) {
        window.location.href = data.checkout_url;
      } else if (data.error) {
        setError(data.error);
      }
    } catch (err) {
      setError('Failed to start checkout');
    } finally {
      setUpgrading(false);
    }
  };

  const handleBuyCredits = async () => {
    await handleUpgrade('credits');
  };

  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      </div>
    );
  }

  const tierInfo = billingStatus ? TIER_INFO[billingStatus.tier] : TIER_INFO.free;
  const TierIcon = tierInfo.icon;

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Billing & Usage</h1>
          <p className="text-muted-foreground">Manage your subscription and view usage statistics</p>
        </div>
        <Button variant="outline" onClick={fetchBillingStatus}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Current Plan & Credits */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Current Plan Card */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${tierInfo.color}`}>
                  <TierIcon className="h-6 w-6 text-white" />
                </div>
                <div>
                  <CardTitle>Current Plan</CardTitle>
                  <CardDescription>Your subscription tier</CardDescription>
                </div>
              </div>
              <Badge variant={billingStatus?.tier === 'enterprise' ? 'default' : 'secondary'} className="text-lg px-3 py-1">
                {tierInfo.name}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-baseline gap-1">
                <span className="text-4xl font-bold">{tierInfo.price}</span>
                <span className="text-muted-foreground">{tierInfo.priceNote}</span>
              </div>
              <ul className="space-y-2">
                {tierInfo.features.slice(0, 3).map((feature, i) => (
                  <li key={i} className="flex items-center gap-2 text-sm">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    {feature}
                  </li>
                ))}
              </ul>
            </div>
          </CardContent>
          {billingStatus?.tier !== 'enterprise' && (
            <CardFooter>
              <Button 
                className="w-full" 
                onClick={() => handleUpgrade(billingStatus?.tier === 'free' ? 'pro' : 'enterprise')}
                disabled={upgrading}
              >
                <Sparkles className="h-4 w-4 mr-2" />
                {upgrading ? 'Processing...' : `Upgrade to ${billingStatus?.tier === 'free' ? 'Pro' : 'Enterprise'}`}
              </Button>
            </CardFooter>
          )}
        </Card>

        {/* Credits Card */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-green-500">
                  <CreditCard className="h-6 w-6 text-white" />
                </div>
                <div>
                  <CardTitle>Credit Balance</CardTitle>
                  <CardDescription>Available for pay-per-use</CardDescription>
                </div>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-baseline gap-1">
                <span className="text-4xl font-bold">${billingStatus?.credits_usd.toFixed(2)}</span>
                <span className="text-muted-foreground">({billingStatus?.credits} cents)</span>
              </div>
              <div className="text-sm text-muted-foreground space-y-1">
                <p>• Queries: ${(billingStatus?.pricing.query_cents || 1) / 100}/each</p>
                <p>• Tools: ${(billingStatus?.pricing.tool_cents || 5) / 100}/each</p>
                <p>• Research: ${(billingStatus?.pricing.research_cents || 10) / 100}/each</p>
              </div>
            </div>
          </CardContent>
          <CardFooter>
            <Button 
              variant="outline" 
              className="w-full"
              onClick={handleBuyCredits}
              disabled={upgrading}
            >
              <CreditCard className="h-4 w-4 mr-2" />
              {upgrading ? 'Processing...' : 'Buy $10 Credits'}
            </Button>
          </CardFooter>
        </Card>
      </div>

      {/* Usage Statistics */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-orange-500">
              <TrendingUp className="h-6 w-6 text-white" />
            </div>
            <div>
              <CardTitle>Usage This Month</CardTitle>
              <CardDescription>Track your API usage against your plan limits</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 md:grid-cols-3">
            {/* Queries */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Queries</span>
                <span className="text-muted-foreground">
                  {billingStatus?.usage.queries_this_month} / {billingStatus?.limits.queries_per_month}
                </span>
              </div>
              <Progress 
                value={(billingStatus?.usage.queries_this_month || 0) / (billingStatus?.limits.queries_per_month || 1) * 100} 
                className="h-2"
              />
              <p className="text-xs text-muted-foreground">
                {Math.max(0, (billingStatus?.limits.queries_per_month || 0) - (billingStatus?.usage.queries_this_month || 0))} remaining
              </p>
            </div>

            {/* Tools */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Tool Generations</span>
                <span className="text-muted-foreground">
                  {billingStatus?.usage.tools_this_month} / {billingStatus?.limits.tools_per_month}
                </span>
              </div>
              <Progress 
                value={(billingStatus?.usage.tools_this_month || 0) / (billingStatus?.limits.tools_per_month || 1) * 100} 
                className="h-2"
              />
              <p className="text-xs text-muted-foreground">
                {Math.max(0, (billingStatus?.limits.tools_per_month || 0) - (billingStatus?.usage.tools_this_month || 0))} remaining
              </p>
            </div>

            {/* Research */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Research Requests</span>
                <span className="text-muted-foreground">
                  {billingStatus?.usage.research_this_month} / {billingStatus?.limits.research_per_month}
                </span>
              </div>
              <Progress 
                value={(billingStatus?.usage.research_this_month || 0) / (billingStatus?.limits.research_per_month || 1) * 100} 
                className="h-2"
              />
              <p className="text-xs text-muted-foreground">
                {Math.max(0, (billingStatus?.limits.research_per_month || 0) - (billingStatus?.usage.research_this_month || 0))} remaining
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Pricing Tiers Comparison */}
      <Card>
        <CardHeader>
          <CardTitle>Compare Plans</CardTitle>
          <CardDescription>Choose the plan that fits your needs</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="free" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="free">Free</TabsTrigger>
              <TabsTrigger value="pro">Pro</TabsTrigger>
              <TabsTrigger value="enterprise">Enterprise</TabsTrigger>
            </TabsList>
            
            {Object.entries(TIER_INFO).map(([key, tier]) => {
              const Icon = tier.icon;
              return (
                <TabsContent key={key} value={key} className="space-y-4">
                  <div className="flex items-center gap-4 p-4 bg-muted/50 rounded-lg">
                    <div className={`p-3 rounded-lg ${tier.color}`}>
                      <Icon className="h-8 w-8 text-white" />
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold">{tier.name}</h3>
                      <p className="text-muted-foreground">
                        <span className="text-3xl font-bold text-foreground">{tier.price}</span>
                        {tier.priceNote}
                      </p>
                    </div>
                  </div>
                  <ul className="space-y-3">
                    {tier.features.map((feature, i) => (
                      <li key={i} className="flex items-center gap-3">
                        <CheckCircle2 className="h-5 w-5 text-green-500 flex-shrink-0" />
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                  {key !== billingStatus?.tier && (
                    <Button 
                      className="w-full mt-4" 
                      variant={key === 'enterprise' ? 'default' : 'outline'}
                      onClick={() => handleUpgrade(key)}
                      disabled={upgrading || (billingStatus?.tier === 'enterprise') || (billingStatus?.tier === 'pro' && key === 'free')}
                    >
                      {billingStatus?.tier === 'enterprise' ? 'Current Plan' : 
                       billingStatus?.tier === 'pro' && key === 'free' ? 'Downgrade not available' :
                       key === billingStatus?.tier ? 'Current Plan' :
                       `Upgrade to ${tier.name}`}
                    </Button>
                  )}
                  {key === billingStatus?.tier && (
                    <div className="flex items-center justify-center gap-2 p-3 bg-green-500/10 text-green-600 rounded-lg">
                      <CheckCircle2 className="h-5 w-5" />
                      <span className="font-medium">This is your current plan</span>
                    </div>
                  )}
                </TabsContent>
              );
            })}
          </Tabs>
        </CardContent>
      </Card>

      {/* Help Section */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Clock className="h-5 w-5 text-muted-foreground" />
              <div>
                <p className="font-medium">Usage resets monthly</p>
                <p className="text-sm text-muted-foreground">Your usage limits reset on the 1st of each month</p>
              </div>
            </div>
            <Button variant="ghost" asChild>
              <a href="/api-docs" target="_blank">View API Documentation →</a>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default BillingPage;
