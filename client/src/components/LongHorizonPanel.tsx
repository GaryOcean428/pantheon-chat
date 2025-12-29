/**
 * LongHorizonPanel - Detailed long-horizon task metrics
 *
 * Full dashboard panel for goal tracking, geodesic efficiency,
 * and geometric error recovery. Used in telemetry dashboard.
 */

import {
  useGoals,
  useEfficiencyStats,
  useEfficiencyDegradation,
  useRecoveryStatus,
  useStuckCheck,
  useTriggerRecovery,
} from '@/hooks';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Badge,
  Progress,
  Button,
  Skeleton,
} from '@/components/ui';
import {
  Flag,
  Route,
  RotateCcw,
  AlertTriangle,
  CheckCircle2,
  TrendingUp,
  TrendingDown,
  Target,
  Loader2,
  RefreshCw,
  ChevronRight,
} from 'lucide-react';

// ============================================================================
// Goal Tracking Panel
// ============================================================================

function GoalTrackingCard() {
  const { data, isLoading, isError } = useGoals();

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-32" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-24 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (isError || !data?.success) {
    return (
      <Card className="border-yellow-500/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Flag className="h-5 w-5 text-yellow-500" />
            Goal Tracking
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Goal tracking service unavailable
          </p>
        </CardContent>
      </Card>
    );
  }

  const { goals, summary } = data;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Flag className="h-5 w-5 text-blue-500" />
          Goal Tracking
          {summary.active_goals > 0 && (
            <Badge variant="secondary" className="ml-2">
              {summary.active_goals} active
            </Badge>
          )}
        </CardTitle>
        <CardDescription>
          Basin-encoded goal hierarchies with Fisher-Rao progress
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Summary metrics */}
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold font-mono">{summary.total_goals}</div>
            <div className="text-xs text-muted-foreground">Total Goals</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold font-mono text-green-500">
              {summary.completed_goals}
            </div>
            <div className="text-xs text-muted-foreground">Completed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold font-mono text-blue-500">
              {(summary.overall_progress * 100).toFixed(0)}%
            </div>
            <div className="text-xs text-muted-foreground">Progress</div>
          </div>
        </div>

        {/* Progress bar */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Overall Progress</span>
            <span className="font-mono">{(summary.overall_progress * 100).toFixed(1)}%</span>
          </div>
          <Progress value={summary.overall_progress * 100} className="h-2" />
        </div>

        {/* Active goals list */}
        {goals.length > 0 && (
          <div className="space-y-2 pt-4 border-t">
            <div className="text-sm font-medium">Active Goals</div>
            {goals
              .filter((g) => !g.completed)
              .slice(0, 5)
              .map((goal) => (
                <div
                  key={goal.goal_id}
                  className="flex items-center justify-between p-2 rounded-lg bg-muted/50"
                >
                  <div className="flex items-center gap-2">
                    <Target className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm truncate max-w-[200px]">
                      {goal.description}
                    </span>
                  </div>
                  <Badge variant="outline" className="text-xs">
                    {goal.steps_taken} steps
                  </Badge>
                </div>
              ))}
          </div>
        )}

        {goals.length === 0 && (
          <div className="text-center py-4 text-muted-foreground">
            <Flag className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No goals defined</p>
            <p className="text-xs">Goals are created during task execution</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Efficiency Tracking Panel
// ============================================================================

function EfficiencyCard() {
  const { data: stats, isLoading: statsLoading } = useEfficiencyStats();
  const { data: degradation, isLoading: degradationLoading } = useEfficiencyDegradation();

  const isLoading = statsLoading || degradationLoading;

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-32" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-24 w-full" />
        </CardContent>
      </Card>
    );
  }

  const effStats = stats?.stats;
  const isDegraded = degradation?.degradation?.is_degraded ?? false;

  const getEfficiencyColor = (eff: number) => {
    if (eff >= 0.8) return 'text-green-500';
    if (eff >= 0.5) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <Card className={isDegraded ? 'border-yellow-500/50' : ''}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Route className="h-5 w-5 text-purple-500" />
          Geodesic Efficiency
          {isDegraded && (
            <Badge className="bg-yellow-500/20 text-yellow-500 ml-2">
              <AlertTriangle className="h-3 w-3 mr-1" />
              Degraded
            </Badge>
          )}
        </CardTitle>
        <CardDescription>
          Path efficiency = optimal distance / actual distance
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {effStats && effStats.count > 0 ? (
          <>
            {/* Main efficiency gauge */}
            <div className="text-center">
              <div
                className={`text-4xl font-bold font-mono ${getEfficiencyColor(effStats.mean_efficiency)}`}
              >
                {(effStats.mean_efficiency * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground">Mean Efficiency</div>
              <Progress
                value={effStats.mean_efficiency * 100}
                className="h-2 mt-2"
              />
            </div>

            {/* Stats grid */}
            <div className="grid grid-cols-3 gap-4 pt-4 border-t">
              <div className="text-center">
                <div className="text-lg font-mono">
                  {(effStats.min_efficiency * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-muted-foreground">Min</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-mono">
                  {(effStats.max_efficiency * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-muted-foreground">Max</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-mono">{effStats.count}</div>
                <div className="text-xs text-muted-foreground">Operations</div>
              </div>
            </div>

            {/* Efficiency by operation type */}
            {stats?.by_operation_type && Object.keys(stats.by_operation_type).length > 0 && (
              <div className="space-y-2 pt-4 border-t">
                <div className="text-sm font-medium">By Operation Type</div>
                {Object.entries(stats.by_operation_type)
                  .slice(0, 5)
                  .map(([type, data]) => (
                    <div
                      key={type}
                      className="flex items-center justify-between text-sm"
                    >
                      <span className="capitalize">{type}</span>
                      <span className={`font-mono ${getEfficiencyColor(data.mean_efficiency)}`}>
                        {(data.mean_efficiency * 100).toFixed(0)}%
                        <span className="text-muted-foreground ml-1">({data.count})</span>
                      </span>
                    </div>
                  ))}
              </div>
            )}

            {/* Degradation alert */}
            {isDegraded && degradation?.degradation && (
              <div className="p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
                <div className="flex items-center gap-2 text-yellow-500">
                  <TrendingDown className="h-4 w-4" />
                  <span className="font-medium">Efficiency Degradation Detected</span>
                </div>
                <p className="text-sm text-muted-foreground mt-1">
                  Current: {(degradation.degradation.current_efficiency * 100).toFixed(0)}%
                  (baseline: {(degradation.degradation.baseline_efficiency * 100).toFixed(0)}%)
                </p>
                <p className="text-xs text-yellow-500/80">
                  {degradation.degradation.drop_percentage.toFixed(1)}% below baseline
                </p>
              </div>
            )}
          </>
        ) : (
          <div className="text-center py-4 text-muted-foreground">
            <Route className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No efficiency data yet</p>
            <p className="text-xs">Operations will be tracked automatically</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Recovery Status Panel
// ============================================================================

function RecoveryCard() {
  const { data: status, isLoading: statusLoading } = useRecoveryStatus();
  const { data: stuckCheck, isLoading: stuckLoading } = useStuckCheck();
  const triggerRecovery = useTriggerRecovery();

  const isLoading = statusLoading || stuckLoading;

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-32" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-24 w-full" />
        </CardContent>
      </Card>
    );
  }

  const stats = status?.stats;
  const isStuck = stuckCheck?.is_stuck ?? false;
  const stuckReason = stuckCheck?.reason ?? '';

  return (
    <Card className={isStuck ? 'border-red-500/50' : ''}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <RotateCcw className="h-5 w-5 text-orange-500" />
          Error Recovery
          {isStuck ? (
            <Badge className="bg-red-500/20 text-red-500 ml-2">
              <AlertTriangle className="h-3 w-3 mr-1" />
              Stuck
            </Badge>
          ) : (
            <Badge className="bg-green-500/20 text-green-500 ml-2">
              <CheckCircle2 className="h-3 w-3 mr-1" />
              Stable
            </Badge>
          )}
        </CardTitle>
        <CardDescription>
          Geometric stuck detection and checkpoint recovery
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {stats ? (
          <>
            {/* Stats grid */}
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-3 rounded-lg bg-muted/50">
                <div className="text-2xl font-bold font-mono">{stats.checkpoint_count}</div>
                <div className="text-xs text-muted-foreground">Checkpoints</div>
              </div>
              <div className="text-center p-3 rounded-lg bg-muted/50">
                <div className="text-2xl font-bold font-mono">{stats.recovery_count}</div>
                <div className="text-xs text-muted-foreground">Recoveries</div>
              </div>
            </div>

            {/* Trajectory info */}
            <div className="space-y-2 pt-4 border-t">
              <div className="flex items-center justify-between text-sm">
                <span>Total Steps</span>
                <span className="font-mono">{stats.total_steps}</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span>Trajectory Length</span>
                <span className="font-mono">{stats.trajectory_length}</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span>Avg Checkpoint Score</span>
                <span className="font-mono">{stats.avg_checkpoint_score.toFixed(3)}</span>
              </div>
            </div>

            {/* Stuck alert with recovery button */}
            {isStuck && (
              <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 space-y-3">
                <div className="flex items-center gap-2 text-red-500">
                  <AlertTriangle className="h-4 w-4" />
                  <span className="font-medium">System Stuck: {stuckReason}</span>
                </div>

                {stuckCheck?.diagnostics && (
                  <div className="text-xs space-y-1">
                    {Object.entries(stuckCheck.diagnostics)
                      .filter(([k]) => !['trajectory_length', 'checkpoints'].includes(k))
                      .slice(0, 3)
                      .map(([key, value]) => (
                        <div key={key} className="flex items-center justify-between">
                          <span className="text-muted-foreground capitalize">
                            {key.replace(/_/g, ' ')}
                          </span>
                          <span className="font-mono">
                            {typeof value === 'number' ? value.toFixed(3) : String(value)}
                          </span>
                        </div>
                      ))}
                  </div>
                )}

                <Button
                  size="sm"
                  variant="destructive"
                  onClick={() => triggerRecovery.mutate()}
                  disabled={triggerRecovery.isPending}
                  className="w-full"
                >
                  {triggerRecovery.isPending ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Recovering...
                    </>
                  ) : (
                    <>
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Trigger Recovery
                    </>
                  )}
                </Button>
              </div>
            )}

            {/* Recent recoveries */}
            {status?.recent_recoveries && status.recent_recoveries.length > 0 && (
              <div className="space-y-2 pt-4 border-t">
                <div className="text-sm font-medium">Recent Recoveries</div>
                {status.recent_recoveries.slice(0, 3).map((recovery, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between p-2 rounded-lg bg-muted/50 text-sm"
                  >
                    <div className="flex items-center gap-2">
                      <RotateCcw className="h-3 w-3 text-muted-foreground" />
                      <span className="capitalize">{recovery.stuck_reason.replace(/_/g, ' ')}</span>
                    </div>
                    <span className="text-muted-foreground">
                      -{recovery.steps_back} steps
                    </span>
                  </div>
                ))}
              </div>
            )}
          </>
        ) : (
          <div className="text-center py-4 text-muted-foreground">
            <RotateCcw className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">Recovery system idle</p>
            <p className="text-xs">States are tracked during task execution</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Main Panel Export
// ============================================================================

export function LongHorizonPanel() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <GoalTrackingCard />
        <EfficiencyCard />
        <RecoveryCard />
      </div>

      {/* Explanation card */}
      <Card>
        <CardHeader>
          <CardTitle>Long-Horizon Task Management</CardTitle>
          <CardDescription>
            QIG-pure geometric tracking for extended task execution
          </CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
          <div>
            <div className="flex items-center gap-2 font-medium mb-2">
              <Flag className="h-4 w-4 text-blue-500" />
              Goal Tracking
            </div>
            <p className="text-muted-foreground">
              Goals are encoded as 64D basin coordinates. Progress is measured via
              Fisher-Rao geodesic distance from current basin to goal basin.
            </p>
          </div>
          <div>
            <div className="flex items-center gap-2 font-medium mb-2">
              <Route className="h-4 w-4 text-purple-500" />
              Geodesic Efficiency
            </div>
            <p className="text-muted-foreground">
              Efficiency = optimal_distance / actual_distance. A value of 1.0 means
              the system followed the perfect geodesic path.
            </p>
          </div>
          <div>
            <div className="flex items-center gap-2 font-medium mb-2">
              <RotateCcw className="h-4 w-4 text-orange-500" />
              Error Recovery
            </div>
            <p className="text-muted-foreground">
              Detects stuck states (basin drift, phi collapse, kappa runaway, progress stall)
              and backtracks to low-curvature checkpoints.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default LongHorizonPanel;
