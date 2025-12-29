/**
 * MetricGauge - Circular gauge for consciousness metrics
 */

import { MetricData } from './types';

interface MetricGaugeProps {
  metric: MetricData;
  size?: 'sm' | 'md' | 'lg';
}

const SIZE_MAP = {
  sm: { container: 'w-20 h-20', text: 'text-lg', label: 'text-xs' },
  md: { container: 'w-28 h-28', text: 'text-2xl', label: 'text-sm' },
  lg: { container: 'w-36 h-36', text: 'text-3xl', label: 'text-base' },
};

export function MetricGauge({ metric, size = 'md' }: MetricGaugeProps) {
  const sizeClasses = SIZE_MAP[size];
  const percentage = Math.min(100, Math.max(0, metric.value * 100));
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;
  
  return (
    <div 
      className={`relative ${sizeClasses.container} flex items-center justify-center`}
      title={metric.description}
      data-testid={`gauge-${metric.label.toLowerCase()}`}
    >
      <svg className="absolute inset-0 transform -rotate-90" viewBox="0 0 100 100">
        {/* Background circle */}
        <circle
          cx="50"
          cy="50"
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth="8"
          className="text-muted/30"
        />
        {/* Progress circle */}
        <circle
          cx="50"
          cy="50"
          r={radius}
          fill="none"
          stroke={metric.color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-500 ease-out"
        />
        {/* Target indicator */}
        {metric.target !== undefined && (
          <circle
            cx="50"
            cy="50"
            r={radius}
            fill="none"
            stroke="white"
            strokeWidth="2"
            strokeDasharray={`2 ${circumference - 2}`}
            strokeDashoffset={circumference - (metric.target * 100 / 100) * circumference}
            opacity={0.5}
          />
        )}
      </svg>
      
      <div className="text-center z-10">
        <span className={`font-bold ${sizeClasses.text}`}>
          {(metric.value * 100).toFixed(0)}
          {metric.unit || '%'}
        </span>
        <p className={`text-muted-foreground ${sizeClasses.label}`}>{metric.label}</p>
      </div>
    </div>
  );
}
