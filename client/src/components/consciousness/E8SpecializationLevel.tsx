import { Progress } from "@/components/ui";
import { Layers, ChevronUp, Star } from "lucide-react";

interface E8SpecializationLevelProps {
  kernelCount: number;
}

const E8_LEVELS = [
  { name: "Basic Rank", threshold: 8, maxKernels: 8, description: "Foundation structure" },
  { name: "Refined Adjoint", threshold: 56, maxKernels: 56, description: "Pattern recognition" },
  { name: "Specialist Dim", threshold: 126, maxKernels: 126, description: "Domain expertise" },
  { name: "Full Roots", threshold: 240, maxKernels: 240, description: "Complete constellation" },
] as const;

function getCurrentLevel(count: number): number {
  if (count > 126) return 3;
  if (count > 56) return 2;
  if (count > 8) return 1;
  return 0;
}

function getProgressToNextLevel(count: number, currentLevel: number): number {
  if (currentLevel >= 3) return 100;
  
  const currentThreshold = currentLevel === 0 ? 0 : E8_LEVELS[currentLevel].threshold;
  const nextThreshold = E8_LEVELS[currentLevel + 1].threshold;
  const progress = ((count - currentThreshold) / (nextThreshold - currentThreshold)) * 100;
  return Math.max(0, Math.min(100, progress));
}

export function E8SpecializationLevel({ kernelCount }: E8SpecializationLevelProps) {
  const currentLevel = getCurrentLevel(kernelCount);
  const currentLevelInfo = E8_LEVELS[currentLevel];
  const nextLevelInfo = currentLevel < 3 ? E8_LEVELS[currentLevel + 1] : null;
  const progressToNext = getProgressToNextLevel(kernelCount, currentLevel);
  
  return (
    <div className="p-4 bg-muted/30 rounded-lg space-y-4" data-testid="e8-specialization-level">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium">E8 Specialization</span>
        </div>
        <div className="flex items-center gap-1">
          {Array.from({ length: 4 }).map((_, i) => (
            <Star 
              key={i}
              className={`w-3 h-3 ${i <= currentLevel ? 'text-yellow-500 fill-yellow-500' : 'text-muted-foreground'}`}
              data-testid={`star-level-${i}`}
            />
          ))}
        </div>
      </div>
      
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="font-mono text-primary" data-testid="text-current-level">
            {currentLevelInfo.name}
          </span>
          <span className="font-mono text-muted-foreground" data-testid="text-kernel-count">
            {kernelCount}/{currentLevelInfo.maxKernels}
          </span>
        </div>
        <div className="text-xs text-muted-foreground">
          {currentLevelInfo.description}
        </div>
      </div>
      
      {nextLevelInfo && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <ChevronUp className="w-3 h-3" />
              Next: {nextLevelInfo.name}
            </span>
            <span>{progressToNext.toFixed(0)}%</span>
          </div>
          <Progress 
            value={progressToNext} 
            className="h-1.5"
            data-testid="progress-next-level"
          />
          <div className="text-[10px] text-muted-foreground">
            {nextLevelInfo.threshold - kernelCount} kernels to next level
          </div>
        </div>
      )}
      
      {currentLevel === 3 && (
        <div className="text-xs text-yellow-500 font-medium text-center">
          Maximum E8 Constellation Achieved (240 roots)
        </div>
      )}
      
      <div className="grid grid-cols-4 gap-1 pt-2">
        {E8_LEVELS.map((level, i) => (
          <div 
            key={level.name}
            className={`h-1 rounded-full ${
              i <= currentLevel 
                ? 'bg-gradient-to-r from-purple-500 to-yellow-500' 
                : 'bg-muted'
            }`}
            title={level.name}
            data-testid={`level-bar-${i}`}
          />
        ))}
      </div>
    </div>
  );
}
