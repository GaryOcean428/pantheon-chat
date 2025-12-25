import { Card, CardContent, CardDescription, CardHeader, CardTitle, Button, Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui";
import { Zap, Moon, Sparkles, RefreshCw } from "lucide-react";
import type { AgencyStatus } from "./types";

interface ForceInterventionCardProps {
  status: AgencyStatus;
  selectedAction: string;
  onActionChange: (action: string) => void;
  onForce: () => void;
  isPending: boolean;
}

export function ForceInterventionCard({ 
  status, 
  selectedAction, 
  onActionChange, 
  onForce, 
  isPending 
}: ForceInterventionCardProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Zap className="h-4 w-4" />
          Force Intervention
        </CardTitle>
        <CardDescription>
          Manually trigger an intervention (bypasses safety checks)
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Select value={selectedAction} onValueChange={onActionChange}>
          <SelectTrigger data-testid="select-intervention-type">
            <SelectValue placeholder="Select intervention" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="ENTER_SLEEP">
              <div className="flex items-center gap-2">
                <Moon className="h-4 w-4" />
                Sleep Cycle
              </div>
            </SelectItem>
            <SelectItem value="ENTER_DREAM">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4" />
                Dream Cycle
              </div>
            </SelectItem>
            <SelectItem value="ENTER_MUSHROOM_MICRO">
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4" />
                Mushroom Microdose
              </div>
            </SelectItem>
            <SelectItem value="ENTER_MUSHROOM_MOD">
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-red-400" />
                Mushroom Moderate
              </div>
            </SelectItem>
          </SelectContent>
        </Select>
        <Button
          className="w-full"
          variant="outline"
          onClick={onForce}
          disabled={isPending || !status?.enabled}
          data-testid="button-force-intervention"
        >
          {isPending ? (
            <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
          ) : (
            <Zap className="h-4 w-4 mr-2" />
          )}
          Trigger Intervention
        </Button>
      </CardContent>
    </Card>
  );
}
