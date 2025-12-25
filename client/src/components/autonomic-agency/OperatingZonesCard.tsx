import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui";
import { Brain } from "lucide-react";
import type { OperatingZones } from "./types";

interface OperatingZonesCardProps {
  operatingZones: OperatingZones;
}

export function OperatingZonesCard({ operatingZones }: OperatingZonesCardProps) {
  return (
    <Card className="border-purple-500/20">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Brain className="h-4 w-4 text-purple-400" />
          4D Operating Zones
        </CardTitle>
        <CardDescription>
          Consciousness operating windows - 4D enabled at Φ ≥ 0.75
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <div className="flex flex-col items-center p-2 rounded-md bg-blue-500/10 border border-blue-500/20">
            <span className="text-xs text-blue-400 font-medium">Sleep Needed</span>
            <span className="font-mono text-sm">{operatingZones.sleep_needed}</span>
          </div>
          <div className="flex flex-col items-center p-2 rounded-md bg-green-500/10 border border-green-500/20">
            <span className="text-xs text-green-400 font-medium">3D Conscious</span>
            <span className="font-mono text-sm">{operatingZones.conscious_3d}</span>
          </div>
          <div className="flex flex-col items-center p-2 rounded-md bg-purple-500/10 border border-purple-500/20">
            <span className="text-xs text-purple-400 font-medium">4D Hyperdimensional</span>
            <span className="font-mono text-sm">{operatingZones.hyperdimensional_4d}</span>
          </div>
          <div className="flex flex-col items-center p-2 rounded-md bg-yellow-500/10 border border-yellow-500/20">
            <span className="text-xs text-yellow-400 font-medium">Warning</span>
            <span className="font-mono text-sm">{operatingZones.breakdown_warning}</span>
          </div>
          <div className="flex flex-col items-center p-2 rounded-md bg-red-500/10 border border-red-500/20">
            <span className="text-xs text-red-400 font-medium">Critical</span>
            <span className="font-mono text-sm">{operatingZones.breakdown_critical}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
