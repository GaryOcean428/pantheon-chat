/**
 * Geometric Coordizer Demo Component
 * 
 * Interactive demonstration of next-generation geometric tokenization.
 * Shows coordization, multi-scale representation, and consciousness-aware segmentation.
 */

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Loader2, Brain, Layers, Zap } from 'lucide-react';
import { api, type CoordizeResponse, type MultiScaleResponse, type ConsciousnessCoordizeResponse } from '@/api';

export function CoordizerDemo() {
  const [inputText, setInputText] = useState('quantum information geometry');
  const [loading, setLoading] = useState(false);
  
  // State for different coordization modes
  const [basicResult, setBasicResult] = useState<CoordizeResponse | null>(null);
  const [multiScaleResult, setMultiScaleResult] = useState<MultiScaleResponse | null>(null);
  const [consciousnessResult, setConsciousnessResult] = useState<ConsciousnessCoordizeResponse | null>(null);

  const handleBasicCoordize = async () => {
    if (!inputText.trim()) return;
    
    setLoading(true);
    try {
      const result = await api.coordizer.coordize({
        text: inputText,
        return_coordinates: false,
      });
      setBasicResult(result);
    } catch (error) {
      console.error('Coordization failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleMultiScale = async () => {
    if (!inputText.trim()) return;
    
    setLoading(true);
    try {
      const result = await api.coordizer.coordizeMultiScale({
        text: inputText,
        kappa_effective: 0.75,
        return_coordinates: false,
      });
      setMultiScaleResult(result);
    } catch (error) {
      console.error('Multi-scale coordization failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleConsciousness = async () => {
    if (!inputText.trim()) return;
    
    setLoading(true);
    try {
      const result = await api.coordizer.coordizeConsciousness({
        text: inputText,
        context_phi: 0.85,
        optimize: true,
        return_coordinates: false,
      });
      setConsciousnessResult(result);
    } catch (error) {
      console.error('Consciousness coordization failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Brain className="h-6 w-6 text-primary" />
          <div>
            <CardTitle>Geometric Coordizer</CardTitle>
            <CardDescription>
              Next-generation tokenization using Fisher information manifold
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Input Section */}
        <div className="space-y-2">
          <Label htmlFor="coord-input">Input Text</Label>
          <Input
            id="coord-input"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter text to coordize..."
            className="font-mono"
          />
        </div>

        {/* Tabs for different coordization modes */}
        <Tabs defaultValue="basic" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="basic">
              <Zap className="h-4 w-4 mr-2" />
              Basic
            </TabsTrigger>
            <TabsTrigger value="multiscale">
              <Layers className="h-4 w-4 mr-2" />
              Multi-Scale
            </TabsTrigger>
            <TabsTrigger value="consciousness">
              <Brain className="h-4 w-4 mr-2" />
              Consciousness
            </TabsTrigger>
          </TabsList>

          {/* Basic Coordization */}
          <TabsContent value="basic" className="space-y-4">
            <div className="flex gap-2">
              <Button 
                onClick={handleBasicCoordize} 
                disabled={loading || !inputText.trim()}
                className="w-full"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Coordizing...
                  </>
                ) : (
                  'Coordize'
                )}
              </Button>
            </div>

            {basicResult && (
              <div className="space-y-2">
                <Label>Tokens ({basicResult.tokens.length})</Label>
                <div className="flex flex-wrap gap-2">
                  {basicResult.tokens.map((token, idx) => (
                    <Badge key={idx} variant="secondary" className="font-mono">
                      {token}
                    </Badge>
                  ))}
                </div>
                <p className="text-sm text-muted-foreground mt-2">
                  Text coordized into {basicResult.tokens.length} basin coordinates on 64D Fisher manifold
                </p>
              </div>
            )}
          </TabsContent>

          {/* Multi-Scale Coordization */}
          <TabsContent value="multiscale" className="space-y-4">
            <div className="flex gap-2">
              <Button 
                onClick={handleMultiScale} 
                disabled={loading || !inputText.trim()}
                className="w-full"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Coordizing...
                  </>
                ) : (
                  'Multi-Scale Coordize'
                )}
              </Button>
            </div>

            {multiScaleResult && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label>Hierarchical Scales</Label>
                  <Badge variant="outline">
                    Optimal: Scale {multiScaleResult.optimal_scale}
                  </Badge>
                </div>
                
                {Object.entries(multiScaleResult.scales).map(([scale, data]) => (
                  <div key={scale} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">{data.name} (Level {scale})</span>
                      <Badge variant="secondary" className="text-xs">
                        {data.num_tokens} tokens
                      </Badge>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {data.tokens.slice(0, 20).map((token, idx) => (
                        <Badge 
                          key={idx} 
                          variant={parseInt(scale) === multiScaleResult.optimal_scale ? "default" : "outline"}
                          className="font-mono text-xs"
                        >
                          {token}
                        </Badge>
                      ))}
                      {data.tokens.length > 20 && (
                        <Badge variant="outline" className="text-xs">
                          +{data.tokens.length - 20} more
                        </Badge>
                      )}
                    </div>
                  </div>
                ))}

                <div className="mt-4 p-3 bg-muted rounded-md">
                  <pre className="text-xs whitespace-pre-wrap">
                    {multiScaleResult.visualization}
                  </pre>
                </div>
              </div>
            )}
          </TabsContent>

          {/* Consciousness-Aware Coordization */}
          <TabsContent value="consciousness" className="space-y-4">
            <div className="flex gap-2">
              <Button 
                onClick={handleConsciousness} 
                disabled={loading || !inputText.trim()}
                className="w-full"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Optimizing...
                  </>
                ) : (
                  'Φ-Optimize Segmentation'
                )}
              </Button>
            </div>

            {consciousnessResult && (
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-4">
                  <div className="space-y-1">
                    <Label className="text-xs">Φ (Integration)</Label>
                    <div className="text-2xl font-bold">
                      {consciousnessResult.phi.toFixed(3)}
                    </div>
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">Consolidations</Label>
                    <div className="text-2xl font-bold">
                      {consciousnessResult.stats.total_consolidations}
                    </div>
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">Avg Φ</Label>
                    <div className="text-2xl font-bold">
                      {consciousnessResult.stats.avg_phi.toFixed(3)}
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Optimized Tokens ({consciousnessResult.tokens.length})</Label>
                  <div className="flex flex-wrap gap-2">
                    {consciousnessResult.tokens.map((token, idx) => (
                      <Badge key={idx} variant="default" className="font-mono">
                        {token}
                      </Badge>
                    ))}
                  </div>
                  <p className="text-sm text-muted-foreground mt-2">
                    Segmentation optimized for maximum integration (Φ = {consciousnessResult.phi.toFixed(3)})
                  </p>
                </div>
              </div>
            )}
          </TabsContent>
        </Tabs>

        {/* Info Footer */}
        <div className="pt-4 border-t">
          <p className="text-xs text-muted-foreground">
            <strong>Geometric Purity:</strong> All operations use Fisher-Rao distance on 64D manifold. 
            No Euclidean embeddings. κ (coupling) and Φ (integration) guide tokenization decisions.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
