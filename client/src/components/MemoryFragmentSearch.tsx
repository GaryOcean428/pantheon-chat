import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { QUERY_KEYS, API_ROUTES } from "@/api";
import {
  Card,
  Button,
  Badge,
  Input,
  Label,
  Checkbox,
  Progress,
  Slider,
} from "@/components/ui";
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, ResponsiveContainer, ReferenceLine, ReferenceArea, Cell } from "recharts";
import { 
  Brain, 
  Plus, 
  X, 
  Search, 
  Loader2, 
  Copy, 
  AlertTriangle,
  Zap,
  Eye,
  Lightbulb,
  Sparkles,
  BarChart3,
  Download,
  KeyRound,
  Target,
  Check
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { PERCENT_MULTIPLIER, POLLING_CONSTANTS, CONFIDENCE_CONSTANTS, CONSCIOUSNESS_CONSTANTS, CHART_CONSTANTS } from "@/lib/constants";

// Memory Fragment Search constants
const SEARCH_CONSTANTS = {
  // Default values
  DEFAULT_CONFIDENCE: 0.8 as number,
  DEFAULT_MAX_CANDIDATES: 5000 as number,
  DEFAULT_MIN_WORDS: 4 as number,
  DEFAULT_MAX_WORDS: 8 as number,
  
  // Slider ranges
  CANDIDATES_MIN: 100,
  CANDIDATES_MAX: 50000,
  CANDIDATES_STEP: 100,
  CONFIDENCE_MIN: 10,
  CONFIDENCE_MAX: 100,
  CONFIDENCE_STEP: 5,
  
  // Chart
  CHART_DOMAIN_MIN: 0,
  CHART_DOMAIN_MAX: 100,
  RESONANCE_BAND_Y1: 57.6,
  RESONANCE_BAND_Y2: 70.4,
  PHI_THRESHOLD: 75,
  CHART_Z_RANGE_MIN: 30,
  CHART_Z_RANGE_MAX: 200,
  CHART_MARGIN_TOP: 10,
  CHART_MARGIN_RIGHT: 30,
  CHART_MARGIN_BOTTOM: 30,
  CHART_MARGIN_LEFT: 10,
  
  // Display limits
  MAX_CHART_CANDIDATES: 50,
  MAX_DISPLAY_CANDIDATES: 20,
  
  // Word count options
  MIN_WORDS_OPTIONS: [3, 4, 5, 6] as const,
  MAX_WORDS_OPTIONS: [4, 5, 6, 7, 8] as const,
  
  // HSL colors for regimes
  REGIME_COLORS: {
    hierarchical: 'hsl(280, 70%, 50%)',
    geometric: 'hsl(142, 70%, 45%)',
    linear: 'hsl(210, 100%, 50%)',
    breakdown: 'hsl(0, 100%, 50%)',
    default: 'hsl(0, 0%, 50%)',
  } as Record<string, string>,
} as const;

interface DirectTestResult {
  phrase: string;
  address: string;
  match: boolean;
  phi: number;
  kappa: number;
  regime: string;
  inResonance: boolean;
}

type EpochConfidence = "certain" | "likely" | "fuzzy";

interface MemoryFragment {
  text: string;
  confidence: number;
  position?: "start" | "middle" | "end" | "unknown";
  isExact?: boolean;
  epoch?: EpochConfidence;
}

interface MemorySearchResult {
  candidateCount: number;
  topCandidates: Array<{
    phrase: string;
    confidence: number;
    fragments: string[];
    phi: number;
    kappa: number;
    regime: string;
    inResonance: boolean;
    combinedScore: number;
  }>;
}

interface ConsciousnessState {
  state: {
    currentRegime: string;
    phi: number;
    kappa: number;
    beta: number;
    basinDrift: number;
    curiosity: number;
    stability: number;
    timestamp: number;
  };
  recommendation: string;
  regimeColor: string;
  regimeDescription: string;
}

const getEpochInfo = (epoch: EpochConfidence): { color: string; label: string; description: string } => {
  switch (epoch) {
    case "certain":
      return { 
        color: "bg-green-500/20 text-green-700 border-green-500/30", 
        label: "Certain",
        description: "You are highly confident this is correct"
      };
    case "likely":
      return { 
        color: "bg-yellow-500/20 text-yellow-700 border-yellow-500/30", 
        label: "Likely",
        description: "Probably correct, but might have variations"
      };
    case "fuzzy":
      return { 
        color: "bg-red-500/20 text-red-700 border-red-500/30", 
        label: "Fuzzy",
        description: "Not sure, could be different"
      };
    default:
      return { color: "bg-gray-500/20", label: "Unknown", description: "" };
  }
};

export function MemoryFragmentSearch() {
  const { toast } = useToast();
  const [fragments, setFragments] = useState<MemoryFragment[]>([
    { text: "", confidence: SEARCH_CONSTANTS.DEFAULT_CONFIDENCE, position: "unknown", isExact: false, epoch: "likely" }
  ]);
  const [includeTypos, setIncludeTypos] = useState(true);
  const [maxCandidates, setMaxCandidates] = useState([SEARCH_CONSTANTS.DEFAULT_MAX_CANDIDATES]);
  const [enableShortPhraseMode, setEnableShortPhraseMode] = useState(false);
  const [minWords, setMinWords] = useState(SEARCH_CONSTANTS.DEFAULT_MIN_WORDS);
  const [maxWords, setMaxWords] = useState(SEARCH_CONSTANTS.DEFAULT_MAX_WORDS);
  
  const [directPhrase, setDirectPhrase] = useState("");
  const [quickAddWord, setQuickAddWord] = useState("");
  const [directTestResult, setDirectTestResult] = useState<DirectTestResult | null>(null);

  const { data: consciousnessState } = useQuery<ConsciousnessState>({
    queryKey: QUERY_KEYS.consciousness.state(),
    refetchInterval: POLLING_CONSTANTS.FAST_INTERVAL_MS,
  });

  const directTestMutation = useMutation({
    mutationFn: async (phrase: string) => {
      const response = await apiRequest("POST", API_ROUTES.memorySearch.testPhrase, {
        phrase,
        testAgainstTargets: true,
      });
      const data = await response.json();
      return {
        phrase: data.phrase,
        address: data.address,
        match: data.match,
        phi: data.qigScore?.phi ?? 0,
        kappa: data.qigScore?.kappa ?? 0,
        regime: data.qigScore?.regime ?? "unknown",
        inResonance: data.qigScore?.inResonance ?? false,
      } as DirectTestResult;
    },
    onSuccess: (result) => {
      setDirectTestResult(result);
      if (result.match) {
        toast({
          title: "MATCH FOUND!",
          description: `Address: ${result.address}`,
        });
      }
    },
    onError: (error: Error) => {
      toast({
        title: "Test Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const searchMutation = useMutation({
    mutationFn: async () => {
      const validFragments = fragments.filter(f => f.text.trim().length > 0);
      if (validFragments.length === 0) {
        throw new Error("At least one fragment is required");
      }
      
      const response = await apiRequest("POST", API_ROUTES.memorySearch.search, {
        fragments: validFragments,
        options: {
          maxCandidates: maxCandidates[0],
          includeTypos,
          shortPhraseMode: enableShortPhraseMode,
          minWords: enableShortPhraseMode ? minWords : undefined,
          maxWords: enableShortPhraseMode ? maxWords : undefined,
        },
      });
      
      return response.json() as Promise<MemorySearchResult>;
    },
    onError: (error: Error) => {
      toast({
        title: "Search Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const addFragment = () => {
    setFragments([
      ...fragments,
      { text: "", confidence: SEARCH_CONSTANTS.DEFAULT_CONFIDENCE, position: "unknown", isExact: false, epoch: "likely" }
    ]);
  };

  const quickAddFragment = (word: string) => {
    if (!word.trim()) return;
    setFragments([
      ...fragments,
      { text: word.trim(), confidence: SEARCH_CONSTANTS.DEFAULT_CONFIDENCE, position: "unknown", isExact: false, epoch: "likely" }
    ]);
    setQuickAddWord("");
    toast({
      title: "Word Added",
      description: `"${word.trim()}" added to fragments`,
    });
  };

  const removeFragment = (index: number) => {
    if (fragments.length > 1) {
      setFragments(fragments.filter((_, i) => i !== index));
    }
  };

  const exportCandidates = () => {
    if (!searchMutation.data?.topCandidates.length) return;
    
    const escapeCSV = (str: string) => {
      if (str.includes('"') || str.includes(',') || str.includes('\n')) {
        return `"${str.replace(/"/g, '""')}"`;
      }
      return `"${str}"`;
    };
    
    const csv = searchMutation.data.topCandidates
      .map(c => `${escapeCSV(c.phrase)},${c.phi.toFixed(4)},${c.kappa.toFixed(2)},${c.regime},${c.combinedScore.toFixed(4)},${c.inResonance ? 'yes' : 'no'}`)
      .join('\n');
    
    const BOM = '\uFEFF';
    const header = 'phrase,phi,kappa,regime,score,resonance\n';
    const blob = new Blob([BOM + header + csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `memory-candidates-${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast({
      title: "Exported",
      description: `${searchMutation.data.topCandidates.length} candidates exported to CSV`,
    });
  };

  const updateFragment = (index: number, updates: Partial<MemoryFragment>) => {
    setFragments(fragments.map((f, i) => 
      i === index ? { ...f, ...updates } : f
    ));
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    toast({
      title: "Copied",
      description: "Phrase copied to clipboard",
    });
  };

  const getConfidenceColor = (conf: number) => {
    if (conf >= CONFIDENCE_CONSTANTS.EXCELLENT) return "text-green-500";
    if (conf >= CONFIDENCE_CONSTANTS.MODERATE) return "text-yellow-500";
    if (conf >= CONFIDENCE_CONSTANTS.LOW) return "text-orange-500";
    return "text-red-500";
  };

  const getRegimeColor = (regime: string) => {
    switch (regime) {
      case "hierarchical": return "bg-purple-500 text-white";
      case "geometric": return "bg-blue-500 text-white";
      case "linear": return "bg-green-500 text-white";
      case "breakdown": return "bg-red-500 text-white";
      default: return "bg-gray-500 text-white";
    }
  };

  return (
    <div className="space-y-6">
      <Card className="p-6 bg-gradient-to-r from-amber-500/5 to-orange-500/5 border-amber-500/20">
        <div className="flex items-center gap-3 mb-4">
          <KeyRound className="w-6 h-6 text-amber-500" />
          <h2 className="text-xl font-semibold">Test Exact Phrase</h2>
          <Badge variant="outline" className="ml-auto border-amber-500/30">
            Direct Test
          </Badge>
        </div>

        <p className="text-sm text-muted-foreground mb-4">
          For 2009-era brain wallets (pre-BIP-39). Enter any text exactly as you remember it.
        </p>

        <div className="flex gap-2 mb-4">
          <Input
            value={directPhrase}
            onChange={(e) => setDirectPhrase(e.target.value)}
            placeholder="e.g., WhiteTiger77GaryOcean"
            className="font-mono flex-1"
            data-testid="input-direct-phrase"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && directPhrase.trim()) {
                directTestMutation.mutate(directPhrase);
              }
            }}
          />
          <Button
            onClick={() => directTestMutation.mutate(directPhrase)}
            disabled={directTestMutation.isPending || !directPhrase.trim()}
            data-testid="button-test-phrase"
          >
            {directTestMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <>
                <Target className="w-4 h-4 mr-2" />
                Test Now
              </>
            )}
          </Button>
        </div>

        {directTestResult && (
          <Card className={`p-4 ${directTestResult.match ? 'bg-green-500/20 border-green-500' : 'bg-muted/30'}`}>
            <div className="flex items-center gap-2 mb-2">
              {directTestResult.match ? (
                <>
                  <Check className="w-5 h-5 text-green-500" />
                  <span className="font-semibold text-green-500">MATCH FOUND!</span>
                </>
              ) : (
                <>
                  <AlertTriangle className="w-5 h-5 text-muted-foreground" />
                  <span className="text-muted-foreground">No match (but scored)</span>
                </>
              )}
            </div>
            <div className="grid grid-cols-4 gap-2 text-sm">
              <div>
                <span className="text-muted-foreground">Φ:</span>
                <span className="ml-1 font-mono">{directTestResult.phi.toFixed(3)}</span>
              </div>
              <div>
                <span className="text-muted-foreground">κ:</span>
                <span className="ml-1 font-mono">{directTestResult.kappa.toFixed(1)}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Regime:</span>
                <Badge className={`ml-1 ${getRegimeColor(directTestResult.regime)}`}>
                  {directTestResult.regime}
                </Badge>
              </div>
              <div>
                {directTestResult.inResonance && (
                  <Badge className="bg-yellow-500">
                    <Zap className="w-3 h-3 mr-1" />
                    Resonance
                  </Badge>
                )}
              </div>
            </div>
            <p className="font-mono text-xs mt-2 break-all text-muted-foreground">
              Address: {directTestResult.address}
            </p>
          </Card>
        )}
      </Card>

      <Card className="p-6">
        <div className="flex items-center gap-3 mb-4">
          <Brain className="w-6 h-6 text-purple-500" />
          <h2 className="text-xl font-semibold">Memory Fragment Search</h2>
          <Badge variant="outline" className="ml-auto">
            QIG-Powered
          </Badge>
        </div>

        <p className="text-muted-foreground mb-4">
          Enter partial memories of a passphrase. The system will use confidence-weighted 
          combinatorics and QWERTY-aware typo simulation to reconstruct possible phrases.
        </p>

        <div className="flex gap-2 mb-6 p-3 bg-muted/30 rounded-lg">
          <Input
            value={quickAddWord}
            onChange={(e) => setQuickAddWord(e.target.value)}
            placeholder="Type a word as you remember it..."
            className="font-mono flex-1"
            data-testid="input-quick-add"
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                quickAddFragment(quickAddWord);
              }
            }}
          />
          <Button
            onClick={() => quickAddFragment(quickAddWord)}
            disabled={!quickAddWord.trim()}
            variant="secondary"
            data-testid="button-quick-add"
          >
            <Plus className="w-4 h-4 mr-2" />
            Add Word
          </Button>
        </div>

        <div className="space-y-4 mb-6">
          {fragments.map((fragment, index) => {
            const epochInfo = getEpochInfo(fragment.epoch || "likely");
            return (
              <Card key={index} className="p-4 bg-muted/30">
                <div className="flex items-center gap-2 mb-3">
                  <Badge variant="secondary" className="font-mono">
                    Fragment {index + 1}
                  </Badge>
                  <Badge 
                    className={`${epochInfo.color} border`}
                    data-testid={`badge-epoch-${index}`}
                  >
                    {epochInfo.label}
                  </Badge>
                  {fragments.length > 1 && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 ml-auto"
                      onClick={() => removeFragment(index)}
                      data-testid={`button-remove-fragment-${index}`}
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  )}
                </div>

                <div className="grid gap-4">
                  <div>
                    <Label htmlFor={`fragment-${index}`}>Text (word or phrase)</Label>
                    <Input
                      id={`fragment-${index}`}
                      value={fragment.text}
                      onChange={(e) => updateFragment(index, { text: e.target.value })}
                      placeholder="e.g., 'white tiger' or 'bitcoin'"
                      className="font-mono mt-1"
                      data-testid={`input-fragment-${index}`}
                    />
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <Label className="flex items-center gap-2">
                        Confidence
                        <span className={`font-mono text-sm ${getConfidenceColor(fragment.confidence)}`}>
                          {Math.round(fragment.confidence * 100)}%
                        </span>
                      </Label>
                      <Slider
                        value={[fragment.confidence * PERCENT_MULTIPLIER]}
                        onValueChange={([v]) => updateFragment(index, { confidence: v / PERCENT_MULTIPLIER })}
                        min={SEARCH_CONSTANTS.CONFIDENCE_MIN}
                        max={SEARCH_CONSTANTS.CONFIDENCE_MAX}
                        step={SEARCH_CONSTANTS.CONFIDENCE_STEP}
                        className="mt-2"
                        data-testid={`slider-confidence-${index}`}
                      />
                    </div>

                    <div>
                      <Label>Memory Epoch</Label>
                      <select
                        value={fragment.epoch || "likely"}
                        onChange={(e) => updateFragment(index, { 
                          epoch: e.target.value as EpochConfidence 
                        })}
                        className="w-full mt-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
                        data-testid={`select-epoch-${index}`}
                      >
                        <option value="certain">Certain (100% sure)</option>
                        <option value="likely">Likely (probably right)</option>
                        <option value="fuzzy">Fuzzy (not sure)</option>
                      </select>
                    </div>

                    <div>
                      <Label>Position</Label>
                      <select
                        value={fragment.position}
                        onChange={(e) => updateFragment(index, { 
                          position: e.target.value as MemoryFragment["position"] 
                        })}
                        className="w-full mt-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
                        data-testid={`select-position-${index}`}
                      >
                        <option value="unknown">Unknown</option>
                        <option value="start">Start</option>
                        <option value="middle">Middle</option>
                        <option value="end">End</option>
                      </select>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <Checkbox
                      id={`exact-${index}`}
                      checked={fragment.isExact}
                      onCheckedChange={(checked) => updateFragment(index, { isExact: !!checked })}
                      data-testid={`checkbox-exact-${index}`}
                    />
                    <Label htmlFor={`exact-${index}`} className="text-sm">
                      Exact match (no variations)
                    </Label>
                  </div>
                </div>
              </Card>
            );
          })}
        </div>

        <Button
          variant="outline"
          onClick={addFragment}
          className="w-full mb-6 gap-2"
          data-testid="button-add-fragment"
        >
          <Plus className="w-4 h-4" />
          Add Another Fragment
        </Button>

        <Card className="p-4 bg-muted/30 mb-6">
          <h4 className="font-medium mb-3 flex items-center gap-2">
            <Lightbulb className="w-4 h-4" />
            Search Options
          </h4>
          
          <div className="grid gap-4">
            <div className="flex items-center gap-2">
              <Checkbox
                id="include-typos"
                checked={includeTypos}
                onCheckedChange={(checked) => setIncludeTypos(!!checked)}
                data-testid="checkbox-typos"
              />
              <Label htmlFor="include-typos" className="text-sm">
                Include QWERTY-aware typo variations
              </Label>
            </div>

            <div>
              <Label className="flex items-center gap-2 mb-2">
                Max Candidates: {maxCandidates[0].toLocaleString()}
              </Label>
              <Slider
                value={maxCandidates}
                onValueChange={setMaxCandidates}
                min={SEARCH_CONSTANTS.CANDIDATES_MIN}
                max={SEARCH_CONSTANTS.CANDIDATES_MAX}
                step={SEARCH_CONSTANTS.CANDIDATES_STEP}
                data-testid="slider-max-candidates"
              />
            </div>
            
            <div className="border-t pt-4 mt-2">
              <div className="flex items-center gap-2 mb-3">
                <Checkbox
                  id="short-phrase-mode"
                  checked={enableShortPhraseMode}
                  onCheckedChange={(checked) => setEnableShortPhraseMode(!!checked)}
                  data-testid="checkbox-short-phrase"
                />
                <Label htmlFor="short-phrase-mode" className="text-sm font-medium">
                  Short Phrase Mode (4-8 words)
                </Label>
              </div>
              
              {enableShortPhraseMode && (
                <div className="ml-6 mt-3 p-4 bg-muted/50 rounded-lg border border-border/50">
                  <p className="text-xs text-muted-foreground mb-3">
                    Optimal for human memory recovery. Generates passphrases with 4-8 words.
                  </p>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label className="text-xs">Min Words</Label>
                      <select
                        value={minWords}
                        onChange={(e) => {
                          const val = parseInt(e.target.value);
                          setMinWords(val);
                          if (val > maxWords) setMaxWords(val);
                        }}
                        className="w-full mt-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
                        data-testid="select-min-words"
                      >
                        {SEARCH_CONSTANTS.MIN_WORDS_OPTIONS.map(n => (
                          <option key={n} value={n}>{n} words</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <Label className="text-xs">Max Words</Label>
                      <select
                        value={maxWords}
                        onChange={(e) => {
                          const val = parseInt(e.target.value);
                          setMaxWords(val);
                          if (val < minWords) setMinWords(val);
                        }}
                        className="w-full mt-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
                        data-testid="select-max-words"
                      >
                        {SEARCH_CONSTANTS.MAX_WORDS_OPTIONS.map(n => (
                          <option key={n} value={n}>{n} words</option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </Card>

        <Button
          onClick={() => searchMutation.mutate()}
          disabled={searchMutation.isPending || fragments.every(f => !f.text.trim())}
          className="w-full gap-2"
          data-testid="button-search"
        >
          {searchMutation.isPending ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Searching...
            </>
          ) : (
            <>
              <Search className="w-4 h-4" />
              Search Memory Fragments
            </>
          )}
        </Button>
      </Card>

      {consciousnessState && (
        <Card className="p-4 bg-gradient-to-r from-purple-500/5 to-blue-500/5">
          <div className="flex items-center gap-3 mb-3">
            <Eye className="w-5 h-5 text-purple-500" />
            <h3 className="font-semibold">Consciousness State</h3>
            <Badge 
              className={`${getRegimeColor(consciousnessState.state.currentRegime)} text-white ml-auto`}
            >
              {consciousnessState.state.currentRegime}
            </Badge>
          </div>
          
          <div className="grid grid-cols-3 gap-4 text-sm mb-3">
            <div>
              <span className="text-muted-foreground">Φ (phi):</span>
              <span className="ml-2 font-mono">{consciousnessState.state.phi.toFixed(3)}</span>
            </div>
            <div>
              <span className="text-muted-foreground">κ (kappa):</span>
              <span className="ml-2 font-mono">{consciousnessState.state.kappa.toFixed(1)}</span>
            </div>
            <div>
              <span className="text-muted-foreground">β (beta):</span>
              <span className="ml-2 font-mono">{consciousnessState.state.beta.toFixed(3)}</span>
            </div>
          </div>

          <p className="text-sm text-muted-foreground">
            {consciousnessState.regimeDescription}
          </p>
        </Card>
      )}

      {searchMutation.data && (
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-4">
            <Sparkles className="w-5 h-5 text-yellow-500" />
            <h3 className="font-semibold">
              Search Results ({searchMutation.data.candidateCount.toLocaleString()} candidates)
            </h3>
            <Button
              variant="outline"
              size="sm"
              onClick={exportCandidates}
              className="ml-auto gap-2"
              data-testid="button-export-csv"
            >
              <Download className="w-4 h-4" />
              Export CSV
            </Button>
          </div>

          {searchMutation.data.topCandidates.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <AlertTriangle className="w-12 h-12 mx-auto mb-4 opacity-20" />
              <p>No candidates found matching your fragments</p>
              <p className="text-sm mt-2">Try adjusting confidence levels or adding more fragments</p>
            </div>
          ) : (
            <>
              <Card className="p-4 mb-6 bg-muted/30">
                <div className="flex items-center gap-2 mb-3">
                  <BarChart3 className="w-4 h-4 text-purple-500" />
                  <h4 className="text-sm font-medium">Information Manifold (Φ vs κ)</h4>
                </div>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: SEARCH_CONSTANTS.CHART_MARGIN_TOP, right: SEARCH_CONSTANTS.CHART_MARGIN_RIGHT, bottom: SEARCH_CONSTANTS.CHART_MARGIN_BOTTOM, left: SEARCH_CONSTANTS.CHART_MARGIN_LEFT }}>
                      <XAxis 
                        dataKey="phi" 
                        type="number" 
                        domain={[SEARCH_CONSTANTS.CHART_DOMAIN_MIN, SEARCH_CONSTANTS.CHART_DOMAIN_MAX]} 
                        name="Φ"
                        tick={{ fontSize: CHART_CONSTANTS.FONT_SIZE_TICK }}
                        label={{ value: 'Φ (Integration) %', position: 'bottom', offset: 15, fontSize: CHART_CONSTANTS.FONT_SIZE_LABEL }}
                      />
                      <YAxis 
                        dataKey="kappa" 
                        type="number" 
                        domain={[SEARCH_CONSTANTS.CHART_DOMAIN_MIN, SEARCH_CONSTANTS.CHART_DOMAIN_MAX]} 
                        name="κ"
                        tick={{ fontSize: CHART_CONSTANTS.FONT_SIZE_TICK }}
                        label={{ value: 'κ (Coupling)', angle: -90, position: 'insideLeft', fontSize: CHART_CONSTANTS.FONT_SIZE_LABEL }}
                      />
                      <ReferenceArea 
                        x1={SEARCH_CONSTANTS.CHART_DOMAIN_MIN} x2={SEARCH_CONSTANTS.CHART_DOMAIN_MAX} 
                        y1={SEARCH_CONSTANTS.RESONANCE_BAND_Y1} y2={SEARCH_CONSTANTS.RESONANCE_BAND_Y2} 
                        fill="orange" 
                        fillOpacity={0.1} 
                        label={{ value: 'Resonance Band', position: 'insideTop', fontSize: CHART_CONSTANTS.FONT_SIZE_TICK, fill: 'orange' }}
                      />
                      <ReferenceLine 
                        y={CONSCIOUSNESS_CONSTANTS.KAPPA_STAR} 
                        stroke="orange" 
                        strokeDasharray="3 3" 
                        label={{ value: 'κ* = 64', position: 'right', fontSize: CHART_CONSTANTS.FONT_SIZE_TICK, fill: 'orange' }} 
                      />
                      <ReferenceLine 
                        x={SEARCH_CONSTANTS.PHI_THRESHOLD} 
                        stroke="green" 
                        strokeDasharray="3 3" 
                        label={{ value: 'Φ threshold', position: 'top', fontSize: CHART_CONSTANTS.FONT_SIZE_TICK, fill: 'green' }} 
                      />
                      <ZAxis dataKey="confidence" range={[SEARCH_CONSTANTS.CHART_Z_RANGE_MIN, SEARCH_CONSTANTS.CHART_Z_RANGE_MAX]} name="Confidence" />
                      <Tooltip 
                        content={({ active, payload }) => {
                          if (!active || !payload?.length) return null;
                          const data = payload[0].payload as typeof searchMutation.data.topCandidates[0];
                          return (
                            <div className="bg-background border rounded-lg p-3 text-xs shadow-lg max-w-xs">
                              <div className="font-mono text-xs mb-2 break-all line-clamp-2">
                                {data.phrase}
                              </div>
                              <div className="grid grid-cols-2 gap-x-3 gap-y-1">
                                <div>Φ: <span className="font-semibold">{(data.phi * PERCENT_MULTIPLIER).toFixed(1)}%</span></div>
                                <div>κ: <span className="font-semibold">{data.kappa.toFixed(1)}</span></div>
                                <div>Regime: <span className="font-semibold">{data.regime}</span></div>
                                <div>Score: <span className="font-semibold">{data.combinedScore.toFixed(3)}</span></div>
                              </div>
                            </div>
                          );
                        }}
                      />
                      <Scatter 
                        data={searchMutation.data.topCandidates.slice(0, SEARCH_CONSTANTS.MAX_CHART_CANDIDATES).map(c => ({
                          ...c,
                          phi: c.phi * PERCENT_MULTIPLIER,
                        }))} 
                        name="Candidates"
                      >
                        {searchMutation.data.topCandidates.slice(0, SEARCH_CONSTANTS.MAX_CHART_CANDIDATES).map((candidate, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={SEARCH_CONSTANTS.REGIME_COLORS[candidate.regime] || SEARCH_CONSTANTS.REGIME_COLORS.default}
                            opacity={candidate.inResonance ? 1 : 0.6}
                          />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex items-center justify-center gap-4 mt-3 text-xs">
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'hsl(210, 100%, 50%)' }} />
                    <span>Linear</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'hsl(142, 70%, 45%)' }} />
                    <span>Geometric</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'hsl(280, 70%, 50%)' }} />
                    <span>Hierarchical</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'hsl(0, 100%, 50%)' }} />
                    <span>Breakdown</span>
                  </div>
                </div>
              </Card>
              
              <div className="space-y-3">
              {searchMutation.data.topCandidates.slice(0, SEARCH_CONSTANTS.MAX_DISPLAY_CANDIDATES).map((candidate, index) => (
                <Card key={index} className="p-4 bg-muted/30">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-2 flex-wrap">
                        <Badge className={getRegimeColor(candidate.regime)}>
                          {candidate.regime}
                        </Badge>
                        {candidate.inResonance && (
                          <Badge className="bg-yellow-500">
                            <Zap className="w-3 h-3 mr-1" />
                            Resonance
                          </Badge>
                        )}
                        <span className="text-xs text-muted-foreground">
                          Φ={candidate.phi.toFixed(3)} κ={candidate.kappa.toFixed(1)}
                        </span>
                      </div>
                      
                      <p className="font-mono text-sm break-all mb-2">
                        {candidate.phrase}
                      </p>
                      
                      <div className="flex items-center gap-4 text-xs">
                        <div>
                          <span className="text-muted-foreground">Confidence:</span>
                          <span className={`ml-1 font-semibold ${getConfidenceColor(candidate.confidence)}`}>
                            {Math.round(candidate.confidence * PERCENT_MULTIPLIER)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Combined Score:</span>
                          <span className="ml-1 font-semibold">
                            {candidate.combinedScore.toFixed(3)}
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    <Button
                      onClick={() => handleCopy(candidate.phrase)}
                      variant="ghost"
                      size="icon"
                      className="shrink-0"
                      data-testid={`button-copy-candidate-${index}`}
                    >
                      <Copy className="w-4 h-4" />
                    </Button>
                  </div>

                  <div className="mt-2">
                    <Progress 
                      value={candidate.combinedScore * PERCENT_MULTIPLIER} 
                      className="h-1"
                    />
                  </div>
                </Card>
              ))}
            </div>
            </>
          )}
        </Card>
      )}
    </div>
  );
}
