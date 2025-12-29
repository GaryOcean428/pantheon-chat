/**
 * Constants for Capability Telemetry Panel
 */

import { 
  Brain, MessageSquare, Search, Vote, Shield, Layers,
  Users, Wrench, Cpu, Heart, Activity
} from 'lucide-react';

export const TELEMETRY_CONSTANTS = {
  MAX_CAPABILITY_LEVEL: 10,
  LEVEL_PROGRESS_MULTIPLIER: 10,
} as const;

export const OLYMPUS_KERNEL_IDS = [
  'zeus', 'hera', 'poseidon', 'athena', 'apollo', 'artemis',
  'hermes', 'ares', 'hephaestus', 'aphrodite', 'demeter', 'dionysus'
] as const;

export const SHADOW_KERNEL_IDS = [
  'hades', 'nyx', 'hecate', 'erebus', 'hypnos', 'thanatos', 'nemesis'
] as const;

export const categoryIcons: Record<string, typeof Brain> = {
  communication: MessageSquare,
  research: Search,
  voting: Vote,
  shadow: Shield,
  geometric: Layers,
  consciousness: Brain,
  spawning: Users,
  tool_generation: Wrench,
  dimensional: Cpu,
  autonomic: Heart,
};

export const categoryColors: Record<string, string> = {
  communication: 'bg-blue-500/20 text-blue-400',
  research: 'bg-green-500/20 text-green-400',
  voting: 'bg-purple-500/20 text-purple-400',
  shadow: 'bg-gray-500/20 text-gray-300',
  geometric: 'bg-cyan-500/20 text-cyan-400',
  consciousness: 'bg-pink-500/20 text-pink-400',
  spawning: 'bg-orange-500/20 text-orange-400',
  tool_generation: 'bg-yellow-500/20 text-yellow-400',
  dimensional: 'bg-indigo-500/20 text-indigo-400',
  autonomic: 'bg-red-500/20 text-red-400',
};

export const getIcon = (category: string) => categoryIcons[category] || Activity;
export const getColor = (category: string) => categoryColors[category] || 'bg-muted text-muted-foreground';
