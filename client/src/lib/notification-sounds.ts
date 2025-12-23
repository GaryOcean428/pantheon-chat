/**
 * Notification Sound Utilities
 * 
 * Play sounds for mesh network events and other notifications.
 * Uses Web Audio API for reliable sound playback.
 */

// Sound frequencies for different notification types
const SOUND_CONFIG = {
  peer_connected: {
    frequency: 880, // A5 - high, pleasant
    duration: 150,
    type: 'sine' as OscillatorType,
  },
  peer_disconnected: {
    frequency: 440, // A4 - lower, subtle
    duration: 200,
    type: 'sine' as OscillatorType,
  },
  knowledge_sync: {
    frequency: 660, // E5 - mid-high, success
    duration: 100,
    type: 'sine' as OscillatorType,
    double: true, // Play twice for emphasis
  },
  capability_update: {
    frequency: 550, // C#5 - mid, informational
    duration: 120,
    type: 'triangle' as OscillatorType,
  },
  success: {
    frequency: 800,
    duration: 100,
    type: 'sine' as OscillatorType,
  },
  error: {
    frequency: 300,
    duration: 300,
    type: 'sawtooth' as OscillatorType,
  },
};

type SoundType = keyof typeof SOUND_CONFIG;

let audioContext: AudioContext | null = null;
let soundEnabled = true;

/**
 * Initialize or get the audio context.
 * Must be called after user interaction due to browser autoplay policies.
 */
function getAudioContext(): AudioContext | null {
  if (!audioContext && typeof window !== 'undefined' && window.AudioContext) {
    try {
      audioContext = new AudioContext();
    } catch (e) {
      console.warn('Web Audio API not supported:', e);
      return null;
    }
  }
  return audioContext;
}

/**
 * Play a notification sound.
 */
export function playNotificationSound(type: SoundType): void {
  if (!soundEnabled) return;
  
  const ctx = getAudioContext();
  if (!ctx) return;
  
  const config = SOUND_CONFIG[type];
  if (!config) return;
  
  // Resume context if suspended (browser autoplay policy)
  if (ctx.state === 'suspended') {
    ctx.resume();
  }
  
  const playTone = (delay: number = 0) => {
    const oscillator = ctx.createOscillator();
    const gainNode = ctx.createGain();
    
    oscillator.type = config.type;
    oscillator.frequency.setValueAtTime(config.frequency, ctx.currentTime + delay);
    
    // Envelope for smooth sound
    gainNode.gain.setValueAtTime(0, ctx.currentTime + delay);
    gainNode.gain.linearRampToValueAtTime(0.3, ctx.currentTime + delay + 0.01);
    gainNode.gain.linearRampToValueAtTime(0, ctx.currentTime + delay + config.duration / 1000);
    
    oscillator.connect(gainNode);
    gainNode.connect(ctx.destination);
    
    oscillator.start(ctx.currentTime + delay);
    oscillator.stop(ctx.currentTime + delay + config.duration / 1000);
  };
  
  playTone();
  
  // Play second tone for double sounds
  if ('double' in config && config.double) {
    playTone(0.15);
  }
}

/**
 * Enable or disable notification sounds.
 */
export function setNotificationSoundEnabled(enabled: boolean): void {
  soundEnabled = enabled;
  // Save preference to localStorage
  if (typeof window !== 'undefined') {
    localStorage.setItem('meshNotificationSounds', enabled ? 'true' : 'false');
  }
}

/**
 * Check if notification sounds are enabled.
 */
export function isNotificationSoundEnabled(): boolean {
  // Load preference from localStorage
  if (typeof window !== 'undefined') {
    const saved = localStorage.getItem('meshNotificationSounds');
    if (saved !== null) {
      soundEnabled = saved === 'true';
    }
  }
  return soundEnabled;
}

/**
 * Play sound for mesh network event type.
 */
export function playMeshEventSound(eventType: string): void {
  const soundMap: Record<string, SoundType> = {
    'peer_connected': 'peer_connected',
    'peer_disconnected': 'peer_disconnected',
    'knowledge_sync': 'knowledge_sync',
    'capability_update': 'capability_update',
  };
  
  const soundType = soundMap[eventType];
  if (soundType) {
    playNotificationSound(soundType);
  }
}
