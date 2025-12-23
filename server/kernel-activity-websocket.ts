/**
 * WebSocket Kernel Activity Streaming
 * 
 * Provides real-time streaming of kernel activity (inter-god communications,
 * debates, discoveries, etc.) to connected clients.
 * 
 * Usage:
 *   Client connects to: ws://localhost:5000/ws/kernel-activity
 *   Client sends: { "type": "subscribe", "filters": { "activityTypes": ["debate", "insight"] } }
 *   Server pushes: { "type": "activity", "data": {...} }
 */

import { WebSocket } from "ws";

export interface KernelActivityItem {
  id: string;
  type: string;
  from: string;
  to: string;
  content: string;
  timestamp: string;
  read: boolean;
  responded: boolean;
  metadata?: Record<string, unknown>;
}

interface ActivitySubscription {
  ws: WebSocket;
  filters?: {
    activityTypes?: string[];
    fromKernels?: string[];
    toKernels?: string[];
  };
  lastEventId: string | null;
}

interface ActivityMessage {
  type: "subscribe" | "unsubscribe" | "heartbeat";
  filters?: {
    activityTypes?: string[];
    fromKernels?: string[];
    toKernels?: string[];
  };
}

interface ActivityUpdate {
  type: "activity" | "activity_batch" | "status" | "subscribed" | "error";
  data?: KernelActivityItem | KernelActivityItem[];
  status?: {
    total_messages: number;
    active_debates: number;
    resolved_debates: number;
    knowledge_transfers: number;
  };
  message?: string;
}

export class KernelActivityStreamer {
  private subscriptions = new Map<string, ActivitySubscription>();
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private activityBuffer: KernelActivityItem[] = [];
  private readonly MAX_BUFFER_SIZE = 100;
  private pollInterval: NodeJS.Timeout | null = null;
  
  constructor() {
    this.startHeartbeat();
    this.startActivityPolling();
  }
  
  /**
   * Start heartbeat to keep connections alive
   */
  private startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      for (const [clientId, sub] of this.subscriptions) {
        if (sub.ws.readyState === WebSocket.OPEN) {
          try {
            sub.ws.ping();
          } catch (err) {
            console.error(`[KernelActivityWS] Heartbeat failed for ${clientId}:`, err);
          }
        }
      }
    }, 30000); // 30 seconds
  }
  
  /**
   * Start polling Python backend for new activity
   * This fetches from the backend and pushes to all WebSocket clients
   */
  private startActivityPolling() {
    // Poll backend every 2 seconds for new activity
    this.pollInterval = setInterval(async () => {
      await this.fetchAndBroadcastActivity();
    }, 2000);
    
    // Initial fetch
    this.fetchAndBroadcastActivity();
  }
  
  /**
   * Fetch activity from Python backend and broadcast to subscribers
   */
  private async fetchAndBroadcastActivity() {
    if (this.subscriptions.size === 0) return;
    
    try {
      const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      const response = await fetch(`${backendUrl}/olympus/pantheon/activity?limit=50`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(5000),
      });
      
      if (!response.ok) {
        return;
      }
      
      const data = await response.json();
      const activities: KernelActivityItem[] = data.activity || [];
      
      // Find new activities not in buffer
      const newActivities = activities.filter(a => 
        !this.activityBuffer.some(b => b.id === a.id)
      );
      
      if (newActivities.length > 0) {
        // Update buffer
        this.activityBuffer = [...newActivities, ...this.activityBuffer].slice(0, this.MAX_BUFFER_SIZE);
        
        // Broadcast new activities to all subscribers
        this.broadcastActivity(newActivities, data.status);
      }
    } catch (err) {
      // Silently ignore fetch errors - backend may be unavailable
    }
  }
  
  /**
   * Broadcast activity to all subscribers based on their filters
   */
  private broadcastActivity(activities: KernelActivityItem[], status?: ActivityUpdate['status']) {
    for (const [clientId, sub] of this.subscriptions) {
      if (sub.ws.readyState !== WebSocket.OPEN) continue;
      
      // Apply filters
      let filteredActivities = activities;
      
      if (sub.filters?.activityTypes?.length) {
        filteredActivities = filteredActivities.filter(a => 
          sub.filters!.activityTypes!.includes(a.type)
        );
      }
      
      if (sub.filters?.fromKernels?.length) {
        filteredActivities = filteredActivities.filter(a => 
          sub.filters!.fromKernels!.includes(a.from.toLowerCase())
        );
      }
      
      if (sub.filters?.toKernels?.length) {
        filteredActivities = filteredActivities.filter(a => 
          sub.filters!.toKernels!.includes(a.to.toLowerCase())
        );
      }
      
      if (filteredActivities.length === 0) continue;
      
      try {
        const update: ActivityUpdate = {
          type: filteredActivities.length === 1 ? "activity" : "activity_batch",
          data: filteredActivities.length === 1 ? filteredActivities[0] : filteredActivities,
          status,
        };
        sub.ws.send(JSON.stringify(update));
        
        // Update last event ID
        sub.lastEventId = filteredActivities[0].id;
      } catch (err) {
        console.error(`[KernelActivityWS] Broadcast failed for ${clientId}:`, err);
      }
    }
  }
  
  /**
   * Manually push activity (called from other parts of the system)
   */
  pushActivity(activity: KernelActivityItem) {
    // Add to buffer
    this.activityBuffer = [activity, ...this.activityBuffer].slice(0, this.MAX_BUFFER_SIZE);
    
    // Broadcast immediately
    this.broadcastActivity([activity]);
  }
  
  /**
   * Handle new WebSocket connection
   */
  handleConnection(ws: WebSocket, clientId: string) {
    console.log(`[KernelActivityWS] New connection: ${clientId}`);
    
    this.subscriptions.set(clientId, {
      ws,
      filters: undefined,
      lastEventId: null,
    });
    
    ws.on("message", (data) => {
      try {
        const message: ActivityMessage = JSON.parse(data.toString());
        
        if (message.type === "subscribe") {
          this.handleSubscribe(clientId, message.filters);
        } else if (message.type === "unsubscribe") {
          this.handleUnsubscribe(clientId);
        } else if (message.type === "heartbeat") {
          // Client heartbeat - just acknowledge
        }
      } catch (err) {
        console.error(`[KernelActivityWS] Message parse error for ${clientId}:`, err);
        
        const errorUpdate: ActivityUpdate = {
          type: "error",
          message: "Invalid message format",
        };
        ws.send(JSON.stringify(errorUpdate));
      }
    });
    
    ws.on("close", () => {
      console.log(`[KernelActivityWS] Connection closed: ${clientId}`);
      this.subscriptions.delete(clientId);
    });
    
    ws.on("error", (err) => {
      console.error(`[KernelActivityWS] Error for ${clientId}:`, err);
    });
    
    ws.on("pong", () => {
      // Heartbeat received
    });
  }
  
  /**
   * Handle subscription request
   */
  private handleSubscribe(clientId: string, filters?: ActivityMessage['filters']) {
    const sub = this.subscriptions.get(clientId);
    if (!sub) return;
    
    sub.filters = filters;
    sub.lastEventId = null;
    
    console.log(`[KernelActivityWS] Client ${clientId} subscribed with filters:`, filters || 'all');
    
    const response: ActivityUpdate = {
      type: "subscribed",
      message: `Subscribed to kernel activity${filters ? ' with filters' : ''}`,
    };
    
    if (sub.ws.readyState === WebSocket.OPEN) {
      sub.ws.send(JSON.stringify(response));
      
      // Send buffered activities immediately
      if (this.activityBuffer.length > 0) {
        let activities = this.activityBuffer;
        
        // Apply filters to buffer
        if (filters?.activityTypes?.length) {
          activities = activities.filter(a => filters.activityTypes!.includes(a.type));
        }
        if (filters?.fromKernels?.length) {
          activities = activities.filter(a => filters.fromKernels!.includes(a.from.toLowerCase()));
        }
        if (filters?.toKernels?.length) {
          activities = activities.filter(a => filters.toKernels!.includes(a.to.toLowerCase()));
        }
        
        if (activities.length > 0) {
          const batchUpdate: ActivityUpdate = {
            type: "activity_batch",
            data: activities,
          };
          sub.ws.send(JSON.stringify(batchUpdate));
        }
      }
    }
  }
  
  /**
   * Handle unsubscribe request
   */
  private handleUnsubscribe(clientId: string) {
    const sub = this.subscriptions.get(clientId);
    if (!sub) return;
    
    sub.filters = undefined;
    sub.lastEventId = null;
    
    console.log(`[KernelActivityWS] Client ${clientId} unsubscribed`);
  }
  
  /**
   * Cleanup on shutdown
   */
  destroy() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }
    
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
    }
    
    // Close all connections
    for (const [clientId, sub] of this.subscriptions) {
      if (sub.ws.readyState === WebSocket.OPEN) {
        sub.ws.close(1000, "Server shutting down");
      }
    }
    
    this.subscriptions.clear();
    console.log("[KernelActivityWS] Streamer destroyed");
  }
  
  /**
   * Get statistics
   */
  getStats() {
    return {
      activeConnections: this.subscriptions.size,
      bufferedActivities: this.activityBuffer.length,
      subscriptions: Array.from(this.subscriptions.entries()).map(([id, sub]) => ({
        clientId: id,
        hasFilters: !!sub.filters,
        lastEventId: sub.lastEventId,
      })),
    };
  }
}

export default KernelActivityStreamer;
