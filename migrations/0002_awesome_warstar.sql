CREATE TABLE "federation_peers" (
	"id" serial PRIMARY KEY NOT NULL,
	"peer_id" varchar(64) NOT NULL,
	"peer_name" varchar(128) NOT NULL,
	"peer_url" text NOT NULL,
	"api_key" text,
	"sync_enabled" boolean DEFAULT true,
	"sync_interval_hours" integer DEFAULT 1,
	"sync_vocabulary" boolean DEFAULT true,
	"sync_knowledge" boolean DEFAULT true,
	"sync_research" boolean DEFAULT false,
	"sync_kernels" boolean DEFAULT true,
	"sync_basins" boolean DEFAULT true,
	"last_sync_at" timestamp with time zone,
	"last_sync_status" varchar(32),
	"last_sync_error" text,
	"sync_count" integer DEFAULT 0,
	"vocabulary_sent" integer DEFAULT 0,
	"vocabulary_received" integer DEFAULT 0,
	"is_reachable" boolean DEFAULT true,
	"consecutive_failures" integer DEFAULT 0,
	"response_time_ms" integer,
	"last_health_check" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now(),
	"updated_at" timestamp with time zone DEFAULT now(),
	CONSTRAINT "federation_peers_peer_id_unique" UNIQUE("peer_id"),
	CONSTRAINT "federation_peers_peer_url_unique" UNIQUE("peer_url")
);
--> statement-breakpoint
CREATE TABLE "governance_proposals" (
	"id" serial PRIMARY KEY NOT NULL,
	"proposal_id" varchar(64) NOT NULL,
	"proposal_type" varchar(32) NOT NULL,
	"status" varchar(32) DEFAULT 'pending' NOT NULL,
	"reason" text,
	"parent_id" varchar(64),
	"parent_phi" double precision,
	"count" integer DEFAULT 1,
	"created_at" timestamp DEFAULT now(),
	"votes_for" jsonb DEFAULT '{}'::jsonb,
	"votes_against" jsonb DEFAULT '{}'::jsonb,
	"audit_log" jsonb DEFAULT '[]'::jsonb,
	CONSTRAINT "governance_proposals_proposal_id_unique" UNIQUE("proposal_id")
);
--> statement-breakpoint
CREATE TABLE "passphrase_vocabulary" (
	"id" varchar(64) PRIMARY KEY DEFAULT 'pv_' || gen_random_uuid()::text NOT NULL,
	"base_item" varchar(100) NOT NULL,
	"item_type" varchar(20) NOT NULL,
	"source" varchar(50) DEFAULT 'manual' NOT NULL,
	"frequency" integer DEFAULT 0,
	"phi_sum" double precision DEFAULT 0,
	"phi_avg" double precision,
	"success_count" integer DEFAULT 0,
	"near_miss_count" integer DEFAULT 0,
	"metadata" jsonb DEFAULT '{}'::jsonb,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "pattern_discoveries" (
	"discovery_id" varchar(64) PRIMARY KEY NOT NULL,
	"god_name" varchar(64) NOT NULL,
	"pattern_type" varchar(32) NOT NULL,
	"description" text NOT NULL,
	"confidence" double precision DEFAULT 0.5,
	"phi_score" double precision DEFAULT 0,
	"basin_coords" double precision[],
	"created_at" timestamp DEFAULT now(),
	"tool_requested" boolean DEFAULT false,
	"tool_request_id" varchar(64)
);
--> statement-breakpoint
CREATE TABLE "qig_metadata" (
	"key" text PRIMARY KEY NOT NULL,
	"value" text NOT NULL,
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "tool_requests" (
	"request_id" varchar(64) PRIMARY KEY NOT NULL,
	"requester_god" varchar(64) NOT NULL,
	"description" text NOT NULL,
	"examples" jsonb DEFAULT '[]'::jsonb,
	"context" jsonb DEFAULT '{}'::jsonb,
	"priority" integer DEFAULT 2,
	"status" varchar(32) DEFAULT 'pending',
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now(),
	"completed_at" timestamp,
	"tool_id" varchar(64),
	"error_message" text,
	"pattern_discoveries" text[]
);
--> statement-breakpoint
CREATE TABLE "vocabulary_stats" (
	"id" serial PRIMARY KEY NOT NULL,
	"total_words" integer NOT NULL,
	"bip39_words" integer NOT NULL,
	"learned_words" integer NOT NULL,
	"high_phi_words" integer NOT NULL,
	"merge_rules" integer NOT NULL,
	"last_updated" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE INDEX "idx_federation_peers_enabled" ON "federation_peers" USING btree ("sync_enabled");--> statement-breakpoint
CREATE INDEX "idx_federation_peers_last_sync" ON "federation_peers" USING btree ("last_sync_at");--> statement-breakpoint
CREATE INDEX "idx_vocab_base_item" ON "passphrase_vocabulary" USING btree ("base_item");--> statement-breakpoint
CREATE INDEX "idx_vocab_type" ON "passphrase_vocabulary" USING btree ("item_type");--> statement-breakpoint
CREATE INDEX "idx_vocab_source" ON "passphrase_vocabulary" USING btree ("source");--> statement-breakpoint
CREATE INDEX "idx_vocab_phi_avg" ON "passphrase_vocabulary" USING btree ("phi_avg");--> statement-breakpoint
CREATE INDEX "idx_vocab_frequency" ON "passphrase_vocabulary" USING btree ("frequency");--> statement-breakpoint
CREATE INDEX "idx_pattern_discoveries_god" ON "pattern_discoveries" USING btree ("god_name");--> statement-breakpoint
CREATE INDEX "idx_pattern_discoveries_confidence" ON "pattern_discoveries" USING btree ("confidence");--> statement-breakpoint
CREATE INDEX "idx_pattern_discoveries_unrequested" ON "pattern_discoveries" USING btree ("tool_requested");--> statement-breakpoint
CREATE INDEX "idx_qig_metadata_key" ON "qig_metadata" USING btree ("key");--> statement-breakpoint
CREATE INDEX "idx_tool_requests_status" ON "tool_requests" USING btree ("status");--> statement-breakpoint
CREATE INDEX "idx_tool_requests_requester" ON "tool_requests" USING btree ("requester_god");--> statement-breakpoint
CREATE INDEX "idx_tool_requests_priority" ON "tool_requests" USING btree ("priority","created_at");