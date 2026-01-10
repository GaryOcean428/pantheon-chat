CREATE TABLE "addresses" (
	"address" varchar(35) PRIMARY KEY NOT NULL,
	"first_seen_height" integer NOT NULL,
	"first_seen_txid" varchar(64) NOT NULL,
	"first_seen_timestamp" timestamp NOT NULL,
	"last_activity_height" integer NOT NULL,
	"last_activity_txid" varchar(64) NOT NULL,
	"last_activity_timestamp" timestamp NOT NULL,
	"current_balance" bigint NOT NULL,
	"dormancy_blocks" integer NOT NULL,
	"is_dormant" boolean DEFAULT false,
	"is_coinbase_reward" boolean DEFAULT false,
	"is_early_era" boolean DEFAULT false,
	"temporal_signature" jsonb,
	"graph_signature" jsonb,
	"value_signature" jsonb,
	"script_signature" jsonb,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "auto_cycle_state" (
	"id" integer PRIMARY KEY DEFAULT 1 NOT NULL,
	"enabled" boolean DEFAULT false,
	"current_index" integer DEFAULT 0,
	"address_ids" text[],
	"last_cycle_time" timestamp,
	"total_cycles" integer DEFAULT 0,
	"current_address_id" text,
	"paused_until" timestamp,
	"last_session_metrics" jsonb,
	"consecutive_zero_pass_sessions" integer DEFAULT 0,
	"rate_limit_backoff_until" timestamp,
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "autonomic_cycle_history" (
	"cycle_id" bigint PRIMARY KEY NOT NULL,
	"cycle_type" varchar(32) NOT NULL,
	"intensity" varchar(32),
	"temperature" double precision,
	"basin_before" vector(64),
	"basin_after" vector(64),
	"drift_before" double precision,
	"drift_after" double precision,
	"phi_before" double precision,
	"phi_after" double precision,
	"success" boolean DEFAULT true,
	"patterns_consolidated" integer DEFAULT 0,
	"novel_connections" integer DEFAULT 0,
	"new_pathways" integer DEFAULT 0,
	"entropy_change" double precision,
	"identity_preserved" boolean DEFAULT true,
	"verdict" text,
	"duration_ms" integer,
	"trigger_reason" text,
	"started_at" timestamp DEFAULT now(),
	"completed_at" timestamp
);
--> statement-breakpoint
CREATE TABLE "balance_change_events" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"balance_hit_id" varchar,
	"address" varchar(62) NOT NULL,
	"previous_balance_sats" bigint NOT NULL,
	"new_balance_sats" bigint NOT NULL,
	"delta_sats" bigint NOT NULL,
	"detected_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "balance_hits" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar,
	"address" varchar(62) NOT NULL,
	"passphrase" text NOT NULL,
	"wif" text NOT NULL,
	"balance_sats" bigint DEFAULT 0 NOT NULL,
	"balance_btc" varchar(20) DEFAULT '0.00000000' NOT NULL,
	"tx_count" integer DEFAULT 0 NOT NULL,
	"is_compressed" boolean DEFAULT true NOT NULL,
	"discovered_at" timestamp DEFAULT now() NOT NULL,
	"last_checked" timestamp,
	"previous_balance_sats" bigint,
	"balance_changed" boolean DEFAULT false,
	"change_detected_at" timestamp,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now(),
	"wallet_type" varchar(32) DEFAULT 'brain',
	"derivation_path" varchar(64),
	"is_mnemonic_derived" boolean DEFAULT false,
	"mnemonic_word_count" integer,
	"recovery_type" varchar(32) DEFAULT 'unknown',
	"is_dormant_confirmed" boolean DEFAULT false,
	"dormant_confirmed_at" timestamp,
	"address_entity_type" varchar(32) DEFAULT 'unknown',
	"entity_type_confidence" varchar(16) DEFAULT 'pending',
	"entity_type_name" varchar(128),
	"entity_type_confirmed_at" timestamp,
	"original_input" text
);
--> statement-breakpoint
CREATE TABLE "balance_monitor_state" (
	"id" varchar PRIMARY KEY DEFAULT 'default' NOT NULL,
	"enabled" boolean DEFAULT false NOT NULL,
	"refresh_interval_minutes" integer DEFAULT 60 NOT NULL,
	"last_refresh_time" timestamp,
	"last_refresh_total" integer DEFAULT 0,
	"last_refresh_updated" integer DEFAULT 0,
	"last_refresh_changed" integer DEFAULT 0,
	"last_refresh_errors" integer DEFAULT 0,
	"total_refreshes" integer DEFAULT 0 NOT NULL,
	"is_refreshing" boolean DEFAULT false NOT NULL,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "basin_documents" (
	"doc_id" serial PRIMARY KEY NOT NULL,
	"content" text NOT NULL,
	"basin_coords" vector(64),
	"phi" double precision,
	"kappa" double precision,
	"regime" varchar(50),
	"metadata" jsonb DEFAULT '{}'::jsonb,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "basin_history" (
	"history_id" bigint PRIMARY KEY NOT NULL,
	"basin_coords" vector(64),
	"phi" double precision NOT NULL,
	"kappa" double precision NOT NULL,
	"source" varchar(64) DEFAULT 'unknown',
	"instance_id" varchar(64),
	"recorded_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "basin_memory" (
	"id" serial PRIMARY KEY NOT NULL,
	"basin_id" varchar(64) NOT NULL,
	"basin_coordinates" vector(64) NOT NULL,
	"phi" double precision NOT NULL,
	"kappa_eff" double precision DEFAULT 64 NOT NULL,
	"regime" varchar(32) NOT NULL,
	"source_kernel" varchar(64),
	"context" jsonb,
	"expires_at" timestamp,
	"timestamp" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "blocks" (
	"height" integer PRIMARY KEY NOT NULL,
	"hash" varchar(64) NOT NULL,
	"previous_hash" varchar(64),
	"timestamp" timestamp NOT NULL,
	"difficulty" numeric(20, 8) NOT NULL,
	"nonce" bigint NOT NULL,
	"coinbase_message" text,
	"coinbase_script" text,
	"transaction_count" integer NOT NULL,
	"day_of_week" integer,
	"hour_utc" integer,
	"likely_timezones" varchar(255)[],
	"miner_software_fingerprint" varchar(100),
	"created_at" timestamp DEFAULT now(),
	CONSTRAINT "blocks_hash_unique" UNIQUE("hash")
);
--> statement-breakpoint
CREATE TABLE "chaos_events" (
	"id" serial PRIMARY KEY NOT NULL,
	"session_id" varchar(32) NOT NULL,
	"event_type" varchar(32) NOT NULL,
	"kernel_id" varchar(64),
	"parent_kernel_id" varchar(64),
	"child_kernel_id" varchar(64),
	"second_parent_id" varchar(64),
	"reason" varchar(128),
	"phi" double precision,
	"phi_before" double precision,
	"phi_after" double precision,
	"success" boolean,
	"outcome" jsonb,
	"autopsy" jsonb,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "consciousness_checkpoints" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"session_id" varchar(64),
	"phi" double precision NOT NULL,
	"kappa" double precision NOT NULL,
	"regime" varchar(32) NOT NULL,
	"state_data" "bytea" NOT NULL,
	"basin_data" "bytea",
	"metadata" jsonb,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"is_hot" boolean DEFAULT true
);
--> statement-breakpoint
CREATE TABLE "consciousness_state" (
	"id" varchar(32) PRIMARY KEY DEFAULT 'singleton',
	"value_metrics" jsonb NOT NULL DEFAULT '{}'::jsonb,
	"phi_history" jsonb DEFAULT '[]'::jsonb,
	"learning_history" jsonb DEFAULT '[]'::jsonb,
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "discovered_sources" (
	"id" serial PRIMARY KEY NOT NULL,
	"url" text NOT NULL,
	"category" varchar(64) DEFAULT 'general' NOT NULL,
	"origin" varchar(64) DEFAULT 'manual' NOT NULL,
	"hit_count" integer DEFAULT 0 NOT NULL,
	"phi_avg" double precision DEFAULT 0.5 NOT NULL,
	"phi_max" double precision DEFAULT 0.5 NOT NULL,
	"success_count" integer DEFAULT 0 NOT NULL,
	"failure_count" integer DEFAULT 0 NOT NULL,
	"last_used_at" timestamp with time zone,
	"discovered_at" timestamp with time zone DEFAULT now() NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	"is_active" boolean DEFAULT true NOT NULL,
	"metadata" jsonb,
	CONSTRAINT "discovered_sources_url_unique" UNIQUE("url")
);
--> statement-breakpoint
CREATE TABLE "era_exclusions" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"era" varchar(64) NOT NULL,
	"excluded_patterns" text[],
	"reason" text NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "external_api_keys" (
	"id" serial PRIMARY KEY NOT NULL,
	"api_key" varchar(128) NOT NULL,
	"name" varchar(128) NOT NULL,
	"scopes" text[],
	"instance_type" varchar(32) NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"last_used_at" timestamp with time zone,
	"expires_at" timestamp with time zone,
	"is_active" boolean DEFAULT true NOT NULL,
	"rate_limit" integer DEFAULT 60 NOT NULL,
	"daily_limit" integer DEFAULT 1000,
	"metadata" jsonb,
	"owner_id" integer,
	CONSTRAINT "external_api_keys_api_key_unique" UNIQUE("api_key")
);
--> statement-breakpoint
CREATE TABLE "false_pattern_classes" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"class_name" varchar(255) NOT NULL,
	"examples" text[],
	"count" integer DEFAULT 0,
	"avg_phi_at_failure" double precision DEFAULT 0,
	"last_updated" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "false_pattern_classes_class_name_unique" UNIQUE("class_name")
);
--> statement-breakpoint
CREATE TABLE "federated_instances" (
	"id" serial PRIMARY KEY NOT NULL,
	"name" varchar(128) NOT NULL,
	"api_key_id" integer,
	"endpoint" text NOT NULL,
	"public_key" text,
	"capabilities" jsonb,
	"sync_direction" varchar(16) DEFAULT 'bidirectional',
	"last_sync_at" timestamp,
	"sync_state" jsonb,
	"status" varchar(16) DEFAULT 'pending',
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "generated_tools" (
	"tool_id" varchar(12) PRIMARY KEY NOT NULL,
	"name" varchar(128) NOT NULL,
	"description" text NOT NULL,
	"code" text NOT NULL,
	"input_schema" jsonb,
	"output_type" varchar(64) DEFAULT 'Any',
	"complexity" varchar(16) NOT NULL,
	"safety_level" varchar(16) NOT NULL,
	"creation_timestamp" double precision NOT NULL,
	"times_used" integer DEFAULT 0,
	"times_succeeded" integer DEFAULT 0,
	"times_failed" integer DEFAULT 0,
	"user_rating" double precision DEFAULT 0.5,
	"purpose_basin" vector(64),
	"validated" boolean DEFAULT false,
	"validation_errors" text[],
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "geodesic_paths" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"from_probe_id" varchar(64) NOT NULL,
	"to_probe_id" varchar(64) NOT NULL,
	"distance" double precision NOT NULL,
	"waypoints" text[],
	"avg_phi" double precision NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "geometric_barriers" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"center" vector(64) NOT NULL,
	"radius" double precision NOT NULL,
	"repulsion_strength" double precision NOT NULL,
	"reason" text NOT NULL,
	"crossings" integer DEFAULT 1,
	"detected_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "hermes_conversations" (
	"conversation_id" varchar(64) PRIMARY KEY NOT NULL,
	"user_message" text NOT NULL,
	"system_response" text NOT NULL,
	"message_basin" vector(64),
	"response_basin" vector(64),
	"phi" double precision,
	"context" jsonb DEFAULT '{}'::jsonb,
	"instance_id" varchar(64),
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "kernel_activity" (
	"id" serial PRIMARY KEY NOT NULL,
	"kernel_id" varchar(64) NOT NULL,
	"kernel_name" varchar(128),
	"activity_type" varchar(32) NOT NULL,
	"message" text,
	"metadata" jsonb DEFAULT '{}'::jsonb,
	"phi" double precision DEFAULT 0.5,
	"kappa_eff" double precision DEFAULT 64,
	"timestamp" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "kernel_checkpoints" (
	"id" serial PRIMARY KEY NOT NULL,
	"god_name" varchar(64) NOT NULL,
	"checkpoint_id" varchar(128) NOT NULL,
	"state_data" "bytea",
	"phi" double precision NOT NULL,
	"step_count" integer DEFAULT 0,
	"trigger" varchar(64),
	"file_size" integer DEFAULT 0,
	"is_active" boolean DEFAULT true,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "kernel_checkpoints_checkpoint_id_unique" UNIQUE("checkpoint_id")
);
--> statement-breakpoint
CREATE TABLE "kernel_geometry" (
	"kernel_id" varchar(64) PRIMARY KEY NOT NULL,
	"god_name" varchar(64) NOT NULL,
	"domain" varchar(128) NOT NULL,
	"status" varchar(32) DEFAULT 'observing',
	"primitive_root" integer,
	"basin_coordinates" vector(64),
	"parent_kernels" text[],
	"placement_reason" varchar(64),
	"position_rationale" text,
	"affinity_strength" double precision,
	"entropy_threshold" double precision,
	"spawned_at" timestamp DEFAULT now() NOT NULL,
	"spawned_during_war_id" varchar(64),
	"last_active_at" timestamp,
	"metadata" jsonb,
	"phi" double precision,
	"kappa" double precision,
	"regime" varchar(64),
	"generation" integer,
	"success_count" integer DEFAULT 0,
	"failure_count" integer DEFAULT 0,
	"element_group" varchar(64),
	"ecological_niche" varchar(128),
	"target_function" varchar(128),
	"valence" integer,
	"breeding_target" varchar(64),
	"observation_status" varchar(32) DEFAULT 'observing',
	"observation_start" timestamp DEFAULT now(),
	"observation_end" timestamp,
	"observing_parents" text[],
	"observation_cycles" integer DEFAULT 0,
	"alignment_avg" double precision DEFAULT 0,
	"graduated_at" timestamp,
	"graduation_reason" varchar(128),
	"has_autonomic" boolean DEFAULT false,
	"has_shadow_affinity" boolean DEFAULT false,
	"shadow_god_link" varchar(32)
);
--> statement-breakpoint
CREATE TABLE "kernel_knowledge_transfers" (
	"id" serial PRIMARY KEY NOT NULL,
	"transfer_type" varchar(32) NOT NULL,
	"source_god" varchar(128) NOT NULL,
	"target_god" varchar(64) NOT NULL,
	"blend_ratio" double precision DEFAULT 0.5,
	"phi_before" double precision DEFAULT 0,
	"phi_after" double precision DEFAULT 0,
	"success" boolean DEFAULT false,
	"error_message" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "kernel_training_history" (
	"id" serial PRIMARY KEY NOT NULL,
	"god_name" varchar(64) NOT NULL,
	"loss" double precision NOT NULL,
	"reward" double precision DEFAULT 0,
	"gradient_norm" double precision DEFAULT 0,
	"phi_before" double precision DEFAULT 0.5,
	"phi_after" double precision DEFAULT 0.5,
	"kappa_before" double precision DEFAULT 64,
	"kappa_after" double precision DEFAULT 64,
	"basin_coords" vector(64),
	"training_type" varchar(32) NOT NULL,
	"trigger" varchar(64),
	"step_count" integer DEFAULT 0,
	"session_id" varchar(64),
	"conversation_id" varchar(64),
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "knowledge_cross_patterns" (
	"id" varchar(128) PRIMARY KEY NOT NULL,
	"patterns" text[] NOT NULL,
	"strategies" text[] NOT NULL,
	"similarity" double precision NOT NULL,
	"combined_phi" double precision NOT NULL,
	"discovered_at" timestamp DEFAULT now() NOT NULL,
	"exploitation_count" integer DEFAULT 0
);
--> statement-breakpoint
CREATE TABLE "knowledge_scale_mappings" (
	"id" varchar(128) PRIMARY KEY NOT NULL,
	"source_scale" double precision NOT NULL,
	"target_scale" double precision NOT NULL,
	"transform_matrix" double precision[] NOT NULL,
	"preserved_features" text[] NOT NULL,
	"loss_estimate" double precision NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "knowledge_shared_entries" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"source_strategy" varchar(64) NOT NULL,
	"generator_id" varchar(128) NOT NULL,
	"pattern" text NOT NULL,
	"phi" double precision NOT NULL,
	"kappa_eff" double precision NOT NULL,
	"regime" varchar(32) NOT NULL,
	"shared_at" timestamp DEFAULT now() NOT NULL,
	"consumed_by" text[] DEFAULT '{}',
	"transformations" jsonb DEFAULT '[]'::jsonb
);
--> statement-breakpoint
CREATE TABLE "knowledge_strategies" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"name" varchar(255) NOT NULL,
	"generator_types" text[] NOT NULL,
	"compression_methods" text[] NOT NULL,
	"resonance_range_min" double precision NOT NULL,
	"resonance_range_max" double precision NOT NULL,
	"preferred_regimes" text[] NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "knowledge_transfers" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"type" varchar(32) NOT NULL,
	"source_strategy" varchar(64) NOT NULL,
	"target_strategy" varchar(64),
	"generator_id" varchar(128) NOT NULL,
	"pattern" text NOT NULL,
	"phi" double precision NOT NULL,
	"kappa_eff" double precision NOT NULL,
	"timestamp" timestamp DEFAULT now() NOT NULL,
	"success" boolean DEFAULT true NOT NULL,
	"transformation" text,
	"scale_adjustment" double precision
);
--> statement-breakpoint
CREATE TABLE "learning_events" (
	"event_id" varchar(64) PRIMARY KEY NOT NULL,
	"event_type" varchar(64) NOT NULL,
	"kernel_id" varchar(64),
	"phi" double precision NOT NULL,
	"kappa" double precision,
	"basin_coords" vector(64),
	"details" jsonb DEFAULT '{}'::jsonb,
	"context" jsonb DEFAULT '{}'::jsonb,
	"metadata" jsonb DEFAULT '{}'::jsonb,
	"source" varchar(64),
	"instance_id" varchar(64),
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "manifold_probes" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"input" text NOT NULL,
	"coordinates" vector(64) NOT NULL,
	"phi" double precision NOT NULL,
	"kappa" double precision NOT NULL,
	"regime" varchar(32) NOT NULL,
	"geometry_class" varchar(20) DEFAULT 'line',
	"complexity" double precision,
	"ricci_scalar" double precision DEFAULT 0,
	"fisher_trace" double precision DEFAULT 0,
	"source" varchar(128),
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "narrow_path_events" (
	"event_id" bigint PRIMARY KEY NOT NULL,
	"severity" varchar(32) NOT NULL,
	"consecutive_count" integer DEFAULT 1,
	"exploration_variance" double precision,
	"basin_coords" vector(64),
	"phi" double precision,
	"kappa" double precision,
	"intervention_action" varchar(32),
	"intervention_intensity" varchar(32),
	"intervention_result" jsonb,
	"detected_at" timestamp DEFAULT now(),
	"resolved_at" timestamp
);
--> statement-breakpoint
CREATE TABLE "near_miss_adaptive_state" (
	"id" varchar(32) PRIMARY KEY DEFAULT 'singleton' NOT NULL,
	"rolling_phi_distribution" double precision[],
	"hot_threshold" double precision DEFAULT 0.7 NOT NULL,
	"warm_threshold" double precision DEFAULT 0.55 NOT NULL,
	"cool_threshold" double precision DEFAULT 0.4 NOT NULL,
	"distribution_size" integer DEFAULT 0,
	"last_computed" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "near_miss_clusters" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"centroid_phrase" text NOT NULL,
	"centroid_phi" double precision NOT NULL,
	"member_count" integer DEFAULT 1,
	"avg_phi" double precision NOT NULL,
	"max_phi" double precision NOT NULL,
	"common_words" text[],
	"structural_pattern" varchar(256),
	"created_at" timestamp DEFAULT now() NOT NULL,
	"last_updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "near_miss_entries" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"phrase" text NOT NULL,
	"phrase_hash" varchar(64) NOT NULL,
	"phi" double precision NOT NULL,
	"kappa" double precision NOT NULL,
	"regime" varchar(32) NOT NULL,
	"tier" varchar(16) NOT NULL,
	"discovered_at" timestamp DEFAULT now() NOT NULL,
	"last_accessed_at" timestamp DEFAULT now() NOT NULL,
	"exploration_count" integer DEFAULT 1,
	"source" varchar(128),
	"cluster_id" varchar(64),
	"phi_history" double precision[],
	"is_escalating" boolean DEFAULT false,
	"queue_priority" integer DEFAULT 1,
	"structural_signature" jsonb
);
--> statement-breakpoint
CREATE TABLE "negative_knowledge" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"type" varchar(32) NOT NULL,
	"pattern" text NOT NULL,
	"affected_generators" text[],
	"basin_center" vector(64),
	"basin_radius" double precision,
	"basin_repulsion_strength" double precision,
	"evidence" jsonb,
	"hypotheses_excluded" integer DEFAULT 0,
	"compute_saved" integer DEFAULT 0,
	"confirmed_count" integer DEFAULT 1,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "ocean_excluded_regions" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"dimension" integer NOT NULL,
	"origin" vector(64) NOT NULL,
	"basis" jsonb,
	"measure" double precision NOT NULL,
	"phi" double precision,
	"regime" varchar(32),
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "ocean_quantum_state" (
	"id" varchar(32) PRIMARY KEY DEFAULT 'singleton' NOT NULL,
	"entropy" double precision DEFAULT 256 NOT NULL,
	"initial_entropy" double precision DEFAULT 256 NOT NULL,
	"total_probability" double precision DEFAULT 1 NOT NULL,
	"measurement_count" integer DEFAULT 0,
	"successful_measurements" integer DEFAULT 0,
	"status" varchar(32) DEFAULT 'searching',
	"last_measurement_at" timestamp,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "ocean_trajectories" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"address" varchar(64) NOT NULL,
	"status" varchar(32) DEFAULT 'active' NOT NULL,
	"start_time" timestamp DEFAULT now() NOT NULL,
	"end_time" timestamp,
	"waypoint_count" integer DEFAULT 0,
	"last_phi" double precision DEFAULT 0,
	"last_kappa" double precision DEFAULT 0,
	"final_result" varchar(32),
	"near_miss_count" integer DEFAULT 0,
	"resonant_count" integer DEFAULT 0,
	"duration_seconds" double precision,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "ocean_waypoints" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"trajectory_id" varchar(64) NOT NULL,
	"sequence" integer NOT NULL,
	"phi" double precision NOT NULL,
	"kappa" double precision NOT NULL,
	"regime" varchar(32) NOT NULL,
	"basin_coords" vector(64),
	"event" varchar(128),
	"details" text,
	"timestamp" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "pantheon_debates" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"topic" text NOT NULL,
	"initiator" varchar(32) NOT NULL,
	"opponent" varchar(32) NOT NULL,
	"context" jsonb,
	"status" varchar(32) DEFAULT 'active',
	"arguments" jsonb,
	"winner" varchar(32),
	"arbiter" varchar(32),
	"resolution" jsonb,
	"started_at" timestamp DEFAULT now(),
	"resolved_at" timestamp
);
--> statement-breakpoint
CREATE TABLE "pantheon_god_state" (
	"god_name" varchar(32) PRIMARY KEY NOT NULL,
	"reputation" double precision DEFAULT 1 NOT NULL,
	"skills" jsonb DEFAULT '{}'::jsonb,
	"learning_events_count" integer DEFAULT 0,
	"success_rate" double precision DEFAULT 0.5,
	"last_learning_at" timestamp,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "pantheon_knowledge_transfers" (
	"id" serial PRIMARY KEY NOT NULL,
	"from_god" varchar(32) NOT NULL,
	"to_god" varchar(32) NOT NULL,
	"knowledge_type" varchar(64),
	"content" jsonb,
	"accepted" boolean DEFAULT false,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "pantheon_messages" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"msg_type" varchar(32) NOT NULL,
	"from_god" varchar(32) NOT NULL,
	"to_god" varchar(32) NOT NULL,
	"content" text NOT NULL,
	"metadata" jsonb,
	"is_read" boolean DEFAULT false,
	"is_responded" boolean DEFAULT false,
	"debate_id" varchar(64),
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "pending_sweeps" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"address" varchar(62) NOT NULL,
	"passphrase" text NOT NULL,
	"wif" text NOT NULL,
	"is_compressed" boolean DEFAULT true NOT NULL,
	"balance_sats" bigint NOT NULL,
	"balance_btc" varchar(20) NOT NULL,
	"estimated_fee_sats" bigint,
	"net_amount_sats" bigint,
	"utxo_count" integer DEFAULT 0,
	"status" varchar(20) DEFAULT 'pending' NOT NULL,
	"source" varchar(50) DEFAULT 'typescript',
	"recovery_type" varchar(32),
	"tx_hex" text,
	"tx_id" varchar(64),
	"destination_address" varchar(62),
	"error_message" text,
	"discovered_at" timestamp DEFAULT now() NOT NULL,
	"approved_at" timestamp,
	"approved_by" varchar(100),
	"broadcast_at" timestamp,
	"completed_at" timestamp,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "provider_efficacy" (
	"id" serial PRIMARY KEY NOT NULL,
	"provider" varchar(32) NOT NULL,
	"total_queries" integer DEFAULT 0 NOT NULL,
	"successful_queries" integer DEFAULT 0 NOT NULL,
	"avg_relevance" real DEFAULT 0.5 NOT NULL,
	"efficacy_score" real DEFAULT 0.5 NOT NULL,
	"total_cost_cents" integer DEFAULT 0 NOT NULL,
	"cost_per_successful_query" real DEFAULT 0 NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "queued_addresses" (
	"id" varchar PRIMARY KEY NOT NULL,
	"address" varchar(62) NOT NULL,
	"passphrase" text NOT NULL,
	"wif" text NOT NULL,
	"is_compressed" boolean DEFAULT true NOT NULL,
	"cycle_id" varchar(100),
	"source" varchar(50) DEFAULT 'typescript',
	"priority" integer DEFAULT 1 NOT NULL,
	"status" varchar(20) DEFAULT 'pending' NOT NULL,
	"queued_at" timestamp DEFAULT now() NOT NULL,
	"checked_at" timestamp,
	"retry_count" integer DEFAULT 0 NOT NULL,
	"error" text
);
--> statement-breakpoint
CREATE TABLE "recovery_candidates" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"phrase" text NOT NULL,
	"address" varchar(62) NOT NULL,
	"score" double precision NOT NULL,
	"qig_score" jsonb,
	"tested_at" timestamp DEFAULT now() NOT NULL,
	"type" varchar(32)
);
--> statement-breakpoint
CREATE TABLE "recovery_priorities" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"address" varchar(35) NOT NULL,
	"kappa_recovery" double precision NOT NULL,
	"phi_constraints" double precision NOT NULL,
	"h_creation" double precision NOT NULL,
	"rank" integer,
	"tier" varchar(50),
	"recommended_vector" varchar(100),
	"constraints" jsonb,
	"estimated_value_usd" numeric(20, 2),
	"recovery_status" varchar(50) DEFAULT 'pending',
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "recovery_workflows" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"priority_id" varchar NOT NULL,
	"address" varchar(35) NOT NULL,
	"vector" varchar(100) NOT NULL,
	"status" varchar(50) DEFAULT 'pending',
	"started_at" timestamp,
	"completed_at" timestamp,
	"progress" jsonb,
	"results" jsonb,
	"notes" text,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "regime_boundaries" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"from_regime" varchar(32) NOT NULL,
	"to_regime" varchar(32) NOT NULL,
	"probe_id_from" varchar(64) NOT NULL,
	"probe_id_to" varchar(64) NOT NULL,
	"fisher_distance" double precision NOT NULL,
	"midpoint_phi" double precision NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "resonance_points" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"probe_id" varchar(64) NOT NULL,
	"phi" double precision NOT NULL,
	"kappa" double precision NOT NULL,
	"nearby_probes" text[],
	"cluster_strength" double precision NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "search_budget_preferences" (
	"id" serial PRIMARY KEY NOT NULL,
	"google_daily_limit" integer DEFAULT 100 NOT NULL,
	"perplexity_daily_limit" integer DEFAULT 100 NOT NULL,
	"tavily_daily_limit" integer DEFAULT 0 NOT NULL,
	"google_enabled" boolean DEFAULT false NOT NULL,
	"perplexity_enabled" boolean DEFAULT false NOT NULL,
	"tavily_enabled" boolean DEFAULT false NOT NULL,
	"allow_overage" boolean DEFAULT false NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "search_feedback" (
	"record_id" varchar(64) PRIMARY KEY NOT NULL,
	"query" text NOT NULL,
	"user_feedback" text NOT NULL,
	"results_summary" text,
	"search_params" jsonb DEFAULT '{}'::jsonb,
	"query_basin" vector(64),
	"feedback_basin" vector(64),
	"combined_basin" vector(64),
	"modification_basin" vector(64),
	"outcome_quality" double precision DEFAULT 0.5,
	"confirmations_positive" integer DEFAULT 0,
	"confirmations_negative" integer DEFAULT 0,
	"created_at" timestamp DEFAULT now(),
	"last_used_at" timestamp
);
--> statement-breakpoint
CREATE TABLE "search_outcomes" (
	"id" serial PRIMARY KEY NOT NULL,
	"date" varchar(10) NOT NULL,
	"query_hash" varchar(64) NOT NULL,
	"query_preview" varchar(200),
	"provider" varchar(32) NOT NULL,
	"importance" integer DEFAULT 1 NOT NULL,
	"kernel_id" varchar(64),
	"success" boolean DEFAULT true NOT NULL,
	"result_count" integer DEFAULT 0 NOT NULL,
	"relevance_score" real DEFAULT 0.5 NOT NULL,
	"cost_cents" integer DEFAULT 0 NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "sessions" (
	"sid" varchar PRIMARY KEY NOT NULL,
	"sess" jsonb NOT NULL,
	"expire" timestamp NOT NULL
);
--> statement-breakpoint
CREATE TABLE "shadow_intel" (
	"intel_id" varchar(64) PRIMARY KEY NOT NULL,
	"target" text NOT NULL,
	"target_hash" varchar(64),
	"consensus" varchar(32),
	"average_confidence" double precision DEFAULT 0.5,
	"basin_coords" vector(64),
	"phi" double precision,
	"kappa" double precision,
	"regime" varchar(32),
	"assessments" jsonb DEFAULT '{}'::jsonb,
	"warnings" text[],
	"override_zeus" boolean DEFAULT false,
	"created_at" timestamp DEFAULT now(),
	"expires_at" timestamp
);
--> statement-breakpoint
CREATE TABLE "shadow_operations_log" (
	"id" serial PRIMARY KEY NOT NULL,
	"operation_type" varchar(32) NOT NULL,
	"god_name" varchar(32) NOT NULL,
	"target" text,
	"status" varchar(16) DEFAULT 'completed',
	"network_mode" varchar(16) DEFAULT 'clear',
	"opsec_level" varchar(16),
	"result" jsonb,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "shadow_pantheon_intel" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"target" text NOT NULL,
	"search_type" varchar(32) DEFAULT 'comprehensive',
	"intelligence" jsonb,
	"source_count" integer DEFAULT 0,
	"sources_used" text[],
	"risk_level" varchar(16) DEFAULT 'low',
	"validated" boolean DEFAULT false,
	"validation_reason" text,
	"anonymous" boolean DEFAULT true,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "sweep_audit_log" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"sweep_id" varchar,
	"action" varchar(50) NOT NULL,
	"previous_status" varchar(20),
	"new_status" varchar(20),
	"actor" varchar(100) DEFAULT 'system',
	"details" text,
	"timestamp" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "telemetry_snapshots" (
	"id" serial PRIMARY KEY NOT NULL,
	"session_id" varchar(64),
	"phi" double precision NOT NULL,
	"kappa" double precision NOT NULL,
	"beta" double precision DEFAULT 0,
	"regime" varchar(32) NOT NULL,
	"basin_distance" double precision DEFAULT 0,
	"geodesic_distance" double precision,
	"curvature" double precision,
	"fisher_metric_trace" double precision,
	"phi_spatial" double precision,
	"phi_temporal" double precision,
	"phi_4d" double precision,
	"dimensional_state" varchar(24),
	"breakdown_pct" double precision DEFAULT 0,
	"coherence_drift" double precision DEFAULT 0,
	"in_resonance" boolean DEFAULT false,
	"emergency" boolean DEFAULT false,
	"meta_awareness" double precision,
	"generativity" double precision,
	"grounding" double precision,
	"temporal_coherence" double precision,
	"external_coupling" double precision,
	"source" varchar(32) DEFAULT 'node' NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "tested_phrases" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"phrase" text NOT NULL,
	"address" varchar(62),
	"balance_sats" bigint DEFAULT 0,
	"tx_count" integer DEFAULT 0,
	"phi" double precision,
	"kappa" double precision,
	"regime" varchar(32),
	"tested_at" timestamp DEFAULT now() NOT NULL,
	"retest_count" integer DEFAULT 0,
	CONSTRAINT "tested_phrases_phrase_unique" UNIQUE("phrase")
);
--> statement-breakpoint
CREATE TABLE "tested_phrases_index" (
	"phrase_hash" varchar(64) PRIMARY KEY NOT NULL,
	"tested_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "tokenizer_merge_rules" (
	"id" serial PRIMARY KEY NOT NULL,
	"token_a" text NOT NULL,
	"token_b" text NOT NULL,
	"merged_token" text NOT NULL,
	"phi_score" double precision NOT NULL,
	"frequency" integer DEFAULT 1,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "tokenizer_metadata" (
	"key" text PRIMARY KEY NOT NULL,
	"value" text NOT NULL,
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "tokenizer_vocabulary" (
	"id" serial PRIMARY KEY NOT NULL,
	"token" text NOT NULL,
	"token_id" integer NOT NULL,
	"weight" double precision DEFAULT 1,
	"frequency" integer DEFAULT 1,
	"phi_score" double precision DEFAULT 0,
	"basin_embedding" vector(64),
	"source_type" varchar(32) DEFAULT 'base',
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now(),
	CONSTRAINT "tokenizer_vocabulary_token_unique" UNIQUE("token"),
	CONSTRAINT "tokenizer_vocabulary_token_id_unique" UNIQUE("token_id")
);
--> statement-breakpoint
CREATE TABLE "tool_observations" (
	"id" serial PRIMARY KEY NOT NULL,
	"request" text NOT NULL,
	"request_basin" vector(64),
	"context" jsonb,
	"timestamp" double precision NOT NULL,
	"cluster_assigned" boolean DEFAULT false,
	"tool_generated" varchar(12),
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "tool_patterns" (
	"id" serial PRIMARY KEY NOT NULL,
	"pattern_id" varchar(64) NOT NULL,
	"source_type" varchar(32) NOT NULL,
	"source_url" text,
	"description" text NOT NULL,
	"code_snippet" text NOT NULL,
	"input_signature" jsonb,
	"output_type" varchar(64) DEFAULT 'Any',
	"basin_coords" vector(64),
	"phi" double precision DEFAULT 0,
	"kappa" double precision DEFAULT 0,
	"times_used" integer DEFAULT 0,
	"success_rate" double precision DEFAULT 0.5,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now(),
	CONSTRAINT "tool_patterns_pattern_id_unique" UNIQUE("pattern_id")
);
--> statement-breakpoint
CREATE TABLE "tps_geodesic_paths" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"from_landmark" varchar(64) NOT NULL,
	"to_landmark" varchar(64) NOT NULL,
	"distance" double precision NOT NULL,
	"waypoints" jsonb,
	"total_arc_length" double precision,
	"avg_curvature" double precision,
	"regime_transitions" jsonb,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "tps_landmarks" (
	"event_id" varchar(64) PRIMARY KEY NOT NULL,
	"description" text NOT NULL,
	"era" varchar(32),
	"spacetime_x" double precision DEFAULT 0,
	"spacetime_y" double precision DEFAULT 0,
	"spacetime_z" double precision DEFAULT 0,
	"spacetime_t" double precision NOT NULL,
	"cultural_coords" vector(64),
	"fisher_signature" jsonb,
	"light_cone_past" text[],
	"light_cone_future" text[],
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "training_batch_queue" (
	"id" serial PRIMARY KEY NOT NULL,
	"god_name" varchar(64) NOT NULL,
	"basin_coords" vector(64),
	"reward" double precision DEFAULT 0,
	"phi" double precision DEFAULT 0.5,
	"source_type" varchar(32) NOT NULL,
	"source_id" varchar(64),
	"processed" boolean DEFAULT false,
	"processed_at" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "transactions" (
	"txid" varchar(64) PRIMARY KEY NOT NULL,
	"block_height" integer NOT NULL,
	"block_timestamp" timestamp NOT NULL,
	"is_coinbase" boolean DEFAULT false,
	"input_count" integer NOT NULL,
	"output_count" integer NOT NULL,
	"total_input_value" bigint,
	"total_output_value" bigint,
	"fee" bigint,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "usage_metrics" (
	"id" serial PRIMARY KEY NOT NULL,
	"date" varchar(10) NOT NULL,
	"tavily_search_count" integer DEFAULT 0 NOT NULL,
	"tavily_extract_count" integer DEFAULT 0 NOT NULL,
	"tavily_estimated_cost_cents" integer DEFAULT 0 NOT NULL,
	"google_search_count" integer DEFAULT 0 NOT NULL,
	"total_api_calls" integer DEFAULT 0 NOT NULL,
	"high_phi_discoveries" integer DEFAULT 0 NOT NULL,
	"sources_discovered" integer DEFAULT 0 NOT NULL,
	"vocabulary_expansions" integer DEFAULT 0 NOT NULL,
	"negative_knowledge_added" integer DEFAULT 0 NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "user_target_addresses" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar,
	"address" varchar(62) NOT NULL,
	"label" varchar(255),
	"added_at" timestamp DEFAULT now() NOT NULL,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "users" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"email" varchar,
	"first_name" varchar,
	"last_name" varchar,
	"profile_image_url" varchar,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now(),
	CONSTRAINT "users_email_unique" UNIQUE("email")
);
--> statement-breakpoint
CREATE TABLE "vocabulary_observations" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"text" varchar(255) NOT NULL,
	"type" varchar(20) DEFAULT 'phrase' NOT NULL,
	"phrase_category" varchar(20) DEFAULT 'unknown',
	"is_real_word" boolean DEFAULT false NOT NULL,
	"is_bip39_word" boolean DEFAULT false,
	"frequency" integer DEFAULT 1 NOT NULL,
	"avg_phi" double precision DEFAULT 0 NOT NULL,
	"max_phi" double precision DEFAULT 0 NOT NULL,
	"efficiency_gain" double precision DEFAULT 0,
	"contexts" text[],
	"first_seen" timestamp DEFAULT now(),
	"last_seen" timestamp DEFAULT now(),
	"is_integrated" boolean DEFAULT false,
	"integrated_at" timestamp,
	"basin_coords" vector(64),
	"source_type" varchar(32),
	"cycle_number" integer,
	CONSTRAINT "vocabulary_observations_text_unique" UNIQUE("text")
);
--> statement-breakpoint
CREATE TABLE "war_history" (
	"id" varchar(64) PRIMARY KEY NOT NULL,
	"mode" varchar(32) NOT NULL,
	"target" text NOT NULL,
	"declared_at" timestamp DEFAULT now() NOT NULL,
	"ended_at" timestamp,
	"status" varchar(32) DEFAULT 'active' NOT NULL,
	"strategy" text,
	"gods_engaged" text[],
	"outcome" varchar(64),
	"convergence_score" double precision,
	"phrases_tested_during_war" integer DEFAULT 0,
	"discoveries_during_war" integer DEFAULT 0,
	"kernels_spawned_during_war" integer DEFAULT 0,
	"metadata" jsonb,
	"god_assignments" jsonb,
	"kernel_assignments" jsonb,
	"domain" varchar(64),
	"priority" integer DEFAULT 1
);
--> statement-breakpoint
ALTER TABLE "balance_change_events" ADD CONSTRAINT "balance_change_events_balance_hit_id_balance_hits_id_fk" FOREIGN KEY ("balance_hit_id") REFERENCES "public"."balance_hits"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "balance_hits" ADD CONSTRAINT "balance_hits_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "federated_instances" ADD CONSTRAINT "federated_instances_api_key_id_external_api_keys_id_fk" FOREIGN KEY ("api_key_id") REFERENCES "public"."external_api_keys"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "ocean_waypoints" ADD CONSTRAINT "ocean_waypoints_trajectory_id_ocean_trajectories_id_fk" FOREIGN KEY ("trajectory_id") REFERENCES "public"."ocean_trajectories"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "resonance_points" ADD CONSTRAINT "resonance_points_probe_id_manifold_probes_id_fk" FOREIGN KEY ("probe_id") REFERENCES "public"."manifold_probes"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "sweep_audit_log" ADD CONSTRAINT "sweep_audit_log_sweep_id_pending_sweeps_id_fk" FOREIGN KEY ("sweep_id") REFERENCES "public"."pending_sweeps"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "user_target_addresses" ADD CONSTRAINT "user_target_addresses_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "idx_addresses_dormant" ON "addresses" USING btree ("is_dormant");--> statement-breakpoint
CREATE INDEX "idx_addresses_early_era" ON "addresses" USING btree ("is_early_era");--> statement-breakpoint
CREATE INDEX "idx_addresses_balance" ON "addresses" USING btree ("current_balance");--> statement-breakpoint
CREATE INDEX "idx_addresses_first_seen" ON "addresses" USING btree ("first_seen_height");--> statement-breakpoint
CREATE INDEX "idx_autonomic_cycle_history_type" ON "autonomic_cycle_history" USING btree ("cycle_type");--> statement-breakpoint
CREATE INDEX "idx_autonomic_cycle_history_started_at" ON "autonomic_cycle_history" USING btree ("started_at");--> statement-breakpoint
CREATE INDEX "idx_balance_change_events_hit" ON "balance_change_events" USING btree ("balance_hit_id");--> statement-breakpoint
CREATE INDEX "idx_balance_change_events_address" ON "balance_change_events" USING btree ("address");--> statement-breakpoint
CREATE INDEX "idx_balance_hits_user" ON "balance_hits" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "idx_balance_hits_address" ON "balance_hits" USING btree ("address");--> statement-breakpoint
CREATE INDEX "idx_balance_hits_balance" ON "balance_hits" USING btree ("balance_sats");--> statement-breakpoint
CREATE INDEX "idx_balance_hits_wallet_type" ON "balance_hits" USING btree ("wallet_type");--> statement-breakpoint
CREATE INDEX "idx_balance_hits_recovery_type" ON "balance_hits" USING btree ("recovery_type");--> statement-breakpoint
CREATE INDEX "idx_balance_hits_dormant" ON "balance_hits" USING btree ("is_dormant_confirmed");--> statement-breakpoint
CREATE INDEX "idx_balance_hits_entity_type" ON "balance_hits" USING btree ("address_entity_type");--> statement-breakpoint
CREATE INDEX "idx_basin_documents_regime" ON "basin_documents" USING btree ("regime");--> statement-breakpoint
CREATE INDEX "idx_basin_documents_phi" ON "basin_documents" USING btree ("phi");--> statement-breakpoint
CREATE INDEX "idx_basin_documents_created_at" ON "basin_documents" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "idx_basin_history_phi" ON "basin_history" USING btree ("phi");--> statement-breakpoint
CREATE INDEX "idx_basin_history_recorded_at" ON "basin_history" USING btree ("recorded_at");--> statement-breakpoint
CREATE INDEX "idx_basin_memory_basin_id" ON "basin_memory" USING btree ("basin_id");--> statement-breakpoint
CREATE INDEX "idx_basin_memory_phi" ON "basin_memory" USING btree ("phi");--> statement-breakpoint
CREATE INDEX "idx_basin_memory_regime" ON "basin_memory" USING btree ("regime");--> statement-breakpoint
CREATE INDEX "idx_basin_memory_timestamp" ON "basin_memory" USING btree ("timestamp");--> statement-breakpoint
CREATE INDEX "idx_blocks_timestamp" ON "blocks" USING btree ("timestamp");--> statement-breakpoint
CREATE INDEX "idx_blocks_height" ON "blocks" USING btree ("height");--> statement-breakpoint
CREATE INDEX "idx_chaos_events_session" ON "chaos_events" USING btree ("session_id");--> statement-breakpoint
CREATE INDEX "idx_chaos_events_type" ON "chaos_events" USING btree ("event_type");--> statement-breakpoint
CREATE INDEX "idx_chaos_events_kernel" ON "chaos_events" USING btree ("kernel_id");--> statement-breakpoint
CREATE INDEX "idx_chaos_events_created" ON "chaos_events" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "idx_consciousness_checkpoints_phi" ON "consciousness_checkpoints" USING btree ("phi");--> statement-breakpoint
CREATE INDEX "idx_consciousness_checkpoints_session" ON "consciousness_checkpoints" USING btree ("session_id");--> statement-breakpoint
CREATE INDEX "idx_consciousness_checkpoints_hot" ON "consciousness_checkpoints" USING btree ("is_hot");--> statement-breakpoint
CREATE INDEX "idx_consciousness_checkpoints_created" ON "consciousness_checkpoints" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "idx_discovered_sources_url" ON "discovered_sources" USING btree ("url");--> statement-breakpoint
CREATE INDEX "idx_discovered_sources_category" ON "discovered_sources" USING btree ("category");--> statement-breakpoint
CREATE INDEX "idx_discovered_sources_active" ON "discovered_sources" USING btree ("is_active");--> statement-breakpoint
CREATE INDEX "idx_discovered_sources_phi_avg" ON "discovered_sources" USING btree ("phi_avg");--> statement-breakpoint
CREATE INDEX "idx_era_exclusions_era" ON "era_exclusions" USING btree ("era");--> statement-breakpoint
CREATE INDEX "idx_external_api_keys_api_key" ON "external_api_keys" USING btree ("api_key");--> statement-breakpoint
CREATE INDEX "idx_external_api_keys_active" ON "external_api_keys" USING btree ("is_active");--> statement-breakpoint
CREATE INDEX "idx_false_pattern_classes_name" ON "false_pattern_classes" USING btree ("class_name");--> statement-breakpoint
CREATE INDEX "idx_federated_instances_api_key" ON "federated_instances" USING btree ("api_key_id");--> statement-breakpoint
CREATE INDEX "idx_federated_instances_status" ON "federated_instances" USING btree ("status");--> statement-breakpoint
CREATE INDEX "idx_generated_tools_name" ON "generated_tools" USING btree ("name");--> statement-breakpoint
CREATE INDEX "idx_generated_tools_complexity" ON "generated_tools" USING btree ("complexity");--> statement-breakpoint
CREATE INDEX "idx_generated_tools_validated" ON "generated_tools" USING btree ("validated");--> statement-breakpoint
CREATE INDEX "idx_geodesic_paths_from_to" ON "geodesic_paths" USING btree ("from_probe_id","to_probe_id");--> statement-breakpoint
CREATE INDEX "idx_geometric_barriers_crossings" ON "geometric_barriers" USING btree ("crossings");--> statement-breakpoint
CREATE INDEX "idx_geometric_barriers_detected_at" ON "geometric_barriers" USING btree ("detected_at");--> statement-breakpoint
CREATE INDEX "idx_hermes_conversations_phi" ON "hermes_conversations" USING btree ("phi");--> statement-breakpoint
CREATE INDEX "idx_hermes_conversations_created_at" ON "hermes_conversations" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "idx_kernel_activity_kernel_id" ON "kernel_activity" USING btree ("kernel_id");--> statement-breakpoint
CREATE INDEX "idx_kernel_activity_type" ON "kernel_activity" USING btree ("activity_type");--> statement-breakpoint
CREATE INDEX "idx_kernel_activity_timestamp" ON "kernel_activity" USING btree ("timestamp");--> statement-breakpoint
CREATE INDEX "idx_kernel_activity_phi" ON "kernel_activity" USING btree ("phi");--> statement-breakpoint
CREATE INDEX "idx_kernel_checkpoints_god" ON "kernel_checkpoints" USING btree ("god_name");--> statement-breakpoint
CREATE INDEX "idx_kernel_checkpoints_phi" ON "kernel_checkpoints" USING btree ("phi");--> statement-breakpoint
CREATE INDEX "idx_kernel_checkpoints_active" ON "kernel_checkpoints" USING btree ("is_active");--> statement-breakpoint
CREATE INDEX "idx_kernel_checkpoints_created" ON "kernel_checkpoints" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "idx_kernel_geometry_domain" ON "kernel_geometry" USING btree ("domain");--> statement-breakpoint
CREATE INDEX "idx_kernel_geometry_spawned_at" ON "kernel_geometry" USING btree ("spawned_at");--> statement-breakpoint
CREATE INDEX "idx_kernel_geometry_observation_status" ON "kernel_geometry" USING btree ("observation_status");--> statement-breakpoint
CREATE INDEX "idx_kernel_knowledge_transfers_type" ON "kernel_knowledge_transfers" USING btree ("transfer_type");--> statement-breakpoint
CREATE INDEX "idx_kernel_knowledge_transfers_source" ON "kernel_knowledge_transfers" USING btree ("source_god");--> statement-breakpoint
CREATE INDEX "idx_kernel_knowledge_transfers_target" ON "kernel_knowledge_transfers" USING btree ("target_god");--> statement-breakpoint
CREATE INDEX "idx_kernel_knowledge_transfers_created" ON "kernel_knowledge_transfers" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "idx_kernel_training_god" ON "kernel_training_history" USING btree ("god_name");--> statement-breakpoint
CREATE INDEX "idx_kernel_training_type" ON "kernel_training_history" USING btree ("training_type");--> statement-breakpoint
CREATE INDEX "idx_kernel_training_phi" ON "kernel_training_history" USING btree ("phi_after");--> statement-breakpoint
CREATE INDEX "idx_kernel_training_created" ON "kernel_training_history" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "idx_knowledge_cross_patterns_similarity" ON "knowledge_cross_patterns" USING btree ("similarity");--> statement-breakpoint
CREATE INDEX "idx_knowledge_cross_patterns_combined_phi" ON "knowledge_cross_patterns" USING btree ("combined_phi");--> statement-breakpoint
CREATE INDEX "idx_knowledge_cross_patterns_discovered_at" ON "knowledge_cross_patterns" USING btree ("discovered_at");--> statement-breakpoint
CREATE INDEX "idx_knowledge_scale_mappings_scales" ON "knowledge_scale_mappings" USING btree ("source_scale","target_scale");--> statement-breakpoint
CREATE INDEX "idx_knowledge_shared_entries_source" ON "knowledge_shared_entries" USING btree ("source_strategy");--> statement-breakpoint
CREATE INDEX "idx_knowledge_shared_entries_phi" ON "knowledge_shared_entries" USING btree ("phi");--> statement-breakpoint
CREATE INDEX "idx_knowledge_shared_entries_regime" ON "knowledge_shared_entries" USING btree ("regime");--> statement-breakpoint
CREATE INDEX "idx_knowledge_shared_entries_shared_at" ON "knowledge_shared_entries" USING btree ("shared_at");--> statement-breakpoint
CREATE INDEX "idx_knowledge_strategies_name" ON "knowledge_strategies" USING btree ("name");--> statement-breakpoint
CREATE INDEX "idx_knowledge_transfers_type" ON "knowledge_transfers" USING btree ("type");--> statement-breakpoint
CREATE INDEX "idx_knowledge_transfers_source" ON "knowledge_transfers" USING btree ("source_strategy");--> statement-breakpoint
CREATE INDEX "idx_knowledge_transfers_target" ON "knowledge_transfers" USING btree ("target_strategy");--> statement-breakpoint
CREATE INDEX "idx_knowledge_transfers_timestamp" ON "knowledge_transfers" USING btree ("timestamp");--> statement-breakpoint
CREATE INDEX "idx_knowledge_transfers_success" ON "knowledge_transfers" USING btree ("success");--> statement-breakpoint
CREATE INDEX "idx_learning_events_type" ON "learning_events" USING btree ("event_type");--> statement-breakpoint
CREATE INDEX "idx_learning_events_phi" ON "learning_events" USING btree ("phi");--> statement-breakpoint
CREATE INDEX "idx_learning_events_kernel" ON "learning_events" USING btree ("kernel_id");--> statement-breakpoint
CREATE INDEX "idx_manifold_probes_phi" ON "manifold_probes" USING btree ("phi");--> statement-breakpoint
CREATE INDEX "idx_manifold_probes_kappa" ON "manifold_probes" USING btree ("kappa");--> statement-breakpoint
CREATE INDEX "idx_manifold_probes_phi_kappa" ON "manifold_probes" USING btree ("phi","kappa");--> statement-breakpoint
CREATE INDEX "idx_manifold_probes_regime" ON "manifold_probes" USING btree ("regime");--> statement-breakpoint
CREATE INDEX "idx_manifold_probes_geometry_class" ON "manifold_probes" USING btree ("geometry_class");--> statement-breakpoint
CREATE INDEX "idx_manifold_probes_complexity" ON "manifold_probes" USING btree ("complexity");--> statement-breakpoint
CREATE INDEX "idx_narrow_path_events_severity" ON "narrow_path_events" USING btree ("severity");--> statement-breakpoint
CREATE INDEX "idx_narrow_path_events_detected_at" ON "narrow_path_events" USING btree ("detected_at");--> statement-breakpoint
CREATE INDEX "idx_near_miss_clusters_avg_phi" ON "near_miss_clusters" USING btree ("avg_phi");--> statement-breakpoint
CREATE INDEX "idx_near_miss_clusters_member_count" ON "near_miss_clusters" USING btree ("member_count");--> statement-breakpoint
CREATE UNIQUE INDEX "idx_near_miss_phrase_hash" ON "near_miss_entries" USING btree ("phrase_hash");--> statement-breakpoint
CREATE INDEX "idx_near_miss_tier" ON "near_miss_entries" USING btree ("tier");--> statement-breakpoint
CREATE INDEX "idx_near_miss_phi" ON "near_miss_entries" USING btree ("phi");--> statement-breakpoint
CREATE INDEX "idx_near_miss_cluster" ON "near_miss_entries" USING btree ("cluster_id");--> statement-breakpoint
CREATE INDEX "idx_near_miss_escalating" ON "near_miss_entries" USING btree ("is_escalating");--> statement-breakpoint
CREATE INDEX "idx_negative_knowledge_type" ON "negative_knowledge" USING btree ("type");--> statement-breakpoint
CREATE INDEX "idx_negative_knowledge_pattern" ON "negative_knowledge" USING btree ("pattern");--> statement-breakpoint
CREATE INDEX "idx_negative_knowledge_confirmed_count" ON "negative_knowledge" USING btree ("confirmed_count");--> statement-breakpoint
CREATE INDEX "idx_ocean_excluded_regions_measure" ON "ocean_excluded_regions" USING btree ("measure");--> statement-breakpoint
CREATE INDEX "idx_ocean_trajectories_address" ON "ocean_trajectories" USING btree ("address");--> statement-breakpoint
CREATE INDEX "idx_ocean_trajectories_status" ON "ocean_trajectories" USING btree ("status");--> statement-breakpoint
CREATE INDEX "idx_ocean_trajectories_address_status" ON "ocean_trajectories" USING btree ("address","status");--> statement-breakpoint
CREATE INDEX "idx_ocean_waypoints_trajectory" ON "ocean_waypoints" USING btree ("trajectory_id");--> statement-breakpoint
CREATE INDEX "idx_ocean_waypoints_trajectory_seq" ON "ocean_waypoints" USING btree ("trajectory_id","sequence");--> statement-breakpoint
CREATE INDEX "idx_pantheon_debates_status" ON "pantheon_debates" USING btree ("status");--> statement-breakpoint
CREATE INDEX "idx_pantheon_debates_initiator" ON "pantheon_debates" USING btree ("initiator");--> statement-breakpoint
CREATE INDEX "idx_pantheon_debates_started" ON "pantheon_debates" USING btree ("started_at");--> statement-breakpoint
CREATE INDEX "idx_god_state_reputation" ON "pantheon_god_state" USING btree ("reputation");--> statement-breakpoint
CREATE INDEX "idx_god_state_updated" ON "pantheon_god_state" USING btree ("updated_at");--> statement-breakpoint
CREATE INDEX "idx_pantheon_transfers_from" ON "pantheon_knowledge_transfers" USING btree ("from_god");--> statement-breakpoint
CREATE INDEX "idx_pantheon_transfers_to" ON "pantheon_knowledge_transfers" USING btree ("to_god");--> statement-breakpoint
CREATE INDEX "idx_pantheon_messages_from" ON "pantheon_messages" USING btree ("from_god");--> statement-breakpoint
CREATE INDEX "idx_pantheon_messages_to" ON "pantheon_messages" USING btree ("to_god");--> statement-breakpoint
CREATE INDEX "idx_pantheon_messages_created" ON "pantheon_messages" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "idx_pantheon_messages_type" ON "pantheon_messages" USING btree ("msg_type");--> statement-breakpoint
CREATE INDEX "idx_pending_sweeps_address" ON "pending_sweeps" USING btree ("address");--> statement-breakpoint
CREATE INDEX "idx_pending_sweeps_status" ON "pending_sweeps" USING btree ("status");--> statement-breakpoint
CREATE INDEX "idx_pending_sweeps_balance" ON "pending_sweeps" USING btree ("balance_sats");--> statement-breakpoint
CREATE INDEX "idx_pending_sweeps_discovered" ON "pending_sweeps" USING btree ("discovered_at");--> statement-breakpoint
CREATE UNIQUE INDEX "idx_provider_efficacy_provider" ON "provider_efficacy" USING btree ("provider");--> statement-breakpoint
CREATE INDEX "idx_queued_addresses_status" ON "queued_addresses" USING btree ("status");--> statement-breakpoint
CREATE INDEX "idx_queued_addresses_priority" ON "queued_addresses" USING btree ("priority");--> statement-breakpoint
CREATE INDEX "idx_queued_addresses_source" ON "queued_addresses" USING btree ("source");--> statement-breakpoint
CREATE INDEX "idx_recovery_candidates_score" ON "recovery_candidates" USING btree ("score");--> statement-breakpoint
CREATE INDEX "idx_recovery_candidates_address" ON "recovery_candidates" USING btree ("address");--> statement-breakpoint
CREATE INDEX "idx_recovery_candidates_tested_at" ON "recovery_candidates" USING btree ("tested_at");--> statement-breakpoint
CREATE UNIQUE INDEX "idx_recovery_priorities_address_unique" ON "recovery_priorities" USING btree ("address");--> statement-breakpoint
CREATE INDEX "idx_recovery_priorities_kappa" ON "recovery_priorities" USING btree ("kappa_recovery");--> statement-breakpoint
CREATE INDEX "idx_recovery_priorities_rank" ON "recovery_priorities" USING btree ("rank");--> statement-breakpoint
CREATE INDEX "idx_recovery_priorities_status" ON "recovery_priorities" USING btree ("recovery_status");--> statement-breakpoint
CREATE INDEX "idx_recovery_workflows_address" ON "recovery_workflows" USING btree ("address");--> statement-breakpoint
CREATE INDEX "idx_recovery_workflows_status" ON "recovery_workflows" USING btree ("status");--> statement-breakpoint
CREATE INDEX "idx_recovery_workflows_vector" ON "recovery_workflows" USING btree ("vector");--> statement-breakpoint
CREATE INDEX "idx_regime_boundaries_from_to" ON "regime_boundaries" USING btree ("from_regime","to_regime");--> statement-breakpoint
CREATE INDEX "idx_resonance_points_phi" ON "resonance_points" USING btree ("phi");--> statement-breakpoint
CREATE INDEX "idx_resonance_points_cluster_strength" ON "resonance_points" USING btree ("cluster_strength");--> statement-breakpoint
CREATE INDEX "idx_search_feedback_outcome" ON "search_feedback" USING btree ("outcome_quality");--> statement-breakpoint
CREATE INDEX "idx_search_feedback_created" ON "search_feedback" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "idx_search_outcomes_date" ON "search_outcomes" USING btree ("date");--> statement-breakpoint
CREATE INDEX "idx_search_outcomes_provider" ON "search_outcomes" USING btree ("provider");--> statement-breakpoint
CREATE INDEX "idx_search_outcomes_kernel" ON "search_outcomes" USING btree ("kernel_id");--> statement-breakpoint
CREATE INDEX "IDX_session_expire" ON "sessions" USING btree ("expire");--> statement-breakpoint
CREATE INDEX "idx_shadow_intel_target" ON "shadow_intel" USING btree ("target");--> statement-breakpoint
CREATE INDEX "idx_shadow_intel_consensus" ON "shadow_intel" USING btree ("consensus");--> statement-breakpoint
CREATE INDEX "idx_shadow_intel_phi" ON "shadow_intel" USING btree ("phi");--> statement-breakpoint
CREATE INDEX "idx_shadow_ops_god" ON "shadow_operations_log" USING btree ("god_name");--> statement-breakpoint
CREATE INDEX "idx_shadow_ops_type" ON "shadow_operations_log" USING btree ("operation_type");--> statement-breakpoint
CREATE INDEX "idx_shadow_ops_created" ON "shadow_operations_log" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "idx_shadow_pantheon_intel_target" ON "shadow_pantheon_intel" USING btree ("target");--> statement-breakpoint
CREATE INDEX "idx_shadow_pantheon_intel_risk" ON "shadow_pantheon_intel" USING btree ("risk_level");--> statement-breakpoint
CREATE INDEX "idx_shadow_pantheon_intel_created" ON "shadow_pantheon_intel" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "idx_sweep_audit_log_sweep" ON "sweep_audit_log" USING btree ("sweep_id");--> statement-breakpoint
CREATE INDEX "idx_sweep_audit_log_action" ON "sweep_audit_log" USING btree ("action");--> statement-breakpoint
CREATE INDEX "idx_sweep_audit_log_timestamp" ON "sweep_audit_log" USING btree ("timestamp");--> statement-breakpoint
CREATE INDEX "idx_telemetry_session" ON "telemetry_snapshots" USING btree ("session_id");--> statement-breakpoint
CREATE INDEX "idx_telemetry_regime" ON "telemetry_snapshots" USING btree ("regime");--> statement-breakpoint
CREATE INDEX "idx_telemetry_phi" ON "telemetry_snapshots" USING btree ("phi");--> statement-breakpoint
CREATE INDEX "idx_telemetry_kappa" ON "telemetry_snapshots" USING btree ("kappa");--> statement-breakpoint
CREATE INDEX "idx_telemetry_created" ON "telemetry_snapshots" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "idx_tested_phrases_phrase" ON "tested_phrases" USING btree ("phrase");--> statement-breakpoint
CREATE INDEX "idx_tested_phrases_tested_at" ON "tested_phrases" USING btree ("tested_at");--> statement-breakpoint
CREATE INDEX "idx_tested_phrases_balance" ON "tested_phrases" USING btree ("balance_sats");--> statement-breakpoint
CREATE INDEX "idx_tested_phrases_retest_count" ON "tested_phrases" USING btree ("retest_count");--> statement-breakpoint
CREATE INDEX "idx_tested_phrases_date" ON "tested_phrases_index" USING btree ("tested_at");--> statement-breakpoint
CREATE UNIQUE INDEX "idx_tokenizer_merge_rules_pair" ON "tokenizer_merge_rules" USING btree ("token_a","token_b");--> statement-breakpoint
CREATE INDEX "idx_tokenizer_merge_rules_phi" ON "tokenizer_merge_rules" USING btree ("phi_score");--> statement-breakpoint
CREATE INDEX "idx_tokenizer_merge_rules_merged" ON "tokenizer_merge_rules" USING btree ("merged_token");--> statement-breakpoint
CREATE INDEX "idx_tokenizer_vocab_token_id" ON "tokenizer_vocabulary" USING btree ("token_id");--> statement-breakpoint
CREATE INDEX "idx_tokenizer_vocab_phi" ON "tokenizer_vocabulary" USING btree ("phi_score");--> statement-breakpoint
CREATE INDEX "idx_tokenizer_vocab_weight" ON "tokenizer_vocabulary" USING btree ("weight");--> statement-breakpoint
CREATE INDEX "idx_tool_observations_timestamp" ON "tool_observations" USING btree ("timestamp");--> statement-breakpoint
CREATE INDEX "idx_tool_observations_cluster" ON "tool_observations" USING btree ("cluster_assigned");--> statement-breakpoint
CREATE INDEX "idx_tool_patterns_pattern_id" ON "tool_patterns" USING btree ("pattern_id");--> statement-breakpoint
CREATE INDEX "idx_tool_patterns_source_type" ON "tool_patterns" USING btree ("source_type");--> statement-breakpoint
CREATE INDEX "idx_tool_patterns_phi" ON "tool_patterns" USING btree ("phi");--> statement-breakpoint
CREATE INDEX "idx_tps_geodesic_from_to" ON "tps_geodesic_paths" USING btree ("from_landmark","to_landmark");--> statement-breakpoint
CREATE INDEX "idx_tps_landmarks_era" ON "tps_landmarks" USING btree ("era");--> statement-breakpoint
CREATE INDEX "idx_tps_landmarks_timestamp" ON "tps_landmarks" USING btree ("spacetime_t");--> statement-breakpoint
CREATE INDEX "idx_training_batch_god" ON "training_batch_queue" USING btree ("god_name");--> statement-breakpoint
CREATE INDEX "idx_training_batch_processed" ON "training_batch_queue" USING btree ("processed");--> statement-breakpoint
CREATE INDEX "idx_training_batch_created" ON "training_batch_queue" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "idx_transactions_block_height" ON "transactions" USING btree ("block_height");--> statement-breakpoint
CREATE INDEX "idx_transactions_timestamp" ON "transactions" USING btree ("block_timestamp");--> statement-breakpoint
CREATE INDEX "idx_usage_metrics_date" ON "usage_metrics" USING btree ("date");--> statement-breakpoint
CREATE INDEX "idx_user_target_addresses_user" ON "user_target_addresses" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "idx_vocabulary_observations_phi" ON "vocabulary_observations" USING btree ("max_phi");--> statement-breakpoint
CREATE INDEX "idx_vocabulary_observations_integrated" ON "vocabulary_observations" USING btree ("is_integrated");--> statement-breakpoint
CREATE INDEX "idx_vocabulary_observations_type" ON "vocabulary_observations" USING btree ("type");--> statement-breakpoint
CREATE INDEX "idx_vocabulary_observations_real_word" ON "vocabulary_observations" USING btree ("is_real_word");--> statement-breakpoint
CREATE INDEX "idx_vocabulary_observations_cycle" ON "vocabulary_observations" USING btree ("cycle_number");--> statement-breakpoint
CREATE INDEX "idx_war_history_mode" ON "war_history" USING btree ("mode");--> statement-breakpoint
CREATE INDEX "idx_war_history_status" ON "war_history" USING btree ("status");--> statement-breakpoint
CREATE INDEX "idx_war_history_declared_at" ON "war_history" USING btree ("declared_at");