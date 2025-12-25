/**
 * Blockchain API Response Types
 *
 * Typed interfaces for responses from blockchain data providers:
 * - Blockstream API
 * - blockchain.info API
 * - Mempool.space API
 * - BlockCypher API
 */

// =============================================================================
// BLOCKSTREAM API TYPES
// =============================================================================

/** Blockstream address stats */
export interface BlockstreamAddressStats {
  funded_txo_count: number;
  funded_txo_sum: number;
  spent_txo_count: number;
  spent_txo_sum: number;
  tx_count: number;
}

/** Blockstream address response */
export interface BlockstreamAddressResponse {
  address: string;
  chain_stats: BlockstreamAddressStats;
  mempool_stats: BlockstreamAddressStats;
}

/** Blockstream transaction output */
export interface BlockstreamTxVout {
  scriptpubkey: string;
  scriptpubkey_asm: string;
  scriptpubkey_type: string;
  scriptpubkey_address?: string;
  value: number;
}

/** Blockstream transaction input prevout */
export interface BlockstreamTxPrevout {
  scriptpubkey: string;
  scriptpubkey_asm: string;
  scriptpubkey_type: string;
  scriptpubkey_address?: string;
  value: number;
}

/** Blockstream transaction input */
export interface BlockstreamTxVin {
  txid: string;
  vout: number;
  prevout?: BlockstreamTxPrevout;
  scriptsig: string;
  scriptsig_asm: string;
  witness?: string[];
  is_coinbase: boolean;
  sequence: number;
}

/** Blockstream transaction status */
export interface BlockstreamTxStatus {
  confirmed: boolean;
  block_height?: number;
  block_hash?: string;
  block_time?: number;
}

/** Blockstream transaction */
export interface BlockstreamTransaction {
  txid: string;
  version: number;
  locktime: number;
  vin: BlockstreamTxVin[];
  vout: BlockstreamTxVout[];
  size: number;
  weight: number;
  fee: number;
  status: BlockstreamTxStatus;
}

// =============================================================================
// BLOCKCHAIN.INFO API TYPES
// =============================================================================

/** blockchain.info transaction output */
export interface BlockchainInfoTxOutput {
  n: number;
  value: number;
  addr?: string;
  tx_index: number;
  script: string;
  spent: boolean;
}

/** blockchain.info transaction input */
export interface BlockchainInfoTxInput {
  sequence: number;
  witness: string;
  script: string;
  index: number;
  prev_out?: BlockchainInfoTxOutput;
}

/** blockchain.info transaction */
export interface BlockchainInfoTransaction {
  hash: string;
  ver: number;
  vin_sz: number;
  vout_sz: number;
  size: number;
  weight: number;
  fee: number;
  relayed_by: string;
  lock_time: number;
  tx_index: number;
  double_spend: boolean;
  time: number;
  block_index?: number;
  block_height?: number;
  inputs: BlockchainInfoTxInput[];
  out: BlockchainInfoTxOutput[];
  // Unified format aliases
  vin?: BlockchainInfoTxInput[];
  vout?: BlockchainInfoTxOutput[];
}

/** blockchain.info raw address response */
export interface BlockchainInfoRawAddressResponse {
  address: string;
  hash160: string;
  n_tx: number;
  n_unredeemed: number;
  total_received: number;
  total_sent: number;
  final_balance: number;
  txs: BlockchainInfoTransaction[];
  // Fallback flags
  _blockstreamFallback?: boolean;
  _mempoolFallback?: boolean;
  _blockcypherFallback?: boolean;
  _formatFallback?: boolean;
  _estimatedEra?: string;
  _estimatedYear?: number;
}

// =============================================================================
// MEMPOOL.SPACE API TYPES
// =============================================================================

/** Mempool.space address stats */
export interface MempoolAddressStats {
  funded_txo_count: number;
  funded_txo_sum: number;
  spent_txo_count: number;
  spent_txo_sum: number;
  tx_count: number;
}

/** Mempool.space address response */
export interface MempoolAddressResponse {
  address: string;
  chain_stats: MempoolAddressStats;
  mempool_stats?: MempoolAddressStats;
}

// =============================================================================
// BLOCKCYPHER API TYPES
// =============================================================================

/** BlockCypher transaction reference */
export interface BlockCypherTxRef {
  tx_hash: string;
  block_height: number;
  tx_input_n: number;
  tx_output_n: number;
  value: number;
  ref_balance: number;
  confirmations: number;
  confirmed: string;
  double_spend: boolean;
}

/** BlockCypher address response */
export interface BlockCypherAddressResponse {
  address: string;
  total_received: number;
  total_sent: number;
  balance: number;
  unconfirmed_balance: number;
  final_balance: number;
  n_tx: number;
  unconfirmed_n_tx: number;
  final_n_tx: number;
  txrefs?: BlockCypherTxRef[];
  tx_url: string;
}

// =============================================================================
// UNIFIED/NORMALIZED TYPES
// =============================================================================

/** Unified transaction format for internal use */
export interface UnifiedTransaction {
  hash: string;
  time?: number;
  block_height?: number;
  vin?: Array<{
    prevout?: {
      scriptpubkey_address?: string;
      scriptpubkey_type?: string;
      scriptpubkey?: string;
      value?: number;
    };
  }>;
  vout?: Array<{
    scriptpubkey_address?: string;
    scriptpubkey_type?: string;
    scriptpubkey?: string;
    value?: number;
  }>;
  value?: number;  // For BlockCypher simplified format
}

/** Unified address data format for internal use */
export interface UnifiedAddressData {
  address: string;
  total_received: number;
  total_sent: number;
  final_balance: number;
  n_tx: number;
  txs: UnifiedTransaction[];
  // Source tracking
  _blockstreamFallback?: boolean;
  _mempoolFallback?: boolean;
  _blockcypherFallback?: boolean;
  _formatFallback?: boolean;
  _estimatedEra?: string;
  _estimatedYear?: number;
}

// =============================================================================
// ADDRESS SIGNATURE TYPES
// =============================================================================

/** Temporal signature for address activity patterns */
export interface TemporalSignature {
  hourPattern?: number[];  // Activity by hour (0-23)
  dayPattern?: number[];   // Activity by day of week (0-6)
  monthPattern?: number[]; // Activity by month (0-11)
  firstSeen?: number;      // Unix timestamp
  lastSeen?: number;       // Unix timestamp
  activityDensity?: number;
}

/** Graph signature for address connectivity */
export interface GraphSignature {
  inputAddresses?: string[];
  outputAddresses?: string[];
  clusterSize?: number;
  clusterIds?: string[];
  commonSpenders?: string[];
  hopDistance?: number;
}

/** Value signature for transaction patterns */
export interface ValueSignature {
  hasRoundNumbers?: boolean;
  roundNumberFrequency?: number;
  avgTxValue?: number;
  maxTxValue?: number;
  minTxValue?: number;
  patternStrength?: number;
  dustOutputCount?: number;
}

/** Script signature for software fingerprinting */
export interface ScriptSignature {
  scriptType?: 'p2pkh' | 'p2sh' | 'p2wpkh' | 'p2wsh' | 'p2pk' | 'p2tr' | 'unknown';
  softwareFingerprint?: string;
  complexity?: number;
  isMultisig?: boolean;
  requiredSigs?: number;
}

/** Combined address signatures */
export interface AddressSignatures {
  temporalSignature?: TemporalSignature;
  graphSignature?: GraphSignature;
  valueSignature?: ValueSignature;
  scriptSignature?: ScriptSignature;
}
