/**
 * STORAGE FACADE
 * 
 * Central access point for all persistence operations.
 * Exposes domain-specific repositories with configurable backends.
 * 
 * Usage:
 *   import { storageFacade } from './persistence';
 *   const candidates = await storageFacade.candidates.getCandidates();
 * 
 * The facade uses concrete Postgres adapters that implement the domain
 * interfaces. New modules should depend on the facade, not on storage.ts
 * or ocean-persistence.ts directly.
 */

import type {
  ICandidateStorage,
  ITargetAddressStorage,
  IUserStorage,
  IOceanProbeStorage,
  ISearchJobStorage,
  StorageConfig,
} from './interfaces';
import {
  UserPostgresAdapter,
} from './adapters';
import { db } from '../db';
import { oceanPersistence } from '../ocean/ocean-persistence';
import type { Candidate, TargetAddress } from '@shared/schema';

class InMemoryCandidateStorage implements ICandidateStorage {
  private candidates: Candidate[] = [];

  async getCandidates(): Promise<Candidate[]> {
    return [...this.candidates];
  }

  async addCandidate(candidate: Candidate): Promise<void> {
    const existing = this.candidates.findIndex(c => c.id === candidate.id);
    if (existing >= 0) {
      this.candidates[existing] = candidate;
    } else {
      this.candidates.push(candidate);
    }
  }

  async clearCandidates(): Promise<void> {
    this.candidates = [];
  }
}

class InMemoryTargetAddressStorage implements ITargetAddressStorage {
  private addresses: TargetAddress[] = [];

  async getTargetAddresses(): Promise<TargetAddress[]> {
    return [...this.addresses];
  }

  async addTargetAddress(address: TargetAddress): Promise<void> {
    const existing = this.addresses.findIndex(a => a.id === address.id);
    if (existing >= 0) {
      this.addresses[existing] = address;
    } else {
      this.addresses.push(address);
    }
  }

  async removeTargetAddress(id: string): Promise<void> {
    this.addresses = this.addresses.filter(a => a.id !== id);
  }
}

class StorageFacade {
  private _candidates: ICandidateStorage;
  private _targetAddresses: ITargetAddressStorage;
  private _users: IUserStorage;
  private _oceanProbes: IOceanProbeStorage | null = null;
  private _searchJobs: ISearchJobStorage;
  private _config: StorageConfig;

  constructor(config?: Partial<StorageConfig>) {
    this._config = {
      backend: config?.backend ?? 'postgres',
      dataDir: config?.dataDir,
    };

    if (this._config.backend !== 'postgres') {
      throw new Error('[StorageFacade] Only postgres backend is supported. Set backend to "postgres".');
    }

    if (!db) {
      throw new Error('[StorageFacade] DATABASE_URL not set - postgres backend is required for persistence');
    }

    this._candidates = new InMemoryCandidateStorage();
    this._targetAddresses = new InMemoryTargetAddressStorage();
    this._users = new UserPostgresAdapter();
    
    this._searchJobs = {
      getSearchJobs: async () => [],
      getSearchJob: async () => null,
      addSearchJob: async () => {},
      updateSearchJob: async () => {},
      appendJobLog: async () => {},
      deleteSearchJob: async () => {},
    };

    if (oceanPersistence.isPersistenceAvailable()) {
      this._oceanProbes = {
        insertProbes: (probes) => oceanPersistence.insertProbes(probes),
        queryProbesByPhiKappa: (range, limit) => oceanPersistence.queryProbesByPhiKappa(range, limit),
        queryProbesByRegime: (regime, limit) => oceanPersistence.queryProbesByRegime(regime, limit),
        getHighPhiProbes: (minPhi, limit) => oceanPersistence.getHighPhiProbes(minPhi, limit),
        getProbeCount: () => oceanPersistence.getProbeCount(),
      };
    }

    console.log(`[StorageFacade] Initialized with backend: ${this._config.backend}`);
  }

  get config(): StorageConfig {
    return { ...this._config };
  }

  get candidates(): ICandidateStorage {
    return this._candidates;
  }

  get targetAddresses(): ITargetAddressStorage {
    return this._targetAddresses;
  }

  get users(): IUserStorage {
    return this._users;
  }

  get oceanProbes(): IOceanProbeStorage | null {
    return this._oceanProbes;
  }

  get searchJobs(): ISearchJobStorage {
    return this._searchJobs;
  }

  isOceanPersistenceAvailable(): boolean {
    return this._oceanProbes !== null;
  }
}

export const storageFacade = new StorageFacade({ backend: 'postgres' });
