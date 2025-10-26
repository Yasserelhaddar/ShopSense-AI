/**
 * Admin API client for ShopSense-AI
 *
 * Provides admin-only operations for:
 * - Data collection management
 * - Training data generation
 * - Model training orchestration
 * - System health monitoring
 */

import { advisoryApi } from '@/lib/api/axios';

export interface CollectionJob {
  job_id: string;
  status: string;
  sources: string[];
  estimated_duration?: string;
}

export interface TrainingJob {
  job_id: string;
  status: string;
  model_name: string;
}

export interface ModelInfo {
  model_id: string;
  name: string;
  created_at: string;
  metrics?: Record<string, any>;
}

export interface SystemHealth {
  advisory_engine: string;
  knowledge_engine: string;
  discovery_engine: string;
  overall: string;
}

export interface SystemStats {
  categories: string[];
  stores: string[];
  total_categories: number;
  total_stores: number;
}

/**
 * Data Collection Operations
 */

export const triggerCollection = async (params: {
  sources: string[];
  categories?: string[];
  max_results?: number;
}): Promise<{ status: string; job: CollectionJob }> => {
  const response = await advisoryApi.post('/admin/data/collect', params);
  return response.data;
};

export const getCollectionStatus = async (jobId: string): Promise<any> => {
  const response = await advisoryApi.get(`/admin/data/collection/${jobId}`);
  return response.data;
};

/**
 * Training Data Operations
 */

export const generateTrainingData = async (params: {
  num_examples?: number;
  domains?: string[];
}): Promise<{ status: string; data_generation: any }> => {
  const response = await advisoryApi.post('/admin/training/generate-data', params);
  return response.data;
};

/**
 * Model Training Operations
 */

export const startTraining = async (params: {
  model_name: string;
  base_model?: string;
  training_params?: Record<string, any>;
}): Promise<{ status: string; training: TrainingJob }> => {
  const response = await advisoryApi.post('/admin/training/start', params);
  return response.data;
};

export const getTrainingStatus = async (jobId: string): Promise<any> => {
  const response = await advisoryApi.get(`/admin/training/status/${jobId}`);
  return response.data;
};

export const listModels = async (): Promise<{ models: ModelInfo[]; total: number }> => {
  const response = await advisoryApi.get('/admin/models');
  return response.data;
};

/**
 * System Monitoring Operations
 */

export const getSystemHealth = async (): Promise<SystemHealth> => {
  const response = await advisoryApi.get('/admin/system/health');
  return response.data;
};

export const getSystemStats = async (): Promise<SystemStats> => {
  const response = await advisoryApi.get('/admin/system/stats');
  return response.data;
};

/**
 * Admin API client object with all methods
 */
const adminApi = {
  // Data collection
  triggerCollection,
  getCollectionStatus,

  // Training data
  generateTrainingData,

  // Model training
  startTraining,
  getTrainingStatus,
  listModels,

  // System monitoring
  getSystemHealth,
  getSystemStats,
};

export default adminApi;
