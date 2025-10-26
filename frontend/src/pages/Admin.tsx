/**
 * Admin Dashboard Page
 *
 * Provides admin-only operations for:
 * - Data collection management
 * - Training data generation
 * - Model training orchestration
 * - System health monitoring
 */

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import adminApi from '@/features/admin/api/adminApi';
import type { SystemHealth, ModelInfo } from '@/features/admin/api/adminApi';

type TabType = 'data' | 'training' | 'models' | 'health';

export function Admin() {
  const [activeTab, setActiveTab] = useState<TabType>('data');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Data Collection State
  const [collectionSources, setCollectionSources] = useState<string[]>(['amazon']);
  const [collectionCategories, setCollectionCategories] = useState('');
  const [maxResults, setMaxResults] = useState('100');
  const [collectionJobId, setCollectionJobId] = useState<string | null>(null);
  const [collectionStatus, setCollectionStatus] = useState<any>(null);

  // Training Data State
  const [trainingExamples, setTrainingExamples] = useState('100');
  const [trainingDomains, setTrainingDomains] = useState('');

  // Model Training State
  const [modelName, setModelName] = useState('');
  const [baseModel, setBaseModel] = useState('meta-llama/Llama-2-7b-hf');
  const [trainingJobId, setTrainingJobId] = useState<string | null>(null);

  // System State
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);

  const showMessage = (type: 'success' | 'error', text: string) => {
    setMessage({ type, text });
    setTimeout(() => setMessage(null), 5000);
  };

  // Data Collection Operations
  const handleTriggerCollection = async () => {
    setLoading(true);
    try {
      const categories = collectionCategories
        ? collectionCategories.split(',').map(c => c.trim())
        : undefined;

      const result = await adminApi.triggerCollection({
        sources: collectionSources,
        categories,
        max_results: parseInt(maxResults),
      });

      setCollectionJobId(result.job.job_id);
      showMessage('success', `Collection job started: ${result.job.job_id}`);
    } catch (error: any) {
      showMessage('error', error.response?.data?.detail || 'Failed to start collection');
    } finally {
      setLoading(false);
    }
  };

  const handleCheckCollectionStatus = async () => {
    if (!collectionJobId) return;

    setLoading(true);
    try {
      const result = await adminApi.getCollectionStatus(collectionJobId);
      setCollectionStatus(result.status);
      showMessage('success', 'Status updated');
    } catch (error: any) {
      showMessage('error', error.response?.data?.detail || 'Failed to check status');
    } finally {
      setLoading(false);
    }
  };

  // Training Data Operations
  const handleGenerateTrainingData = async () => {
    setLoading(true);
    try {
      const domains = trainingDomains
        ? trainingDomains.split(',').map(d => d.trim())
        : undefined;

      const result = await adminApi.generateTrainingData({
        num_examples: parseInt(trainingExamples),
        domains,
      });

      showMessage('success', 'Training data generation started successfully');
    } catch (error: any) {
      showMessage('error', error.response?.data?.detail || 'Failed to generate training data');
    } finally {
      setLoading(false);
    }
  };

  // Model Training Operations
  const handleStartTraining = async () => {
    if (!modelName) {
      showMessage('error', 'Model name is required');
      return;
    }

    setLoading(true);
    try {
      const result = await adminApi.startTraining({
        model_name: modelName,
        base_model: baseModel || undefined,
      });

      setTrainingJobId(result.training.job_id);
      showMessage('success', `Training job started: ${result.training.job_id}`);
    } catch (error: any) {
      showMessage('error', error.response?.data?.detail || 'Failed to start training');
    } finally {
      setLoading(false);
    }
  };

  const handleCheckTrainingStatus = async () => {
    if (!trainingJobId) return;

    setLoading(true);
    try {
      const status = await adminApi.getTrainingStatus(trainingJobId);
      showMessage('success', `Job ${trainingJobId}: ${JSON.stringify(status.status)}`);
    } catch (error: any) {
      showMessage('error', error.response?.data?.detail || 'Failed to check training status');
    } finally {
      setLoading(false);
    }
  };

  // Load models
  const loadModels = async () => {
    setLoading(true);
    try {
      const result = await adminApi.listModels();
      setModels(result.models);
    } catch (error: any) {
      showMessage('error', error.response?.data?.detail || 'Failed to load models');
    } finally {
      setLoading(false);
    }
  };

  // Load system health
  const loadSystemHealth = async () => {
    setLoading(true);
    try {
      const health = await adminApi.getSystemHealth();
      setSystemHealth(health);
    } catch (error: any) {
      showMessage('error', error.response?.data?.detail || 'Failed to load system health');
    } finally {
      setLoading(false);
    }
  };

  // Load data on tab change
  useEffect(() => {
    if (activeTab === 'models') {
      loadModels();
    } else if (activeTab === 'health') {
      loadSystemHealth();
    }
  }, [activeTab]);

  const getHealthBadgeVariant = (status: string) => {
    if (status === 'healthy') return 'default';
    if (status.includes('unhealthy')) return 'destructive';
    return 'secondary';
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Admin Dashboard</h1>
        <p className="text-gray-600">Manage data collection, training, and system operations</p>
      </div>

      {/* Message Alert */}
      {message && (
        <Alert className={`mb-6 ${message.type === 'error' ? 'border-red-500' : 'border-green-500'}`}>
          <AlertDescription>{message.text}</AlertDescription>
        </Alert>
      )}

      {/* Tab Navigation */}
      <div className="flex gap-2 mb-6 border-b">
        <button
          onClick={() => setActiveTab('data')}
          className={`px-4 py-2 font-medium ${
            activeTab === 'data'
              ? 'border-b-2 border-blue-600 text-blue-600'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          Data Collection
        </button>
        <button
          onClick={() => setActiveTab('training')}
          className={`px-4 py-2 font-medium ${
            activeTab === 'training'
              ? 'border-b-2 border-blue-600 text-blue-600'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          Training
        </button>
        <button
          onClick={() => setActiveTab('models')}
          className={`px-4 py-2 font-medium ${
            activeTab === 'models'
              ? 'border-b-2 border-blue-600 text-blue-600'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          Models
        </button>
        <button
          onClick={() => setActiveTab('health')}
          className={`px-4 py-2 font-medium ${
            activeTab === 'health'
              ? 'border-b-2 border-blue-600 text-blue-600'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          System Health
        </button>
      </div>

      {/* Data Collection Tab */}
      {activeTab === 'data' && (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Trigger Product Collection</CardTitle>
              <CardDescription>
                Start data collection from e-commerce sources
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Sources</label>
                <div className="flex gap-2">
                  {['amazon', 'bestbuy', 'walmart', 'ebay'].map(source => (
                    <Button
                      key={source}
                      variant={collectionSources.includes(source) ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => {
                        setCollectionSources(prev =>
                          prev.includes(source)
                            ? prev.filter(s => s !== source)
                            : [...prev, source]
                        );
                      }}
                    >
                      {source}
                    </Button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Categories (comma-separated, optional)
                </label>
                <Input
                  value={collectionCategories}
                  onChange={e => setCollectionCategories(e.target.value)}
                  placeholder="e.g., Electronics, Fashion, Home"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Max Results</label>
                <Input
                  type="number"
                  value={maxResults}
                  onChange={e => setMaxResults(e.target.value)}
                  placeholder="100"
                />
              </div>

              <Button onClick={handleTriggerCollection} disabled={loading || collectionSources.length === 0}>
                {loading ? 'Starting...' : 'Start Collection'}
              </Button>

              {collectionJobId && (
                <div className="mt-4 p-4 bg-gray-50 rounded space-y-3">
                  <p className="text-sm font-medium">Active Job: {collectionJobId}</p>
                  <Button variant="outline" size="sm" onClick={handleCheckCollectionStatus} disabled={loading}>
                    {loading ? 'Checking...' : 'Check Status'}
                  </Button>

                  {collectionStatus && (
                    <div className="mt-3 p-3 bg-white rounded border space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Status:</span>
                        <Badge variant={
                          collectionStatus.status === 'completed' ? 'default' :
                          collectionStatus.status === 'running' ? 'secondary' :
                          collectionStatus.status === 'failed' ? 'destructive' : 'outline'
                        }>
                          {collectionStatus.status}
                        </Badge>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-600">Progress:</span>
                        <span className="font-medium">{collectionStatus.progress || 0}%</span>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-600">Products Collected:</span>
                        <span className="font-medium">{collectionStatus.products_collected || 0}</span>
                      </div>
                      {collectionStatus.errors_count > 0 && (
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-gray-600">Errors:</span>
                          <span className="font-medium text-red-600">{collectionStatus.errors_count}</span>
                        </div>
                      )}
                      {collectionStatus.started_at && (
                        <div className="text-xs text-gray-500 pt-2 border-t">
                          Started: {new Date(collectionStatus.started_at).toLocaleString()}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Training Tab */}
      {activeTab === 'training' && (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Generate Training Data</CardTitle>
              <CardDescription>
                Generate synthetic training data using OpenAI
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Number of Examples</label>
                <Input
                  type="number"
                  value={trainingExamples}
                  onChange={e => setTrainingExamples(e.target.value)}
                  placeholder="100"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Domains (comma-separated, optional)
                </label>
                <Input
                  value={trainingDomains}
                  onChange={e => setTrainingDomains(e.target.value)}
                  placeholder="e.g., electronics, fashion, home"
                />
              </div>

              <Button onClick={handleGenerateTrainingData} disabled={loading}>
                {loading ? 'Generating...' : 'Generate Data'}
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Start Model Training</CardTitle>
              <CardDescription>
                Fine-tune a model using QLoRA
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Model Name *</label>
                <Input
                  value={modelName}
                  onChange={e => setModelName(e.target.value)}
                  placeholder="e.g., shopping_advisor_v3"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Base Model</label>
                <Input
                  value={baseModel}
                  onChange={e => setBaseModel(e.target.value)}
                  placeholder="meta-llama/Llama-2-7b-hf"
                />
              </div>

              <Button onClick={handleStartTraining} disabled={loading || !modelName}>
                {loading ? 'Starting...' : 'Start Training'}
              </Button>

              {trainingJobId && (
                <div className="mt-4 p-4 bg-gray-50 rounded">
                  <p className="text-sm font-medium mb-2">Active Job: {trainingJobId}</p>
                  <Button variant="outline" size="sm" onClick={handleCheckTrainingStatus} disabled={loading}>
                    Check Status
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Models Tab */}
      {activeTab === 'models' && (
        <Card>
          <CardHeader>
            <CardTitle>Available Models</CardTitle>
            <CardDescription>Trained models in the Knowledge Engine</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <p className="text-gray-500">Loading models...</p>
            ) : models.length === 0 ? (
              <p className="text-gray-500">No models available</p>
            ) : (
              <div className="space-y-2">
                {models.map((model, idx) => (
                  <div key={idx} className="p-4 border rounded hover:bg-gray-50">
                    <div className="font-medium">{model.name || model.model_id}</div>
                    <div className="text-sm text-gray-600">{model.model_id}</div>
                    {model.created_at && (
                      <div className="text-xs text-gray-500 mt-1">
                        Created: {new Date(model.created_at).toLocaleString()}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
            <Button
              variant="outline"
              className="mt-4"
              onClick={loadModels}
              disabled={loading}
            >
              Refresh
            </Button>
          </CardContent>
        </Card>
      )}

      {/* System Health Tab */}
      {activeTab === 'health' && (
        <Card>
          <CardHeader>
            <CardTitle>System Health</CardTitle>
            <CardDescription>Health status of all engines</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <p className="text-gray-500">Loading health status...</p>
            ) : systemHealth ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 border rounded">
                  <span className="font-medium">Advisory Engine</span>
                  <Badge variant={getHealthBadgeVariant(systemHealth.advisory_engine)}>
                    {systemHealth.advisory_engine}
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-3 border rounded">
                  <span className="font-medium">Knowledge Engine</span>
                  <Badge variant={getHealthBadgeVariant(systemHealth.knowledge_engine)}>
                    {systemHealth.knowledge_engine}
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-3 border rounded">
                  <span className="font-medium">Discovery Engine</span>
                  <Badge variant={getHealthBadgeVariant(systemHealth.discovery_engine)}>
                    {systemHealth.discovery_engine}
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-3 border rounded bg-gray-50">
                  <span className="font-bold">Overall Status</span>
                  <Badge variant={getHealthBadgeVariant(systemHealth.overall)}>
                    {systemHealth.overall}
                  </Badge>
                </div>
              </div>
            ) : (
              <p className="text-gray-500">No health data available</p>
            )}
            <Button
              variant="outline"
              className="mt-4"
              onClick={loadSystemHealth}
              disabled={loading}
            >
              Refresh
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
