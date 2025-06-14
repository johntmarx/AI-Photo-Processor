import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Progress } from '@/components/ui/Progress';
import {
  CloudDownload,
  PlayArrow,
  CheckCircle,
  AlertCircle,
  Loader2
} from 'lucide-react';

interface BatchSession {
  session_id: string;
  status: string;
  current_stage?: string;
  processed_count: number;
  total_count: number;
  progress: number;
  error?: string;
}

export default function BatchProcessing() {
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [session, setSession] = useState<BatchSession | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Form state
  const [minAestheticScore, setMinAestheticScore] = useState<number | null>(5.0);
  const [minTechnicalScore, setMinTechnicalScore] = useState<number | null>(5.0);
  const [maxPhotos, setMaxPhotos] = useState<number | null>(50);
  
  const [rotationEnabled, setRotationEnabled] = useState(true);
  const [rotationMethod, setRotationMethod] = useState('cv');
  
  const [cropEnabled, setCropEnabled] = useState(true);
  const [cropMethod, setCropMethod] = useState('vlm');
  const [cropAspectRatio, setCropAspectRatio] = useState('original');
  const [cropVlmModel, setCropVlmModel] = useState('qwen2.5-vl:7b');
  
  const [enhanceEnabled, setEnhanceEnabled] = useState(true);
  const [enhanceStrength, setEnhanceStrength] = useState(1.0);

  // Poll for session status
  useEffect(() => {
    if (!sessionId) return;

    const interval = setInterval(async () => {
      try {
        const response = await fetch(`/api/batch/${sessionId}/status`);
        if (response.ok) {
          const data = await response.json();
          setSession(data);
          
          if (data.status === 'completed' || data.status === 'failed') {
            clearInterval(interval);
          }
        }
      } catch (err) {
        console.error('Failed to fetch session status:', err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [sessionId]);

  const startBatchProcessing = async () => {
    setLoading(true);
    setError(null);

    try {
      const request = {
        min_aesthetic_score: minAestheticScore,
        min_technical_score: minTechnicalScore,
        max_photos: maxPhotos,
        rotation_enabled: rotationEnabled,
        rotation_method: rotationMethod,
        crop_enabled: cropEnabled,
        crop_method: cropMethod,
        crop_vlm_model: cropVlmModel,
        crop_aspect_ratio: cropAspectRatio,
        enhance_enabled: enhanceEnabled,
        enhance_strength: enhanceStrength,
        output_format: 'jpeg',
        jpeg_quality: 95
      };

      const response = await fetch(`/api/batch/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });

      if (!response.ok) {
        throw new Error('Failed to start batch processing');
      }

      const data = await response.json();
      setSessionId(data.session_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start batch processing');
    } finally {
      setLoading(false);
    }
  };

  const downloadResults = () => {
    if (!sessionId) return;
    window.open(`/api/batch/${sessionId}/download`, '_blank');
  };

  const getStatusIcon = () => {
    if (!session) return null;
    
    switch (session.status) {
      case 'completed':
        return <CheckCircle className="h-12 w-12 text-green-500" />;
      case 'failed':
        return <AlertCircle className="h-12 w-12 text-red-500" />;
      default:
        return <Loader2 className="h-12 w-12 animate-spin text-primary" />;
    }
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      <div>
        <h1 className="text-3xl font-bold">Batch Processing</h1>
        <p className="text-muted-foreground mt-2">
          Process multiple photos at once with quality filtering and automated enhancements
        </p>
      </div>

      {!sessionId ? (
        <Card>
          <CardHeader>
            <CardTitle>Processing Options</CardTitle>
            <CardDescription>
              Configure your batch processing settings
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Photo Selection */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Photo Selection Criteria</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Minimum Aesthetic Score
                  </label>
                  <Input
                    type="number"
                    value={minAestheticScore || ''}
                    onChange={(e) => setMinAestheticScore(e.target.value ? parseFloat(e.target.value) : null)}
                    min={0}
                    max={10}
                    step={0.5}
                    placeholder="0-10"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Minimum Technical Score
                  </label>
                  <Input
                    type="number"
                    value={minTechnicalScore || ''}
                    onChange={(e) => setMinTechnicalScore(e.target.value ? parseFloat(e.target.value) : null)}
                    min={0}
                    max={10}
                    step={0.5}
                    placeholder="0-10"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Maximum Photos
                  </label>
                  <Input
                    type="number"
                    value={maxPhotos || ''}
                    onChange={(e) => setMaxPhotos(e.target.value ? parseInt(e.target.value) : null)}
                    min={1}
                    max={1000}
                    placeholder="Limit"
                  />
                </div>
              </div>
            </div>

            <div className="border-t pt-4" />

            {/* Rotation Settings */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Rotation Settings</h3>
              <div className="space-y-3">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={rotationEnabled}
                    onChange={(e) => setRotationEnabled(e.target.checked)}
                    className="w-4 h-4 text-primary"
                  />
                  <span className="text-sm font-medium">Enable Rotation Correction</span>
                </label>

                {rotationEnabled && (
                  <div>
                    <label className="text-sm font-medium mb-2 block">
                      Rotation Method
                    </label>
                    <select
                      value={rotationMethod}
                      onChange={(e) => setRotationMethod(e.target.value)}
                      className="w-full px-3 py-2 border rounded-md bg-background"
                    >
                      <option value="cv">Computer Vision (Fast)</option>
                      <option value="vlm">AI Model (Slower, Better)</option>
                    </select>
                  </div>
                )}
              </div>
            </div>

            <div className="border-t pt-4" />

            {/* Crop Settings */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Crop Settings</h3>
              <div className="space-y-3">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={cropEnabled}
                    onChange={(e) => setCropEnabled(e.target.checked)}
                    className="w-4 h-4 text-primary"
                  />
                  <span className="text-sm font-medium">Enable Intelligent Cropping</span>
                </label>

                {cropEnabled && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <label className="text-sm font-medium mb-2 block">
                        Crop Method
                      </label>
                      <select
                        value={cropMethod}
                        onChange={(e) => setCropMethod(e.target.value)}
                        className="w-full px-3 py-2 border rounded-md bg-background"
                      >
                        <option value="cv">Computer Vision</option>
                        <option value="vlm">Vision Language Model</option>
                      </select>
                    </div>

                    <div>
                      <label className="text-sm font-medium mb-2 block">
                        Aspect Ratio
                      </label>
                      <select
                        value={cropAspectRatio}
                        onChange={(e) => setCropAspectRatio(e.target.value)}
                        className="w-full px-3 py-2 border rounded-md bg-background"
                      >
                        <option value="original">Original</option>
                        <option value="16:9">16:9</option>
                        <option value="4:3">4:3</option>
                        <option value="3:2">3:2</option>
                        <option value="1:1">1:1 (Square)</option>
                      </select>
                    </div>

                    {cropMethod === 'vlm' && (
                      <div>
                        <label className="text-sm font-medium mb-2 block">
                          VLM Model
                        </label>
                        <Input
                          value={cropVlmModel}
                          onChange={(e) => setCropVlmModel(e.target.value)}
                          placeholder="Model name"
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            <div className="border-t pt-4" />

            {/* Enhancement Settings */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Enhancement Settings</h3>
              <div className="space-y-3">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={enhanceEnabled}
                    onChange={(e) => setEnhanceEnabled(e.target.checked)}
                    className="w-4 h-4 text-primary"
                  />
                  <span className="text-sm font-medium">Enable Intelligent Enhancement</span>
                </label>

                {enhanceEnabled && (
                  <div>
                    <label className="text-sm font-medium mb-2 block">
                      Enhancement Strength: {enhanceStrength.toFixed(1)}
                    </label>
                    <div className="flex items-center space-x-4">
                      <span className="text-sm">0</span>
                      <input
                        type="range"
                        value={enhanceStrength}
                        onChange={(e) => setEnhanceStrength(parseFloat(e.target.value))}
                        min={0}
                        max={2}
                        step={0.1}
                        className="flex-1"
                      />
                      <span className="text-sm">2</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Start Button */}
            <div className="flex justify-center pt-6">
              <Button
                size="lg"
                onClick={startBatchProcessing}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <PlayArrow className="h-4 w-4 mr-2" />
                    Start Batch Processing
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="py-12">
            <div className="text-center space-y-4">
              {getStatusIcon()}
              
              <h2 className="text-2xl font-semibold">
                {session?.current_stage || 'Processing...'}
              </h2>

              <p className="text-muted-foreground">
                {session?.processed_count || 0} of {session?.total_count || 0} photos processed
              </p>

              <div className="max-w-md mx-auto">
                <Progress value={session?.progress || 0} />
              </div>

              {session?.status === 'completed' && (
                <Button
                  size="lg"
                  onClick={downloadResults}
                  className="mt-6"
                >
                  <CloudDownload className="h-4 w-4 mr-2" />
                  Download Results
                </Button>
              )}

              {session?.error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-red-800 text-sm">{session.error}</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
            <p className="text-red-800">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
}