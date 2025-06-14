import React, { useState, useEffect } from 'react';
import { Brain, Eye, Zap, Target, Info } from 'lucide-react';
import { Card } from '../ui/Card';
import { Progress } from '../ui/Progress';
import { Badge } from '../ui/Badge';

interface AIDecision {
  stage: string;
  description: string;
  confidence: number;
  details: any;
  timestamp: number;
}

interface CropSuggestion {
  name: string;
  description: string;
  aspectRatio: string;
  coordinates: [number, number, number, number];
  score: number;
  reasoning: string;
}

interface AIDecisionVisualizerProps {
  photoId: string;
  operation: string;
}

export function AIDecisionVisualizer({ photoId, operation }: AIDecisionVisualizerProps) {
  const [decisions, setDecisions] = useState<AIDecision[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [cropSuggestions, setCropSuggestions] = useState<CropSuggestion[]>([]);

  useEffect(() => {
    if (operation === 'crop') {
      simulateCropAnalysis();
    }
  }, [photoId, operation]);

  const simulateCropAnalysis = async () => {
    setIsProcessing(true);
    setDecisions([]);
    
    // Stage 1: Scene Analysis
    await addDecision({
      stage: 'Scene Analysis',
      description: 'Qwen2.5-VL analyzing image content and composition',
      confidence: 0,
      details: { analyzing: true }
    });

    await delay(1000);
    
    await addDecision({
      stage: 'Scene Analysis',
      description: 'Detected: Portrait with urban background',
      confidence: 92,
      details: {
        sceneType: 'portrait',
        setting: 'urban',
        lighting: 'golden hour',
        mainSubject: 'person'
      }
    });

    // Stage 2: Object Detection
    await addDecision({
      stage: 'Object Detection',
      description: 'RT-DETR identifying subjects and objects',
      confidence: 0,
      details: { detecting: true }
    });

    await delay(800);

    await addDecision({
      stage: 'Object Detection',
      description: 'Found 1 person, 3 background elements',
      confidence: 96,
      details: {
        subjects: [
          { type: 'person', bbox: [320, 180, 480, 540], confidence: 0.98 },
          { type: 'building', bbox: [0, 0, 200, 400], confidence: 0.85 },
          { type: 'car', bbox: [500, 350, 640, 450], confidence: 0.72 }
        ]
      }
    });

    // Stage 3: Composition Analysis
    await addDecision({
      stage: 'Composition Analysis',
      description: 'Analyzing rule of thirds, balance, and visual flow',
      confidence: 0,
      details: { analyzing: true }
    });

    await delay(600);

    await addDecision({
      stage: 'Composition Analysis',
      description: 'Subject off-center, good negative space distribution',
      confidence: 88,
      details: {
        ruleOfThirds: 'subject at left third',
        balance: 'asymmetric but balanced',
        leadingLines: 'building edge creates vertical line',
        negativeSpace: 'effective use on right side'
      }
    });

    // Stage 4: Generate Crop Suggestions
    await addDecision({
      stage: 'Crop Generation',
      description: 'AI generating optimal crop suggestions',
      confidence: 0,
      details: { generating: true }
    });

    await delay(1200);

    const suggestions: CropSuggestion[] = [
      {
        name: 'Maximum Impact',
        description: 'Tight crop emphasizing subject expression',
        aspectRatio: '4:5',
        coordinates: [280, 150, 520, 480],
        score: 9.2,
        reasoning: 'Eliminates distracting background elements, focuses on facial expression and emotion'
      },
      {
        name: 'Environmental Portrait',
        description: 'Includes urban context',
        aspectRatio: '2:3',
        coordinates: [180, 80, 580, 600],
        score: 8.5,
        reasoning: 'Preserves story-telling elements while maintaining subject prominence'
      },
      {
        name: 'Instagram Story',
        description: 'Vertical crop for social media',
        aspectRatio: '9:16',
        coordinates: [250, 50, 470, 630],
        score: 8.8,
        reasoning: 'Optimized for mobile viewing, maintains subject in upper third for UI clearance'
      },
      {
        name: 'Artistic Tension',
        description: 'Unconventional crop with dynamic composition',
        aspectRatio: '16:9',
        coordinates: [100, 200, 640, 360],
        score: 7.9,
        reasoning: 'Creates visual tension by placing subject at extreme edge, emphasizes environment'
      }
    ];

    setCropSuggestions(suggestions);

    await addDecision({
      stage: 'Crop Generation',
      description: `Generated ${suggestions.length} crop suggestions`,
      confidence: 95,
      details: {
        suggestions: suggestions.map(s => ({
          name: s.name,
          score: s.score,
          aspectRatio: s.aspectRatio
        }))
      }
    });

    setIsProcessing(false);
  };

  const addDecision = async (decision: Omit<AIDecision, 'timestamp'>) => {
    setDecisions(prev => {
      const existing = prev.findIndex(d => d.stage === decision.stage);
      const newDecision = { ...decision, timestamp: Date.now() };
      
      if (existing >= 0) {
        const updated = [...prev];
        updated[existing] = newDecision;
        return updated;
      }
      
      return [...prev, newDecision];
    });
  };

  const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

  return (
    <div className="space-y-4">
      <Card className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold flex items-center">
            <Brain className="h-5 w-5 mr-2" />
            AI Decision Process
          </h3>
          {isProcessing && (
            <Badge variant="secondary" className="animate-pulse">
              <Zap className="h-3 w-3 mr-1" />
              Processing
            </Badge>
          )}
        </div>

        <div className="space-y-3">
          {decisions.map((decision, index) => (
            <div key={`${decision.stage}-${index}`} className="border-l-2 border-blue-200 pl-4">
              <div className="flex items-center justify-between">
                <h4 className="font-medium text-sm">{decision.stage}</h4>
                {decision.confidence > 0 && (
                  <span className="text-xs text-gray-500">
                    {decision.confidence}% confidence
                  </span>
                )}
              </div>
              
              <p className="text-sm text-gray-600 mt-1">{decision.description}</p>
              
              {decision.confidence > 0 && (
                <Progress value={decision.confidence} className="h-1 mt-2" />
              )}
              
              {decision.details && !decision.details.analyzing && (
                <div className="mt-2 text-xs bg-gray-50 p-2 rounded">
                  <pre className="whitespace-pre-wrap">
                    {JSON.stringify(decision.details, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>

      {cropSuggestions.length > 0 && (
        <Card className="p-4">
          <h3 className="font-semibold mb-4 flex items-center">
            <Target className="h-5 w-5 mr-2" />
            AI Crop Suggestions
          </h3>
          
          <div className="grid gap-3">
            {cropSuggestions.map((suggestion, index) => (
              <div key={index} className="border rounded-lg p-3 hover:bg-gray-50">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h4 className="font-medium">{suggestion.name}</h4>
                    <p className="text-sm text-gray-600 mt-1">{suggestion.description}</p>
                    <div className="flex items-center space-x-3 mt-2">
                      <Badge variant="outline" size="sm">
                        {suggestion.aspectRatio}
                      </Badge>
                      <span className="text-xs text-gray-500">
                        Score: {suggestion.score}/10
                      </span>
                    </div>
                  </div>
                  <button className="ml-3 px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600">
                    Apply
                  </button>
                </div>
                
                <div className="mt-3 flex items-start space-x-2">
                  <Info className="h-4 w-4 text-gray-400 mt-0.5 flex-shrink-0" />
                  <p className="text-xs text-gray-500 italic">{suggestion.reasoning}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

// Component to show live AI processing
export function AIProcessingIndicator({ stage, details }: { stage: string; details?: any }) {
  return (
    <div className="fixed bottom-4 right-4 bg-white shadow-lg rounded-lg p-4 max-w-sm">
      <div className="flex items-center space-x-3">
        <div className="animate-spin">
          <Brain className="h-6 w-6 text-blue-500" />
        </div>
        <div className="flex-1">
          <h4 className="font-medium">{stage}</h4>
          {details && (
            <p className="text-sm text-gray-600">{details}</p>
          )}
        </div>
      </div>
    </div>
  );
}