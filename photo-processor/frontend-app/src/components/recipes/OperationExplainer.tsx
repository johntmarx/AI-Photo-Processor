import React from 'react';
import { Info, Brain, Eye, Crop, RotateCw, Sparkles } from 'lucide-react';
import { Card } from '../ui/Card';

interface OperationExplainerProps {
  operation: string;
  isExpanded?: boolean;
}

const operationDetails = {
  crop: {
    icon: Crop,
    aiRequired: true,
    title: 'AI-Powered Smart Crop',
    description: 'Uses visual language models to analyze composition',
    howItWorks: [
      'Qwen2.5-VL analyzes your image composition and content',
      'RT-DETR identifies all subjects and their positions',
      'AI suggests multiple crops for different purposes (impact, balance, social media)',
      'Considers rule of thirds, negative space, and visual tension',
      'Generates platform-specific crops (Instagram 4:5, Stories 9:16, etc.)'
    ],
    whatYouControl: [
      'Choose from AI-suggested crops',
      'Override with manual selection',
      'Set preferred aspect ratios',
      'Adjust crop emphasis (subject vs. environment)'
    ],
    example: 'For a portrait, AI might suggest: tight headshot for impact, environmental portrait for context, off-center crop for artistic tension'
  },
  
  auto_rotate: {
    icon: RotateCw,
    aiRequired: true,
    title: 'AI Rotation Detection',
    description: 'Detects horizon and vertical lines to straighten images',
    howItWorks: [
      'Analyzes image for horizon lines and vertical structures',
      'Detects faces to ensure proper orientation',
      'Calculates optimal rotation angle',
      'Can read EXIF data for camera orientation'
    ],
    whatYouControl: [
      'Enable/disable auto-straightening',
      'Override with manual angle',
      'Set rotation threshold sensitivity'
    ],
    example: 'Automatically fixes tilted horizons in landscapes or straightens architectural photos'
  },

  scene_analysis: {
    icon: Eye,
    aiRequired: true,
    title: 'Scene Understanding',
    description: 'Core AI brain that understands image content',
    howItWorks: [
      'Qwen2.5-VL provides detailed scene understanding',
      'Identifies scene type (portrait, landscape, street, etc.)',
      'Analyzes composition strengths and weaknesses',
      'Suggests optimal processing based on content',
      'Evaluates lighting conditions and color palette'
    ],
    whatYouControl: [
      'Processing style preference',
      'Enhancement intensity',
      'Which suggestions to apply'
    ],
    example: 'Recognizes a sunset landscape and suggests warm color grading with enhanced sky detail'
  },

  quality_assessment: {
    icon: Sparkles,
    aiRequired: true,
    title: 'NIMA Quality Scoring',
    description: 'Neural network evaluates technical and aesthetic quality',
    howItWorks: [
      'NIMA model scores images on technical quality (0-10)',
      'Separate score for aesthetic appeal',
      'Identifies issues like blur, noise, poor exposure',
      'Helps with culling similar images'
    ],
    whatYouControl: [
      'Minimum quality threshold',
      'Weight between technical vs aesthetic',
      'Auto-reject threshold'
    ],
    example: 'Scores a sharp, well-composed image 8.5/10 but a blurry one 3.2/10'
  },

  object_detection: {
    icon: Brain,
    aiRequired: true,
    title: 'RT-DETR Object Detection',
    description: 'Identifies and locates all subjects in the image',
    howItWorks: [
      'Detects people, faces, animals, objects',
      'Creates bounding boxes for each subject',
      'Understands spatial relationships',
      'Feeds data to crop and composition algorithms'
    ],
    whatYouControl: [
      'Which objects to prioritize',
      'Minimum detection confidence',
      'Subject emphasis in crops'
    ],
    example: 'Detects two people and a dog, allowing smart crop to include all subjects'
  },

  enhance: {
    icon: Sparkles,
    aiRequired: false,
    title: 'Basic Enhancements',
    description: 'Manual adjustments without AI analysis',
    howItWorks: [
      'Direct pixel value adjustments',
      'Brightness: multiply all pixels',
      'Contrast: adjust tonal curve',
      'Saturation: boost color channels'
    ],
    whatYouControl: [
      'All adjustment values directly',
      'No AI intervention',
      'Predictable results'
    ],
    example: 'Brightness +10 makes everything 10% brighter uniformly'
  }
};

export function OperationExplainer({ operation, isExpanded = false }: OperationExplainerProps) {
  const details = operationDetails[operation as keyof typeof operationDetails];
  
  if (!details) return null;
  
  const Icon = details.icon;
  
  return (
    <Card className={`p-4 ${details.aiRequired ? 'border-blue-200 bg-blue-50' : 'border-gray-200'}`}>
      <div className="flex items-start space-x-3">
        <div className={`p-2 rounded-lg ${details.aiRequired ? 'bg-blue-100' : 'bg-gray-100'}`}>
          <Icon className="h-5 w-5" />
        </div>
        
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-1">
            <h4 className="font-medium">{details.title}</h4>
            {details.aiRequired && (
              <span className="text-xs bg-blue-200 text-blue-800 px-2 py-0.5 rounded">
                AI Required
              </span>
            )}
          </div>
          
          <p className="text-sm text-gray-600 mb-3">{details.description}</p>
          
          {isExpanded && (
            <>
              <div className="space-y-3">
                <div>
                  <h5 className="text-sm font-medium mb-1 flex items-center">
                    <Info className="h-4 w-4 mr-1" />
                    How it works:
                  </h5>
                  <ul className="text-sm text-gray-600 space-y-1">
                    {details.howItWorks.map((point, i) => (
                      <li key={i} className="flex items-start">
                        <span className="text-blue-500 mr-2">•</span>
                        <span>{point}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div>
                  <h5 className="text-sm font-medium mb-1">What you control:</h5>
                  <ul className="text-sm text-gray-600 space-y-1">
                    {details.whatYouControl.map((point, i) => (
                      <li key={i} className="flex items-start">
                        <span className="text-green-500 mr-2">✓</span>
                        <span>{point}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div className="bg-gray-50 p-3 rounded text-sm">
                  <strong>Example:</strong> {details.example}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </Card>
  );
}

// Usage in Recipe Editor
export function RecipeStepWithExplainer({ step, onChange }: any) {
  const [showExplainer, setShowExplainer] = React.useState(false);
  
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h4 className="font-medium">{step.name}</h4>
        <button
          onClick={() => setShowExplainer(!showExplainer)}
          className="text-sm text-blue-600 hover:text-blue-800"
        >
          {showExplainer ? 'Hide' : 'How does this work?'}
        </button>
      </div>
      
      {showExplainer && (
        <OperationExplainer operation={step.type} isExpanded={true} />
      )}
      
      {/* Regular step parameters UI */}
      <div className="pl-4">
        {/* ... existing parameter controls ... */}
      </div>
    </div>
  );
}