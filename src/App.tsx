import React, { useState } from 'react';
import { Heart } from 'lucide-react';
import { HealthForm } from './components/HealthForm';
import { RiskDisplay } from './components/RiskDisplay';
import { FeatureImportance } from './components/FeatureImportance';
import type { HealthData, PredictionResult } from './types';

// Temporary mock prediction function
const mockPredict = (data: HealthData): PredictionResult => ({
  probability: Math.random(),
  featureImportance: [
    { feature: 'Age', importance: Math.random() },
    { feature: 'Blood Pressure', importance: Math.random() },
    { feature: 'Cholesterol', importance: Math.random() },
    // Add more features...
  ]
});

function App() {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);

  const handleSubmit = (data: HealthData) => {
    // In a real app, this would call your ML backend
    const result = mockPredict(data);
    setPrediction(result);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <Heart className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Heart Disease Risk Assessment
          </h1>
          <p className="text-lg text-gray-600">
            Advanced ML-powered prediction system
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-2xl font-semibold mb-6">Patient Information</h2>
              <HealthForm onSubmit={handleSubmit} />
            </div>
          </div>

          <div className="space-y-6">
            {prediction && (
              <>
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h2 className="text-2xl font-semibold mb-4">Risk Assessment</h2>
                  <RiskDisplay probability={prediction.probability} />
                </div>

                <div className="bg-white rounded-lg shadow-md p-6">
                  <h2 className="text-2xl font-semibold mb-4">Feature Impact</h2>
                  <FeatureImportance data={prediction.featureImportance} />
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;