import React from 'react';

interface RiskDisplayProps {
  probability: number;
}

export const RiskDisplay: React.FC<RiskDisplayProps> = ({ probability }) => {
  const bgColor = probability > 0.5 ? 'bg-red-500' : 'bg-green-500';
  
  return (
    <div className={`${bgColor} rounded-lg p-4 text-white text-center`}>
      <h2 className="text-2xl font-bold mb-1">
        {(probability * 100).toFixed(1)}%
      </h2>
      <p className="text-sm">Probability of Heart Disease</p>
    </div>
  );
};