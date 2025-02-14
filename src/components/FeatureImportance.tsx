import React from 'react';
import Plot from 'react-plotly.js';

interface FeatureImportanceProps {
  data: {
    feature: string;
    importance: number;
  }[];
}

export const FeatureImportance: React.FC<FeatureImportanceProps> = ({ data }) => {
  const sortedData = [...data].sort((a, b) => b.importance - a.importance);

  return (
    <Plot
      data={[
        {
          type: 'bar',
          x: sortedData.map(d => d.importance),
          y: sortedData.map(d => d.feature),
          orientation: 'h',
          marker: {
            color: 'rgb(239, 68, 68)',
          },
        },
      ]}
      layout={{
        title: 'Feature Importance',
        height: 400,
        margin: { l: 150, r: 20, t: 40, b: 20 },
        xaxis: { title: 'Importance' },
        yaxis: { title: '' },
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
    />
  );
};