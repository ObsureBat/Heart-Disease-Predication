import React, { useState } from 'react';
import type { HealthData } from '../types';

interface HealthFormProps {
  onSubmit: (data: HealthData) => void;
}

export const HealthForm: React.FC<HealthFormProps> = ({ onSubmit }) => {
  const [formData, setFormData] = useState<HealthData>({
    age: 45,
    sex: 'M',
    chestPainType: 'typical',
    restingBP: 120,
    cholesterol: 200,
    fastingBS: 0,
    restingECG: 'normal',
    maxHR: 150,
    exerciseAngina: 'N',
    oldpeak: 0,
    stSlope: 'up'
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700">Age</label>
          <input
            type="number"
            value={formData.age}
            onChange={e => setFormData(prev => ({ ...prev, age: parseInt(e.target.value) }))}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-red-500 focus:ring-red-500"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">Sex</label>
          <select
            value={formData.sex}
            onChange={e => setFormData(prev => ({ ...prev, sex: e.target.value as 'M' | 'F' }))}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-red-500 focus:ring-red-500"
          >
            <option value="M">Male</option>
            <option value="F">Female</option>
          </select>
        </div>

        {/* Add other form fields similarly */}
      </div>

      <button
        type="submit"
        className="w-full bg-red-500 text-white py-2 px-4 rounded-md hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-opacity-50"
      >
        Predict Risk
      </button>
    </form>
  );
};