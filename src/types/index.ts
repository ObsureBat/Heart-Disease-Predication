export interface HealthData {
  age: number;
  sex: 'M' | 'F';
  chestPainType: 'typical' | 'atypical' | 'nonAnginal' | 'asymptomatic';
  restingBP: number;
  cholesterol: number;
  fastingBS: number;
  restingECG: 'normal' | 'abnormal' | 'hypertrophy';
  maxHR: number;
  exerciseAngina: 'Y' | 'N';
  oldpeak: number;
  stSlope: 'up' | 'flat' | 'down';
}

export interface PredictionResult {
  probability: number;
  featureImportance: {
    feature: string;
    importance: number;
  }[];
}