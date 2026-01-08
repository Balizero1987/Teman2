'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';
import { NeuronalBackground } from '@/components/ui/NeuronalBackground';

interface NeuralContextType {
  activityLevel: number;
  setActivityLevel: (level: number) => void;
  pulse: () => void;
}

const NeuralContext = createContext<NeuralContextType>({
  activityLevel: 0.1,
  setActivityLevel: () => {},
  pulse: () => {},
});

export const useNeural = () => useContext(NeuralContext);

export const NeuralWrapper = ({ children }: { children: React.ReactNode }) => {
  const [activityLevel, setActivityLevel] = useState(0.1);

  // Auto-decay activity level
  useEffect(() => {
    if (activityLevel > 0.1) {
      const timer = setTimeout(() => {
        setActivityLevel(prev => Math.max(0.1, prev - 0.05));
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [activityLevel]);

  const pulse = () => {
    setActivityLevel(prev => Math.min(1.0, prev + 0.3));
  };

  return (
    <NeuralContext.Provider value={{ activityLevel, setActivityLevel, pulse }}>
      <NeuronalBackground activityLevel={activityLevel} />
      <div className="relative z-10 w-full h-full min-h-screen">
        {children}
      </div>
    </NeuralContext.Provider>
  );
};
