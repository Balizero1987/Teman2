'use client';

import React, { useRef, useEffect } from 'react';

interface NeuronalBackgroundProps {
  activityLevel: number; // 0.0 (idle) to 1.0 (max intense thinking)
}

interface Neuron {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
}

interface Signal {
  startNode: number;
  endNode: number;
  progress: number; // 0.0 to 1.0
  speed: number;
}

export const NeuronalBackground: React.FC<NeuronalBackgroundProps> = ({ activityLevel }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Configuration
  const neuronCount = 60;
  const connectionDistance = 150;
  const baseSpeed = 0.2;
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    // State
    const neurons: Neuron[] = [];
    const signals: Signal[] = [];
    
    // Initialize Neurons
    for (let i = 0; i < neuronCount; i++) {
      neurons.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * baseSpeed,
        vy: (Math.random() - 0.5) * baseSpeed,
        radius: Math.random() * 1.5 + 0.5,
      });
    }

    const handleResize = () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', handleResize);

    let animationFrameId: number;

    const render = () => {
      // Dynamic parameters based on activityLevel
      // activityLevel 0 = idle, 1 = intense thought
      // We smooth the transition logic in the parent or allow jump here?
      // Let's assume passed prop is smoothed or we react instantly.
      
      const speedMultiplier = 1 + (activityLevel * 3); // 1x to 4x speed
      const signalChance = 0.01 + (activityLevel * 0.1); // Chance to spawn signal
      const pulseColor = `rgba(100, 200, 255, ${0.3 + activityLevel * 0.7})`; // Brighter blue
      
      // Clear
      ctx.fillStyle = 'rgba(10, 10, 15, 0.2)'; // Trail effect
      ctx.fillRect(0, 0, width, height);
      // OR full clear for crisp look: ctx.clearRect(0, 0, width, height);
      // Trail effect looks more "fluid".
      
      // Update & Draw Neurons
      ctx.fillStyle = 'rgba(150, 150, 170, 0.5)';
      neurons.forEach(neuron => {
        neuron.x += neuron.vx * speedMultiplier;
        neuron.y += neuron.vy * speedMultiplier;

        // Bounce
        if (neuron.x < 0 || neuron.x > width) neuron.vx *= -1;
        if (neuron.y < 0 || neuron.y > height) neuron.vy *= -1;

        ctx.beginPath();
        ctx.arc(neuron.x, neuron.y, neuron.radius, 0, Math.PI * 2);
        ctx.fill();
      });

      // Draw Connections & Manage Signals
      ctx.lineWidth = 0.5;
      
      for (let i = 0; i < neuronCount; i++) {
        for (let j = i + 1; j < neuronCount; j++) {
          const dx = neurons[i].x - neurons[j].x;
          const dy = neurons[i].y - neurons[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < connectionDistance) {
            // Draw Line
            const opacity = 1 - (dist / connectionDistance);
            ctx.strokeStyle = `rgba(100, 150, 200, ${opacity * 0.2})`;
            ctx.beginPath();
            ctx.moveTo(neurons[i].x, neurons[i].y);
            ctx.lineTo(neurons[j].x, neurons[j].y);
            ctx.stroke();

            // Spawn Signal?
            if (Math.random() < signalChance) {
              signals.push({
                startNode: i,
                endNode: j,
                progress: 0,
                speed: 0.02 + Math.random() * 0.03 + (activityLevel * 0.05)
              });
            }
          }
        }
      }

      // Update & Draw Signals
      for (let i = signals.length - 1; i >= 0; i--) {
        const sig = signals[i];
        sig.progress += sig.speed;
        
        if (sig.progress >= 1) {
          signals.splice(i, 1);
          continue;
        }

        const n1 = neurons[sig.startNode];
        const n2 = neurons[sig.endNode];
        
        // Calculate position
        const sx = n1.x + (n2.x - n1.x) * sig.progress;
        const sy = n1.y + (n2.y - n1.y) * sig.progress;

        // Draw glowing signal
        ctx.fillStyle = pulseColor;
        ctx.shadowBlur = 10;
        ctx.shadowColor = pulseColor;
        ctx.beginPath();
        ctx.arc(sx, sy, 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
    };
  }, [activityLevel]); // Re-init if activityLevel changes drastically? No, logic is inside render.
  // Actually, we don't want to re-init neurons on prop change. 
  // We should use a ref for activityLevel to read inside render loop without re-running effect.
  
  // FIX: Use ref for activityLevel
  const activityRef = useRef(activityLevel);
  useEffect(() => { activityRef.current = activityLevel; }, [activityLevel]);

  // The main effect should depend only on mount
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    const neurons: Neuron[] = [];
    const signals: Signal[] = [];
    
    for (let i = 0; i < neuronCount; i++) {
      neurons.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * baseSpeed,
        vy: (Math.random() - 0.5) * baseSpeed,
        radius: Math.random() * 1.5 + 0.5,
      });
    }

    const handleResize = () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', handleResize);

    let animationFrameId: number;

    const render = () => {
      const currentActivity = activityRef.current;
      const speedMultiplier = 1 + (currentActivity * 3);
      const signalChance = 0.005 + (currentActivity * 0.05); // Tweaked chance
      
      ctx.clearRect(0, 0, width, height); // Clean clear for now
      
      // Update & Draw Neurons
      ctx.fillStyle = 'rgba(150, 150, 170, 0.3)';
      neurons.forEach(neuron => {
        neuron.x += neuron.vx * speedMultiplier;
        neuron.y += neuron.vy * speedMultiplier;

        if (neuron.x < 0) { neuron.x = 0; neuron.vx *= -1; }
        if (neuron.x > width) { neuron.x = width; neuron.vx *= -1; }
        if (neuron.y < 0) { neuron.y = 0; neuron.vy *= -1; }
        if (neuron.y > height) { neuron.y = height; neuron.vy *= -1; }

        ctx.beginPath();
        ctx.arc(neuron.x, neuron.y, neuron.radius, 0, Math.PI * 2);
        ctx.fill();
      });

      // Connections
      ctx.lineWidth = 0.5;
      for (let i = 0; i < neuronCount; i++) {
        for (let j = i + 1; j < neuronCount; j++) {
          const dx = neurons[i].x - neurons[j].x;
          const dy = neurons[i].y - neurons[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < connectionDistance) {
            const opacity = 1 - (dist / connectionDistance);
            ctx.strokeStyle = `rgba(100, 150, 200, ${opacity * 0.15})`;
            ctx.beginPath();
            ctx.moveTo(neurons[i].x, neurons[i].y);
            ctx.lineTo(neurons[j].x, neurons[j].y);
            ctx.stroke();

            if (Math.random() < signalChance) {
              signals.push({
                startNode: i,
                endNode: j,
                progress: 0,
                speed: 0.02 + Math.random() * 0.03 + (currentActivity * 0.05)
              });
            }
          }
        }
      }

      // Signals
      for (let i = signals.length - 1; i >= 0; i--) {
        const sig = signals[i];
        sig.progress += sig.speed;
        
        if (sig.progress >= 1) {
          signals.splice(i, 1);
          continue;
        }

        const n1 = neurons[sig.startNode];
        const n2 = neurons[sig.endNode];
        const sx = n1.x + (n2.x - n1.x) * sig.progress;
        const sy = n1.y + (n2.y - n1.y) * sig.progress;

        ctx.fillStyle = `rgba(100, 200, 255, ${0.5 + currentActivity * 0.5})`;
        ctx.shadowBlur = 8;
        ctx.shadowColor = 'rgba(100, 200, 255, 0.8)';
        ctx.beginPath();
        ctx.arc(sx, sy, 1.5 + (currentActivity), 0, Math.PI * 2); // Larger signals when active
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas 
      ref={canvasRef} 
      className="fixed inset-0 pointer-events-none z-0 opacity-40 mix-blend-screen"
      style={{ background: 'transparent' }}
    />
  );
};
