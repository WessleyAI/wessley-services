import { describe, it, expect } from 'vitest';

// Simple utility functions for testing
export function validateVehicleSignature(signature: string): boolean {
  if (!signature || typeof signature !== 'string') {
    return false;
  }
  
  // Vehicle signature should be in format: make_model_year
  const parts = signature.split('_');
  return parts.length >= 3 && parts.every(part => part.length > 0);
}

export function validatePosition(position: { x: number; y: number; z: number }): boolean {
  if (!position || typeof position !== 'object') {
    return false;
  }
  
  const { x, y, z } = position;
  return (
    typeof x === 'number' && !isNaN(x) && isFinite(x) &&
    typeof y === 'number' && !isNaN(y) && isFinite(y) &&
    typeof z === 'number' && !isNaN(z) && isFinite(z) &&
    z !== undefined // Ensure z property exists
  );
}

export function calculateDistance(
  pos1: { x: number; y: number; z: number },
  pos2: { x: number; y: number; z: number }
): number {
  const dx = pos2.x - pos1.x;
  const dy = pos2.y - pos1.y;
  const dz = pos2.z - pos1.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

describe('Validation Utilities', () => {
  describe('validateVehicleSignature', () => {
    it('should accept valid vehicle signatures', () => {
      expect(validateVehicleSignature('pajero_pinin_2001')).toBe(true);
      expect(validateVehicleSignature('toyota_camry_2020')).toBe(true);
      expect(validateVehicleSignature('ford_f150_1999')).toBe(true);
    });

    it('should reject invalid vehicle signatures', () => {
      expect(validateVehicleSignature('')).toBe(false);
      expect(validateVehicleSignature('invalid')).toBe(false);
      expect(validateVehicleSignature('only_two')).toBe(false);
      expect(validateVehicleSignature('has__empty__parts')).toBe(false);
    });

    it('should reject non-string inputs', () => {
      expect(validateVehicleSignature(null as any)).toBe(false);
      expect(validateVehicleSignature(undefined as any)).toBe(false);
      expect(validateVehicleSignature(123 as any)).toBe(false);
      expect(validateVehicleSignature({} as any)).toBe(false);
    });
  });

  describe('validatePosition', () => {
    it('should accept valid 3D positions', () => {
      expect(validatePosition({ x: 0, y: 0, z: 0 })).toBe(true);
      expect(validatePosition({ x: 100.5, y: -50.2, z: 300.7 })).toBe(true);
      expect(validatePosition({ x: -999, y: 999, z: 0 })).toBe(true);
    });

    it('should reject invalid positions', () => {
      expect(validatePosition(null as any)).toBe(false);
      expect(validatePosition(undefined as any)).toBe(false);
      expect(validatePosition({} as any)).toBe(false);
      expect(validatePosition({ x: 0, y: 0 } as any)).toBe(false);
      expect(validatePosition({ x: NaN, y: 0, z: 0 })).toBe(false);
      expect(validatePosition({ x: 0, y: Infinity, z: 0 })).toBe(false);
    });

    it('should reject non-numeric coordinates', () => {
      expect(validatePosition({ x: '0', y: 0, z: 0 } as any)).toBe(false);
      expect(validatePosition({ x: 0, y: null, z: 0 } as any)).toBe(false);
      expect(validatePosition({ x: 0, y: 0, z: undefined } as any)).toBe(false);
    });
  });

  describe('calculateDistance', () => {
    it('should calculate distance between two points', () => {
      const pos1 = { x: 0, y: 0, z: 0 };
      const pos2 = { x: 3, y: 4, z: 0 };
      
      expect(calculateDistance(pos1, pos2)).toBe(5);
    });

    it('should calculate 3D distance correctly', () => {
      const pos1 = { x: 0, y: 0, z: 0 };
      const pos2 = { x: 1, y: 1, z: 1 };
      
      expect(calculateDistance(pos1, pos2)).toBeCloseTo(Math.sqrt(3), 5);
    });

    it('should return 0 for identical points', () => {
      const pos = { x: 100, y: 200, z: 300 };
      
      expect(calculateDistance(pos, pos)).toBe(0);
    });

    it('should handle negative coordinates', () => {
      const pos1 = { x: -10, y: -20, z: -30 };
      const pos2 = { x: 10, y: 20, z: 30 };
      
      const distance = calculateDistance(pos1, pos2);
      expect(distance).toBeCloseTo(Math.sqrt(20*20 + 40*40 + 60*60), 5);
    });
  });
});