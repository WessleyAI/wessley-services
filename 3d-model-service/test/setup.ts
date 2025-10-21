import { vi } from 'vitest';

// Global test setup
vi.mock('winston', () => ({
  createLogger: vi.fn(() => ({
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
  })),
  format: {
    combine: vi.fn(),
    timestamp: vi.fn(),
    json: vi.fn(),
    colorize: vi.fn(),
    simple: vi.fn(),
  },
  transports: {
    Console: vi.fn(),
    File: vi.fn(),
  },
}));

// Mock environment variables for testing
process.env.NODE_ENV = 'test';
process.env.NEO4J_URI = 'bolt://localhost:7687';
process.env.NEO4J_USERNAME = 'neo4j';
process.env.NEO4J_PASSWORD = 'test';
process.env.REDIS_HOST = 'localhost';
process.env.REDIS_PORT = '6379';
process.env.DEFAULT_VEHICLE_SIGNATURE = 'test_vehicle_2001';