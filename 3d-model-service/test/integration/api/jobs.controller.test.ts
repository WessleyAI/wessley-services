import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import request from 'supertest';

import { JobsController } from '@/controllers/jobs.controller';
import { QueueService } from '@/modules/jobs/queue.service';
import { GraphService } from '@/modules/graph/graph.service';
import { SupabaseService } from '@/modules/realtime/supabase.service';

// Mock services for API testing
class MockQueueService {
  async addJob(data: any) {
    return {
      id: 'job_123',
      requestId: data.requestId,
      vehicleSignature: data.vehicleSignature,
      status: 'waiting',
      progress: 0,
      createdAt: new Date()
    };
  }

  async getJobStatus(jobId: string) {
    return {
      id: jobId,
      status: 'completed',
      progress: 100,
      result: {
        glbUrl: 'https://cdn.example.com/models/test.glb',
        metadata: { componentCount: 5 }
      }
    };
  }

  async getQueueStats() {
    return {
      waiting: 2,
      active: 1,
      completed: 15,
      failed: 0
    };
  }

  async cancelJob(jobId: string) {
    return { id: jobId, status: 'cancelled' };
  }
}

class MockGraphService {
  async validateVehicleData(vehicleSignature: string) {
    if (vehicleSignature === 'invalid_vehicle') {
      return { isValid: false, errors: ['Vehicle not found'] };
    }
    return { 
      isValid: true, 
      componentCount: 5, 
      connectionCount: 4 
    };
  }
}

class MockSupabaseService {
  isConfigured() {
    return true;
  }

  async verifyAndGetUser(token: string) {
    if (token === 'valid_token') {
      return { id: 'user_123', email: 'test@example.com' };
    }
    return null;
  }
}

describe('JobsController API', () => {
  let app: INestApplication;
  let module: TestingModule;

  beforeEach(async () => {
    module = await Test.createTestingModule({
      imports: [
        ConfigModule.forRoot({
          isGlobal: true,
          envFilePath: '.env.test',
        }),
      ],
      controllers: [JobsController],
      providers: [
        { provide: QueueService, useClass: MockQueueService },
        { provide: GraphService, useClass: MockGraphService },
        { provide: SupabaseService, useClass: MockSupabaseService },
      ],
    }).compile();

    app = module.createNestApplication();
    await app.init();
  });

  afterEach(async () => {
    await app.close();
  });

  describe('POST /api/v1/jobs/generate', () => {
    it('should create a new 3D model generation job', async () => {
      const jobData = {
        vehicleSignature: 'test_vehicle_2001',
        options: {
          quality: 'high',
          includeWires: true
        }
      };

      const response = await request(app.getHttpServer())
        .post('/api/v1/jobs/generate')
        .set('Authorization', 'Bearer valid_token')
        .send(jobData)
        .expect(201);

      expect(response.body).toMatchObject({
        success: true,
        data: {
          id: 'job_123',
          requestId: expect.any(String),
          vehicleSignature: 'test_vehicle_2001',
          status: 'waiting',
          progress: 0
        }
      });
    });

    it('should validate vehicle signature', async () => {
      const jobData = {
        vehicleSignature: 'invalid_vehicle',
        options: {}
      };

      const response = await request(app.getHttpServer())
        .post('/api/v1/jobs/generate')
        .set('Authorization', 'Bearer valid_token')
        .send(jobData)
        .expect(400);

      expect(response.body).toMatchObject({
        success: false,
        error: expect.stringContaining('Vehicle not found')
      });
    });

    it('should require authentication', async () => {
      const jobData = {
        vehicleSignature: 'test_vehicle_2001',
        options: {}
      };

      await request(app.getHttpServer())
        .post('/api/v1/jobs/generate')
        .send(jobData)
        .expect(401);
    });

    it('should reject invalid authentication tokens', async () => {
      const jobData = {
        vehicleSignature: 'test_vehicle_2001',
        options: {}
      };

      await request(app.getHttpServer())
        .post('/api/v1/jobs/generate')
        .set('Authorization', 'Bearer invalid_token')
        .send(jobData)
        .expect(401);
    });

    it('should validate request body', async () => {
      await request(app.getHttpServer())
        .post('/api/v1/jobs/generate')
        .set('Authorization', 'Bearer valid_token')
        .send({}) // Missing required fields
        .expect(400);
    });

    it('should handle malformed JSON', async () => {
      await request(app.getHttpServer())
        .post('/api/v1/jobs/generate')
        .set('Authorization', 'Bearer valid_token')
        .set('Content-Type', 'application/json')
        .send('invalid json')
        .expect(400);
    });
  });

  describe('GET /api/v1/jobs/:jobId/status', () => {
    it('should return job status', async () => {
      const response = await request(app.getHttpServer())
        .get('/api/v1/jobs/job_123/status')
        .set('Authorization', 'Bearer valid_token')
        .expect(200);

      expect(response.body).toMatchObject({
        success: true,
        data: {
          id: 'job_123',
          status: 'completed',
          progress: 100,
          result: {
            glbUrl: expect.any(String),
            metadata: expect.any(Object)
          }
        }
      });
    });

    it('should require authentication for job status', async () => {
      await request(app.getHttpServer())
        .get('/api/v1/jobs/job_123/status')
        .expect(401);
    });

    it('should validate job ID format', async () => {
      await request(app.getHttpServer())
        .get('/api/v1/jobs/invalid-job-id/status')
        .set('Authorization', 'Bearer valid_token')
        .expect(400);
    });
  });

  describe('GET /api/v1/jobs/queue/stats', () => {
    it('should return queue statistics (public endpoint)', async () => {
      const response = await request(app.getHttpServer())
        .get('/api/v1/jobs/queue/stats')
        .expect(200);

      expect(response.body).toMatchObject({
        success: true,
        data: {
          waiting: expect.any(Number),
          active: expect.any(Number),
          completed: expect.any(Number),
          failed: expect.any(Number)
        }
      });
    });

    it('should not require authentication for public stats', async () => {
      await request(app.getHttpServer())
        .get('/api/v1/jobs/queue/stats')
        .expect(200);
    });
  });

  describe('POST /api/v1/jobs/:jobId/cancel', () => {
    it('should cancel a job', async () => {
      const response = await request(app.getHttpServer())
        .post('/api/v1/jobs/job_123/cancel')
        .set('Authorization', 'Bearer valid_token')
        .expect(200);

      expect(response.body).toMatchObject({
        success: true,
        data: {
          id: 'job_123',
          status: 'cancelled'
        }
      });
    });

    it('should require authentication to cancel jobs', async () => {
      await request(app.getHttpServer())
        .post('/api/v1/jobs/job_123/cancel')
        .expect(401);
    });
  });

  describe('rate limiting', () => {
    it('should apply rate limiting to job creation', async () => {
      const jobData = {
        vehicleSignature: 'test_vehicle_2001',
        options: {}
      };

      // Make multiple rapid requests
      const requests = Array.from({ length: 10 }, () =>
        request(app.getHttpServer())
          .post('/api/v1/jobs/generate')
          .set('Authorization', 'Bearer valid_token')
          .send(jobData)
      );

      const responses = await Promise.all(requests);
      
      // Some requests should succeed, some should be rate limited
      const successfulRequests = responses.filter(r => r.status === 201);
      const rateLimitedRequests = responses.filter(r => r.status === 429);

      expect(successfulRequests.length).toBeGreaterThan(0);
      expect(rateLimitedRequests.length).toBeGreaterThan(0);
    });
  });

  describe('CORS and security headers', () => {
    it('should include security headers', async () => {
      const response = await request(app.getHttpServer())
        .get('/api/v1/jobs/queue/stats')
        .expect(200);

      expect(response.headers).toHaveProperty('x-content-type-options');
      expect(response.headers).toHaveProperty('x-frame-options');
    });

    it('should handle CORS preflight requests', async () => {
      await request(app.getHttpServer())
        .options('/api/v1/jobs/generate')
        .set('Origin', 'https://wessley.ai')
        .set('Access-Control-Request-Method', 'POST')
        .expect(200);
    });
  });

  describe('error handling', () => {
    it('should return consistent error format', async () => {
      const response = await request(app.getHttpServer())
        .post('/api/v1/jobs/generate')
        .set('Authorization', 'Bearer valid_token')
        .send({}) // Invalid request
        .expect(400);

      expect(response.body).toMatchObject({
        success: false,
        error: expect.any(String),
        code: expect.any(String),
        timestamp: expect.any(String)
      });
    });

    it('should handle internal server errors gracefully', async () => {
      // This would need a mock that throws an error
      // Implementation depends on how errors are handled in the actual controller
    });

    it('should not leak sensitive information in error messages', async () => {
      const response = await request(app.getHttpServer())
        .post('/api/v1/jobs/generate')
        .set('Authorization', 'Bearer invalid_token')
        .send({})
        .expect(401);

      expect(response.body.error).not.toContain('password');
      expect(response.body.error).not.toContain('secret');
      expect(response.body.error).not.toContain('key');
    });
  });

  describe('request validation', () => {
    it('should validate content-type headers', async () => {
      await request(app.getHttpServer())
        .post('/api/v1/jobs/generate')
        .set('Authorization', 'Bearer valid_token')
        .set('Content-Type', 'text/plain')
        .send('not json')
        .expect(400);
    });

    it('should enforce request size limits', async () => {
      const largePayload = {
        vehicleSignature: 'test_vehicle',
        options: {
          data: 'x'.repeat(10 * 1024 * 1024) // 10MB of data
        }
      };

      await request(app.getHttpServer())
        .post('/api/v1/jobs/generate')
        .set('Authorization', 'Bearer valid_token')
        .send(largePayload)
        .expect(413); // Payload too large
    });
  });

  describe('health and monitoring', () => {
    it('should provide health check endpoint', async () => {
      const response = await request(app.getHttpServer())
        .get('/health')
        .expect(200);

      expect(response.body).toMatchObject({
        status: 'ok',
        timestamp: expect.any(String),
        uptime: expect.any(Number)
      });
    });

    it('should provide metrics endpoint', async () => {
      const response = await request(app.getHttpServer())
        .get('/metrics')
        .expect(200);

      expect(response.text).toContain('# HELP');
      expect(response.text).toContain('# TYPE');
    });
  });
});