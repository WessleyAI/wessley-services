import { afterAll, beforeAll } from 'vitest';
import { GenericContainer, StartedTestContainer } from 'testcontainers';

// Global test containers
let neo4jContainer: StartedTestContainer;
let redisContainer: StartedTestContainer;

beforeAll(async () => {
  console.log('🚀 Starting test containers...');

  // Start Neo4j container
  neo4jContainer = await new GenericContainer('neo4j:5.15')
    .withEnvironment({
      NEO4J_AUTH: 'neo4j/testpassword',
      NEO4J_PLUGINS: '["apoc"]',
    })
    .withExposedPorts(7687, 7474)
    .withWaitStrategy({
      type: 'log',
      message: 'Bolt enabled',
    })
    .start();

  // Start Redis container
  redisContainer = await new GenericContainer('redis:7-alpine')
    .withExposedPorts(6379)
    .withWaitStrategy({
      type: 'log',
      message: 'Ready to accept connections',
    })
    .start();

  // Set environment variables for tests
  process.env.NEO4J_URI = `bolt://localhost:${neo4jContainer.getMappedPort(7687)}`;
  process.env.NEO4J_USERNAME = 'neo4j';
  process.env.NEO4J_PASSWORD = 'testpassword';
  process.env.REDIS_HOST = 'localhost';
  process.env.REDIS_PORT = redisContainer.getMappedPort(6379).toString();

  console.log(`✅ Neo4j running on port ${neo4jContainer.getMappedPort(7687)}`);
  console.log(`✅ Redis running on port ${redisContainer.getMappedPort(6379)}`);
}, 120000);

afterAll(async () => {
  console.log('🧹 Cleaning up test containers...');
  
  if (neo4jContainer) {
    await neo4jContainer.stop();
  }
  
  if (redisContainer) {
    await redisContainer.stop();
  }
  
  console.log('✅ Test containers stopped');
}, 30000);