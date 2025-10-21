export const configuration = () => ({
  // Server configuration
  port: parseInt(process.env.PORT, 10) || 3001,
  nodeEnv: process.env.NODE_ENV || 'development',
  apiBaseUrl: process.env.API_BASE_URL || 'http://localhost:3001',
  
  // Neo4j configuration
  neo4j: {
    uri: process.env.NEO4J_URI || 'bolt://localhost:7687',
    username: process.env.NEO4J_USERNAME || 'neo4j',
    password: process.env.NEO4J_PASSWORD || 'password',
    database: process.env.NEO4J_DATABASE || 'neo4j',
    maxConnectionPoolSize: parseInt(process.env.NEO4J_MAX_POOL_SIZE) || 50,
    connectionTimeout: parseInt(process.env.NEO4J_CONNECTION_TIMEOUT) || 30000
  },

  // Redis configuration
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT, 10) || 6379,
    password: process.env.REDIS_PASSWORD,
    db: parseInt(process.env.REDIS_DB, 10) || 0,
    retryAttempts: parseInt(process.env.REDIS_RETRY_ATTEMPTS) || 3,
    retryDelay: parseInt(process.env.REDIS_RETRY_DELAY) || 1000
  },

  // AWS S3 configuration
  s3: {
    accessKeyId: process.env.S3_ACCESS_KEY_ID,
    secretAccessKey: process.env.S3_SECRET_ACCESS_KEY,
    region: process.env.S3_REGION || 'us-east-1',
    bucket: process.env.S3_BUCKET || 'wessley-3d-models',
    endpoint: process.env.S3_ENDPOINT, // For S3-compatible services
    forcePathStyle: process.env.S3_FORCE_PATH_STYLE === 'true'
  },

  // CDN configuration
  cdn: {
    baseUrl: process.env.CDN_BASE_URL || 'https://cdn.wessley.ai',
    distributionId: process.env.CLOUDFRONT_DISTRIBUTION_ID,
    invalidateOnUpload: process.env.CDN_INVALIDATE_ON_UPLOAD !== 'false'
  },

  // Supabase configuration
  supabase: {
    url: process.env.SUPABASE_URL,
    serviceKey: process.env.SUPABASE_SERVICE_KEY,
    anonKey: process.env.SUPABASE_ANON_KEY,
    jwtSecret: process.env.SUPABASE_JWT_SECRET
  },

  // Job queue configuration
  jobs: {
    concurrency: parseInt(process.env.JOB_CONCURRENCY) || 5,
    maxRetries: parseInt(process.env.JOB_MAX_RETRIES) || 3,
    backoffDelay: parseInt(process.env.JOB_BACKOFF_DELAY) || 5000,
    timeout: parseInt(process.env.JOB_TIMEOUT) || 300000, // 5 minutes
    cleanupInterval: parseInt(process.env.JOB_CLEANUP_INTERVAL) || 3600000 // 1 hour
  },

  // 3D generation configuration
  generation: {
    maxFileSize: parseInt(process.env.MAX_GLB_FILE_SIZE) || 52428800, // 50MB
    maxTriangles: parseInt(process.env.MAX_TRIANGLES) || 1000000,
    maxComponents: parseInt(process.env.MAX_COMPONENTS) || 10000,
    defaultQuality: process.env.DEFAULT_QUALITY || 'medium',
    enableLOD: process.env.ENABLE_LOD !== 'false',
    enableOptimization: process.env.ENABLE_OPTIMIZATION !== 'false'
  },

  // Rate limiting
  rateLimit: {
    windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 60000,
    maxRequests: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100,
    skipSuccessfulRequests: process.env.RATE_LIMIT_SKIP_SUCCESS === 'true'
  },

  // CORS configuration
  cors: {
    origin: process.env.CORS_ORIGIN || '*',
    credentials: process.env.CORS_CREDENTIALS === 'true'
  },

  // Logging configuration
  logging: {
    level: process.env.LOG_LEVEL || 'info',
    enableFileLogging: process.env.ENABLE_FILE_LOGGING === 'true',
    logFile: process.env.LOG_FILE || 'logs/app.log'
  },

  // Monitoring and metrics
  monitoring: {
    enableMetrics: process.env.ENABLE_METRICS !== 'false',
    metricsPort: parseInt(process.env.METRICS_PORT) || 9090,
    healthCheckInterval: parseInt(process.env.HEALTH_CHECK_INTERVAL) || 30000
  }
});