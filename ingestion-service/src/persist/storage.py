"""
Storage management for artifacts using S3 and Supabase Storage.
"""
import os
import logging
import asyncio
import json
from typing import Dict, Any, Optional, List, Union, BinaryIO
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
from pathlib import Path
import mimetypes

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    boto3 = None

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    create_client = None
    Client = None


@dataclass
class StorageConfig:
    """Configuration for storage backends."""
    use_s3: bool = False
    use_supabase: bool = True
    s3_bucket: Optional[str] = None
    s3_region: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    supabase_bucket: str = "ingestions"


@dataclass
class StorageResult:
    """Result of storage operation."""
    success: bool
    url: Optional[str] = None
    path: Optional[str] = None
    size_bytes: int = 0
    content_type: Optional[str] = None
    metadata: Dict[str, Any] = None
    error: Optional[str] = None


@dataclass
class ArtifactInfo:
    """Information about a stored artifact."""
    artifact_id: str
    project_id: str
    job_id: str
    artifact_type: str
    filename: str
    content_type: str
    size_bytes: int
    storage_url: str
    created_at: datetime
    metadata: Dict[str, Any]


class StorageManager:
    """
    Unified storage manager supporting S3 and Supabase Storage.
    
    Artifacts stored:
    - text_spans.ndjson: OCR text extraction results
    - components.json: Component detection results
    - netlist.json: Generated netlist
    - netlist.graphml: GraphML format netlist
    - netlist.ndjson: NDJSON format netlist
    - debug_overlay.png: Visual debugging overlay
    - thumbnails/: Page thumbnails
    - processing_logs/: Processing logs and metrics
    """
    
    def __init__(self, config: StorageConfig = None):
        """
        Initialize storage manager.
        
        Args:
            config: Storage configuration
        """
        self.config = config or self._load_config_from_env()
        self.s3_client = None
        self.supabase_client = None
        self.logger = logging.getLogger(__name__)
        
    def _load_config_from_env(self) -> StorageConfig:
        """Load configuration from environment variables."""
        return StorageConfig(
            use_s3=os.getenv("USE_S3", "false").lower() == "true",
            use_supabase=os.getenv("USE_SUPABASE", "true").lower() == "true",
            s3_bucket=os.getenv("S3_BUCKET"),
            s3_region=os.getenv("S3_REGION", "us-east-1"),
            s3_access_key=os.getenv("S3_ACCESS_KEY_ID"),
            s3_secret_key=os.getenv("S3_SECRET_ACCESS_KEY"),
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_key=os.getenv("SUPABASE_SERVICE_KEY"),
            supabase_bucket=os.getenv("SUPABASE_BUCKET", "ingestions")
        )
    
    async def initialize(self) -> bool:
        """
        Initialize storage backends.
        
        Returns:
            True if at least one backend initialized successfully
        """
        success = False
        
        # Initialize S3 if configured
        if self.config.use_s3:
            if await self._initialize_s3():
                success = True
            else:
                self.logger.warning("S3 initialization failed")
        
        # Initialize Supabase if configured
        if self.config.use_supabase:
            if await self._initialize_supabase():
                success = True
            else:
                self.logger.warning("Supabase initialization failed")
        
        if not success:
            self.logger.error("No storage backend initialized successfully")
        
        return success
    
    async def _initialize_s3(self) -> bool:
        """Initialize S3 client."""
        if not S3_AVAILABLE:
            self.logger.error("boto3 not available for S3 storage")
            return False
        
        try:
            self.s3_client = boto3.client(
                's3',
                region_name=self.config.s3_region,
                aws_access_key_id=self.config.s3_access_key,
                aws_secret_access_key=self.config.s3_secret_key
            )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.config.s3_bucket)
            self.logger.info(f"S3 initialized: bucket={self.config.s3_bucket}")
            return True
            
        except Exception as e:
            self.logger.error(f"S3 initialization failed: {e}")
            return False
    
    async def _initialize_supabase(self) -> bool:
        """Initialize Supabase client."""
        if not SUPABASE_AVAILABLE:
            self.logger.error("supabase-py not available for Supabase storage")
            return False
        
        try:
            if not self.config.supabase_url or not self.config.supabase_key:
                self.logger.error("Supabase URL and key required")
                return False
            
            self.supabase_client = create_client(
                self.config.supabase_url,
                self.config.supabase_key
            )
            
            # Test connection by listing buckets
            buckets = self.supabase_client.storage.list_buckets()
            self.logger.info(f"Supabase initialized: bucket={self.config.supabase_bucket}")
            return True
            
        except Exception as e:
            self.logger.error(f"Supabase initialization failed: {e}")
            return False
    
    async def store_artifact(
        self,
        project_id: uuid.UUID,
        job_id: uuid.UUID,
        artifact_type: str,
        content: Union[str, bytes, BinaryIO],
        filename: str,
        content_type: str = None,
        metadata: Dict[str, Any] = None
    ) -> StorageResult:
        """
        Store an artifact in configured storage backend.
        
        Args:
            project_id: Project identifier
            job_id: Job identifier
            artifact_type: Type of artifact (e.g., 'netlist', 'debug_overlay')
            content: Content to store
            filename: Filename for the artifact
            content_type: MIME content type
            metadata: Additional metadata
            
        Returns:
            Storage result with URL and metadata
        """
        # Determine content type if not provided
        if not content_type:
            content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        
        # Generate storage path
        storage_path = self._generate_storage_path(project_id, job_id, artifact_type, filename)
        
        # Convert content to bytes if needed
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        elif hasattr(content, 'read'):
            content_bytes = content.read()
        else:
            content_bytes = content
        
        # Try Supabase first, then S3
        if self.config.use_supabase and self.supabase_client:
            result = await self._store_to_supabase(
                storage_path, content_bytes, content_type, metadata or {}
            )
            if result.success:
                return result
        
        if self.config.use_s3 and self.s3_client:
            result = await self._store_to_s3(
                storage_path, content_bytes, content_type, metadata or {}
            )
            if result.success:
                return result
        
        return StorageResult(
            success=False,
            error="No storage backend available"
        )
    
    async def _store_to_supabase(
        self,
        path: str,
        content: bytes,
        content_type: str,
        metadata: Dict[str, Any]
    ) -> StorageResult:
        """Store content to Supabase Storage."""
        try:
            # Upload file
            result = self.supabase_client.storage.from_(self.config.supabase_bucket).upload(
                path=path,
                file=content,
                file_options={
                    "content-type": content_type,
                    "upsert": True
                }
            )
            
            # Get public URL
            public_url = self.supabase_client.storage.from_(self.config.supabase_bucket).get_public_url(path)
            
            return StorageResult(
                success=True,
                url=public_url,
                path=path,
                size_bytes=len(content),
                content_type=content_type,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Supabase storage failed: {e}")
            return StorageResult(
                success=False,
                error=str(e)
            )
    
    async def _store_to_s3(
        self,
        path: str,
        content: bytes,
        content_type: str,
        metadata: Dict[str, Any]
    ) -> StorageResult:
        """Store content to S3."""
        try:
            # Upload file
            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=path,
                Body=content,
                ContentType=content_type,
                Metadata={k: str(v) for k, v in metadata.items()}
            )
            
            # Generate URL
            url = f"s3://{self.config.s3_bucket}/{path}"
            
            return StorageResult(
                success=True,
                url=url,
                path=path,
                size_bytes=len(content),
                content_type=content_type,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"S3 storage failed: {e}")
            return StorageResult(
                success=False,
                error=str(e)
            )
    
    async def retrieve_artifact(
        self,
        storage_url: str
    ) -> Optional[bytes]:
        """
        Retrieve artifact content by URL.
        
        Args:
            storage_url: Storage URL (S3 or Supabase)
            
        Returns:
            Artifact content as bytes, or None if not found
        """
        try:
            if storage_url.startswith('s3://'):
                return await self._retrieve_from_s3(storage_url)
            elif self.config.supabase_url and self.config.supabase_url in storage_url:
                return await self._retrieve_from_supabase(storage_url)
            else:
                self.logger.error(f"Unsupported storage URL: {storage_url}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve artifact: {e}")
            return None
    
    async def _retrieve_from_s3(self, s3_url: str) -> Optional[bytes]:
        """Retrieve content from S3."""
        if not self.s3_client:
            return None
        
        try:
            # Parse S3 URL
            parts = s3_url.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            key = parts[1]
            
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
            
        except Exception as e:
            self.logger.error(f"S3 retrieval failed: {e}")
            return None
    
    async def _retrieve_from_supabase(self, supabase_url: str) -> Optional[bytes]:
        """Retrieve content from Supabase Storage."""
        if not self.supabase_client:
            return None
        
        try:
            # Extract path from URL
            path = supabase_url.split('/')[-1]  # Simplified path extraction
            
            response = self.supabase_client.storage.from_(self.config.supabase_bucket).download(path)
            return response
            
        except Exception as e:
            self.logger.error(f"Supabase retrieval failed: {e}")
            return None
    
    async def delete_artifacts(
        self,
        project_id: uuid.UUID,
        job_id: uuid.UUID = None
    ) -> bool:
        """
        Delete artifacts for a project or specific job.
        
        Args:
            project_id: Project identifier
            job_id: Optional job identifier (if None, deletes all project artifacts)
            
        Returns:
            True if deletion successful
        """
        try:
            prefix = f"projects/{project_id}"
            if job_id:
                prefix += f"/jobs/{job_id}"
            
            success = True
            
            # Delete from Supabase
            if self.config.use_supabase and self.supabase_client:
                try:
                    files = self.supabase_client.storage.from_(self.config.supabase_bucket).list(prefix)
                    if files:
                        file_paths = [f"{prefix}/{file['name']}" for file in files]
                        self.supabase_client.storage.from_(self.config.supabase_bucket).remove(file_paths)
                except Exception as e:
                    self.logger.error(f"Supabase deletion failed: {e}")
                    success = False
            
            # Delete from S3
            if self.config.use_s3 and self.s3_client:
                try:
                    paginator = self.s3_client.get_paginator('list_objects_v2')
                    pages = paginator.paginate(Bucket=self.config.s3_bucket, Prefix=prefix)
                    
                    for page in pages:
                        if 'Contents' in page:
                            objects = [{'Key': obj['Key']} for obj in page['Contents']]
                            if objects:
                                self.s3_client.delete_objects(
                                    Bucket=self.config.s3_bucket,
                                    Delete={'Objects': objects}
                                )
                except Exception as e:
                    self.logger.error(f"S3 deletion failed: {e}")
                    success = False
            
            return success
            
        except Exception as e:
            self.logger.error(f"Artifact deletion failed: {e}")
            return False
    
    async def list_artifacts(
        self,
        project_id: uuid.UUID,
        job_id: uuid.UUID = None,
        artifact_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        List artifacts for a project or job.
        
        Args:
            project_id: Project identifier
            job_id: Optional job identifier
            artifact_type: Optional artifact type filter
            
        Returns:
            List of artifact metadata
        """
        try:
            prefix = f"projects/{project_id}"
            if job_id:
                prefix += f"/jobs/{job_id}"
            if artifact_type:
                prefix += f"/{artifact_type}"
            
            artifacts = []
            
            # List from primary storage backend
            if self.config.use_supabase and self.supabase_client:
                artifacts.extend(await self._list_from_supabase(prefix))
            elif self.config.use_s3 and self.s3_client:
                artifacts.extend(await self._list_from_s3(prefix))
            
            return artifacts
            
        except Exception as e:
            self.logger.error(f"Artifact listing failed: {e}")
            return []
    
    async def _list_from_supabase(self, prefix: str) -> List[Dict[str, Any]]:
        """List artifacts from Supabase Storage."""
        try:
            files = self.supabase_client.storage.from_(self.config.supabase_bucket).list(prefix)
            
            artifacts = []
            for file_info in files:
                artifacts.append({
                    'name': file_info['name'],
                    'size': file_info.get('metadata', {}).get('size', 0),
                    'created_at': file_info.get('created_at'),
                    'updated_at': file_info.get('updated_at'),
                    'url': self.supabase_client.storage.from_(self.config.supabase_bucket).get_public_url(
                        f"{prefix}/{file_info['name']}"
                    )
                })
            
            return artifacts
            
        except Exception as e:
            self.logger.error(f"Supabase listing failed: {e}")
            return []
    
    async def _list_from_s3(self, prefix: str) -> List[Dict[str, Any]]:
        """List artifacts from S3."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.s3_bucket,
                Prefix=prefix
            )
            
            artifacts = []
            for obj in response.get('Contents', []):
                artifacts.append({
                    'name': obj['Key'].split('/')[-1],
                    'size': obj['Size'],
                    'created_at': obj['LastModified'].isoformat(),
                    'url': f"s3://{self.config.s3_bucket}/{obj['Key']}"
                })
            
            return artifacts
            
        except Exception as e:
            self.logger.error(f"S3 listing failed: {e}")
            return []
    
    def _generate_storage_path(
        self,
        project_id: uuid.UUID,
        job_id: uuid.UUID,
        artifact_type: str,
        filename: str
    ) -> str:
        """Generate hierarchical storage path."""
        return f"projects/{project_id}/jobs/{job_id}/{artifact_type}/{filename}"
    
    async def generate_signed_url(
        self,
        storage_url: str,
        expiry_hours: int = 24
    ) -> Optional[str]:
        """
        Generate signed URL for temporary access.
        
        Args:
            storage_url: Storage URL
            expiry_hours: URL expiry time in hours
            
        Returns:
            Signed URL or None if not supported
        """
        try:
            if storage_url.startswith('s3://') and self.s3_client:
                # Parse S3 URL
                parts = storage_url.replace('s3://', '').split('/', 1)
                bucket = parts[0]
                key = parts[1]
                
                # Generate presigned URL
                signed_url = self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket, 'Key': key},
                    ExpiresIn=expiry_hours * 3600
                )
                return signed_url
            
            elif self.config.supabase_url and self.config.supabase_url in storage_url:
                # Supabase URLs are already public, return as-is
                return storage_url
            
            return None
            
        except Exception as e:
            self.logger.error(f"Signed URL generation failed: {e}")
            return None
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        stats = {
            'backends': [],
            'total_size_bytes': 0,
            'total_objects': 0
        }
        
        try:
            # Supabase stats
            if self.config.use_supabase and self.supabase_client:
                supabase_stats = await self._get_supabase_stats()
                stats['backends'].append(supabase_stats)
                stats['total_size_bytes'] += supabase_stats.get('size_bytes', 0)
                stats['total_objects'] += supabase_stats.get('object_count', 0)
            
            # S3 stats
            if self.config.use_s3 and self.s3_client:
                s3_stats = await self._get_s3_stats()
                stats['backends'].append(s3_stats)
                stats['total_size_bytes'] += s3_stats.get('size_bytes', 0)
                stats['total_objects'] += s3_stats.get('object_count', 0)
            
        except Exception as e:
            self.logger.error(f"Storage stats failed: {e}")
        
        return stats
    
    async def _get_supabase_stats(self) -> Dict[str, Any]:
        """Get Supabase storage statistics."""
        try:
            files = self.supabase_client.storage.from_(self.config.supabase_bucket).list()
            
            total_size = sum(f.get('metadata', {}).get('size', 0) for f in files)
            
            return {
                'backend': 'supabase',
                'bucket': self.config.supabase_bucket,
                'object_count': len(files),
                'size_bytes': total_size
            }
            
        except Exception as e:
            self.logger.error(f"Supabase stats failed: {e}")
            return {'backend': 'supabase', 'error': str(e)}
    
    async def _get_s3_stats(self) -> Dict[str, Any]:
        """Get S3 storage statistics."""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.config.s3_bucket)
            
            total_size = 0
            object_count = 0
            
            for page in pages:
                for obj in page.get('Contents', []):
                    total_size += obj['Size']
                    object_count += 1
            
            return {
                'backend': 's3',
                'bucket': self.config.s3_bucket,
                'object_count': object_count,
                'size_bytes': total_size
            }
            
        except Exception as e:
            self.logger.error(f"S3 stats failed: {e}")
            return {'backend': 's3', 'error': str(e)}


# Convenience function
def create_storage_manager(config: StorageConfig = None) -> StorageManager:
    """Create storage manager with configuration."""
    return StorageManager(config)


# Artifact type constants
class ArtifactType:
    """Standard artifact type constants."""
    TEXT_SPANS = "text_spans"
    COMPONENTS = "components" 
    NETLIST_JSON = "netlist_json"
    NETLIST_GRAPHML = "netlist_graphml"
    NETLIST_NDJSON = "netlist_ndjson"
    DEBUG_OVERLAY = "debug_overlay"
    THUMBNAILS = "thumbnails"
    PROCESSING_LOGS = "processing_logs"
    EMBEDDINGS = "embeddings"
    METRICS = "metrics"