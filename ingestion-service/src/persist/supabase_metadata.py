"""
Supabase metadata persistence for job tracking and project management.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import uuid
import json

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    create_client = None
    Client = None

from ..core.schemas import IngestionStatus, ProcessingMetrics, CreateIngestionRequest


@dataclass
class JobRecord:
    """Job record for database persistence."""
    id: str
    project_id: str
    user_id: str
    status: str
    progress: int
    stage: str
    source_file_id: str
    source_type: str
    processing_modes: Dict[str, Any]
    artifacts: Dict[str, str]
    metrics: Dict[str, Any]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    notify_channel: Optional[str]


@dataclass
class ProjectRecord:
    """Project record for database persistence."""
    id: str
    user_id: str
    name: str
    description: Optional[str]
    vehicle_make: Optional[str]
    vehicle_model: Optional[str]
    vehicle_year: Optional[int]
    settings: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class ArtifactRecord:
    """Artifact record for database persistence."""
    id: str
    job_id: str
    project_id: str
    artifact_type: str
    filename: str
    storage_url: str
    content_type: str
    size_bytes: int
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class UserQuotaRecord:
    """User quota and usage tracking."""
    user_id: str
    plan_type: str
    quota_ingestions_per_month: int
    quota_storage_bytes: int
    usage_ingestions_current_month: int
    usage_storage_bytes: int
    usage_reset_date: datetime
    created_at: datetime
    updated_at: datetime


class SupabaseMetadata:
    """
    Handles job metadata, project management, and user quotas in Supabase.
    
    Database Tables:
    - ingestion_jobs: Job tracking and status
    - projects: Project management
    - artifacts: Artifact metadata and references
    - user_quotas: Usage tracking and limits
    - processing_metrics: Historical metrics
    """
    
    def __init__(
        self,
        url: str = None,
        key: str = None,
        table_prefix: str = ""
    ):
        """
        Initialize Supabase metadata client.
        
        Args:
            url: Supabase URL
            key: Supabase service key
            table_prefix: Optional table name prefix
        """
        if not SUPABASE_AVAILABLE:
            raise ImportError("supabase-py not available. Install with: pip install supabase")
        
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_SERVICE_KEY")
        self.table_prefix = table_prefix
        
        if not self.url or not self.key:
            raise ValueError("Supabase URL and service key required")
        
        self.client: Client = None
        self.logger = logging.getLogger(__name__)
    
    async def connect(self) -> bool:
        """
        Connect to Supabase and verify tables exist.
        
        Returns:
            True if connection successful
        """
        try:
            self.client = create_client(self.url, self.key)
            
            # Test connection by checking tables
            await self._verify_tables()
            
            self.logger.info("Connected to Supabase metadata store")
            return True
            
        except Exception as e:
            self.logger.error(f"Supabase connection failed: {e}")
            return False
    
    async def _verify_tables(self):
        """Verify required tables exist."""
        required_tables = [
            'ingestion_jobs',
            'projects', 
            'artifacts',
            'user_quotas'
        ]
        
        for table in required_tables:
            try:
                # Test table access
                result = self.client.table(self._table_name(table)).select("*").limit(1).execute()
                self.logger.debug(f"Table {table} verified")
            except Exception as e:
                self.logger.warning(f"Table {table} may not exist: {e}")
    
    def _table_name(self, base_name: str) -> str:
        """Get full table name with prefix."""
        return f"{self.table_prefix}{base_name}" if self.table_prefix else base_name
    
    async def create_job(
        self,
        job_id: uuid.UUID,
        project_id: uuid.UUID,
        user_id: str,
        request: CreateIngestionRequest
    ) -> bool:
        """
        Create new ingestion job record.
        
        Args:
            job_id: Unique job identifier
            project_id: Project identifier
            user_id: User identifier
            request: Ingestion request details
            
        Returns:
            True if job created successfully
        """
        try:
            job_data = {
                'id': str(job_id),
                'project_id': str(project_id),
                'user_id': user_id,
                'status': IngestionStatus.QUEUED.value,
                'progress': 0,
                'stage': 'queued',
                'source_file_id': request.source.file_id,
                'source_type': request.source.type,
                'processing_modes': {
                    'ocr': [engine.value for engine in request.modes.ocr],
                    'schematic_parse': request.modes.schematic_parse
                },
                'artifacts': {},
                'metrics': {},
                'error_message': None,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'completed_at': None,
                'notify_channel': request.notify_channel
            }
            
            result = self.client.table(self._table_name('ingestion_jobs')).insert(job_data).execute()
            
            self.logger.info(f"Created job {job_id} for project {project_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create job: {e}")
            return False
    
    async def update_job_status(
        self,
        job_id: uuid.UUID,
        status: IngestionStatus,
        progress: int = None,
        stage: str = None,
        error: str = None,
        artifacts: Dict[str, str] = None,
        metrics: Dict[str, Any] = None
    ) -> bool:
        """
        Update job status and progress.
        
        Args:
            job_id: Job identifier
            status: New status
            progress: Progress percentage (0-100)
            stage: Current processing stage
            error: Error message if failed
            artifacts: Updated artifacts dictionary
            metrics: Processing metrics
            
        Returns:
            True if update successful
        """
        try:
            update_data = {
                'status': status.value,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            if progress is not None:
                update_data['progress'] = progress
            
            if stage is not None:
                update_data['stage'] = stage
            
            if error is not None:
                update_data['error_message'] = error
            
            if artifacts is not None:
                update_data['artifacts'] = artifacts
            
            if metrics is not None:
                update_data['metrics'] = metrics
            
            # Set completion time for completed/failed jobs
            if status in [IngestionStatus.COMPLETED, IngestionStatus.FAILED]:
                update_data['completed_at'] = datetime.now(timezone.utc).isoformat()
            
            result = self.client.table(self._table_name('ingestion_jobs')).update(update_data).eq('id', str(job_id)).execute()
            
            self.logger.debug(f"Updated job {job_id}: {status.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update job status: {e}")
            return False
    
    async def get_job(self, job_id: uuid.UUID) -> Optional[JobRecord]:
        """
        Get job record by ID.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job record or None if not found
        """
        try:
            result = self.client.table(self._table_name('ingestion_jobs')).select("*").eq('id', str(job_id)).execute()
            
            if result.data:
                job_data = result.data[0]
                return JobRecord(
                    id=job_data['id'],
                    project_id=job_data['project_id'],
                    user_id=job_data['user_id'],
                    status=job_data['status'],
                    progress=job_data['progress'],
                    stage=job_data['stage'],
                    source_file_id=job_data['source_file_id'],
                    source_type=job_data['source_type'],
                    processing_modes=job_data['processing_modes'],
                    artifacts=job_data['artifacts'],
                    metrics=job_data['metrics'],
                    error_message=job_data['error_message'],
                    created_at=datetime.fromisoformat(job_data['created_at'].replace('Z', '+00:00')),
                    updated_at=datetime.fromisoformat(job_data['updated_at'].replace('Z', '+00:00')),
                    completed_at=datetime.fromisoformat(job_data['completed_at'].replace('Z', '+00:00')) if job_data['completed_at'] else None,
                    notify_channel=job_data['notify_channel']
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get job: {e}")
            return None
    
    async def list_project_jobs(
        self,
        project_id: uuid.UUID,
        user_id: str = None,
        status: IngestionStatus = None,
        limit: int = 50
    ) -> List[JobRecord]:
        """
        List jobs for a project.
        
        Args:
            project_id: Project identifier
            user_id: Optional user filter
            status: Optional status filter
            limit: Maximum number of results
            
        Returns:
            List of job records
        """
        try:
            query = self.client.table(self._table_name('ingestion_jobs')).select("*").eq('project_id', str(project_id))
            
            if user_id:
                query = query.eq('user_id', user_id)
            
            if status:
                query = query.eq('status', status.value)
            
            result = query.order('created_at', desc=True).limit(limit).execute()
            
            jobs = []
            for job_data in result.data:
                jobs.append(JobRecord(
                    id=job_data['id'],
                    project_id=job_data['project_id'],
                    user_id=job_data['user_id'],
                    status=job_data['status'],
                    progress=job_data['progress'],
                    stage=job_data['stage'],
                    source_file_id=job_data['source_file_id'],
                    source_type=job_data['source_type'],
                    processing_modes=job_data['processing_modes'],
                    artifacts=job_data['artifacts'],
                    metrics=job_data['metrics'],
                    error_message=job_data['error_message'],
                    created_at=datetime.fromisoformat(job_data['created_at'].replace('Z', '+00:00')),
                    updated_at=datetime.fromisoformat(job_data['updated_at'].replace('Z', '+00:00')),
                    completed_at=datetime.fromisoformat(job_data['completed_at'].replace('Z', '+00:00')) if job_data['completed_at'] else None,
                    notify_channel=job_data['notify_channel']
                ))
            
            return jobs
            
        except Exception as e:
            self.logger.error(f"Failed to list project jobs: {e}")
            return []
    
    async def create_project(
        self,
        project_id: uuid.UUID,
        user_id: str,
        name: str,
        description: str = None,
        vehicle_info: Dict[str, Any] = None,
        settings: Dict[str, Any] = None
    ) -> bool:
        """
        Create new project record.
        
        Args:
            project_id: Project identifier
            user_id: User identifier
            name: Project name
            description: Project description
            vehicle_info: Vehicle metadata
            settings: Project settings
            
        Returns:
            True if project created successfully
        """
        try:
            project_data = {
                'id': str(project_id),
                'user_id': user_id,
                'name': name,
                'description': description,
                'vehicle_make': vehicle_info.get('make') if vehicle_info else None,
                'vehicle_model': vehicle_info.get('model') if vehicle_info else None,
                'vehicle_year': vehicle_info.get('year') if vehicle_info else None,
                'settings': settings or {},
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            result = self.client.table(self._table_name('projects')).insert(project_data).execute()
            
            self.logger.info(f"Created project {project_id} for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create project: {e}")
            return False
    
    async def get_project(self, project_id: uuid.UUID) -> Optional[ProjectRecord]:
        """
        Get project record by ID.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Project record or None if not found
        """
        try:
            result = self.client.table(self._table_name('projects')).select("*").eq('id', str(project_id)).execute()
            
            if result.data:
                project_data = result.data[0]
                return ProjectRecord(
                    id=project_data['id'],
                    user_id=project_data['user_id'],
                    name=project_data['name'],
                    description=project_data['description'],
                    vehicle_make=project_data['vehicle_make'],
                    vehicle_model=project_data['vehicle_model'],
                    vehicle_year=project_data['vehicle_year'],
                    settings=project_data['settings'],
                    created_at=datetime.fromisoformat(project_data['created_at'].replace('Z', '+00:00')),
                    updated_at=datetime.fromisoformat(project_data['updated_at'].replace('Z', '+00:00'))
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get project: {e}")
            return None
    
    async def list_user_projects(self, user_id: str, limit: int = 50) -> List[ProjectRecord]:
        """
        List projects for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of results
            
        Returns:
            List of project records
        """
        try:
            result = self.client.table(self._table_name('projects')).select("*").eq('user_id', user_id).order('updated_at', desc=True).limit(limit).execute()
            
            projects = []
            for project_data in result.data:
                projects.append(ProjectRecord(
                    id=project_data['id'],
                    user_id=project_data['user_id'],
                    name=project_data['name'],
                    description=project_data['description'],
                    vehicle_make=project_data['vehicle_make'],
                    vehicle_model=project_data['vehicle_model'],
                    vehicle_year=project_data['vehicle_year'],
                    settings=project_data['settings'],
                    created_at=datetime.fromisoformat(project_data['created_at'].replace('Z', '+00:00')),
                    updated_at=datetime.fromisoformat(project_data['updated_at'].replace('Z', '+00:00'))
                ))
            
            return projects
            
        except Exception as e:
            self.logger.error(f"Failed to list user projects: {e}")
            return []
    
    async def store_artifact_metadata(
        self,
        artifact_id: str,
        job_id: uuid.UUID,
        project_id: uuid.UUID,
        artifact_type: str,
        filename: str,
        storage_url: str,
        content_type: str,
        size_bytes: int,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Store artifact metadata.
        
        Args:
            artifact_id: Artifact identifier
            job_id: Job identifier
            project_id: Project identifier
            artifact_type: Type of artifact
            filename: Original filename
            storage_url: Storage URL
            content_type: MIME content type
            size_bytes: File size in bytes
            metadata: Additional metadata
            
        Returns:
            True if stored successfully
        """
        try:
            artifact_data = {
                'id': artifact_id,
                'job_id': str(job_id),
                'project_id': str(project_id),
                'artifact_type': artifact_type,
                'filename': filename,
                'storage_url': storage_url,
                'content_type': content_type,
                'size_bytes': size_bytes,
                'metadata': metadata or {},
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            result = self.client.table(self._table_name('artifacts')).insert(artifact_data).execute()
            
            self.logger.debug(f"Stored artifact metadata: {artifact_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store artifact metadata: {e}")
            return False
    
    async def list_job_artifacts(self, job_id: uuid.UUID) -> List[ArtifactRecord]:
        """
        List artifacts for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            List of artifact records
        """
        try:
            result = self.client.table(self._table_name('artifacts')).select("*").eq('job_id', str(job_id)).execute()
            
            artifacts = []
            for artifact_data in result.data:
                artifacts.append(ArtifactRecord(
                    id=artifact_data['id'],
                    job_id=artifact_data['job_id'],
                    project_id=artifact_data['project_id'],
                    artifact_type=artifact_data['artifact_type'],
                    filename=artifact_data['filename'],
                    storage_url=artifact_data['storage_url'],
                    content_type=artifact_data['content_type'],
                    size_bytes=artifact_data['size_bytes'],
                    metadata=artifact_data['metadata'],
                    created_at=datetime.fromisoformat(artifact_data['created_at'].replace('Z', '+00:00'))
                ))
            
            return artifacts
            
        except Exception as e:
            self.logger.error(f"Failed to list job artifacts: {e}")
            return []
    
    async def check_user_quota(self, user_id: str) -> Optional[UserQuotaRecord]:
        """
        Check user quota and usage.
        
        Args:
            user_id: User identifier
            
        Returns:
            User quota record or None if not found
        """
        try:
            result = self.client.table(self._table_name('user_quotas')).select("*").eq('user_id', user_id).execute()
            
            if result.data:
                quota_data = result.data[0]
                return UserQuotaRecord(
                    user_id=quota_data['user_id'],
                    plan_type=quota_data['plan_type'],
                    quota_ingestions_per_month=quota_data['quota_ingestions_per_month'],
                    quota_storage_bytes=quota_data['quota_storage_bytes'],
                    usage_ingestions_current_month=quota_data['usage_ingestions_current_month'],
                    usage_storage_bytes=quota_data['usage_storage_bytes'],
                    usage_reset_date=datetime.fromisoformat(quota_data['usage_reset_date'].replace('Z', '+00:00')),
                    created_at=datetime.fromisoformat(quota_data['created_at'].replace('Z', '+00:00')),
                    updated_at=datetime.fromisoformat(quota_data['updated_at'].replace('Z', '+00:00'))
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to check user quota: {e}")
            return None
    
    async def increment_user_usage(
        self,
        user_id: str,
        ingestions: int = 0,
        storage_bytes: int = 0
    ) -> bool:
        """
        Increment user usage counters.
        
        Args:
            user_id: User identifier
            ingestions: Number of ingestions to add
            storage_bytes: Storage bytes to add
            
        Returns:
            True if updated successfully
        """
        try:
            # First check if quota record exists
            quota = await self.check_user_quota(user_id)
            
            if not quota:
                # Create default quota record
                await self._create_default_quota(user_id)
                quota = await self.check_user_quota(user_id)
            
            # Update usage
            update_data = {
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            if ingestions > 0:
                update_data['usage_ingestions_current_month'] = quota.usage_ingestions_current_month + ingestions
            
            if storage_bytes > 0:
                update_data['usage_storage_bytes'] = quota.usage_storage_bytes + storage_bytes
            
            result = self.client.table(self._table_name('user_quotas')).update(update_data).eq('user_id', user_id).execute()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to increment user usage: {e}")
            return False
    
    async def _create_default_quota(self, user_id: str) -> bool:
        """Create default quota record for new user."""
        try:
            next_month = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if next_month.month == 12:
                next_month = next_month.replace(year=next_month.year + 1, month=1)
            else:
                next_month = next_month.replace(month=next_month.month + 1)
            
            quota_data = {
                'user_id': user_id,
                'plan_type': 'free',
                'quota_ingestions_per_month': 10,
                'quota_storage_bytes': 1024 * 1024 * 100,  # 100MB
                'usage_ingestions_current_month': 0,
                'usage_storage_bytes': 0,
                'usage_reset_date': next_month.isoformat(),
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            result = self.client.table(self._table_name('user_quotas')).insert(quota_data).execute()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create default quota: {e}")
            return False
    
    async def delete_job_data(self, job_id: uuid.UUID) -> bool:
        """
        Delete all data for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if deletion successful
        """
        try:
            # Delete artifacts first
            self.client.table(self._table_name('artifacts')).delete().eq('job_id', str(job_id)).execute()
            
            # Delete job record
            self.client.table(self._table_name('ingestion_jobs')).delete().eq('id', str(job_id)).execute()
            
            self.logger.info(f"Deleted job data for {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete job data: {e}")
            return False
    
    async def delete_project_data(self, project_id: uuid.UUID) -> bool:
        """
        Delete all data for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            True if deletion successful
        """
        try:
            # Delete artifacts
            self.client.table(self._table_name('artifacts')).delete().eq('project_id', str(project_id)).execute()
            
            # Delete jobs
            self.client.table(self._table_name('ingestion_jobs')).delete().eq('project_id', str(project_id)).execute()
            
            # Delete project
            self.client.table(self._table_name('projects')).delete().eq('id', str(project_id)).execute()
            
            self.logger.info(f"Deleted project data for {project_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete project data: {e}")
            return False
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        try:
            stats = {}
            
            # Job statistics
            jobs_result = self.client.table(self._table_name('ingestion_jobs')).select("status", count="exact").execute()
            stats['total_jobs'] = jobs_result.count
            
            # Project statistics
            projects_result = self.client.table(self._table_name('projects')).select("id", count="exact").execute()
            stats['total_projects'] = projects_result.count
            
            # Artifact statistics
            artifacts_result = self.client.table(self._table_name('artifacts')).select("size_bytes").execute()
            total_storage = sum(artifact['size_bytes'] for artifact in artifacts_result.data)
            stats['total_storage_bytes'] = total_storage
            stats['total_artifacts'] = len(artifacts_result.data)
            
            # User statistics
            quotas_result = self.client.table(self._table_name('user_quotas')).select("user_id", count="exact").execute()
            stats['total_users'] = quotas_result.count
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get system stats: {e}")
            return {}


# Convenience function
def create_supabase_metadata(**kwargs) -> SupabaseMetadata:
    """Create Supabase metadata client with environment configuration."""
    return SupabaseMetadata(**kwargs)