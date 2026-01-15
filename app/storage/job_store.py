"""
Job storage with SQLite/in-memory support.

Implements Checkpoint pattern: persist job state immediately.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiosqlite

from app.config import get_config
from app.models.schemas import JobState

logger = logging.getLogger(__name__)


class JobStore:
    """
    Persistent job storage.
    
    Supports SQLite for persistence or in-memory for testing.
    Implements Checkpoint pattern from Mandrake.
    """
    
    def __init__(self, db_path: Optional[str] = None, use_memory: bool = False):
        config = get_config()
        
        if use_memory or config.storage.type == "memory":
            self.db_path = ":memory:"
            self._is_memory = True
        else:
            self.db_path = db_path or config.storage.path
            self._is_memory = False
            
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.jobs_dir = config.storage.jobs_dir
        Path(self.jobs_dir).mkdir(parents=True, exist_ok=True)
        
        self._db: Optional[aiosqlite.Connection] = None
        self._memory_store: Dict[str, JobState] = {}
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize database connection and schema."""
        if self._is_memory:
            # Use in-memory dict
            logger.info("Using in-memory job storage")
            return
        
        self._db = await aiosqlite.connect(self.db_path)
        
        # Create tables
        await self._db.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        await self._db.commit()
        
        logger.info(f"Initialized SQLite job storage at {self.db_path}")
    
    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None
    
    async def save_job(self, job: JobState) -> None:
        """
        Save a new job to storage.
        
        Implements Checkpoint pattern: immediate persistence.
        """
        async with self._lock:
            if self._is_memory:
                self._memory_store[job.job_id] = job
                return
            
            data = job.model_dump_json()
            
            await self._db.execute(
                '''
                INSERT INTO jobs (job_id, status, data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ''',
                (
                    job.job_id,
                    job.status,
                    data,
                    job.created_at.isoformat(),
                    job.updated_at.isoformat()
                )
            )
            await self._db.commit()
    
    async def update_job(self, job: JobState) -> None:
        """
        Update existing job in storage.
        
        Implements Checkpoint pattern: immediate persistence.
        """
        job.updated_at = datetime.utcnow()
        
        async with self._lock:
            if self._is_memory:
                self._memory_store[job.job_id] = job
                return
            
            data = job.model_dump_json()
            
            await self._db.execute(
                '''
                UPDATE jobs 
                SET status = ?, data = ?, updated_at = ?
                WHERE job_id = ?
                ''',
                (
                    job.status,
                    data,
                    job.updated_at.isoformat(),
                    job.job_id
                )
            )
            await self._db.commit()
    
    async def get_job(self, job_id: str) -> Optional[JobState]:
        """Get job by ID."""
        async with self._lock:
            if self._is_memory:
                return self._memory_store.get(job_id)
            
            async with self._db.execute(
                'SELECT data FROM jobs WHERE job_id = ?',
                (job_id,)
            ) as cursor:
                row = await cursor.fetchone()
                
            if row:
                return JobState.model_validate_json(row[0])
            return None
    
    async def get_recent_jobs(self, limit: int = 10) -> List[JobState]:
        """Get most recent jobs."""
        async with self._lock:
            if self._is_memory:
                jobs = list(self._memory_store.values())
                jobs.sort(key=lambda j: j.created_at, reverse=True)
                return jobs[:limit]
            
            jobs = []
            async with self._db.execute(
                '''
                SELECT data FROM jobs 
                ORDER BY updated_at DESC 
                LIMIT ?
                ''',
                (limit,)
            ) as cursor:
                async for row in cursor:
                    jobs.append(JobState.model_validate_json(row[0]))
            
            return jobs
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete a job from storage."""
        async with self._lock:
            if self._is_memory:
                if job_id in self._memory_store:
                    del self._memory_store[job_id]
                    return True
                return False
            
            cursor = await self._db.execute(
                'DELETE FROM jobs WHERE job_id = ?',
                (job_id,)
            )
            await self._db.commit()
            return cursor.rowcount > 0
    
    def get_job_dir(self, job_id: str) -> Path:
        """Get directory for job files."""
        job_dir = Path(self.jobs_dir) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir


# Global job store instance
_job_store: Optional[JobStore] = None


async def get_job_store() -> JobStore:
    """Get the global job store instance."""
    global _job_store
    if _job_store is None:
        _job_store = JobStore()
        await _job_store.initialize()
    return _job_store


async def close_job_store() -> None:
    """Close the global job store."""
    global _job_store
    if _job_store:
        await _job_store.close()
        _job_store = None
