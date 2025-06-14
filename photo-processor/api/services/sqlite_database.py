"""
SQLite Database Service for Photo Processor

Provides scalable database operations replacing JSON file storage.
Based on comprehensive data standards documented in NIMA_DATA_STANDARDS.md
"""

import sqlite3
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import uuid
import asyncio
import aiosqlite
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Disable verbose aiosqlite DEBUG logging
aiosqlite_logger = logging.getLogger('aiosqlite')
aiosqlite_logger.setLevel(logging.WARNING)

class SQLiteDatabase:
    """
    Async SQLite database service for photo processing system
    
    This replaces the JSON-based PhotoDatabase with a scalable SQLite solution
    following our documented data standards.
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path / "photo_processor.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database synchronously
        self._init_database()
        
        # Run migrations
        self._run_migrations()
        
        logger.info(f"SQLiteDatabase initialized at {self.db_path}")
    
    def _init_database(self):
        """Initialize database tables synchronously"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
            -- Photos table - main photo records
            CREATE TABLE IF NOT EXISTS photos (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- File paths and metadata
                original_path TEXT NULL,
                processed_path TEXT NULL,
                thumbnail_path TEXT NULL,
                web_path TEXT NULL,
                file_hash TEXT NULL,
                file_size INTEGER DEFAULT 0,
                
                -- Processing metadata
                recipe_id TEXT NULL,
                session_id TEXT NULL,
                processing_time REAL DEFAULT 0,
                error_message TEXT NULL,
                
                -- Additional metadata (JSON)
                metadata TEXT DEFAULT '{}',
                
                UNIQUE(file_hash)
            );
            
            -- AI Analysis table - NIMA and other AI results
            CREATE TABLE IF NOT EXISTS ai_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id TEXT NOT NULL,
                analysis_type TEXT NOT NULL DEFAULT 'nima',
                status TEXT NOT NULL DEFAULT 'pending',
                
                -- Timestamps
                queued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP NULL,
                completed_at TIMESTAMP NULL,
                
                -- NIMA specific fields
                aesthetic_score REAL NULL,
                technical_score REAL NULL,
                quality_level TEXT NULL,
                confidence REAL NULL,
                
                -- Full results (JSON)
                onealign_results TEXT NULL,
                model_info TEXT NULL,
                
                -- Error handling
                error TEXT NULL,
                task_id TEXT NULL,
                
                FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE,
                UNIQUE(photo_id, analysis_type)
            );
            
            -- Processing History table - track all processing events
            CREATE TABLE IF NOT EXISTS processing_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Details
                recipe_id TEXT NULL,
                processing_time REAL NULL,
                details TEXT NULL,
                error_message TEXT NULL,
                
                FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE
            );
            
            -- Upload Sessions table - track batch uploads
            CREATE TABLE IF NOT EXISTS upload_sessions (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP NULL,
                status TEXT DEFAULT 'uploading',
                
                expected_files INTEGER DEFAULT 0,
                uploaded_files INTEGER DEFAULT 0,
                failed_files INTEGER DEFAULT 0,
                total_size INTEGER DEFAULT 0,
                uploaded_size INTEGER DEFAULT 0,
                
                -- Configuration
                auto_process BOOLEAN DEFAULT TRUE,
                recipe_id TEXT NULL,
                
                -- Additional metadata
                metadata TEXT DEFAULT '{}'
            );
            
            -- Upload Session Files table - track files in sessions
            CREATE TABLE IF NOT EXISTS upload_session_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                photo_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_size INTEGER DEFAULT 0,
                status TEXT DEFAULT 'uploaded',
                error_message TEXT NULL,
                
                FOREIGN KEY (session_id) REFERENCES upload_sessions (id) ON DELETE CASCADE,
                FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE
            );
            
            -- Recipes table - store recipe definitions
            CREATE TABLE IF NOT EXISTS recipes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                
                -- Recipe definition (JSON)
                steps TEXT NOT NULL DEFAULT '[]',
                settings TEXT DEFAULT '{}',
                
                -- Usage statistics
                usage_count INTEGER DEFAULT 0,
                last_used TIMESTAMP NULL
            );
            
            -- Create indexes for performance
            CREATE INDEX IF NOT EXISTS idx_photos_status ON photos (status);
            CREATE INDEX IF NOT EXISTS idx_photos_created_at ON photos (created_at);
            CREATE INDEX IF NOT EXISTS idx_photos_file_hash ON photos (file_hash);
            CREATE INDEX IF NOT EXISTS idx_photos_session_id ON photos (session_id);
            
            CREATE INDEX IF NOT EXISTS idx_ai_analysis_photo_id ON ai_analysis (photo_id);
            CREATE INDEX IF NOT EXISTS idx_ai_analysis_status ON ai_analysis (status);
            CREATE INDEX IF NOT EXISTS idx_ai_analysis_type ON ai_analysis (analysis_type);
            
            CREATE INDEX IF NOT EXISTS idx_processing_history_photo_id ON processing_history (photo_id);
            CREATE INDEX IF NOT EXISTS idx_processing_history_timestamp ON processing_history (timestamp);
            
            CREATE INDEX IF NOT EXISTS idx_upload_sessions_status ON upload_sessions (status);
            CREATE INDEX IF NOT EXISTS idx_upload_session_files_session_id ON upload_session_files (session_id);
            
            -- Settings table - store application configuration
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Insert default settings if they don't exist
            INSERT OR IGNORE INTO settings (key, value, category, description) VALUES
                ('immich_server_url', '', 'immich', 'Immich server URL (e.g., http://192.168.1.100:2283)'),
                ('immich_api_key', '', 'immich', 'Immich API key for authentication'),
                ('immich_enabled', 'false', 'immich', 'Enable Immich integration'),
                ('processing_threads', '4', 'processing', 'Number of concurrent processing threads'),
                ('auto_process_uploads', 'true', 'processing', 'Automatically process uploaded photos'),
                ('default_recipe_id', '', 'processing', 'Default recipe to apply to new photos'),
                ('max_file_size_mb', '500', 'upload', 'Maximum file size in megabytes'),
                ('allowed_extensions', '.jpg,.jpeg,.png,.tiff,.tif,.bmp,.nef,.cr2,.cr3,.arw,.dng,.orf,.rw2,.raf', 'upload', 'Comma-separated list of allowed file extensions');
            """)
            
            logger.info("Database tables initialized successfully")
    
    def _run_migrations(self):
        """Run database migrations"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if we need to migrate from nima_results to onealign_results
            cursor.execute("PRAGMA table_info(ai_analysis)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            if 'nima_results' in column_names and 'onealign_results' not in column_names:
                logger.info("Migrating nima_results to onealign_results column...")
                
                # Drop temporary table if it exists from a failed migration
                cursor.execute("DROP TABLE IF EXISTS ai_analysis_new")
                
                # SQLite doesn't support ALTER COLUMN, so we need to recreate the table
                cursor.execute("""
                    CREATE TABLE ai_analysis_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        photo_id TEXT NOT NULL,
                        analysis_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        
                        -- Timestamps
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        queued_at TIMESTAMP NULL,
                        started_at TIMESTAMP NULL,
                        completed_at TIMESTAMP NULL,
                        
                        -- Scores (mapped from NIMA results)
                        aesthetic_score REAL NULL,
                        technical_score REAL NULL,
                        quality_level TEXT NULL,
                        confidence REAL NULL,
                        
                        -- Full results (JSON)
                        onealign_results TEXT NULL,
                        model_info TEXT NULL,
                        
                        -- Error handling
                        error TEXT NULL,
                        task_id TEXT NULL,
                        
                        FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE
                    )
                """)
                
                # Copy data from old table - only copy columns that exist
                # Check which columns actually exist in the old table
                existing_columns = set(column_names)
                
                # Build column list dynamically based on what exists
                copy_columns = []
                select_columns = []
                
                column_mapping = {
                    'id': 'id',
                    'photo_id': 'photo_id',
                    'analysis_type': 'analysis_type',
                    'status': 'status',
                    'created_at': 'created_at',
                    'queued_at': 'queued_at',
                    'started_at': 'started_at',
                    'completed_at': 'completed_at',
                    'aesthetic_score': 'aesthetic_score',
                    'technical_score': 'technical_score',
                    'quality_level': 'quality_level',
                    'confidence': 'confidence',
                    'onealign_results': 'nima_results',  # Map old to new
                    'model_info': 'model_info',
                    'error': 'error',
                    'task_id': 'task_id'
                }
                
                for new_col, old_col in column_mapping.items():
                    if old_col in existing_columns:
                        copy_columns.append(new_col)
                        select_columns.append(old_col)
                    elif new_col == 'created_at':
                        # If created_at doesn't exist, use CURRENT_TIMESTAMP
                        copy_columns.append(new_col)
                        select_columns.append('CURRENT_TIMESTAMP')
                
                columns_str = ', '.join(copy_columns)
                select_str = ', '.join(select_columns)
                
                cursor.execute(f"""
                    INSERT INTO ai_analysis_new ({columns_str})
                    SELECT {select_str}
                    FROM ai_analysis
                """)
                
                # Drop old table and rename new one
                cursor.execute("DROP TABLE ai_analysis")
                cursor.execute("ALTER TABLE ai_analysis_new RENAME TO ai_analysis")
                
                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_analysis_photo_id ON ai_analysis (photo_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_analysis_status ON ai_analysis (status)")
                
                conn.commit()
                logger.info("Migration completed successfully")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get async database connection"""
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            yield conn
    
    # ========================================================================
    # Photo Operations
    # ========================================================================
    
    async def add_photo(self, photo_data: Dict[str, Any]) -> str:
        """Add photo to database"""
        photo_id = photo_data.get('id', str(uuid.uuid4()))
        
        logger.info(f"Adding photo {photo_id} to database: {photo_data.get('filename')}")
        
        async with self.get_connection() as conn:
            # Insert photo record
            await conn.execute("""
                INSERT OR REPLACE INTO photos (
                    id, filename, status, created_at, processed_at,
                    original_path, processed_path, thumbnail_path, web_path,
                    file_hash, file_size, recipe_id, session_id,
                    processing_time, error_message, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                photo_id,
                photo_data.get('filename'),
                photo_data.get('status', 'pending'),
                photo_data.get('created_at', datetime.now()),
                photo_data.get('processed_at'),
                photo_data.get('original_path'),
                photo_data.get('processed_path'),
                photo_data.get('thumbnail_path'),
                photo_data.get('web_path'),
                photo_data.get('file_hash'),
                photo_data.get('file_size', 0),
                photo_data.get('recipe_id'),
                photo_data.get('session_id'),
                photo_data.get('processing_time', 0),
                photo_data.get('error_message'),
                json.dumps(photo_data.get('metadata', {}))
            ))
            
            # Add AI analysis record if provided
            ai_analysis = photo_data.get('ai_analysis')
            if ai_analysis:
                await self.update_photo_ai_analysis(photo_id, ai_analysis, conn)
            
            await conn.commit()
        
        logger.info(f"Successfully added photo {photo_id}")
        return photo_id
    
    async def get_photo(self, photo_id: str) -> Optional[Dict[str, Any]]:
        """Get photo by ID with AI analysis"""
        async with self.get_connection() as conn:
            # Get photo data
            cursor = await conn.execute("""
                SELECT * FROM photos WHERE id = ?
            """, (photo_id,))
            
            photo_row = await cursor.fetchone()
            if not photo_row:
                return None
            
            photo_data = dict(photo_row)
            
            # Parse JSON fields
            photo_data['metadata'] = json.loads(photo_data.get('metadata', '{}'))
            
            # Get AI analysis
            cursor = await conn.execute("""
                SELECT * FROM ai_analysis WHERE photo_id = ? ORDER BY completed_at DESC
            """, (photo_id,))
            
            ai_rows = await cursor.fetchall()
            if ai_rows:
                # Use most recent analysis
                ai_row = ai_rows[0]
                ai_data = dict(ai_row)
                
                # Parse JSON fields
                ai_data['onealign_results'] = json.loads(ai_data.get('onealign_results') or '{}')
                ai_data['model_info'] = json.loads(ai_data.get('model_info') or '{}')
                
                photo_data['ai_analysis'] = ai_data
            else:
                photo_data['ai_analysis'] = {'status': 'not_available'}
            
            return photo_data
    
    async def update_photo(self, photo_id: str, updates: Dict[str, Any]):
        """Update photo data"""
        if not updates:
            return
        
        # Separate regular fields from AI analysis
        ai_analysis = updates.pop('ai_analysis', None)
        
        if updates:
            # Build dynamic UPDATE query
            set_clauses = []
            values = []
            
            for key, value in updates.items():
                if key == 'metadata':
                    value = json.dumps(value)
                set_clauses.append(f"{key} = ?")
                values.append(value)
            
            # Add updated_at
            set_clauses.append("updated_at = ?")
            values.append(datetime.now())
            values.append(photo_id)
            
            async with self.get_connection() as conn:
                await conn.execute(f"""
                    UPDATE photos SET {', '.join(set_clauses)} WHERE id = ?
                """, values)
                await conn.commit()
        
        # Update AI analysis if provided
        if ai_analysis:
            await self.update_photo_ai_analysis(photo_id, ai_analysis)
    
    async def update_photo_status(self, photo_id: str, status: str, processing_message: str = None):
        """Update photo status and add to processing history"""
        updates = {'status': status}
        if status == 'completed':
            updates['processed_at'] = datetime.now()
        
        await self.update_photo(photo_id, updates)
        
        # Add to processing history
        async with self.get_connection() as conn:
            await conn.execute("""
                INSERT INTO processing_history (photo_id, action, status, details)
                VALUES (?, ?, ?, ?)
            """, (photo_id, 'status_update', status, processing_message))
            await conn.commit()
    
    async def update_photo_ai_analysis(self, photo_id: str, ai_analysis: Dict[str, Any], conn=None):
        """Update AI analysis for a photo"""
        if conn is None:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT OR REPLACE INTO ai_analysis (
                        photo_id, analysis_type, status, queued_at, started_at, completed_at,
                        aesthetic_score, technical_score, quality_level, confidence,
                        onealign_results, model_info, error, task_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    photo_id,
                    ai_analysis.get('analysis_type', 'nima'),
                    ai_analysis.get('status', 'pending'),
                    ai_analysis.get('queued_at'),
                    ai_analysis.get('started_at'),
                    ai_analysis.get('completed_at'),
                    ai_analysis.get('aesthetic_score'),
                    ai_analysis.get('technical_score'),
                    ai_analysis.get('quality_level'),
                    ai_analysis.get('confidence'),
                    json.dumps(ai_analysis.get('onealign_results', {})),
                    json.dumps(ai_analysis.get('model_info', {})),
                    ai_analysis.get('error'),
                    ai_analysis.get('task_id')
                ))
                await conn.commit()
        else:
            await conn.execute("""
                INSERT OR REPLACE INTO ai_analysis (
                    photo_id, analysis_type, status, queued_at, started_at, completed_at,
                    aesthetic_score, technical_score, quality_level, confidence,
                    onealign_results, model_info, error, task_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                photo_id,
                ai_analysis.get('analysis_type', 'nima'),
                ai_analysis.get('status', 'pending'),
                ai_analysis.get('queued_at'),
                ai_analysis.get('started_at'),
                ai_analysis.get('completed_at'),
                ai_analysis.get('aesthetic_score'),
                ai_analysis.get('technical_score'),
                ai_analysis.get('quality_level'),
                ai_analysis.get('confidence'),
                json.dumps(ai_analysis.get('onealign_results', {})),
                json.dumps(ai_analysis.get('model_info', {})),
                ai_analysis.get('error'),
                ai_analysis.get('task_id')
            ))
    
    async def update_photo_metadata(self, photo_id: str, metadata: Dict[str, Any]):
        """Update photo metadata"""
        # Get existing metadata
        photo = await self.get_photo(photo_id)
        if photo:
            existing_metadata = photo.get('metadata', {})
            existing_metadata.update(metadata)
            await self.update_photo(photo_id, {'metadata': existing_metadata})
    
    async def delete_photo(self, photo_id: str) -> bool:
        """Delete photo from database"""
        async with self.get_connection() as conn:
            cursor = await conn.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
            await conn.commit()
            return cursor.rowcount > 0
    
    async def list_photos(self, filters: Dict[str, Any] = None, 
                         page: int = 1, page_size: int = 20,
                         sort_by: str = "created_at", order: str = "desc") -> Tuple[List[Dict[str, Any]], int]:
        """List photos with pagination and filters"""
        filters = filters or {}
        
        # Build WHERE clause
        where_clauses = []
        values = []
        join_clauses = []
        
        if filters.get('status'):
            where_clauses.append("p.status = ?")
            values.append(filters['status'])
        
        if filters.get('session_id'):
            where_clauses.append("p.session_id = ?")
            values.append(filters['session_id'])
        
        if filters.get('recipe_id'):
            where_clauses.append("p.recipe_id = ?")
            values.append(filters['recipe_id'])
        
        if filters.get('search'):
            where_clauses.append("p.filename LIKE ?")
            values.append(f"%{filters['search']}%")
        
        # Handle score filters - need to join with ai_analysis table
        score_filters = False
        need_ai_join = False
        
        if any(filters.get(f) is not None for f in ['min_aesthetic_score', 'max_aesthetic_score', 'min_technical_score', 'max_technical_score']):
            score_filters = True
            need_ai_join = True
            
            if filters.get('min_aesthetic_score') is not None:
                where_clauses.append("a.aesthetic_score >= ?")
                values.append(filters['min_aesthetic_score'])
            
            if filters.get('max_aesthetic_score') is not None:
                where_clauses.append("a.aesthetic_score <= ?")
                values.append(filters['max_aesthetic_score'])
            
            if filters.get('min_technical_score') is not None:
                where_clauses.append("a.technical_score >= ?")
                values.append(filters['min_technical_score'])
            
            if filters.get('max_technical_score') is not None:
                where_clauses.append("a.technical_score <= ?")
                values.append(filters['max_technical_score'])
        
        # Handle sorting
        if sort_by == 'aesthetic':
            need_ai_join = True
            sort_by = "a.aesthetic_score"
        elif sort_by == 'technical':
            need_ai_join = True
            sort_by = "a.technical_score"
        elif sort_by == 'combined':
            need_ai_join = True
            sort_by = "(COALESCE(a.aesthetic_score, 0) + COALESCE(a.technical_score, 0)) / 2"
        elif sort_by == 'created_at':
            sort_by = "p.created_at"
        else:
            sort_by = f"p.{sort_by}"
        
        # Add the join if needed
        if need_ai_join and "LEFT JOIN ai_analysis a" not in " ".join(join_clauses):
            join_clauses.append("LEFT JOIN ai_analysis a ON p.id = a.photo_id AND a.status = 'completed'")
        
        join_sql = " ".join(join_clauses)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        order_sql = f"ORDER BY {sort_by} {order.upper()}"
        
        async with self.get_connection() as conn:
            # Get total count - need to use join if we have score filters
            if score_filters:
                cursor = await conn.execute(f"""
                    SELECT COUNT(DISTINCT p.id) FROM photos p {join_sql} {where_sql}
                """, values)
            else:
                cursor = await conn.execute(f"""
                    SELECT COUNT(*) FROM photos p {where_sql}
                """, values)
            total = (await cursor.fetchone())[0]
            
            # Get paginated results
            offset = (page - 1) * page_size
            
            if need_ai_join:
                # Include AI analysis fields in query when we have the join
                cursor = await conn.execute(f"""
                    SELECT p.*, a.aesthetic_score, a.technical_score, a.quality_level, a.confidence, a.status as ai_status
                    FROM photos p {join_sql} {where_sql} {order_sql} LIMIT ? OFFSET ?
                """, values + [page_size, offset])
            else:
                # Simple query without AI fields
                cursor = await conn.execute(f"""
                    SELECT p.* FROM photos p {join_sql} {where_sql} {order_sql} LIMIT ? OFFSET ?
                """, values + [page_size, offset])
            
            rows = await cursor.fetchall()
            photos = []
            
            for row in rows:
                photo_data = dict(row)
                photo_data['metadata'] = json.loads(photo_data.get('metadata', '{}'))
                
                # If we already have AI data from the join, use it
                if need_ai_join and 'ai_status' in photo_data and photo_data['ai_status']:
                    ai_data = {
                        'status': photo_data.pop('ai_status'),
                        'aesthetic_score': photo_data.pop('aesthetic_score', None),
                        'technical_score': photo_data.pop('technical_score', None),
                        'quality_level': photo_data.pop('quality_level', None),
                        'confidence': photo_data.pop('confidence', None)
                    }
                    photo_data['ai_analysis'] = ai_data
                else:
                    # Otherwise fetch AI analysis separately
                    ai_cursor = await conn.execute("""
                        SELECT * FROM ai_analysis WHERE photo_id = ? ORDER BY completed_at DESC LIMIT 1
                    """, (photo_data['id'],))
                    
                    ai_row = await ai_cursor.fetchone()
                    if ai_row:
                        ai_data = dict(ai_row)
                        ai_data['onealign_results'] = json.loads(ai_data.get('onealign_results') or '{}')
                        ai_data['model_info'] = json.loads(ai_data.get('model_info') or '{}')
                        photo_data['ai_analysis'] = ai_data
                    else:
                        photo_data['ai_analysis'] = {'status': 'not_available'}
                
                photos.append(photo_data)
            
            return photos, total
    
    async def get_photos_by_ai_status(self, status: str) -> List[Dict[str, Any]]:
        """Get photos by AI analysis status"""
        async with self.get_connection() as conn:
            cursor = await conn.execute("""
                SELECT p.* FROM photos p
                JOIN ai_analysis a ON p.id = a.photo_id
                WHERE a.status = ?
                ORDER BY p.created_at DESC
            """, (status,))
            
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    
    # ========================================================================
    # Upload Session Operations
    # ========================================================================
    
    async def create_upload_session(self, session_data: Dict[str, Any]) -> str:
        """Create new upload session"""
        session_id = session_data.get('id', str(uuid.uuid4()))
        
        async with self.get_connection() as conn:
            await conn.execute("""
                INSERT INTO upload_sessions (
                    id, expected_files, auto_process, recipe_id, metadata
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                session_data.get('expected_files', 0),
                session_data.get('auto_process', True),
                session_data.get('recipe_id'),
                json.dumps(session_data.get('metadata', {}))
            ))
            await conn.commit()
        
        return session_id
    
    async def get_upload_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get upload session data"""
        async with self.get_connection() as conn:
            cursor = await conn.execute("""
                SELECT * FROM upload_sessions WHERE id = ?
            """, (session_id,))
            
            row = await cursor.fetchone()
            if not row:
                return None
            
            session_data = dict(row)
            session_data['metadata'] = json.loads(session_data.get('metadata', '{}'))
            
            # Get session files
            cursor = await conn.execute("""
                SELECT * FROM upload_session_files WHERE session_id = ?
                ORDER BY uploaded_at
            """, (session_id,))
            
            files = [dict(file_row) for file_row in await cursor.fetchall()]
            session_data['files'] = files
            
            return session_data
    
    async def update_upload_session(self, session_id: str, updates: Dict[str, Any]):
        """Update upload session"""
        if not updates:
            return
        
        set_clauses = []
        values = []
        
        for key, value in updates.items():
            if key == 'metadata':
                value = json.dumps(value)
            set_clauses.append(f"{key} = ?")
            values.append(value)
        
        values.append(session_id)
        
        async with self.get_connection() as conn:
            await conn.execute(f"""
                UPDATE upload_sessions SET {', '.join(set_clauses)} WHERE id = ?
            """, values)
            await conn.commit()
    
    async def add_session_file(self, session_id: str, file_data: Dict[str, Any]):
        """Add file to upload session"""
        async with self.get_connection() as conn:
            await conn.execute("""
                INSERT INTO upload_session_files (
                    session_id, photo_id, filename, file_size, status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                file_data.get('photo_id'),
                file_data.get('filename'),
                file_data.get('file_size', 0),
                file_data.get('status', 'uploaded'),
                file_data.get('error_message')
            ))
            await conn.commit()
    
    # ========================================================================
    # Recipe Operations
    # ========================================================================
    
    async def add_recipe(self, recipe_data: Dict[str, Any]) -> str:
        """Add recipe to database"""
        recipe_id = recipe_data.get('id', str(uuid.uuid4()))
        
        async with self.get_connection() as conn:
            await conn.execute("""
                INSERT OR REPLACE INTO recipes (
                    id, name, description, steps, settings, is_active
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                recipe_id,
                recipe_data.get('name'),
                recipe_data.get('description'),
                json.dumps(recipe_data.get('steps', [])),
                json.dumps(recipe_data.get('settings', {})),
                recipe_data.get('is_active', True)
            ))
            await conn.commit()
        
        return recipe_id
    
    async def get_recipe(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        """Get recipe by ID"""
        async with self.get_connection() as conn:
            cursor = await conn.execute("""
                SELECT * FROM recipes WHERE id = ?
            """, (recipe_id,))
            
            row = await cursor.fetchone()
            if not row:
                return None
            
            recipe_data = dict(row)
            recipe_data['steps'] = json.loads(recipe_data.get('steps', '[]'))
            recipe_data['settings'] = json.loads(recipe_data.get('settings', '{}'))
            
            return recipe_data
    
    async def list_recipes(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all recipes"""
        async with self.get_connection() as conn:
            where_sql = "WHERE is_active = TRUE" if active_only else ""
            cursor = await conn.execute(f"""
                SELECT * FROM recipes {where_sql} ORDER BY name
            """)
            
            rows = await cursor.fetchall()
            recipes = []
            
            for row in rows:
                recipe_data = dict(row)
                recipe_data['steps'] = json.loads(recipe_data.get('steps', '[]'))
                recipe_data['settings'] = json.loads(recipe_data.get('settings', '{}'))
                recipes.append(recipe_data)
            
            return recipes
    
    # ========================================================================
    # Statistics and Cleanup
    # ========================================================================
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        async with self.get_connection() as conn:
            # Photo counts by status
            cursor = await conn.execute("""
                SELECT status, COUNT(*) FROM photos GROUP BY status
            """)
            status_counts = dict(await cursor.fetchall())
            
            # AI analysis counts
            cursor = await conn.execute("""
                SELECT status, COUNT(*) FROM ai_analysis GROUP BY status
            """)
            ai_counts = dict(await cursor.fetchall())
            
            # Total file size
            cursor = await conn.execute("""
                SELECT SUM(file_size) FROM photos WHERE file_size > 0
            """)
            total_size = (await cursor.fetchone())[0] or 0
            
            # Recent activity
            cursor = await conn.execute("""
                SELECT COUNT(*) FROM photos 
                WHERE created_at > datetime('now', '-24 hours')
            """)
            photos_last_24h = (await cursor.fetchone())[0]
            
            return {
                'photo_counts': status_counts,
                'ai_analysis_counts': ai_counts,
                'total_file_size': total_size,
                'photos_last_24h': photos_last_24h,
                'last_updated': datetime.now().isoformat()
            }
    
    async def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up old data"""
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_old)
        
        async with self.get_connection() as conn:
            # Clean old processing history
            cursor = await conn.execute("""
                DELETE FROM processing_history 
                WHERE timestamp < ? AND action != 'upload'
            """, (cutoff_date,))
            history_deleted = cursor.rowcount
            
            # Clean old completed upload sessions
            cursor = await conn.execute("""
                DELETE FROM upload_sessions 
                WHERE status = 'completed' AND completed_at < ?
            """, (cutoff_date,))
            sessions_deleted = cursor.rowcount
            
            await conn.commit()
            
            return {
                'processing_history_deleted': history_deleted,
                'upload_sessions_deleted': sessions_deleted
            }
    
    # Settings management methods
    async def get_setting(self, key: str) -> Optional[str]:
        """Get a setting value by key"""
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT value FROM settings WHERE key = ?",
                (key,)
            )
            row = await cursor.fetchone()
            return row[0] if row else None
    
    async def get_settings_by_category(self, category: str) -> Dict[str, Any]:
        """Get all settings in a category"""
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT key, value, description FROM settings WHERE category = ?",
                (category,)
            )
            rows = await cursor.fetchall()
            
            return {
                row[0]: {
                    'value': row[1],
                    'description': row[2]
                }
                for row in rows
            }
    
    async def get_all_settings(self) -> Dict[str, Dict[str, Any]]:
        """Get all settings grouped by category"""
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT key, value, category, description FROM settings ORDER BY category, key"
            )
            rows = await cursor.fetchall()
            
            settings = {}
            for row in rows:
                key, value, category, description = row
                if category not in settings:
                    settings[category] = {}
                settings[category][key] = {
                    'value': value,
                    'description': description
                }
            
            return settings
    
    async def update_setting(self, key: str, value: str) -> bool:
        """Update a setting value"""
        async with self.get_connection() as conn:
            result = await conn.execute(
                """UPDATE settings 
                   SET value = ?, updated_at = CURRENT_TIMESTAMP 
                   WHERE key = ?""",
                (value, key)
            )
            await conn.commit()
            return result.rowcount > 0
    
    async def update_settings_batch(self, settings: Dict[str, str]) -> int:
        """Update multiple settings at once"""
        async with self.get_connection() as conn:
            updated = 0
            for key, value in settings.items():
                result = await conn.execute(
                    """UPDATE settings 
                       SET value = ?, updated_at = CURRENT_TIMESTAMP 
                       WHERE key = ?""",
                    (value, key)
                )
                updated += result.rowcount
            await conn.commit()
            return updated