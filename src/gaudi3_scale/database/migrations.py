"""Database migration management for Gaudi 3 Scale."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy import MetaData, Table, text
from sqlalchemy.engine import Engine

from .connection import DatabaseConnection

logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages database migrations for Gaudi 3 Scale."""
    
    def __init__(self, database_connection: Optional[DatabaseConnection] = None):
        """Initialize migration manager.
        
        Args:
            database_connection: Optional database connection override
        """
        self.db = database_connection or DatabaseConnection()
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Setup Alembic configuration
        self.alembic_cfg = self._setup_alembic_config()
    
    def _setup_alembic_config(self) -> Config:
        """Setup Alembic configuration."""
        # Get the directory containing this file
        current_dir = Path(__file__).parent
        migrations_dir = current_dir / "migrations"
        
        # Create migrations directory if it doesn't exist
        migrations_dir.mkdir(exist_ok=True)
        
        # Create Alembic config
        alembic_cfg = Config()
        alembic_cfg.set_main_option("script_location", str(migrations_dir))
        alembic_cfg.set_main_option("sqlalchemy.url", self.db.database_url)
        
        return alembic_cfg
    
    def init_migrations(self):
        """Initialize Alembic migrations directory."""
        try:
            command.init(self.alembic_cfg, str(self.alembic_cfg.get_main_option("script_location")))
            self.logger.info("Initialized Alembic migrations directory")
        except Exception as e:
            if "already exists" not in str(e):
                self.logger.error(f"Failed to initialize migrations: {e}")
                raise
            else:
                self.logger.info("Migrations directory already exists")
    
    def create_migration(self, message: str, auto_generate: bool = True) -> Optional[str]:
        """Create a new migration.
        
        Args:
            message: Migration message
            auto_generate: Whether to auto-generate migration from model changes
            
        Returns:
            Migration revision ID
        """
        try:
            if auto_generate:
                # Import models to ensure they're loaded
                from .models import Base
                
                # Generate migration
                revision = command.revision(
                    self.alembic_cfg,
                    message=message,
                    autogenerate=True
                )
            else:
                revision = command.revision(
                    self.alembic_cfg,
                    message=message
                )
            
            self.logger.info(f"Created migration: {message}")
            return revision
        except Exception as e:
            self.logger.error(f"Failed to create migration: {e}")
            raise
    
    def upgrade(self, revision: str = "head"):
        """Upgrade database to specified revision.
        
        Args:
            revision: Target revision (default: "head")
        """
        try:
            command.upgrade(self.alembic_cfg, revision)
            self.logger.info(f"Upgraded database to revision: {revision}")
        except Exception as e:
            self.logger.error(f"Failed to upgrade database: {e}")
            raise
    
    def downgrade(self, revision: str):
        """Downgrade database to specified revision.
        
        Args:
            revision: Target revision
        """
        try:
            command.downgrade(self.alembic_cfg, revision)
            self.logger.info(f"Downgraded database to revision: {revision}")
        except Exception as e:
            self.logger.error(f"Failed to downgrade database: {e}")
            raise
    
    def get_current_revision(self) -> Optional[str]:
        """Get current database revision.
        
        Returns:
            Current revision ID
        """
        try:
            with self.db.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                return context.get_current_revision()
        except Exception as e:
            self.logger.error(f"Failed to get current revision: {e}")
            return None
    
    def get_pending_migrations(self) -> List[str]:
        """Get list of pending migrations.
        
        Returns:
            List of pending migration revision IDs
        """
        try:
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)
            current_rev = self.get_current_revision()
            
            if current_rev is None:
                # No migrations applied yet
                return [rev.revision for rev in script_dir.walk_revisions()]
            
            pending = []
            for revision in script_dir.walk_revisions("head", current_rev):
                if revision.revision != current_rev:
                    pending.append(revision.revision)
            
            return pending
        except Exception as e:
            self.logger.error(f"Failed to get pending migrations: {e}")
            return []
    
    def get_migration_history(self) -> List[Dict[str, any]]:
        """Get migration history.
        
        Returns:
            List of migration history entries
        """
        try:
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)
            current_rev = self.get_current_revision()
            
            history = []
            for revision in script_dir.walk_revisions():
                is_applied = False
                if current_rev:
                    # Check if this revision is in the current branch
                    try:
                        script_dir.get_revision(current_rev)
                        is_applied = True
                    except:
                        pass
                
                history.append({
                    "revision": revision.revision,
                    "message": revision.doc,
                    "is_applied": is_applied,
                    "branch_labels": revision.branch_labels,
                    "depends_on": revision.depends_on,
                    "down_revision": revision.down_revision
                })
            
            return history
        except Exception as e:
            self.logger.error(f"Failed to get migration history: {e}")
            return []
    
    def validate_database_schema(self) -> Dict[str, any]:
        """Validate database schema against models.
        
        Returns:
            Validation results
        """
        try:
            from .models import Base
            
            # Get current database metadata
            db_metadata = MetaData()
            db_metadata.reflect(bind=self.db.engine)
            
            # Get model metadata
            model_metadata = Base.metadata
            
            # Compare tables
            missing_tables = []
            extra_tables = []
            table_diffs = []
            
            model_tables = set(model_metadata.tables.keys())
            db_tables = set(db_metadata.tables.keys())
            
            missing_tables = list(model_tables - db_tables)
            extra_tables = list(db_tables - model_tables)
            
            # Check columns for existing tables
            for table_name in model_tables.intersection(db_tables):
                model_table = model_metadata.tables[table_name]
                db_table = db_metadata.tables[table_name]
                
                model_columns = set(model_table.columns.keys())
                db_columns = set(db_table.columns.keys())
                
                missing_columns = list(model_columns - db_columns)
                extra_columns = list(db_columns - model_columns)
                
                if missing_columns or extra_columns:
                    table_diffs.append({
                        "table": table_name,
                        "missing_columns": missing_columns,
                        "extra_columns": extra_columns
                    })
            
            return {
                "is_valid": not (missing_tables or extra_tables or table_diffs),
                "missing_tables": missing_tables,
                "extra_tables": extra_tables,
                "table_differences": table_diffs,
                "current_revision": self.get_current_revision(),
                "pending_migrations": self.get_pending_migrations()
            }
        except Exception as e:
            self.logger.error(f"Failed to validate database schema: {e}")
            return {
                "is_valid": False,
                "error": str(e)
            }
    
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """Create database backup.
        
        Args:
            backup_path: Optional backup file path
            
        Returns:
            Path to backup file
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"gaudi3_scale_backup_{timestamp}.sql"
        
        try:
            # Use pg_dump for PostgreSQL backup
            import subprocess
            
            # Extract connection details
            db_url = self.db.database_url
            # This is a simplified version - in production, you'd want more robust URL parsing
            
            cmd = [
                "pg_dump",
                db_url,
                "-f", backup_path,
                "--no-password",
                "--verbose"
            ]
            
            subprocess.run(cmd, check=True)
            self.logger.info(f"Database backup created: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Failed to create database backup: {e}")
            raise
    
    def restore_database(self, backup_path: str):
        """Restore database from backup.
        
        Args:
            backup_path: Path to backup file
        """
        try:
            # Use psql for PostgreSQL restore
            import subprocess
            
            cmd = [
                "psql",
                self.db.database_url,
                "-f", backup_path,
                "--no-password",
                "--verbose"
            ]
            
            subprocess.run(cmd, check=True)
            self.logger.info(f"Database restored from: {backup_path}")
        except Exception as e:
            self.logger.error(f"Failed to restore database: {e}")
            raise
    
    def run_data_migration(self, migration_func, description: str):
        """Run a data migration function.
        
        Args:
            migration_func: Function to run migration
            description: Description of the migration
        """
        try:
            self.logger.info(f"Starting data migration: {description}")
            
            with self.db.engine.begin() as conn:
                migration_func(conn)
            
            self.logger.info(f"Completed data migration: {description}")
        except Exception as e:
            self.logger.error(f"Failed to run data migration '{description}': {e}")
            raise