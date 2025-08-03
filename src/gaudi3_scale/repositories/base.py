"""Base repository with common CRUD operations."""

import logging
from typing import Dict, Generic, List, Optional, Type, TypeVar, Union
from uuid import UUID

from sqlalchemy import and_, desc, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Query, Session

from ..database.connection import DatabaseConnection
from ..database.models import Base

ModelType = TypeVar("ModelType", bound=Base)
logger = logging.getLogger(__name__)


class BaseRepository(Generic[ModelType]):
    """Base repository with common CRUD operations."""
    
    def __init__(self, model_class: Type[ModelType], 
                 db_connection: Optional[DatabaseConnection] = None):
        """Initialize repository.
        
        Args:
            model_class: SQLAlchemy model class
            db_connection: Optional database connection override
        """
        self.model_class = model_class
        self.db = db_connection or DatabaseConnection()
        self.logger = logger.getChild(self.__class__.__name__)
    
    def create(self, session: Session, **kwargs) -> ModelType:
        """Create a new model instance.
        
        Args:
            session: Database session
            **kwargs: Model attributes
            
        Returns:
            Created model instance
        """
        try:
            instance = self.model_class(**kwargs)
            session.add(instance)
            session.commit()
            session.refresh(instance)
            
            self.logger.debug(f"Created {self.model_class.__name__}: {instance.id}")
            return instance
        except IntegrityError as e:
            session.rollback()
            self.logger.error(f"Integrity error creating {self.model_class.__name__}: {e}")
            raise
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error creating {self.model_class.__name__}: {e}")
            raise
    
    def get_by_id(self, session: Session, id: Union[UUID, str]) -> Optional[ModelType]:
        """Get model instance by ID.
        
        Args:
            session: Database session
            id: Model ID
            
        Returns:
            Model instance or None if not found
        """
        try:
            return session.query(self.model_class).filter(self.model_class.id == id).first()
        except Exception as e:
            self.logger.error(f"Error getting {self.model_class.__name__} by ID {id}: {e}")
            raise
    
    def get_all(self, session: Session, limit: Optional[int] = None, 
                offset: int = 0) -> List[ModelType]:
        """Get all model instances.
        
        Args:
            session: Database session
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of model instances
        """
        try:
            query = session.query(self.model_class)
            
            if offset > 0:
                query = query.offset(offset)
            
            if limit is not None:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            self.logger.error(f"Error getting all {self.model_class.__name__}: {e}")
            raise
    
    def update(self, session: Session, id: Union[UUID, str], 
               **kwargs) -> Optional[ModelType]:
        """Update model instance.
        
        Args:
            session: Database session
            id: Model ID
            **kwargs: Attributes to update
            
        Returns:
            Updated model instance or None if not found
        """
        try:
            instance = self.get_by_id(session, id)
            if not instance:
                return None
            
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            
            session.commit()
            session.refresh(instance)
            
            self.logger.debug(f"Updated {self.model_class.__name__}: {id}")
            return instance
        except IntegrityError as e:
            session.rollback()
            self.logger.error(f"Integrity error updating {self.model_class.__name__} {id}: {e}")
            raise
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error updating {self.model_class.__name__} {id}: {e}")
            raise
    
    def delete(self, session: Session, id: Union[UUID, str]) -> bool:
        """Delete model instance.
        
        Args:
            session: Database session
            id: Model ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            instance = self.get_by_id(session, id)
            if not instance:
                return False
            
            session.delete(instance)
            session.commit()
            
            self.logger.debug(f"Deleted {self.model_class.__name__}: {id}")
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error deleting {self.model_class.__name__} {id}: {e}")
            raise
    
    def count(self, session: Session, **filters) -> int:
        """Count model instances with optional filters.
        
        Args:
            session: Database session
            **filters: Filter conditions
            
        Returns:
            Count of matching instances
        """
        try:
            query = session.query(func.count(self.model_class.id))
            
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    query = query.filter(getattr(self.model_class, key) == value)
            
            return query.scalar()
        except Exception as e:
            self.logger.error(f"Error counting {self.model_class.__name__}: {e}")
            raise
    
    def exists(self, session: Session, id: Union[UUID, str]) -> bool:
        """Check if model instance exists.
        
        Args:
            session: Database session
            id: Model ID
            
        Returns:
            True if exists, False otherwise
        """
        try:
            return session.query(
                session.query(self.model_class).filter(self.model_class.id == id).exists()
            ).scalar()
        except Exception as e:
            self.logger.error(f"Error checking existence of {self.model_class.__name__} {id}: {e}")
            raise
    
    def find_by(self, session: Session, limit: Optional[int] = None, 
                offset: int = 0, order_by: Optional[str] = None, 
                **filters) -> List[ModelType]:
        """Find model instances with filters.
        
        Args:
            session: Database session
            limit: Maximum number of results
            offset: Number of results to skip
            order_by: Field name to order by (prefix with '-' for descending)
            **filters: Filter conditions
            
        Returns:
            List of matching model instances
        """
        try:
            query = session.query(self.model_class)
            
            # Apply filters
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    if isinstance(value, list):
                        query = query.filter(getattr(self.model_class, key).in_(value))
                    else:
                        query = query.filter(getattr(self.model_class, key) == value)
            
            # Apply ordering
            if order_by:
                if order_by.startswith('-'):
                    field_name = order_by[1:]
                    if hasattr(self.model_class, field_name):
                        query = query.order_by(desc(getattr(self.model_class, field_name)))
                else:
                    if hasattr(self.model_class, order_by):
                        query = query.order_by(getattr(self.model_class, order_by))
            
            # Apply pagination
            if offset > 0:
                query = query.offset(offset)
            
            if limit is not None:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            self.logger.error(f"Error finding {self.model_class.__name__}: {e}")
            raise
    
    def find_one_by(self, session: Session, **filters) -> Optional[ModelType]:
        """Find single model instance with filters.
        
        Args:
            session: Database session
            **filters: Filter conditions
            
        Returns:
            Matching model instance or None
        """
        try:
            query = session.query(self.model_class)
            
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    query = query.filter(getattr(self.model_class, key) == value)
            
            return query.first()
        except Exception as e:
            self.logger.error(f"Error finding one {self.model_class.__name__}: {e}")
            raise
    
    def bulk_create(self, session: Session, instances: List[Dict]) -> List[ModelType]:
        """Create multiple model instances.
        
        Args:
            session: Database session
            instances: List of dictionaries with model attributes
            
        Returns:
            List of created model instances
        """
        try:
            model_instances = [self.model_class(**data) for data in instances]
            session.add_all(model_instances)
            session.commit()
            
            for instance in model_instances:
                session.refresh(instance)
            
            self.logger.debug(f"Bulk created {len(model_instances)} {self.model_class.__name__} instances")
            return model_instances
        except IntegrityError as e:
            session.rollback()
            self.logger.error(f"Integrity error bulk creating {self.model_class.__name__}: {e}")
            raise
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error bulk creating {self.model_class.__name__}: {e}")
            raise
    
    def bulk_update(self, session: Session, updates: List[Dict]) -> int:
        """Update multiple model instances.
        
        Args:
            session: Database session
            updates: List of dictionaries with id and update data
            
        Returns:
            Number of updated instances
        """
        try:
            updated_count = 0
            
            for update_data in updates:
                if 'id' not in update_data:
                    continue
                
                instance_id = update_data.pop('id')
                result = session.query(self.model_class).filter(
                    self.model_class.id == instance_id
                ).update(update_data)
                
                updated_count += result
            
            session.commit()
            
            self.logger.debug(f"Bulk updated {updated_count} {self.model_class.__name__} instances")
            return updated_count
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error bulk updating {self.model_class.__name__}: {e}")
            raise
    
    def get_paginated(self, session: Session, page: int = 1, 
                     page_size: int = 20, **filters) -> Dict[str, any]:
        """Get paginated results.
        
        Args:
            session: Database session
            page: Page number (1-based)
            page_size: Number of items per page
            **filters: Filter conditions
            
        Returns:
            Dictionary with pagination metadata and results
        """
        try:
            offset = (page - 1) * page_size
            
            # Get total count
            total_count = self.count(session, **filters)
            
            # Get paginated results
            results = self.find_by(
                session, 
                limit=page_size, 
                offset=offset, 
                **filters
            )
            
            total_pages = (total_count + page_size - 1) // page_size
            
            return {
                "results": results,
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        except Exception as e:
            self.logger.error(f"Error getting paginated {self.model_class.__name__}: {e}")
            raise