"""Repository for cluster data access operations."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

try:
    from sqlalchemy import and_, func, or_
    from sqlalchemy.orm import Session, joinedload
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    # Fallback for environments without SQLAlchemy
    SQLALCHEMY_AVAILABLE = False
    
    # Mock functions for fallback
    def and_(*args):
        return None
    
    def or_(*args):
        return None
    
    class func:
        @staticmethod
        def count(*args):
            return 0
    
    class Session:
        def __init__(self, *args, **kwargs):
            pass
    
    def joinedload(*args):
        return None

from ..database.models import ClusterModel, NodeModel, TrainingJobModel
from .base import BaseRepository


class ClusterRepository(BaseRepository[ClusterModel]):
    """Repository for cluster CRUD operations."""
    
    def __init__(self):
        """Initialize cluster repository."""
        super().__init__(ClusterModel)
    
    def create_cluster(self, session: Session, cluster_data: Dict) -> ClusterModel:
        """Create a new cluster with validation.
        
        Args:
            session: Database session
            cluster_data: Cluster configuration data
            
        Returns:
            Created cluster instance
        """
        # Validate cluster data
        required_fields = ['name', 'provider', 'region', 'node_count', 'instance_type', 'total_hpus']
        for field in required_fields:
            if field not in cluster_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Check for duplicate cluster name
        existing = self.find_one_by(session, name=cluster_data['name'])
        if existing:
            raise ValueError(f"Cluster with name '{cluster_data['name']}' already exists")
        
        # Set default values
        cluster_data.setdefault('status', 'pending')
        cluster_data.setdefault('health_status', 'unknown')
        cluster_data.setdefault('estimated_cost_per_hour', 0.0)
        cluster_data.setdefault('actual_cost_to_date', 0.0)
        cluster_data.setdefault('enable_monitoring', True)
        cluster_data.setdefault('enable_spot_instances', False)
        cluster_data.setdefault('enable_auto_scaling', False)
        
        return self.create(session, **cluster_data)
    
    def get_cluster_with_nodes(self, session: Session, cluster_id: UUID) -> Optional[ClusterModel]:
        """Get cluster with all associated nodes.
        
        Args:
            session: Database session
            cluster_id: Cluster ID
            
        Returns:
            Cluster with nodes loaded
        """
        return session.query(ClusterModel).options(
            joinedload(ClusterModel.nodes)
        ).filter(ClusterModel.id == cluster_id).first()
    
    def get_clusters_by_provider(self, session: Session, provider: str) -> List[ClusterModel]:
        """Get all clusters for a specific cloud provider.
        
        Args:
            session: Database session
            provider: Cloud provider name
            
        Returns:
            List of clusters
        """
        return self.find_by(session, provider=provider, order_by='-created_at')
    
    def get_active_clusters(self, session: Session) -> List[ClusterModel]:
        """Get all active (running) clusters.
        
        Args:
            session: Database session
            
        Returns:
            List of active clusters
        """
        return self.find_by(session, status='running', order_by='-created_at')
    
    def get_clusters_by_cost_range(self, session: Session, min_cost: float, 
                                  max_cost: float) -> List[ClusterModel]:
        """Get clusters within a cost range.
        
        Args:
            session: Database session
            min_cost: Minimum hourly cost
            max_cost: Maximum hourly cost
            
        Returns:
            List of clusters within cost range
        """
        return session.query(ClusterModel).filter(
            and_(
                ClusterModel.estimated_cost_per_hour >= min_cost,
                ClusterModel.estimated_cost_per_hour <= max_cost
            )
        ).order_by(ClusterModel.estimated_cost_per_hour).all()
    
    def search_clusters(self, session: Session, search_term: str) -> List[ClusterModel]:
        """Search clusters by name or tags.
        
        Args:
            session: Database session
            search_term: Search term
            
        Returns:
            List of matching clusters
        """
        return session.query(ClusterModel).filter(
            or_(
                ClusterModel.name.ilike(f"%{search_term}%"),
                ClusterModel.tags.astext.ilike(f"%{search_term}%")
            )
        ).order_by(ClusterModel.name).all()
    
    def update_cluster_status(self, session: Session, cluster_id: UUID, 
                             status: str) -> Optional[ClusterModel]:
        """Update cluster status.
        
        Args:
            session: Database session
            cluster_id: Cluster ID
            status: New status
            
        Returns:
            Updated cluster or None if not found
        """
        return self.update(session, cluster_id, status=status)
    
    def update_cluster_health(self, session: Session, cluster_id: UUID,
                             health_status: str) -> Optional[ClusterModel]:
        """Update cluster health status.
        
        Args:
            session: Database session
            cluster_id: Cluster ID
            health_status: New health status
            
        Returns:
            Updated cluster or None if not found
        """
        return self.update(
            session, 
            cluster_id, 
            health_status=health_status,
            last_health_check=datetime.utcnow()
        )
    
    def update_cluster_cost(self, session: Session, cluster_id: UUID,
                           actual_cost: float) -> Optional[ClusterModel]:
        """Update cluster actual cost.
        
        Args:
            session: Database session
            cluster_id: Cluster ID
            actual_cost: Actual cost to date
            
        Returns:
            Updated cluster or None if not found
        """
        return self.update(session, cluster_id, actual_cost_to_date=actual_cost)
    
    def get_cluster_statistics(self, session: Session) -> Dict[str, any]:
        """Get overall cluster statistics.
        
        Args:
            session: Database session
            
        Returns:
            Dictionary with cluster statistics
        """
        total_clusters = self.count(session)
        active_clusters = self.count(session, status='running')
        total_hpus = session.query(func.sum(ClusterModel.total_hpus)).scalar() or 0
        total_cost_per_hour = session.query(func.sum(ClusterModel.estimated_cost_per_hour)).scalar() or 0
        
        # Provider distribution
        provider_stats = session.query(
            ClusterModel.provider,
            func.count(ClusterModel.id).label('count')
        ).group_by(ClusterModel.provider).all()
        
        # Status distribution
        status_stats = session.query(
            ClusterModel.status,
            func.count(ClusterModel.id).label('count')
        ).group_by(ClusterModel.status).all()
        
        return {
            "total_clusters": total_clusters,
            "active_clusters": active_clusters,
            "total_hpus": total_hpus,
            "total_estimated_cost_per_hour": total_cost_per_hour,
            "estimated_monthly_cost": total_cost_per_hour * 24 * 30,
            "provider_distribution": {row.provider: row.count for row in provider_stats},
            "status_distribution": {row.status: row.count for row in status_stats}
        }
    
    def get_clusters_needing_health_check(self, session: Session, 
                                        max_age_minutes: int = 30) -> List[ClusterModel]:
        """Get clusters that need health checks.
        
        Args:
            session: Database session
            max_age_minutes: Maximum age of last health check in minutes
            
        Returns:
            List of clusters needing health checks
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        
        return session.query(ClusterModel).filter(
            and_(
                ClusterModel.status == 'running',
                or_(
                    ClusterModel.last_health_check.is_(None),
                    ClusterModel.last_health_check < cutoff_time
                )
            )
        ).all()
    
    def get_cost_trending_data(self, session: Session, 
                              days: int = 30) -> List[Dict[str, any]]:
        """Get cost trending data for clusters.
        
        Args:
            session: Database session
            days: Number of days to include in trending
            
        Returns:
            List of cost trend data points
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # This is a simplified version - in practice, you'd want to track
        # actual cost data over time in a separate table
        clusters = session.query(ClusterModel).filter(
            ClusterModel.created_at >= cutoff_date
        ).all()
        
        trend_data = []
        for cluster in clusters:
            if cluster.created_at:
                days_running = (datetime.utcnow() - cluster.created_at).days
                estimated_total_cost = cluster.estimated_cost_per_hour * 24 * days_running
                
                trend_data.append({
                    "cluster_id": str(cluster.id),
                    "cluster_name": cluster.name,
                    "days_running": days_running,
                    "estimated_total_cost": estimated_total_cost,
                    "actual_cost_to_date": cluster.actual_cost_to_date,
                    "cost_efficiency": (
                        cluster.actual_cost_to_date / estimated_total_cost 
                        if estimated_total_cost > 0 else 0
                    )
                })
        
        return sorted(trend_data, key=lambda x: x["estimated_total_cost"], reverse=True)
    
    def delete_cluster_cascade(self, session: Session, cluster_id: UUID) -> bool:
        """Delete cluster and all associated data.
        
        Args:
            session: Database session
            cluster_id: Cluster ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            cluster = self.get_by_id(session, cluster_id)
            if not cluster:
                return False
            
            # Delete associated training jobs
            session.query(TrainingJobModel).filter(
                TrainingJobModel.cluster_id == cluster_id
            ).delete()
            
            # Delete associated nodes
            session.query(NodeModel).filter(
                NodeModel.cluster_id == cluster_id
            ).delete()
            
            # Delete the cluster itself
            session.delete(cluster)
            session.commit()
            
            self.logger.info(f"Deleted cluster {cluster_id} and all associated data")
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error deleting cluster {cluster_id}: {e}")
            raise