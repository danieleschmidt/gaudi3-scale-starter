"""Async service layer with high-performance I/O operations."""

import asyncio

try:
    import aiohttp
except ImportError:
    # Fallback for environments without aiohttp
    aiohttp = None

try:
    import aiofiles
except ImportError:
    # Fallback for environments without aiofiles
    aiofiles = None
import logging
from typing import Any, Dict, List, Optional, Union, Callable, AsyncIterator
from dataclasses import dataclass
from enum import Enum
import json
import time
from pathlib import Path
import weakref

from ..cache.distributed_cache import get_distributed_cache
from ..database.async_pool import AsyncConnectionPool, AsyncConnectionFactory

logger = logging.getLogger(__name__)

class ServiceState(Enum):
    """Service states."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    SHUTDOWN = "shutdown"

@dataclass
class AsyncServiceConfig:
    """Configuration for async services."""
    max_concurrent_requests: int = 100
    request_timeout: float = 30.0
    connection_pool_size: int = 10
    enable_retries: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_caching: bool = True
    cache_ttl: int = 300
    enable_metrics: bool = True
    batch_size: int = 50
    batch_timeout: float = 0.1

class AsyncHTTPService:
    """High-performance async HTTP client service."""
    
    def __init__(self, config: Optional[AsyncServiceConfig] = None):
        if aiohttp is None:
            raise ImportError("aiohttp is required for AsyncHTTPService. Install with: pip install aiohttp")
        
        self.config = config or AsyncServiceConfig()
        self.logger = logger.getChild(self.__class__.__name__)
        
        # HTTP session management
        self._session: Optional['aiohttp.ClientSession'] = None
        self._session_lock = asyncio.Lock()
        
        # Request limiting
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Caching
        self._cache = get_distributed_cache() if self.config.enable_caching else None
        
        # Metrics
        self._request_count = 0
        self._error_count = 0
        self._total_response_time = 0.0
        self._metrics_lock = asyncio.Lock()
        
        # State
        self._state = ServiceState.INITIALIZING
        self._shutdown_event = asyncio.Event()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=self.config.connection_pool_size,
                        limit_per_host=self.config.connection_pool_size // 2,
                        ttl_dns_cache=300,
                        use_dns_cache=True,
                        enable_cleanup_closed=True
                    )
                    
                    timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
                    
                    self._session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        headers={'User-Agent': 'Gaudi3Scale/1.0'}
                    )
                    
                    self._state = ServiceState.HEALTHY
        
        return self._session
    
    async def _make_request_with_retry(self, 
                                      method: str,
                                      url: str,
                                      **kwargs) -> aiohttp.ClientResponse:
        """Make HTTP request with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                session = await self._get_session()
                
                async with session.request(method, url, **kwargs) as response:
                    # Read response content to avoid warnings
                    await response.read()
                    return response
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Request failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"Request failed after {self.config.max_retries} retries: {e}")
                    break
        
        if last_exception:
            raise last_exception
        
        raise RuntimeError("Request failed with unknown error")
    
    async def get(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make async GET request.
        
        Args:
            url: Request URL
            **kwargs: Additional request parameters
            
        Returns:
            JSON response data
        """
        # Check cache first
        cache_key = f"http_get:{hash(url + str(sorted(kwargs.items())))}"
        
        if self._cache and self.config.enable_caching:
            cached_response = self._cache.get(cache_key)
            if cached_response:
                return cached_response
        
        async with self._semaphore:
            start_time = time.time()
            
            try:
                response = await self._make_request_with_retry('GET', url, **kwargs)
                response.raise_for_status()
                
                data = await response.json()
                
                # Cache successful response
                if self._cache and response.status == 200:
                    self._cache.set(cache_key, data, ttl=self.config.cache_ttl)
                
                # Update metrics
                await self._update_metrics(time.time() - start_time, success=True)
                
                return data
                
            except Exception as e:
                await self._update_metrics(time.time() - start_time, success=False)
                self.logger.error(f"GET request failed for {url}: {e}")
                raise
    
    async def post(self, url: str, data: Any = None, json_data: Any = None, **kwargs) -> Dict[str, Any]:
        """Make async POST request.
        
        Args:
            url: Request URL
            data: Form data
            json_data: JSON data
            **kwargs: Additional request parameters
            
        Returns:
            JSON response data
        """
        async with self._semaphore:
            start_time = time.time()
            
            try:
                if json_data is not None:
                    kwargs['json'] = json_data
                elif data is not None:
                    kwargs['data'] = data
                
                response = await self._make_request_with_retry('POST', url, **kwargs)
                response.raise_for_status()
                
                result = await response.json()
                
                # Update metrics
                await self._update_metrics(time.time() - start_time, success=True)
                
                return result
                
            except Exception as e:
                await self._update_metrics(time.time() - start_time, success=False)
                self.logger.error(f"POST request failed for {url}: {e}")
                raise
    
    async def put(self, url: str, data: Any = None, json_data: Any = None, **kwargs) -> Dict[str, Any]:
        """Make async PUT request."""
        async with self._semaphore:
            start_time = time.time()
            
            try:
                if json_data is not None:
                    kwargs['json'] = json_data
                elif data is not None:
                    kwargs['data'] = data
                
                response = await self._make_request_with_retry('PUT', url, **kwargs)
                response.raise_for_status()
                
                result = await response.json()
                
                await self._update_metrics(time.time() - start_time, success=True)
                return result
                
            except Exception as e:
                await self._update_metrics(time.time() - start_time, success=False)
                raise
    
    async def delete(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make async DELETE request."""
        async with self._semaphore:
            start_time = time.time()
            
            try:
                response = await self._make_request_with_retry('DELETE', url, **kwargs)
                response.raise_for_status()
                
                result = await response.json() if response.content_length > 0 else {}
                
                await self._update_metrics(time.time() - start_time, success=True)
                return result
                
            except Exception as e:
                await self._update_metrics(time.time() - start_time, success=False)
                raise
    
    async def batch_get(self, urls: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Make multiple GET requests concurrently.
        
        Args:
            urls: List of URLs to request
            **kwargs: Additional request parameters
            
        Returns:
            List of response data
        """
        semaphore = asyncio.Semaphore(self.config.batch_size)
        
        async def fetch_one(url: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self.get(url, **kwargs)
                except Exception as e:
                    self.logger.error(f"Batch GET failed for {url}: {e}")
                    return {'error': str(e), 'url': url}
        
        tasks = [fetch_one(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    async def _update_metrics(self, response_time: float, success: bool) -> None:
        """Update request metrics."""
        async with self._metrics_lock:
            self._request_count += 1
            self._total_response_time += response_time
            
            if not success:
                self._error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get HTTP service statistics."""
        avg_response_time = (
            self._total_response_time / self._request_count 
            if self._request_count > 0 else 0.0
        )
        
        error_rate = (
            self._error_count / self._request_count * 100 
            if self._request_count > 0 else 0.0
        )
        
        return {
            'state': self._state.value,
            'request_count': self._request_count,
            'error_count': self._error_count,
            'error_rate': error_rate,
            'avg_response_time': avg_response_time,
            'session_closed': self._session is None or self._session.closed,
            'config': {
                'max_concurrent_requests': self.config.max_concurrent_requests,
                'request_timeout': self.config.request_timeout,
                'connection_pool_size': self.config.connection_pool_size,
                'enable_retries': self.config.enable_retries,
                'max_retries': self.config.max_retries
            }
        }
    
    async def close(self) -> None:
        """Close HTTP service and cleanup resources."""
        self._state = ServiceState.SHUTDOWN
        self._shutdown_event.set()
        
        if self._session and not self._session.closed:
            await self._session.close()
        
        self.logger.info("HTTP service closed")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class AsyncFileService:
    """High-performance async file I/O service."""
    
    def __init__(self, config: Optional[AsyncServiceConfig] = None):
        if aiofiles is None:
            raise ImportError("aiofiles is required for AsyncFileService. Install with: pip install aiofiles")
        
        self.config = config or AsyncServiceConfig()
        self.logger = logger.getChild(self.__class__.__name__)
        
        # File operation limiting
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Metrics
        self._read_count = 0
        self._write_count = 0
        self._error_count = 0
        self._total_io_time = 0.0
        self._metrics_lock = asyncio.Lock()
        
        self._state = ServiceState.HEALTHY
    
    async def read_file(self, file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
        """Read file asynchronously.
        
        Args:
            file_path: Path to file
            encoding: File encoding
            
        Returns:
            File content as string
        """
        async with self._semaphore:
            start_time = time.time()
            
            try:
                async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                    content = await f.read()
                
                await self._update_metrics(time.time() - start_time, 'read', success=True)
                return content
                
            except Exception as e:
                await self._update_metrics(time.time() - start_time, 'read', success=False)
                self.logger.error(f"Failed to read file {file_path}: {e}")
                raise
    
    async def write_file(self, file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> None:
        """Write file asynchronously.
        
        Args:
            file_path: Path to file
            content: Content to write
            encoding: File encoding
        """
        async with self._semaphore:
            start_time = time.time()
            
            try:
                # Ensure directory exists
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                
                async with aiofiles.open(file_path, 'w', encoding=encoding) as f:
                    await f.write(content)
                
                await self._update_metrics(time.time() - start_time, 'write', success=True)
                
            except Exception as e:
                await self._update_metrics(time.time() - start_time, 'write', success=False)
                self.logger.error(f"Failed to write file {file_path}: {e}")
                raise
    
    async def read_json_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Read JSON file asynchronously.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        content = await self.read_file(file_path)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from {file_path}: {e}")
            raise
    
    async def write_json_file(self, file_path: Union[str, Path], data: Dict[str, Any], indent: int = 2) -> None:
        """Write JSON file asynchronously.
        
        Args:
            file_path: Path to JSON file
            data: Data to write
            indent: JSON indentation
        """
        content = json.dumps(data, indent=indent, default=str)
        await self.write_file(file_path, content)
    
    async def read_lines(self, file_path: Union[str, Path], encoding: str = 'utf-8') -> AsyncIterator[str]:
        """Read file lines asynchronously.
        
        Args:
            file_path: Path to file
            encoding: File encoding
            
        Yields:
            File lines
        """
        async with self._semaphore:
            try:
                async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                    async for line in f:
                        yield line.rstrip('\n')
            except Exception as e:
                self.logger.error(f"Failed to read lines from {file_path}: {e}")
                raise
    
    async def append_file(self, file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> None:
        """Append to file asynchronously.
        
        Args:
            file_path: Path to file
            content: Content to append
            encoding: File encoding
        """
        async with self._semaphore:
            start_time = time.time()
            
            try:
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                
                async with aiofiles.open(file_path, 'a', encoding=encoding) as f:
                    await f.write(content)
                
                await self._update_metrics(time.time() - start_time, 'write', success=True)
                
            except Exception as e:
                await self._update_metrics(time.time() - start_time, 'write', success=False)
                self.logger.error(f"Failed to append to file {file_path}: {e}")
                raise
    
    async def batch_read_files(self, file_paths: List[Union[str, Path]]) -> Dict[str, str]:
        """Read multiple files concurrently.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary mapping file paths to content
        """
        semaphore = asyncio.Semaphore(self.config.batch_size)
        
        async def read_one(file_path: Union[str, Path]) -> tuple:
            async with semaphore:
                try:
                    content = await self.read_file(file_path)
                    return str(file_path), content
                except Exception as e:
                    self.logger.error(f"Batch read failed for {file_path}: {e}")
                    return str(file_path), f"ERROR: {e}"
        
        tasks = [read_one(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        return dict(results)
    
    async def copy_file(self, src: Union[str, Path], dst: Union[str, Path]) -> None:
        """Copy file asynchronously.
        
        Args:
            src: Source file path
            dst: Destination file path
        """
        content = await self.read_file(src)
        await self.write_file(dst, content)
    
    async def file_exists(self, file_path: Union[str, Path]) -> bool:
        """Check if file exists asynchronously.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file exists
        """
        return Path(file_path).exists()
    
    async def get_file_size(self, file_path: Union[str, Path]) -> int:
        """Get file size asynchronously.
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in bytes
        """
        return Path(file_path).stat().st_size
    
    async def _update_metrics(self, io_time: float, operation: str, success: bool) -> None:
        """Update I/O metrics."""
        async with self._metrics_lock:
            self._total_io_time += io_time
            
            if operation == 'read':
                self._read_count += 1
            elif operation == 'write':
                self._write_count += 1
            
            if not success:
                self._error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get file service statistics."""
        total_operations = self._read_count + self._write_count
        avg_io_time = (
            self._total_io_time / total_operations 
            if total_operations > 0 else 0.0
        )
        
        error_rate = (
            self._error_count / total_operations * 100 
            if total_operations > 0 else 0.0
        )
        
        return {
            'state': self._state.value,
            'read_count': self._read_count,
            'write_count': self._write_count,
            'total_operations': total_operations,
            'error_count': self._error_count,
            'error_rate': error_rate,
            'avg_io_time': avg_io_time,
            'config': {
                'max_concurrent_requests': self.config.max_concurrent_requests,
                'batch_size': self.config.batch_size
            }
        }
    
    async def close(self) -> None:
        """Close file service."""
        self._state = ServiceState.SHUTDOWN
        self.logger.info("File service closed")


class AsyncServiceManager:
    """Manager for async services with lifecycle management."""
    
    def __init__(self):
        self.services: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Default services
        self._http_service: Optional[AsyncHTTPService] = None
        self._file_service: Optional[AsyncFileService] = None
    
    async def get_http_service(self, config: Optional[AsyncServiceConfig] = None) -> AsyncHTTPService:
        """Get HTTP service instance.
        
        Args:
            config: Optional service configuration
            
        Returns:
            HTTP service instance
        """
        if self._http_service is None:
            async with self._lock:
                if self._http_service is None:
                    self._http_service = AsyncHTTPService(config)
                    self.services['http'] = self._http_service
        
        return self._http_service
    
    async def get_file_service(self, config: Optional[AsyncServiceConfig] = None) -> AsyncFileService:
        """Get file service instance.
        
        Args:
            config: Optional service configuration
            
        Returns:
            File service instance
        """
        if self._file_service is None:
            async with self._lock:
                if self._file_service is None:
                    self._file_service = AsyncFileService(config)
                    self.services['file'] = self._file_service
        
        return self._file_service
    
    async def register_service(self, name: str, service: Any) -> None:
        """Register a custom service.
        
        Args:
            name: Service name
            service: Service instance
        """
        async with self._lock:
            self.services[name] = service
            self.logger.info(f"Registered service '{name}'")
    
    async def get_service(self, name: str) -> Any:
        """Get service by name.
        
        Args:
            name: Service name
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service not found
        """
        async with self._lock:
            if name not in self.services:
                raise KeyError(f"Service '{name}' not found")
            return self.services[name]
    
    async def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all services.
        
        Returns:
            Dictionary mapping service names to statistics
        """
        stats = {}
        async with self._lock:
            for name, service in self.services.items():
                if hasattr(service, 'get_stats'):
                    try:
                        stats[name] = service.get_stats()
                    except Exception as e:
                        self.logger.error(f"Failed to get stats for service '{name}': {e}")
                        stats[name] = {'error': str(e)}
                else:
                    stats[name] = {'stats_available': False}
        
        return stats
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all services.
        
        Returns:
            Dictionary mapping service names to health status
        """
        results = {}
        async with self._lock:
            for name, service in self.services.items():
                try:
                    if hasattr(service, 'get_stats'):
                        stats = service.get_stats()
                        results[name] = stats.get('state') in ['healthy', 'degraded']
                    else:
                        results[name] = True  # Assume healthy if no stats
                except Exception as e:
                    self.logger.error(f"Health check failed for service '{name}': {e}")
                    results[name] = False
        
        return results
    
    async def close_all(self) -> None:
        """Close all services."""
        async with self._lock:
            for name, service in list(self.services.items()):
                try:
                    if hasattr(service, 'close'):
                        await service.close()
                    self.logger.info(f"Closed service '{name}'")
                except Exception as e:
                    self.logger.error(f"Error closing service '{name}': {e}")
            
            self.services.clear()
            self._http_service = None
            self._file_service = None
        
        self.logger.info("Closed all async services")


# Global async service manager
_async_service_manager: Optional[AsyncServiceManager] = None

async def get_async_service_manager() -> AsyncServiceManager:
    """Get global async service manager instance.
    
    Returns:
        Async service manager instance
    """
    global _async_service_manager
    if _async_service_manager is None:
        _async_service_manager = AsyncServiceManager()
    return _async_service_manager