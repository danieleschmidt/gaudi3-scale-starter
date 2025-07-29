"""Unit tests for GaudiAccelerator."""

import pytest
from unittest.mock import Mock, patch

from gaudi3_scale.accelerator import GaudiAccelerator


class TestGaudiAccelerator:
    """Test suite for GaudiAccelerator class."""
    
    @patch('gaudi3_scale.accelerator.htorch')
    def test_accelerator_initialization(self, mock_htorch):
        """Test accelerator initializes correctly."""
        mock_htorch.hpu.is_available.return_value = True
        
        accelerator = GaudiAccelerator()
        assert accelerator is not None
    
    @patch('gaudi3_scale.accelerator.htorch')
    def test_is_available_with_habana(self, mock_htorch):
        """Test is_available returns True when Habana is available."""
        mock_htorch.hpu.is_available.return_value = True
        
        assert GaudiAccelerator.is_available() is True
    
    def test_is_available_without_habana(self):
        """Test is_available returns False when Habana is not available."""
        with patch.dict('sys.modules', {'habana_frameworks.torch': None}):
            assert GaudiAccelerator.is_available() is False
    
    @patch('gaudi3_scale.accelerator.htorch')
    def test_get_device_stats(self, mock_htorch):
        """Test device statistics retrieval."""
        mock_htorch.hpu.memory_allocated.return_value = 1024
        mock_htorch.hpu.memory_reserved.return_value = 2048
        
        accelerator = GaudiAccelerator()
        stats = accelerator.get_device_stats("hpu:0")
        
        assert "hpu_memory_allocated" in stats
        assert "hpu_memory_reserved" in stats
        assert stats["hpu_memory_allocated"] == 1024
        assert stats["hpu_memory_reserved"] == 2048
    
    def test_initialization_without_habana_raises_error(self):
        """Test initialization raises error when Habana is not available."""
        with patch.dict('sys.modules', {'habana_frameworks.torch': None}):
            with pytest.raises(RuntimeError, match="Habana frameworks not found"):
                GaudiAccelerator()