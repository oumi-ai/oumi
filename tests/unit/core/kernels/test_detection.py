# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for kernel detection utilities."""

from unittest.mock import MagicMock, patch

import pytest

from oumi.core.kernels.detection import (
    get_available_kernels_info,
    is_flash_attn3_kernel_available,
    is_kernels_available,
    load_flash_attn3_kernel,
)


class TestKernelsDetection:
    """Test kernel availability detection."""

    @patch("oumi.core.kernels.detection.kernels")
    def test_is_kernels_available_true(self, mock_kernels):
        """Test kernels package detection when available."""
        result = is_kernels_available()
        assert result is True

    @patch("oumi.core.kernels.detection.kernels", side_effect=ImportError("No module"))
    def test_is_kernels_available_false(self, mock_kernels):
        """Test kernels package detection when not available."""
        result = is_kernels_available()
        assert result is False

    @patch("oumi.core.kernels.detection.is_kernels_available")
    def test_is_flash_attn3_kernel_available_no_kernels(self, mock_kernels_available):
        """Test FA3 kernel detection when kernels package not available."""
        mock_kernels_available.return_value = False
        
        result = is_flash_attn3_kernel_available()
        
        assert result is False

    @patch("oumi.core.kernels.detection.get_kernel")
    @patch("oumi.core.kernels.detection.is_kernels_available")
    def test_is_flash_attn3_kernel_available_success(self, mock_kernels_available, mock_get_kernel):
        """Test FA3 kernel detection when kernel is available."""
        mock_kernels_available.return_value = True
        mock_get_kernel.return_value = MagicMock()
        
        result = is_flash_attn3_kernel_available()
        
        assert result is True
        mock_get_kernel.assert_called_once_with("kernels-community/flash-attn3")

    @patch("oumi.core.kernels.detection.get_kernel")
    @patch("oumi.core.kernels.detection.is_kernels_available")
    def test_is_flash_attn3_kernel_available_exception(self, mock_kernels_available, mock_get_kernel):
        """Test FA3 kernel detection when kernel loading fails."""
        mock_kernels_available.return_value = True
        mock_get_kernel.side_effect = Exception("Kernel not found")
        
        result = is_flash_attn3_kernel_available()
        
        assert result is False

    @patch("oumi.core.kernels.detection.is_flash_attn3_kernel_available")
    @patch("oumi.core.kernels.detection.is_kernels_available")
    def test_get_available_kernels_info(self, mock_kernels_available, mock_flash_attn3_available):
        """Test getting kernel availability information."""
        mock_kernels_available.return_value = True
        mock_flash_attn3_available.return_value = False
        
        info = get_available_kernels_info()
        
        assert info == {
            "kernels_package": True,
            "flash_attn3_kernel": False,
        }


class TestLoadFlashAttn3Kernel:
    """Test Flash Attention 3 kernel loading."""

    @patch("oumi.core.kernels.detection.is_kernels_available")
    def test_load_flash_attn3_kernel_no_kernels_package(self, mock_kernels_available):
        """Test kernel loading when kernels package not available."""
        mock_kernels_available.return_value = False
        
        with pytest.raises(ImportError, match="kernels package is required"):
            load_flash_attn3_kernel()

    @patch("oumi.core.kernels.detection.get_kernel")
    @patch("oumi.core.kernels.detection.is_kernels_available")
    def test_load_flash_attn3_kernel_success(self, mock_kernels_available, mock_get_kernel):
        """Test successful kernel loading."""
        mock_kernels_available.return_value = True
        mock_kernel = MagicMock()
        mock_get_kernel.return_value = mock_kernel
        
        result = load_flash_attn3_kernel()
        
        assert result is mock_kernel
        mock_get_kernel.assert_called_once_with("kernels-community/flash-attn3")

    @patch("oumi.core.kernels.detection.get_kernel")
    @patch("oumi.core.kernels.detection.is_kernels_available")
    def test_load_flash_attn3_kernel_returns_none(self, mock_kernels_available, mock_get_kernel):
        """Test kernel loading when get_kernel returns None."""
        mock_kernels_available.return_value = True
        mock_get_kernel.return_value = None
        
        with pytest.raises(RuntimeError, match="Failed to load flash-attn3 kernel"):
            load_flash_attn3_kernel()

    @patch("oumi.core.kernels.detection.get_kernel")
    @patch("oumi.core.kernels.detection.is_kernels_available")
    def test_load_flash_attn3_kernel_exception(self, mock_kernels_available, mock_get_kernel):
        """Test kernel loading when get_kernel raises exception."""
        mock_kernels_available.return_value = True
        mock_get_kernel.side_effect = Exception("Network error")
        
        with pytest.raises(RuntimeError, match="Failed to load flash-attn3 kernel: Network error"):
            load_flash_attn3_kernel()