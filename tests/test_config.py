"""Tests for Elmer configuration system."""

import os
import tempfile

import pytest

from core.config import (
    ElmerConfig,
    HardwareConfig,
    SocketsConfig,
    NGEcosystemConfig,
    PipelinesConfig,
    load_config,
)


class TestElmerConfig:
    def test_defaults(self):
        cfg = ElmerConfig()
        assert cfg.module_id == "elmer"
        assert cfg.version == "0.1.0"
        assert cfg.log_level == "INFO"

    def test_hardware_defaults(self):
        cfg = ElmerConfig()
        assert cfg.hardware.prefer_gpu is True
        assert cfg.hardware.gpu_memory_fraction == 0.5
        assert cfg.hardware.cpu_threads == 0

    def test_sockets_defaults(self):
        cfg = ElmerConfig()
        assert cfg.sockets.max_sockets == 16
        assert cfg.sockets.health_check_interval == 30.0

    def test_ng_ecosystem_defaults(self):
        cfg = ElmerConfig()
        assert cfg.ng_ecosystem.module_id == "elmer"
        assert cfg.ng_ecosystem.peer_bridge_enabled is True
        assert cfg.ng_ecosystem.tier3_upgrade_enabled is True

    def test_pipelines_defaults(self):
        cfg = ElmerConfig()
        assert cfg.pipelines.sensory_enabled is True
        assert cfg.pipelines.inference_enabled is True
        assert cfg.pipelines.health_enabled is True
        assert cfg.pipelines.memory_enabled is True
        assert cfg.pipelines.identity_enabled is True


class TestLoadConfig:
    def test_load_nonexistent_file(self):
        cfg = load_config("/nonexistent/config.yaml")
        assert cfg.module_id == "elmer"  # Falls back to defaults

    def test_load_from_project_config(self):
        cfg = load_config()
        assert cfg.module_id == "elmer"
        assert cfg.version == "0.1.0"

    def test_load_custom_yaml(self):
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("module_id: test_module\nversion: '9.9.9'\n")
            f.write("hardware:\n  prefer_gpu: false\n")
            tmp_path = f.name

        try:
            cfg = load_config(tmp_path)
            assert cfg.module_id == "test_module"
            assert cfg.version == "9.9.9"
            assert cfg.hardware.prefer_gpu is False
            # Unchanged defaults
            assert cfg.sockets.max_sockets == 16
        finally:
            os.unlink(tmp_path)

    def test_env_var_interpolation(self):
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        os.environ["ELMER_TEST_VERSION"] = "1.2.3"
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write("version: '${ELMER_TEST_VERSION}'\n")
                tmp_path = f.name

            try:
                cfg = load_config(tmp_path)
                assert cfg.version == "1.2.3"
            finally:
                os.unlink(tmp_path)
        finally:
            del os.environ["ELMER_TEST_VERSION"]
