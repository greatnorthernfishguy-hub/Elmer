"""#426: late hosted engines join one body through the existing monitor.

Changelog: 2026-09-11 Codex — isolated body/engine fixtures; no model loads.
"""
from types import SimpleNamespace
from pathlib import Path
import pytest
from core.brain_switcher import BrainSwitcher


class Engine:
    def __init__(self):
        self._shared_body = None
        self.events = []
        self.accept = True

    def set_body_lock(self, lock):
        self.lock = lock
        self.events.append("lock")

    def set_lock_file(self, path):
        self.path = path
        self.events.append("file")

    def offer_shared_body(self, body, *, blocking=True):
        assert blocking is False
        assert self.lock is not None and Path(self.path).is_file()
        self.events.append("offer")
        if self.accept:
            self._shared_body = body
        return self.accept

    def revoke_shared_body(self):
        self._shared_body = None
        self.events.append("revoke")


@pytest.fixture
def switcher(monkeypatch, tmp_path):
    body = object()
    proto = SimpleNamespace(_loaded=True, _brain=SimpleNamespace(transformer_body=body))
    manager = SimpleNamespace(get_socket=lambda name: proto)
    switch = BrainSwitcher(manager)
    switch._active_brain = "proto_only"
    monkeypatch.setattr(switch, "_get_lock_file_path", lambda: str(tmp_path / "body.lock"))
    monkeypatch.setattr(switch, "_check_resources", lambda: {"free_ram_mb": 9000, "cpu_load_1m": 0.1})
    return switch, proto, body


def test_engine_arrives_after_body_and_registers_once(switcher):
    switch, proto, body = switcher
    ready = [None]
    switch.register_tonic_engine_provider("cc", lambda: ready[0])
    assert switch._tonic_engines == []
    ready[0] = Engine()
    switch._evaluate_and_switch()
    assert ready[0]._shared_body is body
    assert ready[0].events.index("file") < ready[0].events.index("offer")
    switch._evaluate_and_switch()
    assert switch._tonic_engines == [ready[0]]
    assert ready[0].events.count("offer") == 1


def test_body_arrives_after_engine_and_two_consumers_share_identity(switcher):
    switch, proto, body = switcher
    proto._loaded = False
    syl, cc = Engine(), Engine()
    switch.register_tonic_engine(syl)
    switch.register_tonic_engine_provider("cc", lambda: cc)
    assert cc._shared_body is None and syl._shared_body is None
    proto._loaded = True
    switch._evaluate_and_switch()
    assert cc._shared_body is syl._shared_body is body
    assert cc.lock is syl.lock is switch._body_lock


def test_failed_offer_retries_and_shed_rejoins_without_duplicate(switcher):
    switch, proto, body = switcher
    cc = Engine()
    cc.accept = False
    switch.register_tonic_engine_provider("cc", lambda: cc)
    assert cc._shared_body is None
    cc.accept = True
    switch._evaluate_and_switch()
    assert cc._shared_body is body
    switch._revoke_body_from_tonic()
    assert cc._shared_body is None
    replacement = object()
    proto._brain.transformer_body = replacement
    switch._evaluate_and_switch()
    assert cc._shared_body is replacement
    assert switch._tonic_engines == [cc]


def test_failing_provider_does_not_prevent_other_registration(switcher):
    switch, proto, body = switcher
    def unavailable():
        raise RuntimeError("not ready")
    switch.register_tonic_engine_provider("unready", unavailable)
    cc = Engine()
    switch.register_tonic_engine_provider("cc", lambda: cc)
    switch._evaluate_and_switch()
    assert cc._shared_body is body


def test_restored_proto_writer_gets_same_lock_before_loading(monkeypatch, tmp_path):
    sockets = {}
    manager = SimpleNamespace(
        get_socket=lambda name: sockets.get(name),
        register=lambda proto: sockets.update({"elmer:proto_unibrain": proto}),
    )
    switch = BrainSwitcher(manager)
    class Proto:
        _loaded = False
        def set_body_lock(self, lock):
            self.lock = lock
        def load(self, path):
            assert self.lock is switch._body_lock
            self._loaded = True
            self._brain = SimpleNamespace(transformer_body=object())
            return True
    switch._proto_brain_socket_cls = Proto
    monkeypatch.setattr(switch, "_get_lock_file_path", lambda: str(tmp_path / "body.lock"))
    monkeypatch.setattr(switch, "_write_proto_body_status", lambda available: None)
    monkeypatch.setattr(switch, "_wire_neural_comprehension", lambda: None)
    cc = Engine()
    switch.register_tonic_engine(cc)
    switch._add_proto_unibrain()
    assert cc._shared_body is sockets["elmer:proto_unibrain"]._brain.transformer_body
    assert cc.lock is sockets["elmer:proto_unibrain"].lock
