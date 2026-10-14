"""#1113 pin: Elmer's NG recording embeds the input CONTENT, not its description."""
import types
from runtime import graph_encoder as ge


def test_encode_embeds_given_text(monkeypatch):
    seen = []
    enc = ge.GraphEncoder()
    monkeypatch.setattr(enc, "_embed", lambda t: seen.append(t) or [0.0])
    sig = types.SimpleNamespace(description="Inference on: Sensory input: 12 chars", signal_type="coherence",
                                module_id="elmer", signal_id="s", coherence_score=1.0, confidence=1.0)
    enc.encode(sig, text="the actual words")
    assert seen == ["the actual words"]


def test_encode_falls_back_to_description(monkeypatch):
    seen = []
    enc = ge.GraphEncoder()
    monkeypatch.setattr(enc, "_embed", lambda t: seen.append(t) or [0.0])
    sig = types.SimpleNamespace(description="desc", signal_type="coherence", module_id="elmer", signal_id="s",
                                coherence_score=1.0, confidence=1.0)
    enc.encode(sig)
    enc.encode(sig, text="  ")
    assert seen == ["desc", "desc"]


def test_engine_passes_content():
    src = open(ge.__file__.replace("graph_encoder.py", "engine.py")).read()
    assert "self._graph_encoder.encode(final_signal, text=text)" in src
