import subprocess
import sys

import pytest


@pytest.mark.parametrize("reverse", [False, True])
def test_identity_in_both_orders(reverse):
    script = """
import importlib
pairs = [
 ('mobilerun.agents.agent.droid.events', 'mobilerun.agents.workflows.droid.events', 'ResultEvent'),
 ('mobilerun.agents.agent.droid.droid_agent', 'mobilerun.agents.workflows.droid.droid_agent', 'MobileAgent'),
 ('mobilerun.agents.errors.timeout', 'mobilerun.agents.errors.agent_timeout', 'AgentTimeout'),
 ('mobilerun.agents.agent.usage', 'mobilerun.agents.llm.usage', None),
 ('mobilerun.agents.operator', 'mobilerun.agents.operators.agent', None),
]
for old, new, symbol in pairs:
    names = [old, new]
    if REVERSE: names.reverse()
    a, b = [importlib.import_module(name) for name in names]
    assert a is b, (old, new)
    if symbol: assert getattr(a, symbol) is getattr(b, symbol)
old = importlib.import_module('mobilerun.agents.operator')
new = importlib.import_module('mobilerun.agents.operators.agent')
sentinel = object()
old.MobileAgent = sentinel
assert new.MobileAgent is sentinel
legacy = importlib.import_module('mobilerun.agent.droid.events')
canonical = importlib.import_module('mobilerun.agents.workflows.droid.events')
assert legacy.ResultEvent is canonical.ResultEvent
""".replace("REVERSE", repr(reverse))
    subprocess.run([sys.executable, "-c", script], check=True)
