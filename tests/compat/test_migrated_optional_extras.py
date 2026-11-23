"""Optional integrations fail with add-on installation guidance, not at import time."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("case", ["basic", "anthropic", "langfuse"])
def test_missing_extras_in_fresh_process(case):
    script = r"""
import importlib.abc, sys
class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'anthropic' or fullname.startswith('anthropic.') or fullname == 'langfuse' or fullname.startswith('langfuse.') or fullname == 'llama_index.llms.anthropic' or fullname.startswith('llama_index.llms.anthropic.'):
            raise ModuleNotFoundError('blocked optional dependency', name=fullname)
sys.meta_path.insert(0, BlockExtras())
case = sys.argv[1]
if case == 'basic':
    from mobilerun.agents import MobileConfig
    from mobilerun.agents.operators.agent import AgentOperator
    from click.testing import CliRunner
    from mobilerun.cli.app import cli
    assert MobileConfig and AgentOperator
    for args in (['agent', '--help'], ['agent', 'llm', '--help'], ['run', '--help']):
        result = CliRunner().invoke(cli, args)
        assert result.exit_code == 0, (args, result.output, result.exception)
elif case == 'anthropic':
    from mobilerun.agents.llm.llm_picker import load_llm
    try:
        load_llm('Anthropic', model='test')
    except ImportError as error:
        assert 'mobilerun-agents[anthropic]' in str(error), str(error)
    else:
        raise AssertionError('missing anthropic should fail')
else:
    import logging
    from mobilerun.agents.config.user_config import TracingConfig
    from mobilerun.agents.telemetry.setup import _setup_langfuse_tracing
    class Capture(logging.Handler):
        def emit(self, record): messages.append(record.getMessage())
    messages = []
    log = logging.getLogger('mobilerun')
    handler = Capture()
    log.addHandler(handler)
    try:
        assert _setup_langfuse_tracing(TracingConfig(), None) is False
        assert any('mobilerun-agents[langfuse]' in message for message in messages), messages
    finally:
        log.removeHandler(handler)
"""
    result = subprocess.run([sys.executable, "-c", script, case], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_anthropic_preserves_unrelated_missing_dependency():
    script = r"""
import importlib.abc, sys
from mobilerun.agents.llm.llm_picker import load_llm
missing = 'llama_index.core.llms.utils'
sys.modules.pop(missing, None)
original = ModuleNotFoundError('unrelated dependency missing', name=missing)
class BlockDependency(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == missing:
            raise original
sys.meta_path.insert(0, BlockDependency())
try:
    load_llm('Anthropic', model='test')
except ModuleNotFoundError as error:
    assert error is original
    assert error.name == missing
    assert str(error) == 'unrelated dependency missing'
else:
    raise AssertionError('missing unrelated dependency should fail')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
