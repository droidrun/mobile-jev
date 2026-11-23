import pytest

NAMES = "ContextProvider StepContext AppCardContextProvider AgentOperator MobileAgent MobileAgentState ResultEvent load_llm AgentMacroOperator MobileConfig AgentConfig FastAgentConfig ManagerConfig ExecutorConfig AppCardConfig ToolsConfig CredentialsConfig LLMProfile DelegationOperator".split()


def test_public_exports():
    import mobilerun.agents as agents

    assert set(agents.__all__) == set(NAMES)
    for name in NAMES:
        assert getattr(agents, name) is not None
    with pytest.raises(AttributeError):
        getattr(agents, "no_such_" + "export")
