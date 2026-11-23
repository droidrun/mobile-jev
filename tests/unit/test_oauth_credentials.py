import json

from mobilerun.agents.clients.oauth.credentials import oauth_credentials_present


def test_oauth_credentials_present_handles_missing_wrong_slot_and_token(tmp_path):
    path = tmp_path / "auth.json"
    assert not oauth_credentials_present(str(path), "openai_oauth")

    path.write_text(json.dumps({"anthropicOauth": {"accessToken": "token"}}))
    assert not oauth_credentials_present(str(path), "openai_oauth")

    path.write_text(json.dumps({"openaiOauth": {"access": "token"}}))
    assert oauth_credentials_present(str(path), "openai_oauth")


def test_unknown_oauth_variant_falls_back_to_file_presence(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text("{}")
    assert oauth_credentials_present(str(path), "future_variant")
