import pytest

from mobilerun_agents.agent.tools.helpers.images import ImageStore


def test_image_store_numbers_files_without_overwriting(monkeypatch, tmp_path):
    root = tmp_path / "images"
    monkeypatch.setenv("MOBILERUN_AGENT_IMAGES_DIR", str(root))
    store = ImageStore("run", keep=True)
    assert not (root / "run").exists()
    first = store.save(b"first")
    second = store.save(b"second")
    third = ImageStore("run", keep=True).save(b"third")
    assert [first, second, third] == [
        root / "run/0001.png",
        root / "run/0002.png",
        root / "run/0003.png",
    ]
    assert [path.read_bytes() for path in (first, second, third)] == [
        b"first",
        b"second",
        b"third",
    ]
    assert sorted((root / "run").iterdir()) == [first, second, third]


def test_removing_a_saved_image_does_not_reuse_its_number(monkeypatch, tmp_path):
    monkeypatch.setenv("MOBILERUN_AGENT_IMAGES_DIR", str(tmp_path))
    store = ImageStore("run", keep=True)
    hidden = store.save(b"screenshot to remove")
    hidden.unlink()
    visible = store.save(b"screenshot to keep")
    store.close()
    assert not hidden.exists()
    assert visible == tmp_path / "run/0002.png"
    assert visible.read_bytes() == b"screenshot to keep"


@pytest.mark.parametrize("run_id", ["", ".", "..", "../other", "/other"])
def test_invalid_image_run_ids_cannot_target_other_folders(run_id):
    with pytest.raises(ValueError, match="directory name"):
        ImageStore(run_id, keep=False)


@pytest.mark.parametrize("keep", [False, True])
def test_close_only_keeps_images_when_requested(monkeypatch, tmp_path, keep):
    root = tmp_path / "images"
    monkeypatch.setenv("MOBILERUN_AGENT_IMAGES_DIR", str(root))
    sibling = root / "other"
    sibling.mkdir(parents=True)
    store = ImageStore("run", keep=keep)
    store.save(b"captured image")
    store.close()
    assert (root / "run").exists() is keep
    assert (root / "run/0001.png").is_file() is keep
    assert sibling.is_dir()
