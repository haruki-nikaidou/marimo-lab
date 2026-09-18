import pytest

from marimo_lab import DATA_DIR, PROJECT_ROOT, data_path


def test_data_path_is_independent_of_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert data_path("raw", "sample.csv") == DATA_DIR.resolve() / "raw" / "sample.csv"


def test_project_root_holds_the_pyproject():
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


@pytest.mark.parametrize("escape", ["../secrets.env", "raw/../../etc/passwd"])
def test_data_path_rejects_escapes(escape):
    with pytest.raises(ValueError):
        data_path(escape)
