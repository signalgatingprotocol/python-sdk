import pytest

from scripts import check_release_version
from scripts.check_release_version import check_versions


@pytest.mark.parametrize("tag", [None, "0.1.0", "v0.1.0"])
def test_check_versions_accepts_matching_versions(tag: str | None) -> None:
    check_versions("0.1.0", "0.1.0", tag)


def test_check_versions_rejects_runtime_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match=r"distribution version 0\.2\.0 does not match runtime version 0\.1\.0",
    ):
        check_versions("0.2.0", "0.1.0")


def test_check_versions_rejects_release_tag_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match=r"release tag v0\.2\.0 does not match distribution version 0\.1\.0",
    ):
        check_versions("0.1.0", "0.1.0", "v0.2.0")


def test_main_reports_release_tag_mismatch_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(check_release_version, "version", lambda _: "0.1.0")
    monkeypatch.setattr(check_release_version.signal_gating, "__version__", "0.1.0")
    monkeypatch.setattr("sys.argv", ["check_release_version.py", "--tag", "v0.2.0"])

    with pytest.raises(SystemExit) as error:
        check_release_version.main()

    assert error.value.code == 1
    assert capsys.readouterr().err == (
        "error: release tag v0.2.0 does not match distribution version 0.1.0; "
        "publish tag 0.1.0 or v0.1.0\n"
    )
