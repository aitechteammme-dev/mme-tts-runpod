"""ci/conda_diagnostics.py against fake site-packages (no conda, no pip, no network)."""
import importlib.util
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("conda_diagnostics", ROOT / "ci" / "conda_diagnostics.py")
cd = importlib.util.module_from_spec(spec)
sys.modules["conda_diagnostics"] = cd
spec.loader.exec_module(cd)
REAL_DISTRIBUTIONS = metadata.distributions


def fake_dist(site: Path, name: str, version: str, record=True, installer="conda"):
    d = site / f"{name}-{version}.dist-info"
    d.mkdir(parents=True)
    (d / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")
    if record:
        (d / "RECORD").write_text("", encoding="utf-8")
    if installer:
        (d / "INSTALLER").write_text(installer + "\n", encoding="utf-8")


def setup(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    meta = tmp_path / "conda-meta"
    meta.mkdir()
    fake_dist(site, "Replaced_With_Record", "1.0", record=True)          # differs from the lock, has RECORD: replaceable
    fake_dist(site, "replaced-no-record", "1.0", record=False)           # differs from the lock, no RECORD: pip cannot uninstall it
    fake_dist(site, "same-version-no-record", "2.0", record=False)       # same as the lock: pip leaves it alone
    fake_dist(site, "not-in-lock", "9.9", record=False)
    fake_dist(site, "pip-owned", "1.0", record=True, installer="pip")
    (meta / "replaced_with_record-1.0-pyh_0.json").write_text("{}")
    (meta / "replaced-no-record-1.0-py311_1.json").write_text("{}")
    lock = tmp_path / "lock"
    lock.write_text("# c\nReplaced.With.Record==2.0 \\\n    --hash=sha256:" + "a" * 64 + "\nreplaced_no_record==2.0 \\\n    --hash=sha256:" + "b" * 64 +
                    "\nsame-version-no-record==2.0\npip-owned==3.0\nmissing-from-image==1.0\n", encoding="utf-8")
    return site, meta, lock


def test_rows_classify_replacement_ownership_and_record(tmp_path):
    site, meta, lock = setup(tmp_path)
    rows = {r["name"]: r for r in cd.diagnose(cd.lock_pins(lock), metadata.distributions(path=[str(site)]), cd.conda_owned(meta))}
    assert set(rows) == {"replaced-with-record", "replaced-no-record", "same-version-no-record", "pip-owned"}          # not-in-lock and missing-from-image are not overlaps
    assert rows["replaced-with-record"]["replaced"] and rows["replaced-with-record"]["conda_meta"] and not rows["replaced-with-record"]["risk"]
    assert rows["replaced-no-record"]["risk"] and rows["replaced-no-record"]["conda_meta"]
    assert not rows["same-version-no-record"]["replaced"] and not rows["same-version-no-record"]["risk"]
    assert rows["pip-owned"]["installer"] == "pip" and not rows["pip-owned"]["conda_meta"]


def test_main_fails_only_when_pip_is_certain_to_fail(tmp_path, monkeypatch, capsys):
    site, meta, lock = setup(tmp_path)
    monkeypatch.setattr(cd.metadata, "distributions", lambda: REAL_DISTRIBUTIONS(path=[str(site)]))
    assert cd.main([str(lock), "--conda-meta", str(meta)]) == 1
    captured = capsys.readouterr()
    assert "replaced-no-record" in captured.err and "replaced-with-record" not in captured.err
    assert "pip cannot uninstall this" in captured.out
    (site / "replaced-no-record-1.0.dist-info" / "RECORD").write_text("")              # now it has RECORD
    assert cd.main([str(lock), "--conda-meta", str(meta)]) == 0


def test_main_reports_the_whole_overlap_even_without_conda_meta(tmp_path, monkeypatch, capsys):
    site, meta, lock = setup(tmp_path)
    (site / "replaced-no-record-1.0.dist-info" / "RECORD").write_text("")
    monkeypatch.setattr(cd.metadata, "distributions", lambda: REAL_DISTRIBUTIONS(path=[str(site)]))
    assert cd.main([str(lock), "--conda-meta", str(tmp_path / "absent")]) == 0
    out = capsys.readouterr().out
    assert "conda-meta absent" in out and "overlapping installed distributions 4" in out and "pip will not re-check its hash" in out


def test_the_diagnostic_does_not_modify_anything(tmp_path, monkeypatch):
    site, meta, lock = setup(tmp_path)
    before = sorted(p.name for p in tmp_path.rglob("*"))
    lock_text = lock.read_text()
    monkeypatch.setattr(cd.metadata, "distributions", lambda: REAL_DISTRIBUTIONS(path=[str(site)]))
    cd.main([str(lock), "--conda-meta", str(meta)])
    assert sorted(p.name for p in tmp_path.rglob("*")) == before and lock.read_text() == lock_text
