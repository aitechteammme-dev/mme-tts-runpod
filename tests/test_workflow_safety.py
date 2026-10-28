"""Static guards: the build-only workflow can never publish an image, the publishing workflow stays confined to tts-v3, and the Dockerfile/lock keep their integrity controls. No network, no Docker."""
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
BUILD_ONLY = ROOT / ".github" / "workflows" / "build-only-v4.yml"
PUBLISH = ROOT / ".github" / "workflows" / "build-image.yml"


def load(p):
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def code(p):
    """The workflow with its comments removed (parsed and re-dumped), so explanatory comments can say 'no docker push' without tripping the scans."""
    return yaml.safe_dump(load(p), width=10 ** 6)


def triggers(wf):
    return wf.get("on", wf.get(True))          # PyYAML (YAML 1.1) reads the key `on` as the boolean True


def all_steps(wf):
    return [s for j in wf["jobs"].values() for s in j["steps"]]


def test_build_only_triggers_are_manual_or_the_v4_branch_only():
    t = triggers(load(BUILD_ONLY))
    assert set(t) == {"workflow_dispatch", "push"}
    assert t["push"]["branches"] == ["tts-v4-streaming"]


def test_build_only_has_read_only_permissions_and_no_secrets():
    wf = load(BUILD_ONLY)
    assert wf["permissions"] == {"contents": "read"}
    for job in wf["jobs"].values():
        assert "permissions" not in job                                   # no job-level widening
    text = code(BUILD_ONLY)
    assert "secrets." not in text and "packages: write" not in text and "id-token" not in text


@pytest.mark.parametrize("forbidden", ["docker push", "docker login", "login-action", "build-push-action", "push: true", "docker tag", "ghcr.io", "docker.io/", "--push", "buildx imagetools", "skopeo", "crane "])
def test_build_only_contains_no_publishing_construct(forbidden):
    assert forbidden not in code(BUILD_ONLY)


def test_build_only_image_tag_is_local_only_and_never_a_release_tag():
    env = load(BUILD_ONLY)["env"]
    assert re.fullmatch(r"mme-tts-v4-buildcheck:run-\$\{\{ github\.run_id \}\}-\$\{\{ github\.run_attempt \}\}", env["IMAGE"])      # no registry host, no slash
    text = code(BUILD_ONLY)
    for tag in (":v4", ":latest", ":v2", ":v3"):
        assert tag not in text.replace("mme-tts-v4-buildcheck", "")


def test_build_only_actions_are_the_two_allowed_ones_pinned_by_sha():
    uses = [s["uses"] for s in all_steps(load(BUILD_ONLY)) if "uses" in s]
    assert {u.split("@")[0] for u in uses} == {"actions/checkout", "actions/upload-artifact"}
    assert all(re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", u) for u in uses)
    checkout = next(s for s in all_steps(load(BUILD_ONLY)) if s.get("uses", "").startswith("actions/checkout"))
    assert checkout["with"]["persist-credentials"] is False


def test_build_only_has_time_and_disk_safeguards():
    wf = load(BUILD_ONLY)
    job = wf["jobs"]["build-only"]
    assert 0 < job["timeout-minutes"] <= 90 and all("timeout-minutes" in s for s in job["steps"] if s.get("name", "").startswith(("Build", "Image checks", "Test suite")))
    pre = next(s for s in job["steps"] if s.get("name", "").startswith("Preflight"))
    assert "MIN_FREE_GB" in pre["run"] and "exit 1" in pre["run"] and int(wf["env"]["MIN_FREE_GB"]) >= 30
    assert wf["concurrency"]["cancel-in-progress"] is True


def test_failures_are_never_masked_and_evidence_is_kept():
    wf = load(BUILD_ONLY)
    steps = all_steps(wf)
    assert "continue-on-error" not in code(BUILD_ONLY)                                  # a failing step must fail the job
    build = next(s for s in steps if s.get("id") == "build")
    assert "if" not in build                                                            # default success(): skipped if the preflight failed
    by_name = {s["name"]: s for s in steps}
    for name in ("Collect image evidence (pip freeze, pip check, docker history)", "Image checks (CPU only, container has no network)", "Test suite inside the image (real baked assets, container has no network)"):
        assert by_name[name]["if"] == "${{ !cancelled() && steps.build.outcome == 'success' }}", name      # run whenever an image exists, independent of each other
    assert by_name["Summary"]["if"] == "always()" and by_name["Remove the local image"]["if"] == "always()"
    upload = next(s for s in steps if s.get("uses", "").startswith("actions/upload-artifact"))
    assert upload["if"] == "always()" and upload["with"]["if-no-files-found"] == "warn"
    order = [s.get("name") for s in steps]
    assert order.index("Upload text logs (workflow artifact, not a registry)") > order.index("Test suite inside the image (real baked assets, container has no network)")


def test_check_containers_have_no_network():
    text = "\n".join(s.get("run", "") for s in all_steps(load(BUILD_ONLY)))      # shell code only, comments of the YAML excluded
    runs = [m.group(0) for m in re.finditer(r"docker run[^\n]*(\\\n[^\n]*)*", text)]
    with_net = [r for r in runs if "--network none" not in r]
    assert len(runs) >= 5 and len(with_net) == 1 and "pip install --quiet --target /pt pytest==9.2.0" in with_net[0]      # only the test-tooling install has network
    assert "skip" not in text.lower().replace("skopeo", "")


def test_publishing_workflow_is_unchanged_in_scope():
    wf = load(PUBLISH)
    assert triggers(wf)["push"]["branches"] == ["tts-v3"]
    job = next(iter(wf["jobs"].values()))
    assert job["if"] == "github.ref == 'refs/heads/tts-v3'"
    tags = [s["with"]["tags"] for s in job["steps"] if s.get("uses", "").startswith("docker/build-push-action")][0]
    assert sorted(t.strip() for t in tags.strip().splitlines()) == ["ghcr.io/aitechteammme-dev/mme-tts-runpod:v4", "ghcr.io/aitechteammme-dev/mme-tts-runpod:v4-${{ github.sha }}"]


def test_dockerfile_integrity_controls():
    d = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^FROM pytorch/pytorch:2\.7\.1-cuda12\.8-cudnn9-runtime@sha256:[0-9a-f]{64}$", d, re.M)
    assert "pip install --no-cache-dir --require-hashes --no-deps -r requirements.lock" in d
    assert "RUN --network=none python -c" in d and "verify_tokenizer_offline" in d
    assert "sys.version_info[:2] == (3, 11)" in d
    # no torch reinstall: explicit exact assertions instead, before and after the installs (CUDA 12.8 build, torch and torchaudio 2.7.1+cu128)
    assert "--force-reinstall" not in d and "download.pytorch.org" not in d
    assert d.count("assert torch.__version__ == '2.7.1+cu128' and torch.version.cuda == '12.8'") == 2 and d.count("assert torchaudio.__version__ == '2.7.1+cu128'") == 2
    assert d.index("assert torch.__version__") < d.index("pip install --no-cache-dir --require-hashes") < d.rindex("assert torch.__version__")
    # conda diagnostic runs before the hash-locked install
    assert d.index("RUN python /tmp/conda_diagnostics.py requirements.lock") < d.index("pip install --no-cache-dir --require-hashes")


def test_every_lock_entry_has_hashes():
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    entries = re.split(r"\n(?=[A-Za-z0-9_.\-]+==)", lock)[1:]
    assert len(entries) >= 150
    bad = [e.split("==")[0] for e in entries if not re.search(r"--hash=sha256:[0-9a-f]{64}", e)]
    assert not bad, bad
