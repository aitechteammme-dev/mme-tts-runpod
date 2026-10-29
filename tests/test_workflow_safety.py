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


def docker_runs(shell: str) -> list[str]:
    """Every `docker run ...` command, with backslash-continued lines joined into one."""
    return re.findall(r"docker run[^\n]*", shell.replace("\\\n", " "))


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
    built = "${{ !cancelled() && steps.build.outcome == 'success' }}"
    for name in ("Collect image evidence (pip freeze, pip check, docker history)", "Image checks (CPU only, container has no network)", "Prepare test tooling (the only network use after the build)"):
        assert by_name[name]["if"] == built, name                                        # run whenever an image exists, independent of each other
    tests = by_name["Test suite inside the image (real baked assets, container has no network)"]
    assert tests["if"] == "${{ !cancelled() && steps.build.outcome == 'success' && steps.tooling.outcome == 'success' }}"     # needs the tooling, but a tooling failure still fails the job
    assert by_name["Summary"]["if"] == "always()" and by_name["Remove the local image"]["if"] == "always()"
    upload = next(s for s in steps if s.get("uses", "").startswith("actions/upload-artifact"))
    assert upload["if"] == "always()" and upload["with"]["if-no-files-found"] == "warn"
    order = [s.get("name") for s in steps]
    assert order.index("Upload text logs (workflow artifact, not a registry)") > order.index("Test suite inside the image (real baked assets, container has no network)")


def test_check_containers_have_no_network():
    text = "\n".join(s.get("run", "") for s in all_steps(load(BUILD_ONLY)))      # shell code only, comments of the YAML excluded
    runs = docker_runs(text)
    with_net = [r for r in runs if "--network none" not in r]
    assert len(runs) >= 6 and len(with_net) == 1                                         # only the tooling installation has network
    prep = with_net[0]
    assert "--no-deps --require-hashes --target /pt -r /work/ci/requirements-test.txt" in prep and '-v "$PWD":/work:ro' in prep and "PYTHONPATH" not in prep
    assert "skip" not in text.lower().replace("skopeo", "")


def test_test_tooling_path_is_the_same_interpreter_mounted_read_only_and_never_prepended():
    wf = load(BUILD_ONLY)
    def shell(s):      # the shell commands only: explanatory comment lines may mention the words asserted absent below
        return "\n".join(line for line in s.splitlines() if not line.strip().startswith("#"))
    steps = {s["name"]: shell(s["run"]) for s in all_steps(wf) if "run" in s}
    prep = steps["Prepare test tooling (the only network use after the build)"]
    tests = steps["Test suite inside the image (real baked assets, container has no network)"]
    # the container that RUNS the tests must mount the very folder the tooling step filled (this was missing: PYTHONPATH pointed at an unmounted /pt), read-only
    assert '-v "$RUNNER_TEMP/pt":/pt:ro' in tests and "ci/run_tests.py" in tests and "--network none" in tests
    assert "PYTHONPATH" not in prep and "PYTHONPATH" not in tests and "python -m pytest" not in tests         # PYTHONPATH would PREPEND the tooling folder and could shadow locked packages
    # both containers start the image's own `python` (conda's), so pip installs for the interpreter that runs the tests
    assert '"$IMAGE" sh -c \'python -c' in prep and 'python -m pip install' in prep and '"$IMAGE" sh -c \'python ci/run_tests.py' in tests
    assert "pytest==" not in prep                                                      # versions live only in the hash-checked requirements file
    verify = [m for m in docker_runs(prep) if "--network none" in m]
    assert len(verify) == 1 and ":/pt:ro" in verify[0] and ":/work:ro" in verify[0] and "ci/run_tests.py --check-tooling" in verify[0]      # proven before the tests start
    # REGRESSION (CI run 40671308987): the check read `iniconfig.__version__`, which iniconfig 2.3.1 does not define. Versions must come from package metadata, never from a module attribute.
    for name in ("pytest", "pluggy", "iniconfig"):
        assert f"{name}.__version__" not in prep and f"{name}.__version__" not in tests
    assert ".__version__" not in prep and ".__version__" not in tests


def test_runner_gives_the_tooling_precedence_without_exposing_the_folder():
    """REGRESSION (CI): a sys.path entry for /pt loses to the base image's site-packages (pluggy 1.5.0 won over the pinned 1.7.0). Precedence must come from a meta-path finder for the tooling names only."""
    src = (ROOT / "ci" / "run_tests.py").read_text(encoding="utf-8")
    code_lines = "\n".join(line for line in src.splitlines() if not line.strip().startswith("#"))
    assert "sys.meta_path.insert(0" in code_lines
    assert not re.search(r"sys\.path\.(append|insert)\(\s*TOOLING_DIR", code_lines) and "PYTHONPATH" not in code_lines
    assert 'TOOLING_MODULES = {"pytest", "_pytest", "pluggy", "iniconfig", "py"}' in src                         # exactly the tooling names, nothing from the lock
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8").lower()
    for name in ("pytest", "_pytest", "pluggy", "iniconfig"):
        assert f"\n{name}==" not in lock


def test_test_requirements_are_pinned_hashed_and_cannot_replace_locked_packages():
    text = (ROOT / "ci" / "requirements-test.txt").read_text(encoding="utf-8")
    entries = re.split(r"\n(?=[A-Za-z0-9_.\-]+==)", text)[1:]
    names = [e.split("==")[0].lower() for e in entries]
    assert names == ["pytest", "pluggy", "iniconfig"]
    assert all(re.search(r"--hash=sha256:[0-9a-f]{64}", e) for e in entries)
    assert re.search(r"^pytest==9\.2\.0 ", text, re.M)
    lock = {m.group(1).lower().replace("_", "-") for m in re.finditer(r"^([A-Za-z0-9_.\-]+)==", (ROOT / "requirements.lock").read_text(encoding="utf-8"), re.M)}
    assert not set(names) & lock                                                       # nothing the image already locks is (re)installed by the tooling step
    assert "pytest" not in lock and "pytest" not in (ROOT / "requirements.txt").read_text(encoding="utf-8")     # the image itself carries no test tooling


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
