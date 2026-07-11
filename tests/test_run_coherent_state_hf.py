from pathlib import Path
import subprocess


def test_git_porcelain_all_lists_leaf_run_files_for_resume(tmp_path):
    subprocess.run(["git", "init", "-q", "-b", "trunk"], cwd=tmp_path,
                   check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"],
                   cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path,
                   check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("x")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    run = tmp_path / "results" / "coherent_state" / "run_001"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text("{}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=tmp_path, text=True).splitlines()
    assert status == ["?? results/coherent_state/run_001/manifest.json"]
    raw = status[0][3:]
    path = (tmp_path / raw).resolve()
    assert run.resolve() in path.parents
