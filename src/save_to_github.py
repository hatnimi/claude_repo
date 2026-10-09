"""Colab에서 지금 화면의 노트북과 실행 결과(outputs/)를 GitHub에 저장한다.

노트북 셀에서 이렇게 부른다.
    import sys; sys.path.insert(0, "/content/claude_repo/src")
    import importlib, save_to_github; importlib.reload(save_to_github); save_to_github.save()

저장 방식
  1. 이전 저장이 중간에 멈춘 흔적(rebase)이 있으면 정리한다.
  2. 화면의 노트북을 파일로 쓴다. 출력에 찍힌 Google API 키는 가린다.
  3. GitHub의 최신 상태를 받아 그 위에 '노트북과 outputs/만' 얹어 커밋하고 올린다.
     코드·문서는 GitHub 쪽 최신 버전을 그대로 쓰므로, 다른 곳에서 고친 내용과 부딪히지 않는다.
  4. Colab의 코드·문서도 GitHub 최신 상태로 맞춘다.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = "hatnimi/claude_repo"
BRANCH = "claude/clever-lamport-uucwuo"
WORKDIR = Path("/content/claude_repo")
NOTEBOOK = "tts_vowel_length.ipynb"
SAVE_CELL = '''import sys; sys.path.insert(0, "/content/claude_repo/src")
import importlib, save_to_github; importlib.reload(save_to_github)
save_to_github.save(message="Colab에서 실행 결과 저장")   # 커밋 메시지는 바꿔도 된다'''
API_KEY = re.compile(r"AIza[0-9A-Za-z_\-]{35}")


def _upgrade_save_cell(nb: dict) -> bool:
    """예전 저장 셀(코드가 셀 안에 길게 들어 있던 것)을 이 모듈을 부르는 짧은 셀로 바꾼다."""
    changed = False
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", ""))
        if cell.get("cell_type") == "code" and "get_ipynb" in src and "save_to_github" not in src:
            cell["source"] = SAVE_CELL
            changed = True
        if cell.get("cell_type") == "markdown" and "`docs/탐구노트.md`, `outputs/`" in src:
            cell["source"] = src.replace("`docs/탐구노트.md`, `outputs/` 결과 파일을", "`outputs/` 결과 파일을")
            changed = True
    return changed


def notebook_text(nb: dict) -> tuple[str, int]:
    """노트북을 저장할 글자로 바꾸고, 출력에 남은 API 키를 가린다."""
    return API_KEY.subn("AIza***(가림)", json.dumps(nb, ensure_ascii=False, indent=1))


def save(message: str = "Colab에서 실행 결과 저장", workdir: Path = WORKDIR, nb: dict | None = None,
         token: str | None = None, remote: str | None = None) -> bool:
    if nb is None or token is None:
        from google.colab import _message, userdata
        nb = nb or _message.blocking_request("get_ipynb", timeout_sec=120)["ipynb"]
        token = token or userdata.get("GITHUB_TOKEN")
    remote = remote or f"https://x-access-token:{token}@github.com/{REPO}.git"

    def git(*args: str, quiet: bool = False) -> int:
        r = subprocess.run(["git", "-C", str(workdir), *args], capture_output=True, text=True)
        out = (r.stdout + r.stderr).replace(token, "***").strip()
        if out and not quiet:
            print(out)
        return r.returncode

    # 1) 이전 저장이 중간에 멈춘 흔적 정리
    #    rebase를 취소하면 그 사이에 생긴 결과 파일이 지워질 수 있어서, outputs/를 먼저 따로 복사해 둔다
    backup = Path(tempfile.mkdtemp()) / "outputs"
    if (workdir / "outputs").exists():
        shutil.copytree(workdir / "outputs", backup)
    git("rebase", "--abort", quiet=True)
    git("merge", "--abort", quiet=True)
    for d in ("rebase-merge", "rebase-apply"):
        shutil.rmtree(workdir / ".git" / d, ignore_errors=True)
    if backup.exists():
        shutil.copytree(backup, workdir / "outputs", dirs_exist_ok=True)

    # 2) 화면의 노트북 → 파일 (예전 저장 셀은 새 셀로 교체, API 키는 가림)
    if _upgrade_save_cell(nb):
        print("🔧 저장 셀을 새 버전으로 바꿔서 저장합니다.")
    text, n_keys = notebook_text(nb)
    if n_keys:
        print(f"🔒 노트북 출력에 있던 API 키 {n_keys}곳을 가리고 저장합니다.")
    (workdir / NOTEBOOK).write_text(text, encoding="utf-8")

    # 3) GitHub 최신 상태 위에 노트북과 outputs/만 얹어서 커밋
    git("config", "user.name", "hatnimi")
    git("config", "user.email", "176273713+hatnimi@users.noreply.github.com")
    if git("fetch", "-q", remote, BRANCH) != 0:
        print("❌ GitHub에서 최신 상태를 받지 못했습니다. GITHUB_TOKEN을 확인하세요.")
        return False
    git("reset", "-q", "--mixed", "FETCH_HEAD")        # 기록은 GitHub 최신으로, 내 파일은 그대로 둔다
    git("add", NOTEBOOK, "outputs")
    if git("diff", "--cached", "--quiet", quiet=True) == 0:
        print("새로 바뀐 내용이 없습니다.")
    else:
        git("commit", "-q", "-m", message)
    ok = git("push", "-q", remote, f"HEAD:{BRANCH}") == 0

    # 4) 코드·문서도 GitHub 최신과 맞추고, 브랜치에 다시 연결 (다음에 첫 셀의 git pull이 되도록)
    git("checkout", "-q", "-B", BRANCH, quiet=True)
    git("reset", "-q", "--hard", "HEAD")
    git("branch", "-q", f"--set-upstream-to=origin/{BRANCH}", quiet=True)
    print("✅ 저장 완료" if ok else "❌ 저장 실패: 위 메시지 확인")
    return ok
