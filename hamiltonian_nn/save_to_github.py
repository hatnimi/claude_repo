"""Colab에서 지금 화면의 노트북과 실행 결과(outputs/)를 GitHub에 저장한다.

노트북 마지막 셀에서 이렇게 부른다.
    import importlib, save_to_github; importlib.reload(save_to_github); save_to_github.save()

필요한 것: Colab 왼쪽 🔑 보안 비밀에 GITHUB_TOKEN (저장소 쓰기 권한)

저장 방식
  1. 화면의 노트북(실행 결과 포함)을 파일로 쓴다.
  2. GitHub의 최신 상태를 받아 그 위에 '노트북과 outputs/만' 얹어 커밋하고 올린다.
     다른 파일은 GitHub 쪽 최신 버전을 그대로 쓰므로, 다른 곳에서 고친 내용과 부딪히지 않는다.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = "hatnimi/claude_repo"
BRANCH = "claude/relaxed-knuth-q3zflq"
WORKDIR = Path("/content/hamiltonian_repo")
NOTEBOOK = "hamiltonian_nn/hamiltonian_nn.ipynb"
OUTPUTS = "hamiltonian_nn/outputs"


def save(message: str = "해밀턴 경로 신경망: Colab 실행 결과 저장", workdir: Path = WORKDIR,
         nb: dict | None = None, token: str | None = None, remote: str | None = None) -> bool:
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

    # 1) 화면의 노트북 → 파일
    (workdir / NOTEBOOK).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    # 2) GitHub 최신 상태 위에 노트북과 outputs/만 얹어서 커밋
    git("config", "user.name", "hatnimi")
    git("config", "user.email", "176273713+hatnimi@users.noreply.github.com")
    if git("fetch", "-q", remote, BRANCH) != 0:
        print("❌ GitHub에서 최신 상태를 받지 못했습니다. GITHUB_TOKEN을 확인하세요.")
        return False
    git("reset", "-q", "--mixed", "FETCH_HEAD")      # 기록은 GitHub 최신으로, 내 파일은 그대로 둔다
    git("add", NOTEBOOK)
    git("add", "--ignore-removal", OUTPUTS)          # Colab에 없는 결과 파일을 GitHub에서 지우지 않는다
    if git("diff", "--cached", "--quiet", quiet=True) == 0:
        print("새로 바뀐 내용이 없습니다.")
    else:
        git("commit", "-q", "-m", message)
    ok = git("push", "-q", remote, f"HEAD:{BRANCH}") == 0

    # 3) 나머지 파일도 GitHub 최신과 맞추고 브랜치에 다시 연결 (다음 실행 때 첫 셀의 git pull이 되도록)
    git("checkout", "-q", "-B", BRANCH, quiet=True)
    git("reset", "-q", "--hard", "HEAD")
    git("branch", "-q", f"--set-upstream-to=origin/{BRANCH}", quiet=True)
    print("✅ 저장 완료" if ok else "❌ 저장 실패: 위 메시지 확인")
    return ok
