"""Deploy lên Hugging Face Spaces (Docker SDK).

Chạy: make deploy-hf SPACE=<username>/tasco-semantic-search
Cần đăng nhập trước:  .venv/bin/huggingface-cli login   (hoặc export HF_TOKEN=hf_...)

Upload toàn bộ repo (trừ artefact local) rồi ghi đè README.md bằng bản có
front-matter Space (deploy/README_SPACE.md). Build diễn ra server-side theo
Dockerfile — model + embedding cache bake lúc build, runtime offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent

IGNORE = [
    ".git*", ".venv*", "__pycache__", "*.pyc",
    "data/cache/*", "eval/reports/*.json", ".env", ".dockerignore",
]


def main() -> None:
    if len(sys.argv) < 2 or "/" not in sys.argv[1]:
        sys.exit("Cách dùng: python deploy/deploy_hf.py <username>/<space-name>")
    repo_id = sys.argv[1]

    api = HfApi()
    who = api.whoami()  # fail sớm nếu chưa login
    print(f"Đăng nhập HF: {who['name']}")

    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker",
                    exist_ok=True)
    print(f"Space: https://huggingface.co/spaces/{repo_id}")

    print("Upload repo …")
    api.upload_folder(folder_path=str(ROOT), repo_id=repo_id, repo_type="space",
                      ignore_patterns=IGNORE,
                      commit_message="deploy: tasco-semantic-search")
    # README của Space (front-matter sdk: docker) đè lên README submission
    api.upload_file(path_or_fileobj=str(ROOT / "deploy" / "README_SPACE.md"),
                    path_in_repo="README.md", repo_id=repo_id, repo_type="space",
                    commit_message="deploy: Space README (front-matter)")

    print("\n✓ Đã push. HF đang build image (~5-10 phút — model 450MB bake lúc build).")
    print(f"  Theo dõi build log: https://huggingface.co/spaces/{repo_id} → tab Logs")
    print(f"  Khi chạy: UI tại https://huggingface.co/spaces/{repo_id} , ")
    print(f"  API trực tiếp: https://{repo_id.replace('/', '-')}.hf.space/v1/search?q=cafe")


if __name__ == "__main__":
    main()
