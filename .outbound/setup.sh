echo "TORCH_NUM_THREADS=1" >> /etc/environment

cd /opt/workspace

git clone https://token:$GITHUB_TOKEN@github.com/brausepulver/pref_rl.git .

apt update && apt install -y git-lfs
git lfs install
git lfs pull

export UV_CACHE_DIR=.cache/
uv sync
