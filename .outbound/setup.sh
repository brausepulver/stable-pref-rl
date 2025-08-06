echo "TORCH_NUM_THREADS=1" >> /etc/environment

cd /opt/workspace

git clone https://token:$GITHUB_TOKEN@github.com/brausepulver/pref_rl.git .
git checkout exp/synth-feedback-2

snap install --classic aws-cli
aws sts get-caller-identity  # Verify AWS CLI installation
aws s3 cp s3://$S3_BUCKET/validation_data.pkl /opt/workspace/data/validation_data.pkl

apt update && apt install -y pipx
pipx install uv
pipx ensurepath

export UV_CACHE_DIR=.cache/
/root/.local/bin/uv sync
