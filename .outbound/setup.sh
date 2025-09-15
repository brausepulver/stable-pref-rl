echo "TORCH_NUM_THREADS=1" >> /etc/environment

apt update && apt install -y pipx unzip

cd /root

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

cd /opt/workspace

git clone https://token:$GITHUB_TOKEN@github.com/brausepulver/pref_rl.git .

aws sts get-caller-identity  # Verify AWS CLI installation
aws s3 cp s3://$S3_BUCKET/validation_data.pkl /opt/workspace/data/validation_data.pkl

pipx install uv
pipx ensurepath

export UV_CACHE_DIR=.cache/
/root/.local/bin/uv sync
