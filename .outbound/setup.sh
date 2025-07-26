echo "TORCH_NUM_THREADS=1" >> /etc/environment

cd /opt/workspace

git clone https://token:$GITHUB_TOKEN@github.com/brausepulver/pref_rl.git .
aws s3 cp s3://$S3_BUCKET/validation_data.pkl /opt/workspace/data/validation_data.pkl

export UV_CACHE_DIR=.cache/
uv sync
