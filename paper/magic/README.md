# Reproducing SOFT Paper Results with UCC

This directory contains scripts to reproduce Table III and Table IV
from the SOFT paper (Li et al., arXiv 2512.23037) using UCC as a
CPU-only alternative to the SOFT GPU simulator.

## Quick Start (Local)

```bash
# Validate discard rates against Table IV (all 6 noise levels, ~2 min)
uv run python paper/magic/run_vs_soft.py --shots 1000000

# Local correctness check (~6 hours, observes ~5 logical errors)
uv run python paper/magic/validate_local.py --max-errors 5
```

## Cloud Reproduction (AWS)

The full Table III reproduction requires billions of shots. The
workflow has two phases: a quick **benchmark** (30 min, ~$0.75)
to measure real throughput, then optional **production runs**.

### Prerequisites

- An AWS account with EC2 access
- AWS CLI installed locally (`brew install awscli` or `pip install awscli`)
- A GitHub fine-grained personal access token with read access to
  `unitaryfoundation/ucc-next` (repo is private)
- An EC2 key pair in your target region

### 1. Create a GitHub Token

1. Go to https://github.com/settings/tokens?type=beta
2. Click **Generate new token**
3. Name: `ucc-cloud-run` (or similar)
4. Repository access: **Only select repositories** → `unitaryfoundation/ucc-next`
5. Permissions: **Contents: Read-only**
6. Generate and save the token (starts with `github_pat_...`)

### 2. Launch the EC2 Instance

```bash
# Find the latest Amazon Linux 2023 AMI
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=al2023-ami-2023.*-x86_64" \
            "Name=state,Values=available" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text \
  --region us-east-1)

echo "Using AMI: $AMI_ID"

# Launch a spot instance (c7i.24xlarge: 48 vCPUs, 192 GB RAM)
# ~$1.50/hr spot vs ~$4.08/hr on-demand
aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type c7i.24xlarge \
  --key-name your-key-pair-name \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ucc-soft-repro}]' \
  --region us-east-1
```

> **Note:** Replace `your-key-pair-name` with your EC2 key pair.
> The spot instance is set to "stop" on interruption (not terminate),
> so the EBS volume and resume files are preserved. Just restart it.

### 3. Set Up the Instance

SSH in and run the setup script:

```bash
ssh -i your-key.pem ec2-user@<instance-ip>

# Set your GitHub token
export GITHUB_TOKEN=github_pat_...

# Install git (not included in Amazon Linux 2023 by default)
sudo dnf install -y git

# Clone and set up
git clone "https://${GITHUB_TOKEN}@github.com/unitaryfoundation/ucc-next.git"
cd ucc-next
bash paper/magic/cloud_setup.sh
```

**512-qubit build:** To compile with support for up to 512 qubits
(wider Pauli bitmasks, needed for future larger circuits), pass
`--max-qubits 512`. This increases `BitMask` from 8 bytes to 64
bytes, which may slightly slow frame-heavy operations:

```bash
bash paper/magic/cloud_setup.sh --max-qubits 512
```

### 4. Run the Benchmark

Measure real throughput before committing to long runs (~10 min):

```bash
cd ~/ucc-next
uv run python paper/magic/benchmark_cloud.py --duration 60
```

This prints measured shots/s at each noise level and projected
costs. Use these numbers to decide on error targets.

**You can stop here.** The benchmark gives you the information
needed to decide whether to proceed with production runs, run
locally on a fast laptop instead, or both.

---

## Production Runs (Optional)

If the benchmark numbers look good and you want to run production
sampling, continue with the steps below.

### 5. Set Up S3 Background Sync (Optional)

For long runs, background S3 sync provides crash insurance beyond
Sinter's local resume files. This requires an S3 bucket and an
IAM role.

**Create an S3 bucket:**

```bash
# Pick a unique bucket name
export S3_BUCKET=ucc-soft-results-yourname

aws s3 mb s3://$S3_BUCKET --region us-east-1
```

**Create an IAM role** (lets the instance write to S3 without
storing AWS credentials on disk):

```bash
# Create the trust policy (allows EC2 to assume this role)
cat > /tmp/ec2-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

# Create the role
aws iam create-role \
  --role-name ucc-cloud-runner \
  --assume-role-policy-document file:///tmp/ec2-trust-policy.json

# Create a policy that only allows writing to our bucket
cat > /tmp/s3-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::$S3_BUCKET",
      "arn:aws:s3:::$S3_BUCKET/*"
    ]
  }]
}
EOF

aws iam put-role-policy \
  --role-name ucc-cloud-runner \
  --policy-name s3-results-access \
  --policy-document file:///tmp/s3-policy.json

# Create an instance profile and attach the role
aws iam create-instance-profile --instance-profile-name ucc-cloud-runner
aws iam add-role-to-instance-profile \
  --instance-profile-name ucc-cloud-runner \
  --role-name ucc-cloud-runner

# Wait for IAM propagation, then attach to the running instance
sleep 10
aws ec2 associate-iam-instance-profile \
  --instance-id <instance-id> \
  --iam-instance-profile Name=ucc-cloud-runner
```

**On the instance**, set up the sync timer:

```bash
bash paper/magic/setup_s3_sync.sh $S3_BUCKET
```

This installs a systemd timer that syncs `results/` to S3 every
5 minutes at idle I/O priority. It will never slow down the
sampling runs.

Verify it's running:
```bash
systemctl status ucc-s3-sync.timer
aws s3 ls s3://$S3_BUCKET/$(hostname)/
```

### 6. Run Production Sampling

Use `tmux` so runs survive SSH disconnects:

```bash
tmux new -s ucc

# Fastest point first
uv run python paper/magic/run_cloud.py --noise 0.002 --max-errors 100

# Then (adjust --max-errors based on benchmark cost projections)
uv run python paper/magic/run_cloud.py --noise 0.001 --max-errors 50
uv run python paper/magic/run_cloud.py --noise 0.0005 --max-errors 8
```

Each run:
- Auto-detects physical cores and uses them as Sinter workers
- Saves progress to `results/resume_d5_p<noise>.csv` (crash-resilient)
- Prints stats every 60 seconds
- On completion, writes `results/results_d5_p<noise>.csv`

If the spot instance is interrupted, just restart it and re-run the
same command — Sinter picks up from the resume file automatically.

### 7. Download Results

Results are continuously synced to S3 (if configured). Download:

```bash
# From S3
aws s3 sync s3://$S3_BUCKET/ ./cloud_results/

# Or directly from the instance
scp -i your-key.pem ec2-user@<instance-ip>:~/ucc-next/results/*.csv ./cloud_results/
```

### 8. Clean Up

```bash
# Terminate the instance
aws ec2 terminate-instances --instance-ids <instance-id>

# If you created S3/IAM resources:
aws s3 rb s3://$S3_BUCKET --force
aws iam remove-role-from-instance-profile \
  --instance-profile-name ucc-cloud-runner --role-name ucc-cloud-runner
aws iam delete-instance-profile --instance-profile-name ucc-cloud-runner
aws iam delete-role-policy --role-name ucc-cloud-runner --policy-name s3-results-access
aws iam delete-role --role-name ucc-cloud-runner

# Revoke the GitHub token at:
# https://github.com/settings/tokens
```

## Cost Reference

**Instance:** c7i.24xlarge (48 vCPUs / 24 physical cores, Sapphire
Rapids, AVX-512 + BMI2)

| Pricing | $/hr |
|---|---|
| On-demand | ~$4.08 |
| Spot | ~$1.50 (varies) |

Estimated costs depend on measured throughput. Run
`benchmark_cloud.py` for projections with real numbers. Rough
estimates from dev VM (extrapolated):

| Configuration | Wall time (24 cores) | Spot cost |
|---|---|---|
| p=0.002, 100 errors | ~16h | ~$24 |
| p=0.001, 49 errors | ~23h | ~$34 |
| p=0.0005, 8 errors | ~80h | ~$120 |

## Scripts Reference

| Script | Purpose |
|---|---|
| `run_vs_soft.py` | Quick local validation of discard rates (Table IV) |
| `validate_local.py` | Local correctness check (observe logical errors) |
| `benchmark_cloud.py` | Measure cloud throughput, project costs |
| `cloud_setup.sh` | Bootstrap EC2 instance |
| `setup_s3_sync.sh` | Background S3 sync (systemd timer) |
| `run_cloud.py` | Production runner for a single noise level |
| `ucc_soft_sampler.py` | Sinter adapter for UCC |

## Circuits

Vendored from https://github.com/haoliri0/SOFT (Apache-2.0):

| File | Qubits | Detectors | Notes |
|---|---|---|---|
| `circuit_d5_p0.0005.stim` | 42 | 107 | Table III |
| `circuit_d5_p0.001.stim` | 42 | 107 | Table III |
| `circuit_d5_p0.002.stim` | 42 | 107 | Table III |
| `circuit_d5_p0.003.stim` | 42 | 107 | Table IV only |
| `circuit_d5_p0.004.stim` | 42 | 107 | Table IV only |
| `circuit_d5_p0.005.stim` | 42 | 107 | Table IV only |
| `circuit_d3_p0.001.stim` | 15 | 20 | Not in paper |

All d=5 circuits contain T/T_DAG gates (non-Clifford), which Stim
cannot simulate but UCC handles natively.

## Appendix: Benchmark Results

### Apple M4 Pro (14 cores: 10P + 4E)

MacBook Pro, battery power. Workers pinned to 10 performance cores.

| p | Workers | Shots/s | Surv/s | Discard% | Scaling |
|---|---|---|---|---|---|
| 0.0020 | 1 | 216,660 | 4,528 | 97.91% | -- |
| 0.0020 | 10 | 1,957,573 | 40,782 | 97.92% | 9.0x |
| 0.0010 | 1 | 82,847 | 11,941 | 85.59% | -- |
| 0.0010 | 10 | 757,733 | 109,032 | 85.61% | 9.1x |
| 0.0005 | 1 | 46,861 | 17,778 | 62.06% | -- |
| 0.0005 | 10 | 415,890 | 157,723 | 62.08% | 8.9x |

### AWS c7i.16xlarge (64 vCPUs / 32 physical cores, Sapphire Rapids)

Amazon Linux 2023, AVX-512 + BMI2. 60s per test. 64-qubit build.

| p | Workers | Shots/s | Surv/s | Discard% | Scaling |
|---|---|---|---|---|---|
| 0.0020 | 1 | 141,838 | 2,954 | 97.92% | -- |
| 0.0020 | 32 | 4,062,958 | 84,762 | 97.91% | 28.6x |
| 0.0010 | 1 | 57,718 | 8,299 | 85.62% | -- |
| 0.0010 | 32 | 1,692,923 | 243,630 | 85.61% | 29.3x |
| 0.0005 | 1 | 33,039 | 12,525 | 62.09% | -- |
| 0.0005 | 32 | 964,100 | 365,619 | 62.08% | 29.2x |

### AWS c7i.24xlarge (96 vCPUs / 48 physical cores, Sapphire Rapids)

Amazon Linux 2023, AVX-512 + BMI2. 60s per test. 512-qubit build
(`--max-qubits 512`, 64-byte BitMask).

| p | Workers | Shots/s | Surv/s | Discard% | Scaling |
|---|---|---|---|---|---|
| 0.0020 | 1 | 175,466 | 3,655 | 97.92% | -- |
| 0.0020 | 48 | 8,129,818 | 169,219 | 97.92% | 46.3x |
| 0.0010 | 1 | 68,566 | 9,859 | 85.62% | -- |
| 0.0010 | 48 | 3,260,017 | 469,252 | 85.61% | 47.5x |
| 0.0005 | 1 | 37,964 | 14,401 | 62.07% | -- |
| 0.0005 | 48 | 1,813,658 | 687,772 | 62.08% | 47.8x |

### Per-Core Comparison

| p | M4 Pro | c7i (Sapphire Rapids) | Ratio |
|---|---|---|---|
| 0.0020 | 216,660 | 175,466 | M4 Pro 1.23x faster |
| 0.0010 | 82,847 | 68,566 | M4 Pro 1.21x faster |
| 0.0005 | 46,861 | 37,964 | M4 Pro 1.23x faster |

### Cost Projections (c7i.24xlarge, 48 workers, 512-qubit build)

Spot price ~$1.50/hr for c7i.24xlarge.

| Target | p=0.002 | p=0.001 | p=0.0005 | Total |
|---|---|---|---|---|
| Match paper (22/49/8 errors) | 1h, $2 | 6h, $9 | 21h, $31 | **28h, $42** |
| 100 errors each | 5h, $7 | 13h, $19 | 257h, $386 | **275h, $413** |
