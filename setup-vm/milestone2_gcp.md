# Milestone 2 — GCP GPU setup (L4, the spec's preferred GPU)

Use this if you still have GCP credits from milestone 1 and want the
spec-specified L4 GPU.

## 1. Create a VM

[GCP Console → Compute Engine → Create instance](https://console.cloud.google.com/compute/instancesAdd):

- **Region/zone:** `us-central1-a` (or any region with L4 stock — try `europe-west4`, `us-east4`, `asia-southeast1` if `us-central1` is empty)
- **Machine configuration → GPU:**
  - GPU type: **NVIDIA L4**
  - Number of GPUs: **1**
  - Machine type: **g2-standard-8** (8 vCPU, 32 GB) is the canonical pairing.
- **Provisioning model:** **Spot** is ~30% cheaper and totally fine for benchmarking.
- **Boot disk:**
  - Click **Change** → **Public images** → **Deep Learning on Linux** → **Deep Learning VM with CUDA 12.4** (or newer) — comes pre-loaded with PyTorch and CUDA.
  - Disk size: **150 GB** (model + flash-attn build).
- **Firewall:** allow HTTPS (and HTTP if you need it).
- **Create.**

> **Quota:** if `gpus_all_regions` is 0, request a bump to 1 in **IAM & Admin → Quotas**. Approval is automatic and takes a few minutes.

## 2. Push your local code to your fork

On your **Mac**, from the repo:

```bash
git status
git add miniengine/ tests/ pyproject.toml docs/ milestone2_report.md milestone2_report.pdf setup-vm/
git commit -m "milestone 2: paged KV pool + flash-attn + torch.compile + cuda graphs"
git push origin main
```

## 3. SSH and run the setup script

```bash
gcloud compute ssh miniengine-l4 --zone=<ZONE>
# (or use the SSH-in-browser button in the console)

git clone https://github.com/HivaMohammadzadeh1/CS349D-miniengine.git
cd CS349D-miniengine
bash setup-vm/setup_milestone2.sh
```

flash-attn build takes ~15–25 min on g2-standard-8.

## 4. Run benchmarks and pull artifacts

Same as AWS — see [milestone2_aws.md](milestone2_aws.md) §4–§6, but use
`gcloud compute scp --recurse miniengine-l4:~/CS349D-miniengine/bench-out ./bench-out --zone=<ZONE>`.

## 5. Stop the VM when done

```bash
gcloud compute instances stop miniengine-l4 --zone=<ZONE>
```

**Spot L4 is ~$0.30/hr; on-demand ~$0.70/hr.** A full benchmark run is
~30 min once setup is done.
