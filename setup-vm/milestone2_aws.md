# Milestone 2 — AWS GPU setup (g5.2xlarge, A10G)

The course's $100 AWS credits + g5.2xlarge are the easiest path. **Note:**
g5.2xlarge has an **A10G**, not an L4 — the spec asks for L4 but the
course-supplied AWS path uses A10G. Use this if you want the easy on-ramp;
absolute numbers will differ slightly from L4. Use [`milestone2_gcp.md`](milestone2_gcp.md)
if you specifically need an L4.

## 1. One-time AWS console setup (~20 min, mostly waiting)

### 1.1 Redeem credits
<https://us-east-1.console.aws.amazon.com/costmanagement/home#/credits/redeemCredits>

Enter the coupon code from the Ed post. Verify under **Cost Management → Credits** that you see **+$100 active**.

### 1.2 Request quota increase
<https://console.aws.amazon.com/servicequotas/>

1. **AWS services → Amazon Elastic Compute Cloud (Amazon EC2)**.
2. Find quota **"Running On-Demand G and VT instances"**.
3. **Request quota increase** → enter at least **8** vCPUs (g5.2xlarge = 8 vCPUs).
4. Wait for the email approval (typically <1 hour, sometimes ~10 min).

### 1.3 Launch a g5.2xlarge

EC2 → **Launch instance**:
- **Name:** `miniengine-l4`
- **AMI:** *Deep Learning AMI GPU PyTorch 2.6 (Ubuntu 22.04)* — search this exact string in the AMI catalog. Pre-baked CUDA + PyTorch + drivers; saves ~30 min of setup.
- **Instance type:** `g5.2xlarge` (1× A10G, 24 GB VRAM, 32 GB RAM, 8 vCPUs)
- **Key pair:** create a new one (`miniengine-key`) and save the `.pem` file. `chmod 400 ~/Downloads/miniengine-key.pem` immediately.
- **Network:** default VPC, default security group is fine — but *edit* it to allow inbound **SSH (22) from My IP**.
- **Storage:** 150 GB gp3 root volume (model weights + flash-attn build cache need room).
- **Launch.**

Wait for **Instance state: running** + **Status check: 2/2 passed** (~2 min). Copy the **Public IPv4 address**.

## 2. Push your local code to your fork

On your **Mac**, from the repo:

```bash
# Verify your branch and commit the milestone-2 work first
git status
git add miniengine/ tests/ pyproject.toml docs/ milestone2_report.md milestone2_report.pdf
git commit -m "milestone 2: paged KV pool + flash-attn + torch.compile + cuda graphs"
git push origin main
```

## 3. SSH and run the setup script

```bash
# From your Mac
ssh -i ~/Downloads/miniengine-key.pem ubuntu@<PUBLIC_IP>

# On the instance
git clone https://github.com/HivaMohammadzadeh1/CS349D-miniengine.git
cd CS349D-miniengine
bash setup-vm/setup_milestone2.sh
```

The setup script verifies CUDA, creates the venv, installs all deps including flash-attn (this is the slow part — ~10–25 min on a g5.2xlarge), and runs a smoke test loading Qwen3-8B onto the GPU.

## 4. Run the benchmarks

```bash
bash setup-vm/run_benchmarks.sh
```

This produces `bench-out/` with terminal capture for every required mode:
`batched`, `paged`, `paged + torch.compile`, `paged + torch.compile + cuda-graph`,
plus the `--page-size 16` and `--page-size 128` sweep, plus accuracy on MMLU.

## 5. Pull artifacts back

On your **Mac**:

```bash
scp -i ~/Downloads/miniengine-key.pem -r \
    ubuntu@<PUBLIC_IP>:~/CS349D-miniengine/bench-out ./bench-out
```

Open the `.txt` files in `bench-out/`, screenshot the relevant summaries, paste them into `milestone2_report.md` (replacing the `<paste>` placeholders), and re-render the PDF:

```bash
pandoc milestone2_report.md -o /tmp/m2.html --standalone
weasyprint /tmp/m2.html milestone2_report.pdf
```

## 6. Stop the instance when done

**This matters — g5.2xlarge is ~$1.20/hr on-demand.**

```bash
# In the AWS console: Instances → select → Instance state → Stop instance
# (Stop, not Terminate — Stop preserves the EBS volume, so you can resume later.
#  Terminate deletes everything.)
```

If you're done forever, *Terminate* the instance and *Delete* the EBS volume.
