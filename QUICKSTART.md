# Quick Start: Get Your Submission Live in 10 Minutes

## ⚡ Fastest Path to Deployment

### Step 1: Prepare Files for HF Space (2 mins)
1. Go to Hugging Face: https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in:
   - **Space name**: `clinical-trial-env`
   - **SDK**: Select **Docker**
   - **Visibility**: Public
   - Click **Create**

### Step 2: Clone & Push Your Files (3 mins)

**Copy-paste these commands into PowerShell/Git Bash**:

```bash
# Clone the Space (replace with YOUR username)
git clone https://huggingface.co/spaces/YOUR_USERNAME/clinical-trial-env
cd clinical-trial-env

# Copy all 8 files from your local project here:
# (Use File Explorer to copy these files into this cloned folder)
# - environment.py
# - app.py
# - inference.py
# - openenv.yaml
# - requirements.txt
# - Dockerfile
# - README.md
# - validate-submission.sh

# Then run:
git add .
git commit -m "Deploy Clinical Trial Environment"
git push origin main
```

### Step 3: Wait for Auto-Build (3-5 mins)
1. After push, Hugging Face auto-builds your Docker image
2. Check progress: Click Space name → **Builds** tab
3. Wait until it says "Successfully built"
4. Once done, your Space URL is live: `https://YOUR_USERNAME-clinical-trial-env.hf.space`

### Step 4: Test Your Space (1 min)

**Test the API is working** (in PowerShell):
```powershell
$url = "https://YOUR_USERNAME-clinical-trial-env.hf.space/reset"
$headers = @{"Content-Type" = "application/json"}
$body = "{}"
Invoke-WebRequest -Uri $url -Method POST -Headers $headers -Body $body
```

Should return **200 OK** with JSON data.

### Step 5: Run Validation Script (1 min)

**In Git Bash or WSL**:
```bash
cd d:\Pytorch-meta-hackathon\clinical_trial_env
bash ./validate-submission.sh https://YOUR_USERNAME-clinical-trial-env.hf.space
```

**Expected output** (all 3 checks pass):
```
========================================
  OpenEnv Submission Validator
========================================
[HH:MM:SS] PASSED -- HF Space is live and responds to /reset
[HH:MM:SS] PASSED -- Docker build succeeded
[HH:MM:SS] PASSED -- openenv validate passed

========================================
  All 3/3 checks passed!
  Your submission is ready to submit.
========================================
```

---

## 🆘 If Something Goes Wrong

### Files Still Don't Appear in HF Space (After git push)
```bash
git log  # Should show your commit
git status  # Should show "nothing to commit"
# If not committed, try:
git add -A
git commit -m "Add files"
git push
```

### Space Build Failed (Check Logs)
1. Click your Space name → **Builds** tab
2. Click the failed build
3. Scroll to see error (usually missing package or syntax issue)
4. Check our `requirements.txt` has: `pydantic`, `fastapi`, `uvicorn`, `openai`

### Validation Script Fails
1. **Step 1 (Ping)**: Ensure Space is running. Refresh page to wake it up.
2. **Step 2 (Docker)**: 
   - Make sure Docker Desktop is **running** (check system tray)
   - If not, start it now
3. **Step 3 (openenv)**: Run `pip install openenv-core` first

### Docker Desktop Not Starting
- Right-click Docker icon → Restart
- Or: Settings → General → "Start Docker on startup"
- Give it 30 seconds to wake up

---

## 📋 Files Checklist (Must Be in HF Space Repo)
- ✅ `environment.py` — Core env logic
- ✅ `app.py` — FastAPI server
- ✅ `inference.py` — Baseline script
- ✅ `openenv.yaml` — Metadata
- ✅ `requirements.txt` — Dependencies
- ✅ `Dockerfile` — Container spec
- ✅ `README.md` — Documentation
- ✅ `validate-submission.sh` — Validation script
- ✅ `.gitignore` — (auto-created by HF)

---

## Your Space URL
**Replace and bookmark this** (once deployed):
```
https://YOUR_USERNAME-clinical-trial-env.hf.space
```

---

## Next: Run Inference (Optional)

Once Space is live, test the baseline:

```bash
set OPENAI_API_KEY=sk-your-key-here
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4

python inference.py
```

Expected output:
```
[START] Task: easy
[STEP] Step: 1, Reward: 0.45, Done: False
[STEP] Step: 2, Reward: 0.52, Done: True
[END] Task: easy, Average Reward: 0.49
```

---

## Success! 🎉
Once validation passes, you're ready to submit the hackathon. The project is:
- ✅ Deployed on HF Spaces
- ✅ OpenEnv spec compliant
- ✅ Docker working
- ✅ Baseline reproducible
- ✅ Novel (CRA trial monitoring task)

Good luck!
