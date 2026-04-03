# Deployment & Validation Guide

## Problem: Files Not Appearing in HF Space

**Root Cause**: You likely cloned the Space but didn't properly `git add`, `commit`, and `push` the files.

---

## Solution (Step-by-Step)

### **Phase 1: Push Files to HF Space** ✅

1. **Find your HF Space Git URL**:
   - Go to your Space settings on Hugging Face
   - Top menu: "Clone repository"
   - Copy the URL (looks like `https://huggingface.co/spaces/yourusername/clinical-trial-env`)

2. **Clone the Space repo locally** (if not already):
   ```bash
   git clone https://huggingface.co/spaces/yourusername/clinical-trial-env
   cd clinical-trial-env
   ```

3. **Copy all project files to the cloned folder**:
   - Copy these files from `d:\Pytorch-meta-hackathon\clinical_trial_env\` to the cloned Space folder:
     - `environment.py`
     - `app.py`
     - `inference.py`
     - `openenv.yaml`
     - `requirements.txt`
     - `Dockerfile`
     - `README.md`
     - `validate-submission.sh`

4. **Commit and Push** (in the cloned Space directory):
   ```bash
   git add .
   git commit -m "Add Clinical Trial Environment files"
   git push origin main
   ```

5. **Verify on HF**:
   - Refresh your Space page
   - All files should now appear in the repo
   - Space should auto-build (check "Builds" tab for progress)
   - Wait 5-10 minutes for deployment

---

### **Phase 2: Test Docker Locally** ✅

Now that Docker is installed:

1. **Navigate to project**:
   ```bash
   cd d:\Pytorch-meta-hackathon\clinical_trial_env
   ```

2. **Build Docker image**:
   ```bash
   docker build -t clinical-trial-env:latest .
   ```
   - Should take 1-3 minutes
   - Watch for `Successfully tagged clinical-trial-env:latest`

3. **Run Docker container** (test locally):
   ```bash
   docker run -p 8000:8000 clinical-trial-env:latest
   ```
   - Should start the FastAPI server on port 8000
   - Should see: `Uvicorn running on http://0.0.0.0:8000`

4. **Test the API** (in another terminal):
   ```bash
   curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d "{}"
   ```
   - Should return JSON with observation data (200 OK)

---

### **Phase 3: Run Validation Script** ✅

Once your HF Space is deployed:

1. **Get your Space URL**:
   - Format: `https://yourusername-clinical-trial-env.hf.space`
   - Find it in Space settings or use direct URL

2. **Test Space is live** (quick check):
   ```bash
   curl -X POST https://yourusername-clinical-trial-env.hf.space/reset -H "Content-Type: application/json" -d "{}"
   ```
   - Should return 200 + JSON

3. **Run the validation script**:
   ```bash
   # Windows PowerShell / Git Bash
   bash ./validate-submission.sh https://yourusername-clinical-trial-env.hf.space d:\Pytorch-meta-hackathon\clinical_trial_env
   ```
   - Should output:
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

## Troubleshooting

### Docker Build Fails
- Error: `FROM python:3.9-slim` not found?
  - Docker needs internet to pull base image
  - Retry: `docker build -t clinical-trial-env:latest .`
- Error: `app.py not found`?
  - Ensure all files are in the same directory

### HF Space Still Empty
- Did you `git push`? Check with: `git log` (should show your commit)
- Check Space "Builds" tab for errors
- Try rebuilding: Space settings → Restart Space

### Space /reset Returns 500
- Check Space logs: Click on the Space name → "Logs"
- Common: Missing Python packages—requirements.txt not read
- Fix: Ensure requirements.txt has all deps (pydantic, fastapi, uvicorn, openai)

### Validation Script Fails at Step 1 (Ping)
- Ensure Space is running (check HF Space status)
- Verify URL format (should end with `.hf.space`, not `.hf.space/`)
- Try manually: `curl -X POST YOUR_SPACE_URL/reset -H "Content-Type: application/json" -d "{}"`

### Validation Script Fails at Step 2 (Docker)
- Docker not installed? Reinstall from https://docs.docker.com/get-docker/
- Dockerfile syntax error? Check our version in repo

### Validation Script Fails at Step 3 (openenv validate)
- openenv-core not installed? Run: `pip install openenv-core`
- openenv.yaml syntax error? Check YAML formatting

---

## Summary Checklist

- [ ] Cloned HF Space repo
- [ ] Copied all 8 files to Space folder
- [ ] Ran `git add .`, `git commit -m "..."`, `git push`
- [ ] Refreshed HF Space page (files now visible)
- [ ] Waited 5-10 min for auto-build
- [ ] Docker installed locally
- [ ] Ran `docker build` (succeeded)
- [ ] Ran Docker container test (API responds on port 8000)
- [ ] Space is live at `https://yourusername-clinical-trial-env.hf.space`
- [ ] Tested Space with manual curl (got 200)
- [ ] Ran validation script (all 3 checks passed)
- [ ] Ready to submit! 🚀

---

## Quick Commands Reference

```bash
# HF Space deployment
git clone https://huggingface.co/spaces/yourusername/clinical-trial-env
cd clinical-trial-env
# [copy files here]
git add .
git commit -m "Add Clinical Trial Environment"
git push origin main

# Docker testing
docker build -t clinical-trial-env:latest .
docker run -p 8000:8000 clinical-trial-env:latest

# API test
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d "{}"

# Validation
bash ./validate-submission.sh https://your-space.hf.space d:\Pytorch-meta-hackathon\clinical_trial_env
```

Good luck! 🎯
