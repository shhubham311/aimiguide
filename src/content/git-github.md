# Version Control: Git & GitHub

## Why Git & GitHub Matter for ML/AI

In machine learning, you're not just managing code—you're managing **datasets, model checkpoints, experiment configurations, hyperparameters, and collaborative notebooks**. Without version control, you lose track of which code produced which results, team members overwrite each other's work, and reproducing experiments becomes impossible.

Git tracks every change to your codebase with full history. GitHub adds collaboration, code review, CI/CD, and project management. Together, they are **non-negotiable skills** for any ML engineer—every tech company expects fluency.

---

## Git Fundamentals

### The Three Areas of Git

```
┌─────────────────────────────────────────────────────────────┐
│                    Git Workflow                               │
│                                                              │
│  Working Directory ──git add──► Staging Area ──git commit──► Repository │
│                                                              │
│  (your files on       (changes ready    (permanent          │
│   your computer)       to commit)        history)            │
│                                                              │
│  ┌──────────┐    git add     ┌──────────┐   git commit   ┌──────────┐
│  │  Unstaged│ ────────────► │  Staged  │ ─────────────► │ Committed│
│  │  Changes │               │  Changes │               │  History │
│  └──────────┘               └──────────┘               └──────────┘
│       │                          │                          │
│  git diff                  git diff --staged           git log
│  (see changes)             (see staged diff)          (see history)
└─────────────────────────────────────────────────────────────┘
```

### Core Commands

```bash
# Initialize a repository
git init
git clone <url>                    # Clone existing repo

# Basic workflow
git status                         # Check state of files
git add <file>                     # Stage a file
git add .                          # Stage all changes
git commit -m "message"            # Commit staged changes

# Viewing history
git log                            # Commit history
git log --oneline --graph          # Compact visual history
git show <commit-hash>             # Show specific commit

# Branching
git branch <name>                  # Create branch
git checkout <name>                # Switch branch
git checkout -b <name>             # Create + switch
git merge <branch>                 # Merge branch into current

# Remote
git remote add origin <url>        # Add remote
git push origin main               # Push to remote
git pull origin main               # Pull + merge from remote
git fetch origin                   # Fetch without merging
```

---

## Branching Strategies for ML Projects

### Feature Branch Workflow

```
                    main (production)
                   /
                 /───── merge
                /
    feature/data-pipeline    ──► develop
                \                  /
                 \───── merge     /
                                  \
                    feature/model-training ──► develop
                                  /
                    feature/evaluation ──► develop
```

### Git Flow for ML Teams

```
┌──────────────────────────────────────────────────┐
│                Git Flow                            │
│                                                    │
│  main (production) ◄────── release/v1.0           │
│       │                       │                    │
│  hotfix/bugfix              develop               │
│       │                       │                    │
│       └───── merge ────────► │                    │
│                               │                   │
│         feature/model-v2 ◄────┤                   │
│         feature/new-data ◄────┤                   │
│         feature/api ◄─────────┤                   │
│                                                    │
│  Rules:                                            │
│  - main: only production-ready code               │
│  - develop: integration branch                     │
│  - feature/*: individual work branches             │
│  - release/*: release preparation                  │
└──────────────────────────────────────────────────┘
```

---

## Handling ML-Specific Files

### .gitignore for ML Projects

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
venv/
.env

# ML Artifacts (use DVC or git-lfs instead)
*.pkl
*.h5
*.pt
*.pth
*.onnx
*.bin
*.safetensors
checkpoints/
models/

# Data (use DVC)
data/raw/
data/processed/
*.csv
*.parquet

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
```

### Git LFS for Large Files

```bash
# Install Git LFS
git lfs install

# Track large model files
git lfs track "*.safetensors"
git lfs track "*.bin"
git lfs track "*.pt"

# This creates .gitattributes
# *.safetensors filter=lfs diff=lfs merge=lfs -text

git add .gitattributes
git commit -m "Configure Git LFS for model files"
```

### DVC (Data Version Control)

```
┌────────────────────────────────────────────────┐
│  DVC: Git for Data                             │
│                                                 │
│  Git tracks:  .dvc files (small metadata)      │
│  DVC tracks:  data/ models/ checkpoints/       │
│                                                 │
│  .dvc/train_data.dvc:                          │
│  md5: abc123                                    │
│  size: 2.5GB                                   │
│  path: data/train.csv                           │
│                                                 │
│  Storage: S3, GCS, Azure Blob, local           │
└────────────────────────────────────────────────┘

# Usage
pip install dvc

dvc init
dvc add data/train.csv              # Track data file
git add data/train.csv.dvc
git commit -m "Track training data"

dvc remote add -d myremote s3://bucket/data
dvc push                            # Upload data to cloud
dvc pull                            # Download data
```

---

## Collaboration with Pull Requests

### Creating a PR

```
Step 1: Fork or branch
  git checkout -b feature/add-bert-evaluation

Step 2: Make changes, commit
  git add .
  git commit -m "Add BERT evaluation pipeline"

Step 3: Push to remote
  git push origin feature/add-bert-evaluation

Step 4: Open PR on GitHub
  → Title: "Add BERT evaluation pipeline"
  → Description: What, Why, How
  → Reviewers: Assign team members
  → CI checks: Tests, linting, formatting
```

### PR Description Template

```markdown
## Description
Add BERT-based evaluation for text classification models.

## Changes
- Added `evaluate_bert()` function in `eval.py`
- Updated `requirements.txt` with `transformers>=4.30`
- Added unit tests for evaluation pipeline

## Testing
```bash
pytest tests/test_eval.py -v  # All 5 tests pass
```

## Related Issues
Closes #42
```

### Code Review Best Practices

```
┌──────────────────────────────────────────┐
│  Good PR Practices                       │
│                                           │
│  ✓ Small, focused PRs (< 400 lines)      │
│  ✓ Clear description with context        │
│  ✓ All CI checks passing                 │
│  ✓ Self-review before requesting review  │
│  ✓ Respond to comments promptly          │
│                                           │
│  ✗ Giant PRs (> 1000 lines)              │
│  ✗ Vague descriptions                    │
│  ✗ Broken CI without explanation         │
│  ✗ Force-pushing to shared branches      │
└──────────────────────────────────────────┘
```

---

## Advanced Git Operations

### Resolving Merge Conflicts

```bash
# Scenario: Both you and colleague edited model.py
git pull origin main
# CONFLICT: Merge conflict in model.py

# Open model.py — you'll see:
<<<<<<< HEAD
learning_rate = 0.001
batch_size = 32
=======
learning_rate = 0.0005
batch_size = 64
>>>>>>> feature/new-hyperparams

# Resolve by choosing the correct version, then:
git add model.py
git commit -m "Resolve merge conflict in model.py"
```

### Undoing Mistakes

```bash
# Unstage a file
git restore --staged <file>

# Discard working directory changes
git restore <file>

# Amend last commit (before pushing!)
git commit --amend -m "Better message"

# Revert a committed change (safe, creates new commit)
git revert <commit-hash>

# Reset to previous commit (dangerous!)
git reset --soft HEAD~1    # Keep changes staged
git reset --hard HEAD~1    # Discard all changes
```

### Interactive Rebase

```bash
# Squash last 3 commits into one
git rebase -i HEAD~3

# Opens editor:
pick  a1b2c3d  Add data loading
squash e4f5g6h  Add preprocessing
squash i7j8k9l  Add augmentation
# → Single clean commit: "Add data pipeline"

# Rebase onto latest main
git checkout feature/my-model
git rebase main
```

---

## GitHub Actions for ML CI/CD

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov black flake8

      - name: Lint
        run: flake8 src/ tests/

      - name: Format check
        run: black --check src/ tests/

      - name: Run tests
        run: pytest tests/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## Exercises

### Exercise 1: Branching Scenario
You're working on a model training script. You discover a bug in the data loader while developing a new feature. What do you do?

**Solution:**
```
1. Stash current feature work:
   git stash

2. Create hotfix branch from main:
   git checkout main
   git checkout -b hotfix/data-loader-bug

3. Fix the bug, commit, push:
   git add data_loader.py
   git commit -m "Fix data loader index out of bounds"
   git push origin hotfix/data-loader-bug

4. Merge fix to main:
   git checkout main
   git merge hotfix/data-loader-bug

5. Return to feature:
   git checkout feature/new-model
   git stash pop   # Restore your work
```

### Exercise 2: Merge Conflict Resolution
You have a config.py with `EPOCHS = 10` on main and `EPOCHS = 50` on your branch. The main branch also has `BATCH_SIZE = 32` that your branch doesn't have. Resolve the conflict keeping both changes.

**Solution:**
```
<<<<<<< HEAD
EPOCHS = 10
BATCH_SIZE = 32
=======
EPOCHS = 50
>>>>>>> feature/new-training

Resolved:
EPOCHS = 50        # From feature branch
BATCH_SIZE = 32    # From main branch

git add config.py
git commit -m "Merge: use 50 epochs from feature, keep batch size from main"
```

### Exercise 3: Design a .gitignore
You're starting a new ML project. What should your .gitignore include?

**Solution:**
```gitignore
# Python artifacts
__pycache__/
*.pyc
.venv/
dist/
*.egg-info/

# ML model files
*.pkl
*.h5
*.pt
*.safetensors
checkpoints/
wandb/

# Data files
data/
*.csv
*.json
*.parquet

# Secrets
.env
credentials.json

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

---

## Key Takeaways

1. **Git tracks changes** across three areas: working directory, staging area, repository
2. **Branching** enables parallel development—always work on feature branches
3. **.gitignore** is essential in ML to exclude large data/model files
4. **DVC** extends Git for versioning datasets and model checkpoints
5. **GitHub Actions** automates testing, linting, and deployment pipelines
