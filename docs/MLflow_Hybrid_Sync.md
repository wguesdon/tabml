# MLflow Hybrid Sync Tutorial: Kaggle File Logging to Local Server

## Overview

This tutorial demonstrates a hybrid approach for MLflow experiment tracking where you:
1. Run experiments on Kaggle with file-based MLflow logging
2. Download the experiment data
3. Sync it with your local MLflow server

This approach is perfect when you want to avoid network configuration complexities while still maintaining a centralized MLflow server.

### Advantages of This Approach
- ✅ No network configuration required (no ngrok, port forwarding, etc.)
- ✅ Works reliably regardless of network restrictions
- ✅ Experiments are portable and backed up
- ✅ Can batch process multiple experiments
- ✅ Full security - no external access to your server

### When to Use This Method
- You prefer simplicity over real-time tracking
- You're working with sensitive data
- Network restrictions prevent direct connections
- You want to review experiments before adding to main server
- You're running long experiments that might exceed tunnel timeouts

## Table of Contents
1. [Part 1: Set Up Local MLflow Server](#part-1-set-up-local-mlflow-server)
2. [Part 2: Configure Kaggle for File-Based Logging](#part-2-configure-kaggle-for-file-based-logging)
3. [Part 3: Download Experiments from Kaggle](#part-3-download-experiments-from-kaggle)
4. [Part 4: Sync to Local MLflow Server](#part-4-sync-to-local-mlflow-server)
5. [Part 5: Automation Scripts](#part-5-automation-scripts)
6. [Advanced Topics](#advanced-topics)
7. [Troubleshooting](#troubleshooting)

---

## Part 1: Set Up Local MLflow Server

### Step 1: Install MLflow on Ubuntu

```bash
# Update system
sudo apt update
sudo apt install python3-pip python3-venv

# Create a virtual environment (recommended)
python3 -m venv mlflow-env
source mlflow-env/bin/activate

# Install MLflow
pip install mlflow pandas scikit-learn

# Verify installation
mlflow --version
```

### Step 2: Create MLflow Directory Structure

```bash
# Create main MLflow directory
mkdir -p ~/mlflow-local
cd ~/mlflow-local

# Create subdirectories
mkdir -p mlruns
mkdir -p artifacts
mkdir -p kaggle-imports
mkdir -p scripts
```

### Step 3: Configure MLflow Server

Create a configuration script:

```bash
nano ~/mlflow-local/scripts/start_mlflow.sh
```

Add the following content:

```bash
#!/bin/bash
# MLflow Server Startup Script

MLFLOW_HOME="$HOME/mlflow-local"
cd $MLFLOW_HOME

# Activate virtual environment if exists
if [ -d "$HOME/mlflow-env" ]; then
    source $HOME/mlflow-env/bin/activate
fi

# Start MLflow server
mlflow server \
    --host 127.0.0.1 \
    --port 5000 \
    --backend-store-uri sqlite:///$MLFLOW_HOME/mlflow.db \
    --default-artifact-root file://$MLFLOW_HOME/artifacts \
    --serve-artifacts
```

Make it executable:

```bash
chmod +x ~/mlflow-local/scripts/start_mlflow.sh
```

### Step 4: Start MLflow Server

```bash
# Start the server
~/mlflow-local/scripts/start_mlflow.sh

# Access UI at http://localhost:5000
```

---

## Part 2: Configure Kaggle for File-Based Logging

### Step 1: Basic MLflow Setup in Kaggle

Create a new Kaggle notebook and add this initialization cell:

```python
# Install MLflow in Kaggle
!pip install mlflow -q

import mlflow
import os
import json
from datetime import datetime

# Configure file-based tracking
MLFLOW_DIR = "/kaggle/working/mlruns"
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")

# Create metadata file for easier syncing
def create_sync_metadata():
    metadata = {
        "kaggle_user": os.environ.get("KAGGLE_USERNAME", "unknown"),
        "kernel_id": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "interactive"),
        "created_at": datetime.now().isoformat(),
        "mlflow_version": mlflow.__version__
    }
    
    with open("/kaggle/working/sync_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

# Initialize metadata
metadata = create_sync_metadata()
print(f"MLflow configured for file-based logging")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Metadata: {metadata}")
```

### Step 2: Complete Kaggle Experiment Example

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Set up MLflow
MLFLOW_DIR = "/kaggle/working/mlruns"
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")

# Create or set experiment
experiment_name = f"kaggle_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
mlflow.set_experiment(experiment_name)

# Log experiment metadata
experiment_metadata = {
    "platform": "kaggle",
    "notebook_name": "hybrid_mlflow_demo",
    "purpose": "classification_model_training"
}

# Generate sample data (replace with your actual Kaggle competition data)
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define hyperparameters to test
hyperparameter_configs = [
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2},
    {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5},
    {"n_estimators": 150, "max_depth": None, "min_samples_split": 2}
]

# Track multiple runs
best_run_id = None
best_accuracy = 0

for config_idx, params in enumerate(hyperparameter_configs):
    with mlflow.start_run(run_name=f"run_{config_idx}") as run:
        # Log tags
        mlflow.set_tags(experiment_metadata)
        mlflow.set_tag("config_index", config_idx)
        
        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Log dataset info
        mlflow.log_param("n_samples_train", len(X_train))
        mlflow.log_param("n_samples_test", len(X_test))
        mlflow.log_param("n_features", X.shape[1])
        
        # Train model
        print(f"\nTraining model with config {config_idx}: {params}")
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mlflow.log_metric("cv_mean_score", cv_scores.mean())
        mlflow.log_metric("cv_std_score", cv_scores.std())
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Log metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "cv_mean": cv_scores.mean()
        }
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"  {metric_name}: {metric_value:.4f}")
        
        # Feature importance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_importance = model.feature_importances_
        indices = np.argsort(feature_importance)[::-1][:10]
        ax.bar(range(10), feature_importance[indices])
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Importance")
        ax.set_title(f"Top 10 Features - Config {config_idx}")
        plt.tight_layout()
        
        # Save and log plot
        plot_path = f"feature_importance_{config_idx}.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
        
        # Save model
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=mlflow.models.infer_signature(X_train, y_pred)
        )
        
        # Also save as joblib for backup
        joblib_path = f"model_{config_idx}.joblib"
        joblib.dump(model, joblib_path)
        mlflow.log_artifact(joblib_path)
        
        # Track best run
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_run_id = run.info.run_id
        
        # Create run summary
        summary = {
            "run_id": run.info.run_id,
            "config": params,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save summary
        summary_path = f"run_summary_{config_idx}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path)

print(f"\n✅ All experiments complete!")
print(f"Best run ID: {best_run_id}")
print(f"Best accuracy: {best_accuracy:.4f}")
print(f"MLflow tracking directory: {MLFLOW_DIR}")

# Create final summary for easy download identification
final_summary = {
    "experiment_name": experiment_name,
    "total_runs": len(hyperparameter_configs),
    "best_run_id": best_run_id,
    "best_accuracy": best_accuracy,
    "completed_at": datetime.now().isoformat()
}

with open("/kaggle/working/experiment_summary.json", "w") as f:
    json.dump(final_summary, f, indent=2)

print("\n📁 Files ready for download:")
print("- mlruns/ (entire directory)")
print("- experiment_summary.json")
print("- sync_metadata.json")
```

### Step 3: Prepare for Download

After running experiments, create a compressed archive:

```python
import shutil
import os

# Create archive of MLflow runs
archive_name = f"mlflow_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.make_archive(
    f"/kaggle/working/{archive_name}",
    'zip',
    "/kaggle/working",
    "mlruns"
)

print(f"✅ Archive created: {archive_name}.zip")
print(f"Size: {os.path.getsize(f'/kaggle/working/{archive_name}.zip') / 1024 / 1024:.2f} MB")

# List all files for download
print("\n📥 Available for download:")
for file in os.listdir("/kaggle/working"):
    if file.endswith('.zip') or file.endswith('.json'):
        size = os.path.getsize(f"/kaggle/working/{file}") / 1024
        print(f"  - {file} ({size:.2f} KB)")
```

---

## Part 3: Download Experiments from Kaggle

### Option 1: Manual Download via Kaggle UI

1. In your Kaggle notebook, click on the **Output** tab
2. Download the ZIP file and JSON files
3. Save them to `~/mlflow-local/kaggle-imports/`

### Option 2: Using Kaggle API

Install and configure Kaggle API locally:

```bash
# Install Kaggle API
pip install kaggle

# Set up credentials (get from Kaggle Account Settings)
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

Create a download script:

```bash
nano ~/mlflow-local/scripts/download_kaggle.py
```

```python
#!/usr/bin/env python3
"""
Download MLflow experiments from Kaggle kernels
"""

import os
import sys
import json
from kaggle import api
from datetime import datetime
import argparse

def download_kernel_output(kernel_slug, username, download_path):
    """
    Download output files from a Kaggle kernel
    
    Args:
        kernel_slug: The kernel identifier
        username: Kaggle username
        download_path: Local directory to save files
    """
    # Create download directory
    os.makedirs(download_path, exist_ok=True)
    
    try:
        # Download kernel output
        api.kernels_output(
            user_name=username,
            kernel_slug=kernel_slug,
            path=download_path
        )
        
        print(f"✅ Downloaded kernel output to: {download_path}")
        
        # List downloaded files
        files = os.listdir(download_path)
        print(f"📁 Downloaded files:")
        for file in files:
            size = os.path.getsize(os.path.join(download_path, file)) / 1024
            print(f"  - {file} ({size:.2f} KB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading kernel output: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download MLflow experiments from Kaggle")
    parser.add_argument("--kernel", required=True, help="Kernel slug")
    parser.add_argument("--user", required=True, help="Kaggle username")
    parser.add_argument("--output", default="~/mlflow-local/kaggle-imports", 
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Expand paths
    output_dir = os.path.expanduser(args.output)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_dir = os.path.join(output_dir, f"{args.kernel}_{timestamp}")
    
    # Download
    success = download_kernel_output(args.kernel, args.user, download_dir)
    
    if success:
        print(f"\n🎯 Next step: Run sync_mlflow.py --import-dir {download_dir}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
```

Usage:
```bash
# Make executable
chmod +x ~/mlflow-local/scripts/download_kaggle.py

# Download kernel output
python ~/mlflow-local/scripts/download_kaggle.py \
    --kernel your-kernel-name \
    --user your-username
```

---

## Part 4: Sync to Local MLflow Server

### Step 1: Create Sync Script

Create the main synchronization script:

```bash
nano ~/mlflow-local/scripts/sync_mlflow.py
```

```python
#!/usr/bin/env python3
"""
Sync Kaggle MLflow experiments to local MLflow server
"""

import os
import sys
import json
import shutil
import zipfile
import argparse
from pathlib import Path
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

class MLflowSyncManager:
    def __init__(self, local_mlflow_uri, import_dir):
        """
        Initialize the sync manager
        
        Args:
            local_mlflow_uri: URI of local MLflow server
            import_dir: Directory containing imported Kaggle experiments
        """
        self.local_uri = local_mlflow_uri
        self.import_dir = Path(import_dir)
        self.client = MlflowClient(tracking_uri=local_mlflow_uri)
        mlflow.set_tracking_uri(local_mlflow_uri)
        
    def extract_archives(self):
        """Extract any ZIP files in the import directory"""
        extracted_dirs = []
        
        for zip_file in self.import_dir.glob("*.zip"):
            print(f"📦 Extracting: {zip_file.name}")
            
            extract_dir = self.import_dir / zip_file.stem
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(extract_dir)
            
            extracted_dirs.append(extract_dir)
            print(f"  ✅ Extracted to: {extract_dir}")
        
        return extracted_dirs
    
    def find_mlruns_directories(self):
        """Find all mlruns directories to import"""
        mlruns_dirs = []
        
        # Look for mlruns directories
        for path in self.import_dir.rglob("mlruns"):
            if path.is_dir():
                mlruns_dirs.append(path)
                print(f"📁 Found mlruns directory: {path}")
        
        return mlruns_dirs
    
    def get_kaggle_experiments(self, mlruns_dir):
        """Get list of experiments from Kaggle mlruns directory"""
        experiments = []
        
        # Default experiment (0) and named experiments
        for exp_dir in mlruns_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name != ".trash":
                meta_file = exp_dir / "meta.yaml"
                if meta_file.exists():
                    experiments.append(exp_dir)
        
        return experiments
    
    def create_local_experiment(self, kaggle_exp_name, metadata=None):
        """Create or get experiment in local MLflow"""
        # Add prefix to distinguish Kaggle imports
        local_exp_name = f"kaggle_{kaggle_exp_name}"
        
        try:
            # Try to create new experiment
            exp_id = self.client.create_experiment(
                local_exp_name,
                tags=metadata or {}
            )
            print(f"✅ Created experiment: {local_exp_name} (ID: {exp_id})")
        except Exception:
            # Experiment exists, get its ID
            exp = self.client.get_experiment_by_name(local_exp_name)
            exp_id = exp.experiment_id
            print(f"📌 Using existing experiment: {local_exp_name} (ID: {exp_id})")
        
        return exp_id
    
    def sync_experiment(self, kaggle_exp_dir, local_exp_id):
        """Sync runs from Kaggle experiment to local experiment"""
        runs_synced = 0
        
        # Iterate through runs in the experiment
        for run_dir in kaggle_exp_dir.iterdir():
            if run_dir.is_dir() and run_dir.name != "meta.yaml":
                try:
                    # Copy run directory to local MLflow
                    local_run_dir = Path(self.local_uri.replace("file://", "")) / local_exp_id / run_dir.name
                    
                    if local_run_dir.exists():
                        print(f"  ⚠️ Run {run_dir.name} already exists, skipping...")
                        continue
                    
                    # Copy entire run directory
                    shutil.copytree(run_dir, local_run_dir)
                    runs_synced += 1
                    print(f"  ✅ Synced run: {run_dir.name}")
                    
                except Exception as e:
                    print(f"  ❌ Error syncing run {run_dir.name}: {e}")
        
        return runs_synced
    
    def sync_all(self):
        """Main sync process"""
        print("\n" + "="*50)
        print("🚀 Starting MLflow Sync Process")
        print("="*50 + "\n")
        
        # Extract archives
        self.extract_archives()
        
        # Find mlruns directories
        mlruns_dirs = self.find_mlruns_directories()
        
        if not mlruns_dirs:
            print("❌ No mlruns directories found!")
            return False
        
        total_experiments = 0
        total_runs = 0
        
        for mlruns_dir in mlruns_dirs:
            print(f"\n📂 Processing: {mlruns_dir}")
            
            # Get experiments
            experiments = self.get_kaggle_experiments(mlruns_dir)
            
            for exp_dir in experiments:
                # Read experiment metadata if available
                metadata = {}
                summary_file = self.import_dir / "experiment_summary.json"
                if summary_file.exists():
                    with open(summary_file) as f:
                        metadata = json.load(f)
                
                # Get experiment name from directory
                exp_name = exp_dir.name
                if exp_name == "0":
                    exp_name = "default"
                
                # Create local experiment
                local_exp_id = self.create_local_experiment(exp_name, metadata)
                
                # Sync runs
                runs_synced = self.sync_experiment(exp_dir, local_exp_id)
                
                total_experiments += 1
                total_runs += runs_synced
        
        print("\n" + "="*50)
        print("📊 Sync Summary")
        print("="*50)
        print(f"✅ Experiments synced: {total_experiments}")
        print(f"✅ Runs synced: {total_runs}")
        print(f"🔗 View at: http://localhost:5000")
        
        return True
    
    def cleanup(self, remove_imports=False):
        """Clean up imported files"""
        if remove_imports:
            print("\n🧹 Cleaning up import directory...")
            shutil.rmtree(self.import_dir)
            print("✅ Import files removed")

def main():
    parser = argparse.ArgumentParser(description="Sync Kaggle MLflow experiments to local server")
    parser.add_argument("--import-dir", required=True, help="Directory with Kaggle exports")
    parser.add_argument("--mlflow-uri", default="file:///home/user/mlflow-local/mlruns",
                       help="Local MLflow tracking URI")
    parser.add_argument("--cleanup", action="store_true", help="Remove import files after sync")
    
    args = parser.parse_args()
    
    # Expand paths
    import_dir = os.path.expanduser(args.import_dir)
    mlflow_uri = args.mlflow_uri.replace("~", os.path.expanduser("~"))
    
    # Initialize sync manager
    sync_manager = MLflowSyncManager(mlflow_uri, import_dir)
    
    # Run sync
    success = sync_manager.sync_all()
    
    # Cleanup if requested
    if success and args.cleanup:
        sync_manager.cleanup(remove_imports=True)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
```

### Step 2: Alternative Simple Copy Method

For a simpler approach, you can directly copy the mlruns folder:

```bash
# Create simple sync script
nano ~/mlflow-local/scripts/simple_sync.sh
```

```bash
#!/bin/bash
# Simple MLflow sync script

IMPORT_DIR="$1"
MLFLOW_HOME="$HOME/mlflow-local"

if [ -z "$IMPORT_DIR" ]; then
    echo "Usage: $0 <import_directory>"
    exit 1
fi

echo "🔄 Syncing MLflow experiments..."

# Extract if ZIP file
if [ -f "$IMPORT_DIR"/*.zip ]; then
    echo "📦 Extracting archives..."
    cd "$IMPORT_DIR"
    unzip -q *.zip
fi

# Find and copy mlruns
if [ -d "$IMPORT_DIR/mlruns" ]; then
    echo "📁 Copying experiments..."
    cp -r "$IMPORT_DIR/mlruns/"* "$MLFLOW_HOME/mlruns/"
    echo "✅ Sync complete!"
else
    echo "❌ No mlruns directory found!"
    exit 1
fi

echo "🔗 View experiments at: http://localhost:5000"
```

Make executable:
```bash
chmod +x ~/mlflow-local/scripts/simple_sync.sh
```

---

## Part 5: Automation Scripts

### Complete Workflow Automation

Create a master automation script:

```bash
nano ~/mlflow-local/scripts/kaggle_mlflow_workflow.py
```

```python
#!/usr/bin/env python3
"""
Automated workflow for Kaggle MLflow sync
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

class KaggleMLflowWorkflow:
    def __init__(self, config_file=None):
        """Initialize workflow with configuration"""
        self.config = self.load_config(config_file)
        self.mlflow_home = Path(self.config["mlflow_home"])
        
    def load_config(self, config_file):
        """Load configuration from file or use defaults"""
        default_config = {
            "mlflow_home": "~/mlflow-local",
            "mlflow_uri": "http://localhost:5000",
            "kaggle_username": os.environ.get("KAGGLE_USERNAME", ""),
            "import_dir": "~/mlflow-local/kaggle-imports",
            "auto_cleanup": False,
            "start_mlflow_server": True
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file) as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        # Expand paths
        for key in ["mlflow_home", "import_dir"]:
            default_config[key] = os.path.expanduser(default_config[key])
        
        return default_config
    
    def start_mlflow_server(self):
        """Start local MLflow server if not running"""
        # Check if already running
        result = subprocess.run(
            ["pgrep", "-f", "mlflow server"],
            capture_output=True
        )
        
        if result.returncode == 0:
            print("✅ MLflow server already running")
            return True
        
        print("🚀 Starting MLflow server...")
        script_path = self.mlflow_home / "scripts" / "start_mlflow.sh"
        
        if script_path.exists():
            subprocess.Popen([str(script_path)])
            time.sleep(3)  # Wait for server to start
            print("✅ MLflow server started")
            return True
        else:
            print(f"⚠️ Start script not found: {script_path}")
            return False
    
    def monitor_kaggle_outputs(self, kernel_slug, interval=60):
        """Monitor Kaggle kernel for completion"""
        print(f"👀 Monitoring kernel: {kernel_slug}")
        print(f"   Checking every {interval} seconds...")
        
        while True:
            # Check kernel status (requires Kaggle API)
            try:
                result = subprocess.run(
                    ["kaggle", "kernels", "status", kernel_slug],
                    capture_output=True,
                    text=True
                )
                
                if "complete" in result.stdout.lower():
                    print("✅ Kernel execution complete!")
                    return True
                
                print(f"   Status: {result.stdout.strip()}")
                
            except Exception as e:
                print(f"⚠️ Error checking status: {e}")
            
            time.sleep(interval)
    
    def download_and_sync(self, kernel_slug):
        """Download kernel output and sync to MLflow"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_dir = Path(self.config["import_dir"]) / f"{kernel_slug}_{timestamp}"
        
        # Download
        print(f"📥 Downloading kernel output...")
        download_cmd = [
            "python",
            str(self.mlflow_home / "scripts" / "download_kaggle.py"),
            "--kernel", kernel_slug,
            "--user", self.config["kaggle_username"],
            "--output", str(download_dir)
        ]
        
        result = subprocess.run(download_cmd)
        if result.returncode != 0:
            print("❌ Download failed!")
            return False
        
        # Sync
        print(f"🔄 Syncing to MLflow...")
        sync_cmd = [
            "python",
            str(self.mlflow_home / "scripts" / "sync_mlflow.py"),
            "--import-dir", str(download_dir),
            "--mlflow-uri", f"file://{self.mlflow_home}/mlruns"
        ]
        
        if self.config["auto_cleanup"]:
            sync_cmd.append("--cleanup")
        
        result = subprocess.run(sync_cmd)
        return result.returncode == 0
    
    def run_workflow(self, kernel_slug, monitor=False):
        """Execute complete workflow"""
        print("\n" + "="*60)
        print("🎯 Kaggle MLflow Workflow")
        print("="*60 + "\n")
        
        # Start MLflow if configured
        if self.config["start_mlflow_server"]:
            self.start_mlflow_server()
        
        # Monitor if requested
        if monitor:
            self.monitor_kaggle_outputs(kernel_slug)
        
        # Download and sync
        success = self.download_and_sync(kernel_slug)
        
        if success:
            print("\n✅ Workflow complete!")
            print(f"🔗 View results at: {self.config['mlflow_uri']}")
        else:
            print("\n❌ Workflow failed!")
        
        return success

def main():
    parser = argparse.ArgumentParser(description="Automated Kaggle MLflow workflow")
    parser.add_argument("kernel", help="Kaggle kernel slug")
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--monitor", action="store_true", 
                       help="Monitor kernel until completion")
    
    args = parser.parse_args()
    
    # Initialize workflow
    workflow = KaggleMLflowWorkflow(args.config)
    
    # Run workflow
    success = workflow.run_workflow(args.kernel, monitor=args.monitor)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
```

### Configuration File

Create a configuration file for the workflow:

```bash
nano ~/mlflow-local/config.json
```

```json
{
    "mlflow_home": "~/mlflow-local",
    "mlflow_uri": "http://localhost:5000",
    "kaggle_username": "your_kaggle_username",
    "import_dir": "~/mlflow-local/kaggle-imports",
    "auto_cleanup": true,
    "start_mlflow_server": true
}
```

### Cron Job for Regular Sync

Set up automatic daily sync:

```bash
# Edit crontab
crontab -e

# Add daily sync at 2 AM
0 2 * * * /home/user/mlflow-local/scripts/sync_daily.sh >> /home/user/mlflow-local/logs/sync.log 2>&1
```

Create the daily sync script:

```bash
nano ~/mlflow-local/scripts/sync_daily.sh
```

```bash
#!/bin/bash
# Daily sync script for Kaggle MLflow experiments

MLFLOW_HOME="$HOME/mlflow-local"
IMPORT_DIR="$MLFLOW_HOME/kaggle-imports/daily"
LOG_FILE="$MLFLOW_HOME/logs/sync_$(date +%Y%m%d).log"

echo "=====================================" >> $LOG_FILE
echo "Starting daily sync: $(date)" >> $LOG_FILE
echo "=====================================" >> $LOG_FILE

# List of kernels to sync (add your kernel slugs here)
KERNELS=(
    "username/kernel-name-1"
    "username/kernel-name-2"
)

for kernel in "${KERNELS[@]}"; do
    echo "Syncing kernel: $kernel" >> $LOG_FILE
    python $MLFLOW_HOME/scripts/kaggle_mlflow_workflow.py "$kernel" \
        --config $MLFLOW_HOME/config.json >> $LOG_FILE 2>&1
done

echo "Daily sync complete: $(date)" >> $LOG_FILE
```

---

## Advanced Topics

### Handling Large Experiments

For large experiments with many artifacts:

```python
# In Kaggle - compress artifacts before logging
import zipfile
import os

def compress_and_log_artifacts(artifact_dir, mlflow_client):
    """Compress large artifacts before logging"""
    
    # Create archive
    archive_path = f"{artifact_dir}.zip"
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(artifact_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, artifact_dir)
                zf.write(file_path, arc_name)
    
    # Log compressed archive
    mlflow.log_artifact(archive_path)
    
    # Clean up
    os.remove(archive_path)
    
    return f"Compressed {len(os.listdir(artifact_dir))} files"
```

### Incremental Sync

Only sync new experiments:

```python
def get_last_sync_time(sync_file="~/.mlflow_last_sync"):
    """Get timestamp of last sync"""
    sync_file = os.path.expanduser(sync_file)
    if os.path.exists(sync_file):
        with open(sync_file) as f:
            return datetime.fromisoformat(f.read().strip())
    return None

def update_sync_time(sync_file="~/.mlflow_last_sync"):
    """Update last sync timestamp"""
    sync_file = os.path.expanduser(sync_file)
    with open(sync_file, 'w') as f:
        f.write(datetime.now().isoformat())

def sync_incremental(mlruns_dir, last_sync_time):
    """Only sync runs created after last_sync_time"""
    new_runs = []
    
    for exp_dir in Path(mlruns_dir).iterdir():
        if exp_dir.is_dir():
            for run_dir in exp_dir.iterdir():
                if run_dir.is_dir():
                    # Check run creation time
                    meta_file = run_dir / "meta.yaml"
                    if meta_file.exists():
                        create_time = datetime.fromtimestamp(
                            meta_file.stat().st_mtime
                        )
                        if not last_sync_time or create_time > last_sync_time:
                            new_runs.append(run_dir)
    
    return new_runs
```

### Sync Validation

Validate synced experiments:

```python
def validate_sync(kaggle_mlruns, local_mlruns):
    """Validate that all experiments were synced correctly"""
    
    validation_report = {
        "timestamp": datetime.now().isoformat(),
        "kaggle_experiments": 0,
        "local_experiments": 0,
        "missing_runs": [],
        "corrupt_runs": []
    }
    
    # Count Kaggle experiments and runs
    kaggle_runs = set()
    for exp_dir in Path(kaggle_mlruns).iterdir():
        if exp_dir.is_dir() and exp_dir.name != ".trash":
            validation_report["kaggle_experiments"] += 1
            for run_dir in exp_dir.iterdir():
                if run_dir.is_dir():
                    kaggle_runs.add(run_dir.name)
    
    # Check local experiments
    local_runs = set()
    for exp_dir in Path(local_mlruns).iterdir():
        if exp_dir.is_dir() and exp_dir.name != ".trash":
            validation_report["local_experiments"] += 1
            for run_dir in exp_dir.iterdir():
                if run_dir.is_dir():
                    local_runs.add(run_dir.name)
    
    # Find missing runs
    validation_report["missing_runs"] = list(kaggle_runs - local_runs)
    
    # Save report
    with open("sync_validation_report.json", "w") as f:
        json.dump(validation_report, f, indent=2)
    
    return len(validation_report["missing_runs"]) == 0
```

---

## Troubleshooting

### Common Issues

#### 1. MLflow Server Won't Start

```bash
# Check if port is in use
sudo lsof -i :5000

# Kill existing process
sudo kill -9 $(sudo lsof -t -i:5000)

# Start with different port
mlflow server --port 5001
```

#### 2. Import Fails with Permission Errors

```bash
# Fix permissions
chmod -R 755 ~/mlflow-local/mlruns
chown -R $USER:$USER ~/mlflow-local
```

#### 3. Experiments Not Showing in UI

```bash
# Rebuild MLflow database
cd ~/mlflow-local
mlflow db upgrade sqlite:///mlflow.db

# Restart server
pkill -f "mlflow server"
~/mlflow-local/scripts/start_mlflow.sh
```

#### 4. Large Files Fail to Download

```python
# In Kaggle - split large artifacts
import os

def split_large_file(file_path, chunk_size_mb=95):
    """Split files larger than Kaggle's limit"""
    
    chunk_size = chunk_size_mb * 1024 * 1024
    file_size = os.path.getsize(file_path)
    
    if file_size <= chunk_size:
        return [file_path]
    
    chunks = []
    with open(file_path, 'rb') as f:
        chunk_num = 0
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break
            
            chunk_path = f"{file_path}.part{chunk_num:03d}"
            with open(chunk_path, 'wb') as chunk_file:
                chunk_file.write(chunk_data)
            
            chunks.append(chunk_path)
            chunk_num += 1
    
    return chunks
```

#### 5. Kaggle API Authentication Issues

```bash
# Verify API credentials
kaggle config view

# Test API connection
kaggle competitions list

# Re-download credentials from Kaggle settings
# Place in ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Debug Mode

Enable detailed logging:

```python
# In sync script
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlflow_sync.log'),
        logging.StreamHandler()
    ]
)
```

---

## Best Practices

### 1. Organize Experiments

```python
# In Kaggle - use meaningful experiment names
experiment_name = f"{competition_name}_{model_type}_{datetime.now().strftime('%Y%m%d')}"
mlflow.set_experiment(experiment_name)
```

### 2. Tag Everything

```python
# Add comprehensive tags
mlflow.set_tags({
    "competition": "titanic",
    "kernel_version": "v3",
    "dataset_version": "2024-01",
    "cv_strategy": "stratified_kfold",
    "feature_engineering": "v2_polynomial"
})
```

### 3. Log Data Samples

```python
# Log sample of training data for reference
train_sample = X_train[:100].copy()
sample_df = pd.DataFrame(train_sample)
sample_df.to_csv("train_sample.csv", index=False)
mlflow.log_artifact("train_sample.csv")
```

### 4. Version Control Integration

```bash
# In local setup - track sync history
cd ~/mlflow-local
git init
git add scripts/ config.json
git commit -m "MLflow sync configuration"
```

### 5. Backup Strategy

```bash
# Regular backups of MLflow data
# Add to crontab
0 3 * * * tar -czf ~/backups/mlflow_$(date +\%Y\%m\%d).tar.gz ~/mlflow-local/mlruns
```

---

## Summary

This hybrid approach provides a robust solution for tracking Kaggle experiments in your local MLflow server without complex networking:

✅ **No network configuration** - Works behind firewalls and NATs  
✅ **Reliable** - No timeout or connection issues  
✅ **Secure** - No external access to your server  
✅ **Flexible** - Can batch process multiple experiments  
✅ **Portable** - Experiments are backed up as files  

The workflow is:
1. Run experiments on Kaggle with file-based MLflow tracking
2. Download the experiment files (manually or via API)
3. Sync to your local MLflow server
4. View and compare in the MLflow UI

This approach is perfect for data scientists who want the benefits of MLflow tracking without the complexity of real-time connectivity.

---

## Quick Start Checklist

- [ ] Install MLflow locally
- [ ] Create directory structure
- [ ] Set up MLflow server startup script
- [ ] Configure Kaggle notebook with file tracking
- [ ] Run experiments on Kaggle
- [ ] Download experiment files
- [ ] Run sync script
- [ ] View in MLflow UI at http://localhost:5000

---

*Last updated: 2024 | Version: 1.0*
