# MLflow + ngrok Tutorial: Track Kaggle Experiments on Your Local Server

## Overview

This tutorial shows you how to set up MLflow on your Ubuntu mini PC and track machine learning experiments from Kaggle notebooks using ngrok as a secure tunnel.

### What You'll Learn
- Install and configure MLflow on Ubuntu
- Set up ngrok for secure external access
- Connect Kaggle notebooks to your local MLflow server
- Track experiments, parameters, metrics, and models
- Access the MLflow UI from anywhere

### Prerequisites
- Ubuntu mini PC with Python 3.7+
- Kaggle account
- Basic familiarity with terminal commands
- Internet connection with ability to forward ports

## Table of Contents
1. [Part 1: Set Up MLflow on Ubuntu](#part-1-set-up-mlflow-on-ubuntu)
2. [Part 2: Set Up ngrok](#part-2-set-up-ngrok)
3. [Part 3: Configure Kaggle Notebook](#part-3-configure-kaggle-notebook)
4. [Part 4: Access MLflow UI](#part-4-access-mlflow-ui)
5. [Part 5: Advanced Setup](#part-5-advanced-setup)
6. [Troubleshooting](#troubleshooting)
7. [Security Considerations](#security-considerations)

---

## Part 1: Set Up MLflow on Ubuntu

### Step 1: Install MLflow

Open a terminal on your Ubuntu machine and run:

```bash
# Update system packages
sudo apt update

# Install pip if not already installed
sudo apt install python3-pip

# Install MLflow
pip install mlflow

# Verify installation
mlflow --version
```

### Step 2: Create MLflow Directory Structure

Set up a dedicated directory for MLflow data:

```bash
# Create a directory for MLflow
mkdir ~/mlflow-server
cd ~/mlflow-server

# Create subdirectories for artifacts and database
mkdir artifacts
mkdir mlruns
```

### Step 3: Start MLflow Server

Launch the MLflow tracking server:

```bash
# Start MLflow with specific backend and artifact stores
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --default-artifact-root ./artifacts \
    --backend-store-uri sqlite:///mlflow.db
```

**Note:** Keep this terminal window open - MLflow is now running locally on port 5000.

You should see output like:
```
[2024-XX-XX XX:XX:XX +0000] [12345] [INFO] Starting gunicorn 20.1.0
[2024-XX-XX XX:XX:XX +0000] [12345] [INFO] Listening at: http://0.0.0.0:5000
```

---

## Part 2: Set Up ngrok

### Step 1: Install ngrok

Open a **new terminal window** (keep MLflow running in the first one):

#### Option 1: Install using snap (recommended)
```bash
sudo snap install ngrok
```

#### Option 2: Download directly
```bash
# Download ngrok
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz

# Extract the archive
tar xvzf ngrok-v3-stable-linux-amd64.tgz

# Move to system path
sudo mv ngrok /usr/local/bin/

# Verify installation
ngrok version
```

### Step 2: Create ngrok Account (Free)

1. Go to [https://ngrok.com](https://ngrok.com)
2. Click "Sign up" and create a free account
3. After logging in, navigate to "Your Authtoken" in the dashboard
4. Copy your authentication token

### Step 3: Configure ngrok

Add your authentication token to ngrok:

```bash
# Replace YOUR_AUTH_TOKEN_HERE with your actual token
ngrok config add-authtoken YOUR_AUTH_TOKEN_HERE
```

### Step 4: Create Tunnel to MLflow

Start the ngrok tunnel:

```bash
# Create public tunnel to your MLflow server
ngrok http 5000
```

You'll see output similar to:
```
ngrok                                                           (Ctrl+C to quit)

Session Status                online
Account                       your_email@example.com (Plan: Free)
Version                       3.0.0
Region                        United States (us)
Latency                       50ms
Web Interface                 http://127.0.0.1:4040
Forwarding                    https://abc123xyz.ngrok-free.app -> http://localhost:5000

Connections                   ttl     opn     rt1     rt5     p50     p90
                              0       0       0.00    0.00    0.00    0.00
```

**⚠️ IMPORTANT:** Copy the HTTPS forwarding URL (e.g., `https://abc123xyz.ngrok-free.app`) - you'll need this for Kaggle!

---

## Part 3: Configure Kaggle Notebook

### Step 1: Install MLflow in Kaggle

In a new Kaggle notebook cell, install MLflow:

```python
!pip install mlflow -q
import mlflow
print(f"MLflow version: {mlflow.__version__}")
```

### Step 2: Test Connection to Your Server

Verify the connection to your MLflow server:

```python
import mlflow
import requests

# Replace with your actual ngrok URL
MLFLOW_TRACKING_URI = "https://abc123xyz.ngrok-free.app"

# Test connection
try:
    response = requests.get(f"{MLFLOW_TRACKING_URI}/health")
    if response.status_code == 200:
        print("✅ Successfully connected to MLflow server!")
    else:
        print(f"❌ Connection failed with status: {response.status_code}")
except Exception as e:
    print(f"❌ Connection error: {e}")

# Set the tracking URI for MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"📍 MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
```

### Step 3: Complete Training Example

Here's a complete example that tracks a machine learning experiment:

```python
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configure MLflow
MLFLOW_TRACKING_URI = "https://abc123xyz.ngrok-free.app"  # Your ngrok URL
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Set experiment name
experiment_name = "kaggle-experiment-" + datetime.now().strftime("%Y%m%d")
mlflow.set_experiment(experiment_name)

# Generate sample data (replace with your actual dataset)
# For real Kaggle competitions, load your data here:
# df = pd.read_csv('/kaggle/input/your-competition/train.csv')

print("📊 Generating sample dataset...")
n_samples = 1000
n_features = 20
X = np.random.rand(n_samples, n_features)
y = np.random.randint(0, 2, n_samples)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Start MLflow run
with mlflow.start_run(run_name="rf_classifier_" + datetime.now().strftime("%H%M%S")):
    
    # Log tags
    mlflow.set_tag("platform", "kaggle")
    mlflow.set_tag("model_type", "random_forest")
    mlflow.set_tag("developer", "your_name")
    
    # Define hyperparameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42
    }
    
    # Log parameters
    for param, value in params.items():
        mlflow.log_param(param, value)
    
    # Log dataset info
    mlflow.log_param("n_samples", n_samples)
    mlflow.log_param("n_features", n_features)
    mlflow.log_param("test_size", 0.2)
    
    # Train model
    print("🎯 Training Random Forest model...")
    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_precision": precision_score(y_test, y_pred_test, average='weighted'),
        "test_recall": recall_score(y_test, y_pred_test, average='weighted'),
        "test_f1": f1_score(y_test, y_pred_test, average='weighted')
    }
    
    # Log metrics
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Create and log feature importance plot
    plt.figure(figsize=(10, 6))
    feature_importance = rf.feature_importances_
    indices = np.argsort(feature_importance)[::-1][:10]  # Top 10 features
    plt.bar(range(10), feature_importance[indices])
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()
    
    # Log model
    print("💾 Logging model to MLflow...")
    mlflow.sklearn.log_model(
        rf, 
        "random_forest_model",
        registered_model_name="KaggleRandomForest"
    )
    
    # Log additional information as artifacts
    with open("model_info.txt", "w") as f:
        f.write(f"Model Training Report\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Platform: Kaggle\n")
        f.write(f"Model Type: Random Forest\n")
        f.write(f"\nMetrics:\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"  {metric_name}: {metric_value:.4f}\n")
    
    mlflow.log_artifact("model_info.txt")
    
    # Get run info
    run_id = mlflow.active_run().info.run_id
    
    print("\n" + "=" * 50)
    print("✅ MLflow run completed successfully!")
    print(f"📊 Run ID: {run_id}")
    print(f"🔗 View in MLflow UI: {MLFLOW_TRACKING_URI}")
    print("=" * 50)
```

### Step 4: Hyperparameter Tuning Example

Track multiple runs for hyperparameter tuning:

```python
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Configure MLflow
MLFLOW_TRACKING_URI = "https://abc123xyz.ngrok-free.app"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("hyperparameter-tuning")

# Sample data
X = np.random.rand(500, 10)
y = np.random.randint(0, 2, 500)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search with MLflow tracking
for n_est in param_grid['n_estimators']:
    for max_d in param_grid['max_depth']:
        for min_split in param_grid['min_samples_split']:
            with mlflow.start_run():
                # Log parameters
                mlflow.log_param("n_estimators", n_est)
                mlflow.log_param("max_depth", max_d)
                mlflow.log_param("min_samples_split", min_split)
                
                # Train model
                rf = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    min_samples_split=min_split,
                    random_state=42
                )
                rf.fit(X, y)
                
                # Log score
                score = rf.score(X, y)
                mlflow.log_metric("accuracy", score)
                
                print(f"Params: n_est={n_est}, max_depth={max_d}, "
                      f"min_split={min_split} → Accuracy: {score:.4f}")

print("\n✅ Hyperparameter tuning complete! Check MLflow UI for results.")
```

---

## Part 4: Access MLflow UI

### Viewing Your Experiments

1. Open your web browser
2. Navigate to your ngrok URL: `https://abc123xyz.ngrok-free.app`
3. You'll see the MLflow UI with:
   - **Experiments**: List of all your experiments
   - **Runs**: Individual training runs with parameters and metrics
   - **Models**: Registered models and versions
   - **Metrics**: Interactive plots of metrics over time

### MLflow UI Features

- **Compare Runs**: Select multiple runs and click "Compare" to see side-by-side metrics
- **Download Artifacts**: Access saved models, plots, and files
- **Search Runs**: Use the search box with queries like `metrics.accuracy > 0.9`
- **Model Registry**: Promote models to staging or production

---

## Part 5: Advanced Setup

### Auto-start MLflow with systemd

Create a service to automatically start MLflow on boot:

```bash
# Create service file
sudo nano /etc/systemd/system/mlflow.service
```

Add the following content (replace `your_username` with your actual username):

```ini
[Unit]
Description=MLflow Server
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/your_username/mlflow-server
Environment="PATH=/home/your_username/.local/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/your_username/.local/bin/mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --default-artifact-root /home/your_username/mlflow-server/artifacts \
    --backend-store-uri sqlite:////home/your_username/mlflow-server/mlflow.db
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable mlflow

# Start the service
sudo systemctl start mlflow

# Check status
sudo systemctl status mlflow
```

### Create ngrok Configuration File

For easier ngrok management, create a configuration file:

```bash
# Create ngrok config directory
mkdir ~/.ngrok2

# Edit configuration
nano ~/.ngrok2/ngrok.yml
```

Add:

```yaml
version: "2"
authtoken: YOUR_AUTH_TOKEN_HERE
tunnels:
  mlflow:
    proto: http
    addr: 5000
    inspect: false
    bind_tls: true
```

Now you can start ngrok with:
```bash
ngrok start mlflow
```

### Persistent URL (ngrok Paid Feature)

For production use, consider ngrok's paid features:

1. **Reserved Subdomain**: Get a permanent URL like `your-name.ngrok.io`
2. **Custom Domain**: Use your own domain
3. **IP Restrictions**: Limit access to specific IP addresses

To use a reserved subdomain:
```bash
ngrok http 5000 --subdomain=your-subdomain
```

### PostgreSQL Backend (Production Setup)

For production, use PostgreSQL instead of SQLite:

```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE mlflow;
CREATE USER mlflow_user WITH ENCRYPTED PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow_user;
\q

# Install PostgreSQL Python adapter
pip install psycopg2-binary

# Start MLflow with PostgreSQL
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --default-artifact-root ./artifacts \
    --backend-store-uri postgresql://mlflow_user:your_password@localhost/mlflow
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Connection Refused Error in Kaggle

**Symptoms**: 
- `ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))`

**Solutions**:
- Verify MLflow is running: `curl http://localhost:5000/health`
- Check ngrok status - should show "online"
- Ensure you're using the HTTPS URL from ngrok, not HTTP
- Try restarting both MLflow and ngrok

#### 2. 404 Not Found When Accessing UI

**Symptoms**:
- Browser shows 404 error when accessing ngrok URL

**Solutions**:
- Make sure you're using the root URL without any path
- Try: `https://your-ngrok-url.ngrok-free.app/` (with trailing slash)
- Check ngrok terminal for any error messages

#### 3. Experiments Not Showing in UI

**Symptoms**:
- Runs complete in Kaggle but don't appear in MLflow UI

**Solutions**:
- Refresh the MLflow UI page (F5)
- Check Kaggle notebook output for error messages
- Verify tracking URI is set correctly
- Check MLflow server logs for errors

#### 4. ngrok Session Expires

**Symptoms**:
- Connection stops working after ~2 hours

**Solutions**:
- Free tier has 2-hour session limit
- Restart ngrok: `ngrok http 5000`
- Consider upgrading for longer sessions
- Use systemd to auto-restart ngrok

#### 5. Large Artifacts Fail to Upload

**Symptoms**:
- Error when logging large models or files

**Solutions**:
- Check available disk space on Ubuntu machine
- Increase ngrok upload limits (paid feature)
- Consider using cloud storage for artifacts (S3, Azure Blob)

### Debug Commands

```bash
# Check if MLflow is running
ps aux | grep mlflow

# Check MLflow logs (if using systemd)
sudo journalctl -u mlflow -f

# Test MLflow locally
curl http://localhost:5000/health

# Check ngrok status
curl http://localhost:4040/api/tunnels

# View ngrok inspection interface
# Open browser to http://localhost:4040

# Check disk space
df -h ~/mlflow-server

# Check SQLite database
sqlite3 ~/mlflow-server/mlflow.db ".tables"
```

---

## Security Considerations

### Current Setup Security

⚠️ **Important**: The basic ngrok setup makes your MLflow server publicly accessible to anyone with the URL.

### Security Best Practices

#### 1. Basic Authentication with nginx

Install and configure nginx as a reverse proxy with authentication:

```bash
# Install nginx and password utilities
sudo apt install nginx apache2-utils

# Create password file
sudo htpasswd -c /etc/nginx/.htpasswd mlflow_user

# Create nginx configuration
sudo nano /etc/nginx/sites-available/mlflow
```

Add configuration:
```nginx
server {
    listen 5001;
    server_name localhost;
    
    auth_basic "MLflow Access";
    auth_basic_user_file /etc/nginx/.htpasswd;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Enable and restart:
```bash
sudo ln -s /etc/nginx/sites-available/mlflow /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Now use ngrok with port 5001 instead
ngrok http 5001
```

#### 2. IP Whitelisting (ngrok Paid)

With ngrok paid plans, restrict access to specific IPs:
```bash
ngrok http 5000 --ip-policy-file=whitelist.json
```

#### 3. VPN Alternative

For maximum security, use a VPN instead of ngrok:
- Set up WireGuard or OpenVPN on your Ubuntu machine
- Connect to VPN from Kaggle notebook
- Access MLflow directly through VPN tunnel

#### 4. Environment Variables for Secrets

Never hardcode URLs or credentials in notebooks:

```python
import os
from kaggle_secrets import UserSecretsClient

# For Kaggle secrets
user_secrets = UserSecretsClient()
mlflow_uri = user_secrets.get_secret("MLFLOW_URI")

# Or use environment variables
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
```

---

## Next Steps

### 1. Experiment Tracking Best Practices

- **Consistent Naming**: Use descriptive experiment and run names
- **Tag Everything**: Add tags for dataset version, model type, etc.
- **Log Artifacts**: Save confusion matrices, ROC curves, SHAP values
- **Track Data Versions**: Log dataset hashes or versions

### 2. Model Registry Workflow

```python
# Register a model
mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name="ProductionModel"
)

# Transition model stages
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="ProductionModel",
    version=1,
    stage="Staging"
)
```

### 3. Integrate with CI/CD

- Automate model training with GitHub Actions
- Deploy models from MLflow to production
- Set up alerts for metric thresholds

### 4. Scale Your Setup

- **Cloud Migration**: Move MLflow to AWS, GCP, or Azure
- **Kubernetes**: Deploy MLflow on K8s for scalability
- **Managed Services**: Use Databricks MLflow or Amazon SageMaker

---

## Resources

### Official Documentation
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [ngrok Documentation](https://ngrok.com/docs)
- [Kaggle Notebooks Guide](https://www.kaggle.com/docs/notebooks)

### Community Resources
- [MLflow GitHub Repository](https://github.com/mlflow/mlflow)
- [MLflow Examples](https://github.com/mlflow/mlflow/tree/master/examples)
- [ngrok Community Forum](https://ngrok.com/community)

### Troubleshooting Resources
- [MLflow Troubleshooting Guide](https://mlflow.org/docs/latest/troubleshooting.html)
- [ngrok Error Reference](https://ngrok.com/docs/errors)

---

## Conclusion

You now have a fully functional MLflow tracking server that can receive experiments from Kaggle notebooks! This setup enables you to:

✅ Track all your Kaggle experiments in one place  
✅ Compare model performance across runs  
✅ Store and version models  
✅ Share results with team members  
✅ Build a comprehensive experiment history  

Remember to:
- Keep your ngrok URL private
- Regularly backup your MLflow database
- Monitor disk space for artifacts
- Consider upgrading to production-ready solutions as you scale

Happy experimenting! 🚀

---

## Quick Reference

### Essential Commands

```bash
# Start MLflow
mlflow server --host 0.0.0.0 --port 5000

# Start ngrok
ngrok http 5000

# Check services
systemctl status mlflow
ps aux | grep ngrok

# View logs
journalctl -u mlflow -f
```

### Python Quick Start

```python
import mlflow

# Configure
mlflow.set_tracking_uri("https://your-ngrok-url.ngrok-free.app")
mlflow.set_experiment("my-experiment")

# Track
with mlflow.start_run():
    mlflow.log_param("param_name", value)
    mlflow.log_metric("metric_name", value)
    mlflow.sklearn.log_model(model, "model")
```

---

*Last updated: 2024 | Version: 1.0*
