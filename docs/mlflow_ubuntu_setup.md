# MLflow Server Setup on Ubuntu

This guide walks you through setting up a dedicated MLflow tracking server on your Ubuntu machine with proper virtual environment isolation and network accessibility.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Step 1: Create MLflow User (Optional but Recommended)](#step-1-create-mlflow-user-optional-but-recommended)
- [Step 2: Set Up Virtual Environment](#step-2-set-up-virtual-environment)
- [Step 3: Install MLflow](#step-3-install-mlflow)
- [Step 4: Create Directory Structure](#step-4-create-directory-structure)
- [Step 5: Start MLflow Server](#step-5-start-mlflow-server)
- [Step 6: Create Systemd Service](#step-6-create-systemd-service)
- [Step 7: Configure Firewall](#step-7-configure-firewall)
- [Step 8: Configure Client .env File](#step-8-configure-client-env-file)
- [Step 9: Test the Connection](#step-9-test-the-connection)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Ubuntu 20.04 or later
- Python 3.8 or later
- sudo access
- Network access to the Ubuntu machine

## Step 1: Create MLflow User (Optional but Recommended)

For better security and isolation, create a dedicated user for MLflow:

```bash
# Create a new user for mlflow
sudo useradd -m -s /bin/bash mlflow

# Set password for the mlflow user (optional)
sudo passwd mlflow

# Add to sudo group if needed (optional)
sudo usermod -aG sudo mlflow

# Switch to mlflow user
sudo su - mlflow
```

If you prefer to use your existing user, skip this step.

## Step 2: Set Up Virtual Environment

```bash
# Install python3-venv if not already installed
sudo apt update
sudo apt install python3-venv python3-pip

# Create a directory for MLflow
mkdir -p ~/mlflow-server
cd ~/mlflow-server

# Create virtual environment
python3 -m venv mlflow_env

# Activate virtual environment
source mlflow_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Step 3: Install MLflow

```bash
# Make sure virtual environment is activated
source ~/mlflow-server/mlflow_env/bin/activate

# Install MLflow 2.x (for DagsHub compatibility)
pip install "mlflow>=2.8.0,<3.0"

# Install additional dependencies for better performance
pip install psycopg2-binary  # For PostgreSQL backend (optional)
pip install boto3             # For S3 artifact storage (optional)
pip install azure-storage-blob # For Azure blob storage (optional)

# Verify installation
mlflow --version
```

## Step 4: Create Directory Structure

```bash
# Create directories for MLflow data
mkdir -p ~/mlflow-server/mlruns        # For experiment data
mkdir -p ~/mlflow-server/artifacts     # For artifact storage
mkdir -p ~/mlflow-server/database      # For backend database

# Create a configuration directory
mkdir -p ~/mlflow-server/config
```

## Step 5: Start MLflow Server

### Basic Setup (SQLite Backend)

```bash
# Activate virtual environment
source ~/mlflow-server/mlflow_env/bin/activate

# Start MLflow server
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///home/$USER/mlflow-server/database/mlflow.db \
    --default-artifact-root /home/$USER/mlflow-server/artifacts \
    --serve-artifacts
```

### Production Setup (PostgreSQL Backend)

First, install and configure PostgreSQL:

```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE mlflow_db;
CREATE USER mlflow_user WITH ENCRYPTED PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;
EOF

# Start MLflow with PostgreSQL
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri postgresql://mlflow_user:your_secure_password@localhost/mlflow_db \
    --default-artifact-root /home/$USER/mlflow-server/artifacts \
    --serve-artifacts
```

## Step 6: Create Systemd Service

Create a systemd service to automatically start MLflow on boot:

```bash
# Create service file
sudo tee /etc/systemd/system/mlflow.service << 'EOF'
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
User=mlflow
Group=mlflow
WorkingDirectory=/home/mlflow/mlflow-server
Environment="PATH=/home/mlflow/mlflow-server/mlflow_env/bin"
ExecStart=/home/mlflow/mlflow-server/mlflow_env/bin/mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///home/mlflow/mlflow-server/database/mlflow.db \
    --default-artifact-root /home/mlflow/mlflow-server/artifacts \
    --serve-artifacts
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd daemon
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable mlflow

# Start the service
sudo systemctl start mlflow

# Check service status
sudo systemctl status mlflow

# View logs
sudo journalctl -u mlflow -f
```

## Step 7: Configure Firewall

Allow MLflow port through the firewall:

```bash
# If using ufw (Ubuntu's default firewall)
sudo ufw allow 5000/tcp
sudo ufw reload

# Or if you want to restrict to local network only (recommended)
# Replace 192.168.1.0/24 with your network range
sudo ufw allow from 192.168.1.0/24 to any port 5000

# Check firewall status
sudo ufw status
```

## Step 8: Configure Client .env File

On your client machines (where you run TabML), create or update the `.env` file:

### Find Your Server's IP Address

```bash
# On the Ubuntu server, find the IP address
ip addr show | grep inet | grep -v 127.0.0.1
# or
hostname -I
```

### Create .env File on Client

```bash
# Copy the example file
cp .env.example .env
```

Edit the `.env` file with your server details:

```bash
# ============================================================================
# MLflow Configuration for Local Network Server
# ============================================================================

# MLflow tracking server URI
# Replace with your Ubuntu server's IP address
MLFLOW_TRACKING_URI="http://192.168.1.100:5000"

# Alternative: Use hostname if you've configured DNS/hosts file
# MLFLOW_TRACKING_URI="http://ml-server.local:5000"

# Experiment name (optional - can be set in code)
MLFLOW_EXPERIMENT_NAME="tabml-experiments"

# For authentication (if you've configured it)
# MLFLOW_TRACKING_USERNAME="your-username"
# MLFLOW_TRACKING_PASSWORD="your-password"
```

### Configure Hostname (Optional)

For easier access, you can configure hostname resolution:

```bash
# On client machines, edit /etc/hosts (Linux/Mac) or C:\Windows\System32\drivers\etc\hosts (Windows)
# Add this line (replace with your server's IP):
192.168.1.100   ml-server.local

# Then you can use:
MLFLOW_TRACKING_URI="http://ml-server.local:5000"
```

## Step 9: Test the Connection

### From Client Machine

```python
# test_mlflow_connection.py
from dotenv import load_dotenv
import os
import mlflow

# Load environment variables
load_dotenv()

# Set tracking URI from environment
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Create a test experiment
experiment_name = "test-connection"
mlflow.create_experiment(experiment_name, artifact_location="/tmp/mlflow-test")

# Start a test run
with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id):
    mlflow.log_param("test_param", "test_value")
    mlflow.log_metric("test_metric", 0.99)
    print(f"Successfully connected to MLflow at {os.getenv('MLFLOW_TRACKING_URI')}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

print("\nTest completed! Check the MLflow UI to see your test run.")
```

### Access MLflow UI

Open a web browser and navigate to:
- `http://192.168.1.100:5000` (replace with your server's IP)
- or `http://ml-server.local:5000` (if hostname configured)

## Advanced Configuration

### Enable Authentication

For production environments, consider adding authentication:

```bash
# Install nginx for reverse proxy with authentication
sudo apt install nginx apache2-utils

# Create password file
sudo htpasswd -c /etc/nginx/.htpasswd mlflow_user

# Configure nginx
sudo tee /etc/nginx/sites-available/mlflow << 'EOF'
server {
    listen 80;
    server_name ml-server.local;

    auth_basic "MLflow Login";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/mlflow /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Use S3 for Artifact Storage

```bash
# Install boto3 in virtual environment
pip install boto3

# Start MLflow with S3
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///home/mlflow/mlflow-server/database/mlflow.db \
    --default-artifact-root s3://your-bucket/mlflow-artifacts
```

### Backup and Maintenance

Create a backup script:

```bash
#!/bin/bash
# backup_mlflow.sh

BACKUP_DIR="/home/mlflow/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup database
cp /home/mlflow/mlflow-server/database/mlflow.db $BACKUP_DIR/mlflow_${TIMESTAMP}.db

# Backup artifacts (if local)
tar -czf $BACKUP_DIR/artifacts_${TIMESTAMP}.tar.gz /home/mlflow/mlflow-server/artifacts

# Keep only last 7 days of backups
find $BACKUP_DIR -type f -mtime +7 -delete

echo "Backup completed: $TIMESTAMP"
```

Add to crontab for daily backups:

```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * /home/mlflow/backup_mlflow.sh
```

## Troubleshooting

### Common Issues and Solutions

1. **Connection Refused Error**
   ```bash
   # Check if MLflow is running
   sudo systemctl status mlflow
   
   # Check if port is listening
   sudo netstat -tlnp | grep 5000
   
   # Check firewall
   sudo ufw status
   ```

2. **Permission Denied Errors**
   ```bash
   # Fix ownership
   sudo chown -R mlflow:mlflow /home/mlflow/mlflow-server
   
   # Fix permissions
   sudo chmod -R 755 /home/mlflow/mlflow-server
   ```

3. **Virtual Environment Not Found**
   ```bash
   # Recreate virtual environment
   cd ~/mlflow-server
   python3 -m venv mlflow_env
   source mlflow_env/bin/activate
   pip install "mlflow>=2.8.0,<3.0"
   ```

4. **Database Lock Errors (SQLite)**
   ```bash
   # Switch to PostgreSQL for concurrent access
   # Or ensure only one MLflow server instance is running
   sudo systemctl stop mlflow
   sudo systemctl start mlflow
   ```

5. **Check Logs**
   ```bash
   # System logs
   sudo journalctl -u mlflow -n 100
   
   # MLflow logs (if not using systemd)
   tail -f ~/mlflow-server/mlflow.log
   ```

### Performance Tuning

For better performance with multiple users:

```bash
# Start with more workers (gunicorn)
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri postgresql://mlflow_user:password@localhost/mlflow_db \
    --default-artifact-root /home/mlflow/mlflow-server/artifacts \
    --workers 4 \
    --serve-artifacts
```

## Security Recommendations

1. **Use PostgreSQL** instead of SQLite for production
2. **Enable authentication** via nginx reverse proxy
3. **Use HTTPS** with SSL certificates (Let's Encrypt)
4. **Restrict network access** via firewall rules
5. **Regular backups** of database and artifacts
6. **Monitor disk space** for artifacts storage
7. **Set up log rotation** to prevent disk fill

## Next Steps

1. Access the MLflow UI at `http://your-server-ip:5000`
2. Create your first experiment using TabML
3. Configure your team members' `.env` files
4. Set up automated backups
5. Consider adding authentication for production use

## Useful Commands

```bash
# Start MLflow service
sudo systemctl start mlflow

# Stop MLflow service
sudo systemctl stop mlflow

# Restart MLflow service
sudo systemctl restart mlflow

# View service status
sudo systemctl status mlflow

# View logs
sudo journalctl -u mlflow -f

# Check disk usage
df -h /home/mlflow/mlflow-server

# List experiments
mlflow experiments list --tracking-uri http://localhost:5000

# Delete old runs (cleanup)
mlflow gc --tracking-uri http://localhost:5000 --older-than "30d"
```

## Support

For TabML-specific MLflow integration issues, refer to:
- [TabML Documentation](../README.md)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [TabML GitHub Issues](https://github.com/wguesdon/tabml/issues)