"""GitHub Actions integration for CI/CD workflows."""

import logging
import os
import tempfile
import zipfile
from typing import Dict, Any, List, Optional
from pathlib import Path

from .client import GitHubClient

logger = logging.getLogger(__name__)


class GitHubActionsIntegration:
    """Integration with GitHub Actions for CI/CD workflows.
    
    Provides functionality to interact with GitHub Actions workflows,
    download artifacts, and manage CI/CD pipeline integration.
    """
    
    def __init__(self, github_client: GitHubClient):
        """Initialize GitHub Actions integration.
        
        Args:
            github_client: GitHub API client instance
        """
        self.github_client = github_client
    
    def get_training_workflows(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """Get workflows related to training.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            List of training-related workflows
        """
        try:
            url = f"{self.github_client.base_url}/repos/{owner}/{repo}/actions/workflows"
            response = self.github_client.session.get(url)
            response.raise_for_status()
            
            workflows = response.json()["workflows"]
            
            # Filter for training-related workflows
            training_workflows = []
            for workflow in workflows:
                name = workflow["name"].lower()
                if any(keyword in name for keyword in ["train", "model", "gaudi", "hpu"]):
                    training_workflows.append(workflow)
            
            return training_workflows
        except Exception as e:
            logger.error(f"Failed to get training workflows: {e}")
            return []
    
    def get_workflow_artifacts(self, owner: str, repo: str, run_id: int) -> List[Dict[str, Any]]:
        """Get artifacts from a workflow run.
        
        Args:
            owner: Repository owner
            repo: Repository name
            run_id: Workflow run ID
            
        Returns:
            List of artifacts from the workflow run
        """
        try:
            url = f"{self.github_client.base_url}/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts"
            response = self.github_client.session.get(url)
            response.raise_for_status()
            
            return response.json()["artifacts"]
        except Exception as e:
            logger.error(f"Failed to get workflow artifacts: {e}")
            return []
    
    def download_model_artifacts(self, owner: str, repo: str, run_id: int, 
                               download_dir: Optional[str] = None) -> List[str]:
        """Download model artifacts from a training workflow run.
        
        Args:
            owner: Repository owner
            repo: Repository name
            run_id: Workflow run ID
            download_dir: Directory to download artifacts (default: temp dir)
            
        Returns:
            List of downloaded artifact paths
        """
        if download_dir is None:
            download_dir = tempfile.mkdtemp(prefix="gaudi3_artifacts_")
        
        download_dir = Path(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
        
        artifacts = self.get_workflow_artifacts(owner, repo, run_id)
        downloaded_files = []
        
        for artifact in artifacts:
            artifact_name = artifact["name"]
            
            # Filter for model-related artifacts
            if any(keyword in artifact_name.lower() for keyword in ["model", "checkpoint", "weights"]):
                try:
                    artifact_path = download_dir / f"{artifact_name}.zip"
                    self.github_client.download_artifact(owner, repo, artifact["id"], str(artifact_path))
                    
                    # Extract if it's a zip file
                    if artifact_path.suffix == ".zip":
                        extract_dir = download_dir / artifact_name
                        extract_dir.mkdir(exist_ok=True)
                        
                        with zipfile.ZipFile(artifact_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_dir)
                        
                        # Remove the zip file and add extracted directory
                        artifact_path.unlink()
                        downloaded_files.append(str(extract_dir))
                    else:
                        downloaded_files.append(str(artifact_path))
                    
                    logger.info(f"Downloaded artifact: {artifact_name}")
                except Exception as e:
                    logger.error(f"Failed to download artifact {artifact_name}: {e}")
        
        return downloaded_files
    
    def get_training_status(self, owner: str, repo: str, workflow_name: str) -> Dict[str, Any]:
        """Get the status of the latest training workflow run.
        
        Args:
            owner: Repository owner
            repo: Repository name
            workflow_name: Name of the training workflow
            
        Returns:
            Training status information
        """
        try:
            runs = self.github_client.get_workflow_runs(owner, repo, workflow_name, limit=1)
            
            if not runs:
                return {"status": "no_runs", "message": "No workflow runs found"}
            
            latest_run = runs[0]
            
            return {
                "status": latest_run["status"],
                "conclusion": latest_run.get("conclusion"),
                "run_id": latest_run["id"],
                "run_number": latest_run["run_number"],
                "created_at": latest_run["created_at"],
                "updated_at": latest_run["updated_at"],
                "html_url": latest_run["html_url"],
                "head_sha": latest_run["head_sha"],
                "head_branch": latest_run["head_branch"]
            }
        except Exception as e:
            logger.error(f"Failed to get training status: {e}")
            return {"status": "error", "message": str(e)}
    
    def create_training_workflow_file(self, workflow_path: str, config: Dict[str, Any]) -> str:
        """Create a GitHub Actions workflow file for training.
        
        Args:
            workflow_path: Path to save the workflow file
            config: Training workflow configuration
            
        Returns:
            Path to the created workflow file
        """
        workflow_content = f"""name: Gaudi 3 Training Workflow

on:
  push:
    branches: [ {config.get('trigger_branch', 'main')} ]
  pull_request:
    branches: [ {config.get('trigger_branch', 'main')} ]
  workflow_dispatch:
    inputs:
      model_name:
        description: 'Model name to train'
        required: true
        default: '{config.get('default_model', 'llama-7b')}'
      epochs:
        description: 'Number of training epochs'
        required: true
        default: '{config.get('default_epochs', '3')}'

jobs:
  train-model:
    runs-on: [self-hosted, gaudi3]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Setup Gaudi environment
      run: |
        source /opt/habanalabs/init_env.sh
        pip install -r requirements.txt
    
    - name: Verify Gaudi availability
      run: |
        python -c "import habana_frameworks.torch as htorch; print(f'HPUs available: {{htorch.hpu.device_count()}}')"
    
    - name: Run training
      env:
        WANDB_PROJECT: ${{{{ config.get('wandb_project', 'gaudi3-scale') }}}}
        WANDB_API_KEY: ${{{{ secrets.WANDB_API_KEY }}}}
      run: |
        gaudi3-train \\
          --model-name ${{{{ github.event.inputs.model_name || '{config.get('default_model', 'llama-7b')}' }}}} \\
          --epochs ${{{{ github.event.inputs.epochs || '{config.get('default_epochs', '3')}' }}}} \\
          --output-dir ./models \\
          --log-level info
    
    - name: Upload model artifacts
      uses: actions/upload-artifact@v4
      if: success()
      with:
        name: trained-model-${{{{ github.sha }}}}
        path: ./models/
        retention-days: 30
    
    - name: Upload training logs
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: training-logs-${{{{ github.sha }}}}
        path: ./logs/
        retention-days: 7
    
    - name: Report training completion
      if: success()
      run: |
        echo "Training completed successfully"
        echo "Model artifacts uploaded"
        echo "Run ID: ${{{{ github.run_id }}}}"
    
    - name: Report training failure
      if: failure()
      run: |
        echo "Training failed"
        echo "Check logs for details"
        exit 1
"""
        
        with open(workflow_path, 'w') as f:
            f.write(workflow_content)
        
        logger.info(f"Created training workflow at {workflow_path}")
        return workflow_path
    
    def setup_training_secrets(self, owner: str, repo: str, secrets: Dict[str, str]) -> bool:
        """Setup repository secrets required for training workflows.
        
        Note: This requires admin permissions and uses GitHub CLI or API.
        
        Args:
            owner: Repository owner
            repo: Repository name
            secrets: Dictionary of secret names and values
            
        Returns:
            True if secrets were set successfully
        """
        try:
            # This would require GitHub CLI or advanced API permissions
            # For now, we'll log the required secrets
            logger.info(f"Required secrets for {owner}/{repo}:")
            for secret_name in secrets.keys():
                logger.info(f"  - {secret_name}")
            
            logger.warning("Secrets must be manually configured in repository settings")
            logger.info("Required secrets: WANDB_API_KEY, HUGGINGFACE_TOKEN")
            
            return True
        except Exception as e:
            logger.error(f"Failed to setup training secrets: {e}")
            return False
    
    def monitor_training_run(self, owner: str, repo: str, run_id: int) -> Dict[str, Any]:
        """Monitor a training workflow run and return status updates.
        
        Args:
            owner: Repository owner
            repo: Repository name
            run_id: Workflow run ID
            
        Returns:
            Monitoring information including logs and status
        """
        try:
            # Get run details
            url = f"{self.github_client.base_url}/repos/{owner}/{repo}/actions/runs/{run_id}"
            response = self.github_client.session.get(url)
            response.raise_for_status()
            
            run_data = response.json()
            
            # Get job details
            jobs_url = f"{url}/jobs"
            jobs_response = self.github_client.session.get(jobs_url)
            jobs_response.raise_for_status()
            
            jobs = jobs_response.json()["jobs"]
            
            return {
                "run_status": run_data["status"],
                "conclusion": run_data.get("conclusion"),
                "started_at": run_data.get("run_started_at"),
                "jobs": [
                    {
                        "name": job["name"],
                        "status": job["status"],
                        "conclusion": job.get("conclusion"),
                        "started_at": job.get("started_at"),
                        "completed_at": job.get("completed_at")
                    }
                    for job in jobs
                ]
            }
        except Exception as e:
            logger.error(f"Failed to monitor training run: {e}")
            return {"error": str(e)}