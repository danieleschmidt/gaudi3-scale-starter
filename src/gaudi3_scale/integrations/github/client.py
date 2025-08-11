"""GitHub API client for integration with training workflows."""

import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    # Fallback for environments without requests
    requests = None

try:
    from pydantic import BaseModel, ConfigDict
except ImportError:
    # Fallback for environments without pydantic
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class ConfigDict:
        def __init__(self, **kwargs):
            pass

logger = logging.getLogger(__name__)


class GitHubRepository(BaseModel):
    """GitHub repository information."""
    model_config = ConfigDict(extra='forbid')
    
    name: str
    full_name: str
    description: Optional[str] = None
    private: bool
    clone_url: str
    ssh_url: str
    html_url: str
    default_branch: str


class GitHubCommit(BaseModel):
    """GitHub commit information."""
    model_config = ConfigDict(extra='forbid')
    
    sha: str
    message: str
    author: str
    timestamp: datetime
    url: str


class GitHubArtifact(BaseModel):
    """GitHub Actions artifact information."""
    model_config = ConfigDict(extra='forbid')
    
    id: int
    name: str
    size_in_bytes: int
    url: str
    expired: bool
    created_at: datetime


class GitHubClient:
    """GitHub API client for training workflow integration.
    
    Provides methods to interact with GitHub repositories, commits,
    and Actions artifacts for model training automation.
    """
    
    def __init__(self, token: Optional[str] = None):
        """Initialize GitHub client.
        
        Args:
            token: GitHub personal access token. If None, will try to get from environment.
        """
        if requests is None:
            raise ImportError("requests library is required for GitHub client. Install with: pip install requests")
        
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GitHub token not provided and GITHUB_TOKEN env var not set")
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "gaudi3-scale/0.1.0"
        })
        self.base_url = "https://api.github.com"
    
    def get_repository(self, owner: str, repo: str) -> GitHubRepository:
        """Get repository information.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Repository information
            
        Raises:
            requests.HTTPError: If API request fails
        """
        url = f"{self.base_url}/repos/{owner}/{repo}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            return GitHubRepository(
                name=data["name"],
                full_name=data["full_name"],
                description=data.get("description"),
                private=data["private"],
                clone_url=data["clone_url"],
                ssh_url=data["ssh_url"],
                html_url=data["html_url"],
                default_branch=data["default_branch"]
            )
        except requests.RequestException as e:
            logger.error(f"Failed to get repository {owner}/{repo}: {e}")
            raise
    
    def get_commits(self, owner: str, repo: str, branch: str = "main", limit: int = 10) -> List[GitHubCommit]:
        """Get recent commits from repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name (default: main)
            limit: Maximum number of commits to retrieve
            
        Returns:
            List of recent commits
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/commits"
        params = {
            "sha": branch,
            "per_page": min(limit, 100)
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            commits = []
            for commit_data in response.json():
                commit = GitHubCommit(
                    sha=commit_data["sha"],
                    message=commit_data["commit"]["message"],
                    author=commit_data["commit"]["author"]["name"],
                    timestamp=datetime.fromisoformat(
                        commit_data["commit"]["author"]["date"].replace("Z", "+00:00")
                    ),
                    url=commit_data["html_url"]
                )
                commits.append(commit)
            
            return commits
        except requests.RequestException as e:
            logger.error(f"Failed to get commits for {owner}/{repo}: {e}")
            raise
    
    def create_issue(self, owner: str, repo: str, title: str, body: str, labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a new issue in the repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            title: Issue title
            body: Issue body
            labels: Optional list of labels
            
        Returns:
            Created issue information
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/issues"
        data = {
            "title": title,
            "body": body,
        }
        if labels:
            data["labels"] = labels
        
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to create issue in {owner}/{repo}: {e}")
            raise
    
    def get_workflow_runs(self, owner: str, repo: str, workflow_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get workflow runs for a specific workflow.
        
        Args:
            owner: Repository owner
            repo: Repository name
            workflow_id: Workflow ID or filename
            limit: Maximum number of runs to retrieve
            
        Returns:
            List of workflow runs
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs"
        params = {"per_page": min(limit, 100)}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()["workflow_runs"]
        except requests.RequestException as e:
            logger.error(f"Failed to get workflow runs for {owner}/{repo}: {e}")
            raise
    
    def download_artifact(self, owner: str, repo: str, artifact_id: int, download_path: str) -> str:
        """Download a workflow artifact.
        
        Args:
            owner: Repository owner
            repo: Repository name
            artifact_id: Artifact ID
            download_path: Local path to save the artifact
            
        Returns:
            Path to downloaded file
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/actions/artifacts/{artifact_id}/zip"
        
        try:
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            with open(download_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded artifact {artifact_id} to {download_path}")
            return download_path
        except requests.RequestException as e:
            logger.error(f"Failed to download artifact {artifact_id}: {e}")
            raise
    
    def trigger_workflow(self, owner: str, repo: str, workflow_id: str, ref: str = "main", inputs: Optional[Dict[str, Any]] = None) -> bool:
        """Trigger a workflow dispatch event.
        
        Args:
            owner: Repository owner
            repo: Repository name
            workflow_id: Workflow ID or filename
            ref: Git reference (branch, tag, or commit SHA)
            inputs: Workflow inputs
            
        Returns:
            True if workflow was triggered successfully
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches"
        data = {"ref": ref}
        if inputs:
            data["inputs"] = inputs
        
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
            logger.info(f"Triggered workflow {workflow_id} for {owner}/{repo}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to trigger workflow {workflow_id}: {e}")
            return False
    
    def create_training_issue(self, owner: str, repo: str, model_name: str, 
                            training_config: Dict[str, Any], error_message: Optional[str] = None) -> Dict[str, Any]:
        """Create a training-specific issue with structured information.
        
        Args:
            owner: Repository owner
            repo: Repository name
            model_name: Name of the model being trained
            training_config: Training configuration details
            error_message: Optional error message if training failed
            
        Returns:
            Created issue information
        """
        if error_message:
            title = f"Training Failed: {model_name}"
            body = f"""## Training Failure Report

**Model:** {model_name}
**Timestamp:** {datetime.now(timezone.utc).isoformat()}

### Error Details
```
{error_message}
```

### Training Configuration
```yaml
"""
            for key, value in training_config.items():
                body += f"{key}: {value}\n"
            body += "```\n\n### Next Steps\n- [ ] Review error logs\n- [ ] Check configuration\n- [ ] Restart training if needed"
            labels = ["training-failure", "gaudi3", "bug"]
        else:
            title = f"Training Started: {model_name}"
            body = f"""## Training Started

**Model:** {model_name}
**Timestamp:** {datetime.now(timezone.utc).isoformat()}

### Configuration
```yaml
"""
            for key, value in training_config.items():
                body += f"{key}: {value}\n"
            body += "```\n\nThis issue will track the progress of this training run."
            labels = ["training-in-progress", "gaudi3"]
        
        return self.create_issue(owner, repo, title, body, labels)