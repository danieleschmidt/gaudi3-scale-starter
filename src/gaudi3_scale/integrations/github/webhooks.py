"""GitHub webhook handlers for training automation."""

import hashlib
import hmac
import json
import logging
from typing import Dict, Any, Optional, Callable

from ...optional_deps import FASTAPI, OptionalDependencyError, require_optional_dep

# Import FastAPI components conditionally
if FASTAPI:
    from fastapi import HTTPException, Header, Request
else:
    # Create stub classes for when FastAPI is not available
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)
    
    class Header:
        def __init__(self, default=None, **kwargs):
            self.default = default
    
    class Request:
        async def body(self):
            raise OptionalDependencyError('fastapi', 'GitHub webhook request handling')

logger = logging.getLogger(__name__)


class GitHubWebhookHandler:
    """Handler for GitHub webhook events.
    
    Processes webhook events from GitHub to trigger training workflows
    based on repository events like pushes, releases, and pull requests.
    """
    
    def __init__(self, secret: Optional[str] = None):
        """Initialize webhook handler.
        
        Args:
            secret: Webhook secret for signature verification
        """
        self.secret = secret
        self.handlers: Dict[str, Callable] = {}
    
    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify GitHub webhook signature.
        
        Args:
            payload: Raw request payload
            signature: GitHub signature header
            
        Returns:
            True if signature is valid
        """
        if not self.secret:
            logger.warning("No webhook secret configured - skipping signature verification")
            return True
        
        if not signature.startswith("sha256="):
            return False
        
        expected_signature = hmac.new(
            self.secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        received_signature = signature[7:]  # Remove "sha256=" prefix
        
        return hmac.compare_digest(expected_signature, received_signature)
    
    def register_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register an event handler.
        
        Args:
            event_type: GitHub event type (e.g., 'push', 'pull_request')
            handler: Handler function that takes event payload
        """
        self.handlers[event_type] = handler
        logger.info(f"Registered handler for {event_type} events")
    
    @require_optional_dep('fastapi', 'GitHub webhook handling')
    async def handle_webhook(
        self,
        request: Request,
        x_github_event: str = Header(...),
        x_hub_signature_256: Optional[str] = Header(None)
    ) -> Dict[str, str]:
        """Handle incoming GitHub webhook.
        
        Args:
            request: FastAPI request object
            x_github_event: GitHub event type header
            x_hub_signature_256: GitHub signature header
            
        Returns:
            Response message
            
        Raises:
            HTTPException: If signature verification fails or handler errors
        """
        payload = await request.body()
        
        # Verify signature if secret is configured
        if x_hub_signature_256 and not self.verify_signature(payload, x_hub_signature_256):
            logger.error("Invalid webhook signature")
            raise HTTPException(status_code=403, detail="Invalid signature")
        
        try:
            event_data = json.loads(payload)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON payload: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        
        # Handle the event
        if x_github_event in self.handlers:
            try:
                await self.handlers[x_github_event](event_data)
                logger.info(f"Successfully handled {x_github_event} event")
                return {"status": "success", "event": x_github_event}
            except Exception as e:
                logger.error(f"Error handling {x_github_event} event: {e}")
                raise HTTPException(status_code=500, detail=f"Handler error: {str(e)}")
        else:
            logger.info(f"No handler registered for {x_github_event} event")
            return {"status": "ignored", "event": x_github_event}


class TrainingWorkflowTrigger:
    """Triggers training workflows based on GitHub events."""
    
    def __init__(self, github_client, training_service):
        """Initialize training workflow trigger.
        
        Args:
            github_client: GitHub API client
            training_service: Training service instance
        """
        self.github_client = github_client
        self.training_service = training_service
    
    async def handle_push_event(self, event_data: Dict[str, Any]) -> None:
        """Handle push events to trigger training.
        
        Args:
            event_data: GitHub push event payload
        """
        repository = event_data["repository"]
        ref = event_data["ref"]
        commits = event_data["commits"]
        
        logger.info(f"Push to {repository['full_name']} on {ref} with {len(commits)} commits")
        
        # Check if push is to main/master branch
        if ref in ["refs/heads/main", "refs/heads/master"]:
            # Look for training configuration in commit messages
            for commit in commits:
                message = commit["message"]
                if "[train]" in message.lower() or "trigger training" in message.lower():
                    await self._trigger_training_from_commit(repository, commit)
    
    async def handle_release_event(self, event_data: Dict[str, Any]) -> None:
        """Handle release events to trigger training.
        
        Args:
            event_data: GitHub release event payload
        """
        if event_data["action"] == "published":
            repository = event_data["repository"]
            release = event_data["release"]
            
            logger.info(f"Release {release['tag_name']} published in {repository['full_name']}")
            
            # Trigger training for the release
            training_config = {
                "model_name": f"{repository['name']}-{release['tag_name']}",
                "git_ref": release["tag_name"],
                "repository_url": repository["clone_url"],
                "trigger": "release",
                "release_notes": release["body"]
            }
            
            await self.training_service.start_training(training_config)
    
    async def handle_pull_request_event(self, event_data: Dict[str, Any]) -> None:
        """Handle pull request events for training validation.
        
        Args:
            event_data: GitHub pull request event payload
        """
        action = event_data["action"]
        pull_request = event_data["pull_request"]
        repository = event_data["repository"]
        
        if action == "opened" and "[train]" in pull_request["title"].lower():
            logger.info(f"Training PR opened: {pull_request['title']}")
            
            # Trigger validation training on PR branch
            training_config = {
                "model_name": f"pr-{pull_request['number']}-{repository['name']}",
                "git_ref": pull_request["head"]["ref"],
                "repository_url": repository["clone_url"],
                "trigger": "pull_request",
                "validation": True,
                "pr_number": pull_request["number"]
            }
            
            await self.training_service.start_training(training_config)
    
    async def _trigger_training_from_commit(self, repository: Dict[str, Any], commit: Dict[str, Any]) -> None:
        """Trigger training from a specific commit.
        
        Args:
            repository: Repository information
            commit: Commit information
        """
        training_config = {
            "model_name": f"{repository['name']}-{commit['id'][:8]}",
            "git_ref": commit["id"],
            "repository_url": repository["clone_url"],
            "trigger": "commit",
            "commit_message": commit["message"],
            "author": commit["author"]["name"]
        }
        
        await self.training_service.start_training(training_config)
        logger.info(f"Triggered training for commit {commit['id'][:8]} in {repository['full_name']}")


def setup_github_webhooks(webhook_handler: GitHubWebhookHandler, github_client, training_service) -> None:
    """Setup GitHub webhook handlers for training automation.
    
    Args:
        webhook_handler: Webhook handler instance
        github_client: GitHub API client
        training_service: Training service instance
    """
    trigger = TrainingWorkflowTrigger(github_client, training_service)
    
    # Register event handlers
    webhook_handler.register_handler("push", trigger.handle_push_event)
    webhook_handler.register_handler("release", trigger.handle_release_event)
    webhook_handler.register_handler("pull_request", trigger.handle_pull_request_event)
    
    logger.info("GitHub webhook handlers configured for training automation")