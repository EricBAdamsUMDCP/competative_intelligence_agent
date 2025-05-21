# core/security/manager.py
import os
import json
import hashlib
import hmac
import base64
import time
import uuid
import logging
import jwt
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from functools import wraps
from fastapi import HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SecurityManager:
    """Manages security features for the agent system."""
    
    def __init__(self, security_config: Dict[str, Any] = None):
        """Initialize the security manager.
        
        Args:
            security_config: Configuration for security features
        """
        self.logger = logging.getLogger("security.manager")
        
        # Load configuration
        self.config = security_config or {}
        self.secret_key = self.config.get("secret_key", os.environ.get("SECRET_KEY", "default_insecure_key"))
        self.jwt_algorithm = self.config.get("jwt_algorithm", "HS256")
        self.token_expiry_minutes = self.config.get("token_expiry_minutes", 60)  # 1 hour by default
        
        # API key configuration
        self.api_key = self.config.get("api_key", os.environ.get("API_KEY", "dev_key"))
        
        # Encryption key
        self.encryption_key = self._generate_encryption_key(
            self.secret_key, 
            self.config.get("encryption_salt", "default_salt")
        )
        self.cipher = Fernet(self.encryption_key)
        
        # Role-based access control
        self.roles = self.config.get("roles", {
            "admin": {"permissions": ["*"]},
            "user": {"permissions": ["read:*", "write:own", "execute:workflow"]},
            "reader": {"permissions": ["read:*"]},
            "service": {"permissions": ["execute:*"]}
        })
        
        # User permissions
        self.users = self.config.get("users", {})
        
        # Token blacklist
        self.token_blacklist = set()
        
        # API key header definition for FastAPI
        self.api_key_header = APIKeyHeader(name="X-API-Key")
    
    def _generate_encryption_key(self, password: str, salt: str) -> bytes:
        """Generate an encryption key from a password and salt.
        
        Args:
            password: Password to derive key from
            salt: Salt for key derivation
            
        Returns:
            Encryption key
        """
        if isinstance(password, str):
            password = password.encode()
        
        if isinstance(salt, str):
            salt = salt.encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def encrypt_data(self, data: Union[str, bytes, Dict, List]) -> str:
        """Encrypt data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        if isinstance(data, (dict, list)):
            data = json.dumps(data)
        
        if isinstance(data, str):
            data = data.encode()
        
        encrypted = self.cipher.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_data(self, encrypted_data: str) -> Union[str, Dict, List]:
        """Decrypt data.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data)
            decrypted = self.cipher.decrypt(encrypted_bytes)
            
            # Try to parse as JSON
            try:
                return json.loads(decrypted)
            except:
                # Return as string if not JSON
                return decrypted.decode()
        except Exception as e:
            self.logger.error(f"Error decrypting data: {str(e)}")
            raise ValueError("Invalid encrypted data")
    
    def create_jwt_token(self, user_id: str, role: str = None, 
                       additional_claims: Dict[str, Any] = None) -> str:
        """Create a JWT token.
        
        Args:
            user_id: User ID
            role: User role
            additional_claims: Additional claims to include in the token
            
        Returns:
            JWT token
        """
        # Get role from user if not provided
        if not role and user_id in self.users:
            role = self.users[user_id].get("role")
        
        # Default to reader role if not specified
        role = role or "reader"
        
        # Check if role exists
        if role not in self.roles:
            raise ValueError(f"Invalid role: {role}")
        
        # Get permissions for this role
        permissions = self.roles[role].get("permissions", [])
        
        # Create JWT payload
        now = datetime.utcnow()
        payload = {
            "sub": user_id,
            "role": role,
            "permissions": permissions,
            "iat": now,
            "exp": now + timedelta(minutes=self.token_expiry_minutes),
            "jti": str(uuid.uuid4())
        }
        
        # Add additional claims
        if additional_claims:
            payload.update(additional_claims)
        
        # Create token
        token = jwt.encode(payload, self.secret_key, algorithm=self.jwt_algorithm)
        
        return token
    
    def validate_jwt_token(self, token: str) -> Dict[str, Any]:
        """Validate a JWT token.
        
        Args:
            token: JWT token
            
        Returns:
            Decoded token payload
            
        Raises:
            ValueError: If token is invalid or expired
        """
        try:
            # Check if token is blacklisted
            if token in self.token_blacklist:
                raise ValueError("Token has been revoked")
            
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.jwt_algorithm])
            
            return payload
        
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
    
    def revoke_token(self, token: str):
        """Revoke a JWT token.
        
        Args:
            token: JWT token
        """
        try:
            # Add to blacklist
            self.token_blacklist.add(token)
            
            # Get token ID
            payload = jwt.decode(token, self.secret_key, algorithms=[self.jwt_algorithm])
            token_id = payload.get("jti")
            
            self.logger.info(f"Revoked token {token_id}")
        
        except Exception as e:
            self.logger.error(f"Error revoking token: {str(e)}")
    
    def has_permission(self, user_id: str, required_permission: str) -> bool:
        """Check if a user has a specific permission.
        
        Args:
            user_id: User ID
            required_permission: Required permission
            
        Returns:
            True if user has permission, False otherwise
        """
        # Get user role
        if user_id not in self.users:
            return False
        
        role = self.users[user_id].get("role")
        
        # Get permissions for this role
        if role not in self.roles:
            return False
        
        permissions = self.roles[role].get("permissions", [])
        
        # Check for wildcard permission
        if "*" in permissions:
            return True
        
        # Check for exact permission
        if required_permission in permissions:
            return True
        
        # Check for prefix permission (e.g., "read:*")
        prefix = required_permission.split(":")[0] + ":*"
        if prefix in permissions:
            return True
        
        return False
    
    def create_hmac_signature(self, data: str) -> str:
        """Create an HMAC signature for data.
        
        Args:
            data: Data to sign
            
        Returns:
            HMAC signature
        """
        if isinstance(data, str):
            data = data.encode()
        
        signature = hmac.new(
            self.secret_key.encode(),
            data,
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_hmac_signature(self, data: str, signature: str) -> bool:
        """Verify an HMAC signature.
        
        Args:
            data: Data that was signed
            signature: HMAC signature
            
        Returns:
            True if signature is valid, False otherwise
        """
        expected_signature = self.create_hmac_signature(data)
        return hmac.compare_digest(expected_signature, signature)
    
    def validate_api_key(self, api_key: str = Security(APIKeyHeader(name="X-API-Key"))) -> bool:
        """Validate an API key.
        
        Args:
            api_key: API key
            
        Returns:
            True if API key is valid
            
        Raises:
            HTTPException: If API key is invalid
        """
        if api_key != self.api_key:
            raise HTTPException(status_code=403, detail="Invalid API key")
        
        return True
    
    def require_permission(self, required_permission: str):
        """Decorator to require a specific permission for an API endpoint.
        
        Args:
            required_permission: Required permission
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(request: Request, *args, **kwargs):
                # Get token from authorization header
                auth_header = request.headers.get("Authorization")
                if not auth_header or not auth_header.startswith("Bearer "):
                    raise HTTPException(status_code=401, detail="Authentication required")
                
                token = auth_header.split(" ")[1]
                
                try:
                    # Validate token
                    payload = self.validate_jwt_token(token)
                    
                    # Check permissions
                    user_id = payload.get("sub")
                    permissions = payload.get("permissions", [])
                    
                    if "*" in permissions or required_permission in permissions:
                        # User has permission
                        return await func(request, *args, **kwargs)
                    
                    # Check for prefix permission
                    prefix = required_permission.split(":")[0] + ":*"
                    if prefix in permissions:
                        return await func(request, *args, **kwargs)
                    
                    # No permission
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                except ValueError as e:
                    raise HTTPException(status_code=401, detail=str(e))
            
            return wrapper
        
        return decorator
    
    def load_security_config(self, config_file: str):
        """Load security configuration from a file.
        
        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Update configuration
            self.config.update(config)
            
            # Update key components
            if "secret_key" in config:
                self.secret_key = config["secret_key"]
            
            if "jwt_algorithm" in config:
                self.jwt_algorithm = config["jwt_algorithm"]
            
            if "token_expiry_minutes" in config:
                self.token_expiry_minutes = config["token_expiry_minutes"]
            
            if "api_key" in config:
                self.api_key = config["api_key"]
            
            if "roles" in config:
                self.roles = config["roles"]
            
            if "users" in config:
                self.users = config["users"]
            
            if "encryption_salt" in config:
                # Regenerate encryption key with new salt
                self.encryption_key = self._generate_encryption_key(
                    self.secret_key, 
                    config["encryption_salt"]
                )
                self.cipher = Fernet(self.encryption_key)
            
            self.logger.info(f"Loaded security configuration from {config_file}")
        
        except Exception as e:
            self.logger.error(f"Error loading security configuration: {str(e)}")
            raise

# Create a global security manager instance
_security_manager = None

def get_security_manager() -> SecurityManager:
    """Get the global security manager instance.
    
    Returns:
        Global security manager instance
    """
    global _security_manager
    if _security_manager is None:
        # Try to load configuration from environment variable
        config_file = os.environ.get("SECURITY_CONFIG_FILE")
        
        if config_file and os.path.exists(config_file):
            # Load configuration from file
            try:
                with open(config_file, 'r') as f:
                    security_config = json.load(f)
            except Exception as e:
                logging.error(f"Error loading security configuration: {str(e)}")
                security_config = {}
        else:
            # Use default configuration
            security_config = {}
        
        _security_manager = SecurityManager(security_config)
    
    return _security_manager


# Secure the agent's run method to enforce permissions
def secure_agent_execution(func):
    """Decorator to secure agent execution with permission checks.
    
    Args:
        func: Function to secure
        
    Returns:
        Secured function
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Get the current user from context
        user_id = kwargs.get("user_id")
        if not user_id:
            # No user specified, use default service account
            return await func(self, *args, **kwargs)
        
        # Check permission for executing this agent
        security_manager = get_security_manager()
        permission = f"execute:agent:{self.agent_id}"
        
        if not security_manager.has_permission(user_id, permission):
            self.logger.warning(f"User {user_id} does not have permission {permission}")
            raise PermissionError(f"User {user_id} does not have permission to execute agent {self.agent_id}")
        
        # Execute the function
        return await func(self, *args, **kwargs)
    
    return wrapper


# Secure workflow execution to enforce permissions
def secure_workflow_execution(func):
    """Decorator to secure workflow execution with permission checks.
    
    Args:
        func: Function to secure
        
    Returns:
        Secured function
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Get the current user from context
        user_id = kwargs.get("user_id")
        if not user_id:
            # No user specified, use default service account
            return await func(self, *args, **kwargs)
        
        # Get workflow type
        workflow_def = kwargs.get("workflow_def", {})
        workflow_type = workflow_def.get("name", "unknown")
        
        # Check permission for executing this workflow
        security_manager = get_security_manager()
        permission = f"execute:workflow:{workflow_type}"
        
        if not security_manager.has_permission(user_id, permission):
            self.logger.warning(f"User {user_id} does not have permission {permission}")
            raise PermissionError(f"User {user_id} does not have permission to execute workflow {workflow_type}")
        
        # Execute the function
        return await func(self, *args, **kwargs)
    
    return wrapper


# Example of securing an API endpoint
def secure_endpoint(permission: str):
    """Decorator to secure an API endpoint with permission checks.
    
    Args:
        permission: Required permission
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Get API key
            api_key = request.headers.get("X-API-Key")
            
            # Validate API key
            security_manager = get_security_manager()
            if api_key != security_manager.api_key:
                raise HTTPException(status_code=403, detail="Invalid API key")
            
            # If using JWT authentication as well
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                
                try:
                    # Validate token
                    payload = security_manager.validate_jwt_token(token)
                    
                    # Check permissions
                    permissions = payload.get("permissions", [])
                    
                    if not ("*" in permissions or permission in permissions or 
                          any(p.endswith(":*") and permission.startswith(p[:-1]) for p in permissions)):
                        # No permission
                        raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                except ValueError as e:
                    raise HTTPException(status_code=401, detail=str(e))
            
            # Execute the function
            return await func(request, *args, **kwargs)
        
        return wrapper
    
    return decorator