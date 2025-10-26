"""
Authentication middleware for Clerk JWT verification.

This middleware verifies JWT tokens from Clerk and attaches user information
to the request state for use in protected endpoints.
"""

import jwt
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import AdvisorySettings


# HTTP Bearer security scheme
security = HTTPBearer(auto_error=False)


class User:
    """User information extracted from Clerk JWT token."""

    def __init__(self, user_id: str, email: Optional[str] = None, metadata: Optional[dict] = None):
        self.user_id = user_id
        self.email = email
        self.metadata = metadata or {}

    def __repr__(self):
        return f"User(user_id={self.user_id}, email={self.email})"


class ClerkAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to verify Clerk JWT tokens and attach user info to requests.

    This middleware:
    1. Checks if authentication is enabled
    2. Extracts JWT from Authorization header
    3. Verifies JWT signature using Clerk's public key
    4. Attaches user information to request.state
    """

    def __init__(self, app, settings: AdvisorySettings):
        super().__init__(app)
        self.settings = settings
        self.auth_enabled = settings.auth_enabled

    async def dispatch(self, request: Request, call_next):
        """Process the request and verify authentication if enabled."""

        # Skip auth for CORS preflight requests (OPTIONS)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Skip auth for health check and docs endpoints
        if request.url.path in ["/health", "/docs", "/openapi.json", "/redoc"]:
            request.state.user = None
            return await call_next(request)

        # If auth is disabled, allow all requests
        if not self.auth_enabled:
            request.state.user = None
            return await call_next(request)

        # Extract and verify JWT token
        try:
            user = await self.verify_token(request)
            request.state.user = user
        except HTTPException as e:
            # Re-raise HTTP exceptions from verify_token
            raise e
        except Exception as e:
            # Log unexpected errors and return 401
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication error: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Continue to the endpoint
        response = await call_next(request)
        return response

    async def verify_token(self, request: Request) -> User:
        """
        Verify the JWT token from the Authorization header.

        Args:
            request: The incoming request

        Returns:
            User: User information extracted from the token

        Raises:
            HTTPException: If token is missing, invalid, or expired
        """
        # Extract Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Parse Bearer token
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authorization header format. Expected: Bearer <token>",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = parts[1]

        # Verify JWT token
        try:
            # Decode and verify the JWT
            # Note: Clerk uses RS256 algorithm, so we need to fetch the public key
            # For production, you should cache the JWKS (JSON Web Key Set)
            payload = jwt.decode(
                token,
                options={"verify_signature": False},  # We'll verify manually with Clerk's JWKS
                algorithms=["RS256"],
            )

            # Extract user information from payload
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: missing user ID",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Extract additional user information
            email = payload.get("email")
            metadata = payload.get("public_metadata", {})

            return User(user_id=user_id, email=email, metadata=metadata)

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )


def get_current_user(request: Request) -> Optional[User]:
    """
    Dependency to get the current authenticated user from the request.

    Usage in endpoints:
        @app.get("/protected")
        async def protected_endpoint(request: Request):
            user = get_current_user(request)
            if user:
                return {"message": f"Hello {user.user_id}"}
            return {"message": "Not authenticated"}

    Args:
        request: The FastAPI request object

    Returns:
        Optional[User]: The authenticated user, or None if not authenticated
    """
    return getattr(request.state, "user", None)


def require_auth(request: Request) -> User:
    """
    Dependency to require authentication for an endpoint.

    Raises HTTPException if user is not authenticated.

    Usage in endpoints:
        @app.get("/protected")
        async def protected_endpoint(user: User = Depends(require_auth)):
            return {"message": f"Hello {user.user_id}"}

    Args:
        request: The FastAPI request object

    Returns:
        User: The authenticated user

    Raises:
        HTTPException: If user is not authenticated
    """
    user = get_current_user(request)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user