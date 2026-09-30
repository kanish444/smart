import os
from typing import Optional, List, Dict, Any, Callable
from fastapi import Header, Query, Cookie, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.core.security import decode_access_token, validate_access_token
from app.models.user import UserModel, UserRole, UserStatus
from app.services.user_service import UserService

# Backward-compatibility operator token support
OPERATOR_TOKEN = os.getenv("OPERATOR_TOKEN", "smartclass_operator_2026")

bearer_scheme = HTTPBearer(auto_error=False)

_user_service: Optional[UserService] = None


def get_user_service() -> UserService:
    global _user_service
    if _user_service is None:
        _user_service = UserService()
    return _user_service


class AuthContext(BaseModel):
    """Context object describing authenticated caller."""
    user_id: str
    name: str
    role: str
    department: Optional[str] = None
    year: Optional[str] = None
    section: Optional[str] = None
    assigned_classroom: Optional[str] = None
    is_hod: bool = False
    is_faculty: bool = False
    is_advisor: bool = False
    is_operator: bool = False


def get_current_user(
    auth_header: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    access_token_cookie: Optional[str] = Cookie(None, alias="access_token"),
    token_query: Optional[str] = Query(None, alias="token"),
    x_operator_token: Optional[str] = Header(None, alias="X-Operator-Token"),
    operator_token: Optional[str] = Query(None, alias="operator_token"),
    user_service: UserService = Depends(get_user_service)
) -> UserModel:
    """
    Extracts and authenticates the user from JWT Bearer Header, Cookie, Query, or Operator Token.
    Validates token signature, expiration, user existence, and account active status.
    """
    token = None
    if auth_header and auth_header.credentials:
        token = auth_header.credentials
    elif access_token_cookie:
        token = access_token_cookie
    elif token_query:
        token = token_query

    if token:
        try:
            payload = validate_access_token(token)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"}
            )

        if not payload or "sub" not in payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token payload.",
                headers={"WWW-Authenticate": "Bearer"}
            )

        user_id = payload["sub"]
        user = user_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account not found.",
                headers={"WWW-Authenticate": "Bearer"}
            )
        if user.status != UserStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is disabled. Contact system administrator."
            )
        return user

    # Fallback legacy operator token support
    op_tok = x_operator_token or operator_token
    if op_tok and op_tok == OPERATOR_TOKEN:
        # Generate virtual operator HOD user
        return UserModel(
            user_id="OPERATOR",
            name="System Operator",
            role=UserRole.HOD,
            password_hash="",
            status=UserStatus.ACTIVE
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Please provide a valid Bearer token.",
        headers={"WWW-Authenticate": "Bearer"}
    )


# =============================================================================
# Reusable Role Dependency Factory
# =============================================================================

def require_roles(*allowed_roles: UserRole) -> Callable:
    """Dependency factory restricting endpoint access to one or more UserRole enums."""
    def role_checker(user: UserModel = Depends(get_current_user)) -> UserModel:
        if user.role not in allowed_roles:
            if len(allowed_roles) == 1 and allowed_roles[0] == UserRole.HOD:
                msg = "Access forbidden: HOD role required."
            elif len(allowed_roles) == 1 and allowed_roles[0] == UserRole.CLASS_ADVISOR:
                msg = "Access forbidden: Class Advisor role required."
            elif len(allowed_roles) == 1 and allowed_roles[0] == UserRole.FACULTY:
                msg = "Access forbidden: Faculty role required."
            else:
                allowed_names = ", ".join(r.value.upper() for r in allowed_roles)
                msg = f"Access forbidden: requires one of [{allowed_names}] role(s)."
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=msg
            )
        return user
    return role_checker


# Single & Multi-Role Dependencies
require_authenticated_user = get_current_user
require_any_authenticated_user = get_current_user
require_hod = require_roles(UserRole.HOD)
require_faculty = require_roles(UserRole.FACULTY)
require_class_advisor = require_roles(UserRole.CLASS_ADVISOR)
require_hod_or_faculty = require_roles(UserRole.HOD, UserRole.FACULTY)
require_hod_or_advisor = require_roles(UserRole.HOD, UserRole.CLASS_ADVISOR)

# Semantic Functional Dependencies (Aligned with Phase 12 Permission Matrix)
require_dashboard_access = require_roles(UserRole.HOD, UserRole.FACULTY, UserRole.CLASS_ADVISOR)
require_live_classroom_access = require_roles(UserRole.HOD, UserRole.FACULTY, UserRole.CLASS_ADVISOR)
require_attendance_access = require_roles(UserRole.HOD, UserRole.FACULTY, UserRole.CLASS_ADVISOR)
require_sensor_access = require_roles(UserRole.HOD, UserRole.FACULTY, UserRole.CLASS_ADVISOR)
require_student_management = require_roles(UserRole.HOD, UserRole.CLASS_ADVISOR)
require_faculty_management = require_roles(UserRole.HOD)
require_classroom_management = require_roles(UserRole.HOD)
require_session_control = require_roles(UserRole.HOD, UserRole.FACULTY)


def get_current_role(
    x_operator_token: Optional[str] = Header(None, alias="X-Operator-Token"),
    operator_token: Optional[str] = Query(None, alias="operator_token"),
    auth_header: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    access_token_cookie: Optional[str] = Cookie(None, alias="access_token"),
    token_query: Optional[str] = Query(None, alias="token"),
    user_service: UserService = Depends(get_user_service)
) -> AuthContext:
    """Legacy compatibility helper extracting AuthContext."""
    token = None
    if auth_header and auth_header.credentials:
        token = auth_header.credentials
    elif access_token_cookie:
        token = access_token_cookie
    elif token_query:
        token = token_query

    if token:
        payload = decode_access_token(token)
        if payload and "sub" in payload:
            user = user_service.get_user_by_id(payload["sub"])
            if user and user.status == UserStatus.ACTIVE:
                is_hod = (user.role == UserRole.HOD)
                is_faculty = (user.role == UserRole.FACULTY)
                is_advisor = (user.role == UserRole.CLASS_ADVISOR)
                return AuthContext(
                    user_id=user.user_id,
                    name=user.name,
                    role=user.role.value,
                    department=user.department,
                    year=user.year,
                    section=user.section,
                    assigned_classroom=user.assigned_classroom,
                    is_hod=is_hod,
                    is_faculty=is_faculty,
                    is_advisor=is_advisor,
                    is_operator=is_hod
                )

    op_tok = x_operator_token or operator_token
    if op_tok and op_tok == OPERATOR_TOKEN:
        return AuthContext(user_id="OPERATOR", name="Operator", role="hod", is_hod=True, is_operator=True)

    return AuthContext(
        user_id="ANONYMOUS",
        name="Viewer",
        role="viewer",
        is_hod=False,
        is_faculty=False,
        is_advisor=False,
        is_operator=False
    )


def require_operator(
    auth: AuthContext = Depends(get_current_role)
) -> AuthContext:
    """Legacy compatibility wrapper enforcing operator / HOD privileges."""
    if not auth.is_operator:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operator authorization required for this operation."
        )
    return auth

