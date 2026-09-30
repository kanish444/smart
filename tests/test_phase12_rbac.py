import os
import datetime
import pytest
import jwt
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.user_service import UserService
from app.models.user import UserModel, UserRole, UserStatus
from app.core.security import hash_password, create_access_token, JWT_SECRET_KEY, JWT_ALGORITHM
from app.auth import get_user_service


@pytest.fixture(scope="module")
def rbac_client(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("rbac_phase12")
    test_db = str(temp_dir / "smartclass_rbac.sqlite")
    
    user_svc = UserService(db_path=test_db, mongo_uri="mongodb://localhost:27017")

    # Create Advisor ADV001
    adv = UserModel(
        user_id="ADV001",
        name="Prof. Advisor Ramesh",
        role=UserRole.CLASS_ADVISOR,
        password_hash=hash_password("advisor123"),
        email="advisor@smartclass.edu",
        department="AI&DS",
        year="3rd Year",
        section="B",
        assigned_classroom="AIDS-B",
        status=UserStatus.ACTIVE
    )
    user_svc.create_user(adv)

    # Create a Disabled Faculty account to test disabled account rejection
    disabled_faculty = UserModel(
        user_id="FAC_DISABLED",
        name="Disabled Faculty",
        role=UserRole.FACULTY,
        password_hash=hash_password("password123"),
        email="disabled@smartclass.edu",
        department="AI&DS",
        status=UserStatus.DISABLED
    )
    user_svc.create_user(disabled_faculty)

    app = create_app()
    app.dependency_overrides[get_user_service] = lambda: user_svc

    with TestClient(app) as client:
        yield {
            "client": client,
            "user_service": user_svc,
            "db_path": test_db
        }


# =============================================================================
# Helper token generators
# =============================================================================

def _get_token_for(client, user_id, password):
    resp = client.post("/api/auth/login", json={"user_id": user_id, "password": password})
    assert resp.status_code == 200, f"Login failed for {user_id}: {resp.text}"
    token = resp.json()["data"]["access_token"]
    # Clear client cookie after login so headers take explicit precedence
    client.cookies.clear()
    return token


# =============================================================================
# 1. Valid HOD Login
# =============================================================================
def test_01_valid_hod_login(rbac_client):
    c = rbac_client["client"]
    c.cookies.clear()
    resp = c.post("/api/auth/login", json={"user_id": "HOD001", "password": "admin123"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["user"]["role"] == "hod"
    assert data["user"]["user_id"] == "HOD001"
    assert data["redirect_url"] == "/hod/dashboard"
    assert "access_token" in data
    assert resp.cookies.get("access_token") is not None
    c.cookies.clear()


# =============================================================================
# 2. Valid Faculty Login
# =============================================================================
def test_02_valid_faculty_login(rbac_client):
    c = rbac_client["client"]
    c.cookies.clear()
    resp = c.post("/api/auth/login", json={"user_id": "FAC001", "password": "faculty123"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["user"]["role"] == "faculty"
    assert data["user"]["user_id"] == "FAC001"
    assert data["redirect_url"] == "/faculty/dashboard"
    assert "access_token" in data
    c.cookies.clear()


# =============================================================================
# 3. Valid Class Advisor Login
# =============================================================================
def test_03_valid_class_advisor_login(rbac_client):
    c = rbac_client["client"]
    c.cookies.clear()
    resp = c.post("/api/auth/login", json={"user_id": "ADV001", "password": "advisor123"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["user"]["role"] == "class_advisor"
    assert data["user"]["user_id"] == "ADV001"
    assert data["redirect_url"] == "/advisor/dashboard"
    assert "access_token" in data
    c.cookies.clear()


# =============================================================================
# 4. Invalid Username
# =============================================================================
def test_04_invalid_username(rbac_client):
    c = rbac_client["client"]
    c.cookies.clear()
    resp = c.post("/api/auth/login", json={"user_id": "NON_EXISTENT_USER", "password": "anypassword"})
    assert resp.status_code == 401
    body = resp.json()
    assert body["success"] is False
    assert "Invalid User ID or password" in body["error"]["message"]


# =============================================================================
# 5. Invalid Password
# =============================================================================
def test_05_invalid_password(rbac_client):
    c = rbac_client["client"]
    c.cookies.clear()
    resp = c.post("/api/auth/login", json={"user_id": "HOD001", "password": "wrong_password_123"})
    assert resp.status_code == 401
    body = resp.json()
    assert body["success"] is False
    assert "Invalid User ID or password" in body["error"]["message"]


# =============================================================================
# 6. Missing Authentication
# =============================================================================
def test_06_missing_authentication(rbac_client):
    c = rbac_client["client"]
    c.cookies.clear()
    resp = c.get("/api/auth/me")
    assert resp.status_code == 401
    body = resp.json()
    assert body["success"] is False
    assert "Authentication required" in body["error"]["message"]


# =============================================================================
# 7. Invalid JWT Token
# =============================================================================
def test_07_invalid_jwt(rbac_client):
    c = rbac_client["client"]
    c.cookies.clear()
    resp = c.get("/api/auth/me", headers={"Authorization": "Bearer completely.invalid.jwt.token"})
    assert resp.status_code == 401
    body = resp.json()
    assert body["success"] is False


# =============================================================================
# 8. Expired JWT Token
# =============================================================================
def test_08_expired_jwt(rbac_client):
    c = rbac_client["client"]
    c.cookies.clear()
    expired_payload = {
        "sub": "HOD001",
        "role": "hod",
        "name": "Dr. Arunkumar HOD",
        "exp": datetime.datetime.utcnow() - datetime.timedelta(hours=2),
        "iat": datetime.datetime.utcnow() - datetime.timedelta(hours=3)
    }
    expired_token = jwt.encode(expired_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    resp = c.get("/api/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
    assert resp.status_code == 401
    body = resp.json()
    assert body["success"] is False
    assert "expired" in body["error"]["message"].lower()


# =============================================================================
# 9. HOD Accessing HOD-Protected Endpoint
# =============================================================================
def test_09_hod_accessing_hod_protected_endpoint(rbac_client):
    c = rbac_client["client"]
    c.cookies.clear()
    token = _get_token_for(c, "HOD001", "admin123")

    # HOD can access faculty list
    resp_fac = c.get("/api/hod/faculty", headers={"Authorization": f"Bearer {token}"})
    assert resp_fac.status_code == 200
    assert resp_fac.json()["success"] is True

    # HOD can access advisors list
    resp_adv = c.get("/api/hod/advisors", headers={"Authorization": f"Bearer {token}"})
    assert resp_adv.status_code == 200
    assert resp_adv.json()["success"] is True


# =============================================================================
# 10. Faculty Accessing Faculty Endpoint
# =============================================================================
def test_10_faculty_accessing_faculty_endpoint(rbac_client):
    c = rbac_client["client"]
    c.cookies.clear()
    token = _get_token_for(c, "FAC001", "faculty123")

    # Faculty can access dashboard role summary
    resp_dash = c.get("/api/dashboard/role-summary", headers={"Authorization": f"Bearer {token}"})
    assert resp_dash.status_code == 200
    data = resp_dash.json()["data"]
    assert data["role"] == "faculty"
    assert data["faculty_name"] == "Dr. Anand Kumar"

    # Faculty can access sensor status
    resp_sensors = c.get("/api/sensors/status", headers={"Authorization": f"Bearer {token}"})
    assert resp_sensors.status_code == 200
    assert resp_sensors.json()["data"]["status"] == "ONLINE"



# =============================================================================
# 11. Class Advisor Accessing Advisor Endpoint
# =============================================================================
def test_11_class_advisor_accessing_advisor_endpoint(rbac_client):
    c = rbac_client["client"]
    token = _get_token_for(c, "ADV001", "advisor123")

    # Advisor can access classroom status
    resp_status = c.get("/api/advisor/classroom-status", headers={"Authorization": f"Bearer {token}"})
    assert resp_status.status_code == 200
    assert resp_status.json()["success"] is True

    # Advisor can access scoped students
    resp_students = c.get("/api/scoped/students", headers={"Authorization": f"Bearer {token}"})
    assert resp_students.status_code == 200
    assert resp_students.json()["success"] is True


# =============================================================================
# 12. Faculty Attempting HOD-Only Endpoint
# =============================================================================
def test_12_faculty_attempting_hod_only_endpoint(rbac_client):
    c = rbac_client["client"]
    token = _get_token_for(c, "FAC001", "faculty123")

    # Faculty must NOT be able to list advisors (HOD only)
    resp = c.get("/api/hod/advisors", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403
    assert "HOD role required" in resp.json()["error"]["message"]

    # Faculty must NOT be able to create new faculty
    new_faculty = {
        "user_id": "FAC_NEW",
        "name": "Unauthorized Faculty",
        "password": "password123",
        "department": "AI&DS"
    }
    resp_create = c.post("/api/hod/faculty", json=new_faculty, headers={"Authorization": f"Bearer {token}"})
    assert resp_create.status_code == 403
    assert "HOD role required" in resp_create.json()["error"]["message"]


# =============================================================================
# 13. Class Advisor Attempting HOD-Only Endpoint
# =============================================================================
def test_13_class_advisor_attempting_hod_only_endpoint(rbac_client):
    c = rbac_client["client"]
    token = _get_token_for(c, "ADV001", "advisor123")

    # Advisor must NOT be able to access faculty management
    resp = c.get("/api/hod/faculty", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403
    assert "HOD role required" in resp.json()["error"]["message"]

    # Advisor must NOT be able to reset advisor passwords
    resp_reset = c.post(
        "/api/hod/advisors/ADV001/reset-password",
        json={"new_password": "newpass"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp_reset.status_code == 403
    assert "HOD role required" in resp_reset.json()["error"]["message"]


# =============================================================================
# 14. Unauthenticated Request to Protected Endpoints
# =============================================================================
def test_14_unauthenticated_request_to_protected_endpoints(rbac_client):
    c = rbac_client["client"]
    c.cookies.clear()
    protected_endpoints = [
        "/api/auth/me",
        "/api/hod/faculty",
        "/api/hod/advisors",
        "/api/scoped/students",
        "/api/sensors/status"
    ]
    for ep in protected_endpoints:
        resp = c.get(ep)
        assert resp.status_code == 401, f"Expected 401 for unauthenticated {ep}, got {resp.status_code}"
        assert resp.json()["success"] is False


# =============================================================================
# 15. Malformed or Invalid Authorization Header
# =============================================================================
def test_15_malformed_authorization_header(rbac_client):
    c = rbac_client["client"]
    c.cookies.clear()
    malformed_headers = [
        {"Authorization": "Basic YWRtaW46cGFzc3dvcmQ="},  # Non-bearer scheme
        {"Authorization": "Bearer"},                     # Empty bearer token
        {"Authorization": "NotBearerTokenAtAll"},         # Garbage header
    ]
    for h in malformed_headers:
        resp = c.get("/api/auth/me", headers=h)
        assert resp.status_code == 401
        assert resp.json()["success"] is False



# =============================================================================
# 16. Role Extraction From Authenticated User
# =============================================================================
def test_16_role_extraction_from_authenticated_user(rbac_client):
    c = rbac_client["client"]

    # 1. HOD profile check
    hod_tok = _get_token_for(c, "HOD001", "admin123")
    res_hod = c.get("/api/auth/me", headers={"Authorization": f"Bearer {hod_tok}"})
    assert res_hod.status_code == 200
    assert res_hod.json()["data"]["role"] == "hod"

    # 2. Faculty profile check
    fac_tok = _get_token_for(c, "FAC001", "faculty123")
    res_fac = c.get("/api/auth/me", headers={"Authorization": f"Bearer {fac_tok}"})
    assert res_fac.status_code == 200
    assert res_fac.json()["data"]["role"] == "faculty"

    # 3. Advisor profile check
    adv_tok = _get_token_for(c, "ADV001", "advisor123")
    res_adv = c.get("/api/auth/me", headers={"Authorization": f"Bearer {adv_tok}"})
    assert res_adv.status_code == 200
    assert res_adv.json()["data"]["role"] == "class_advisor"


# =============================================================================
# 17. All Three Roles Reaching Permitted Shared Endpoints
# =============================================================================
def test_17_all_three_roles_reaching_permitted_shared_endpoints(rbac_client):
    c = rbac_client["client"]
    tokens = {
        "hod": _get_token_for(c, "HOD001", "admin123"),
        "faculty": _get_token_for(c, "FAC001", "faculty123"),
        "class_advisor": _get_token_for(c, "ADV001", "advisor123")
    }

    shared_endpoints = [
        "/api/dashboard/role-summary",
        "/api/sensors/status"
    ]

    for role_name, token in tokens.items():
        for ep in shared_endpoints:
            resp = c.get(ep, headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == 200, f"Role {role_name} failed to access shared endpoint {ep}: {resp.status_code}"
            assert resp.json()["success"] is True


# =============================================================================
# 18. No Accidental Privilege Escalation
# =============================================================================
def test_18_no_accidental_privilege_escalation(rbac_client):
    c = rbac_client["client"]
    fac_tok = _get_token_for(c, "FAC001", "faculty123")
    adv_tok = _get_token_for(c, "ADV001", "advisor123")

    # Faculty CANNOT access student directory management
    resp_fac_students = c.get("/api/scoped/students", headers={"Authorization": f"Bearer {fac_tok}"})
    assert resp_fac_students.status_code == 403

    # Advisor CANNOT access faculty management
    resp_adv_fac = c.get("/api/hod/faculty", headers={"Authorization": f"Bearer {adv_tok}"})
    assert resp_adv_fac.status_code == 403

    # Disabled account cannot login
    resp_dis_login = c.post("/api/auth/login", json={"user_id": "FAC_DISABLED", "password": "password123"})
    assert resp_dis_login.status_code == 403
    assert "disabled" in resp_dis_login.json()["error"]["message"].lower()

    # Disabled account cannot use previously issued token
    disabled_token = create_access_token(user_id="FAC_DISABLED", role="faculty", name="Disabled Faculty")
    resp_dis_tok = c.get("/api/auth/me", headers={"Authorization": f"Bearer {disabled_token}"})
    assert resp_dis_tok.status_code == 403
    assert "disabled" in resp_dis_tok.json()["error"]["message"].lower()
