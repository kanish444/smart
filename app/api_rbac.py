import os
import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Response, UploadFile, File, Form
from pydantic import BaseModel
from app.services.excel_service import ExcelService

from app.models.user import (
    UserModel,
    UserRole,
    UserStatus,
    UserPublicProfile,
    LoginRequest,
    LoginResponse,
    CreateAdvisorRequest,
    UpdateAdvisorRequest,
    ResetAdvisorPasswordRequest,
    CreateFacultyRequest,
    UpdateFacultyRequest,
    ResetFacultyPasswordRequest,
    ChangePasswordRequest
)
from app.core.security import verify_password, hash_password, create_access_token
from app.auth import (
    get_user_service,
    get_current_user,
    require_hod,
    require_faculty,
    require_class_advisor,
    require_any_authenticated_user,
    require_dashboard_access,
    require_live_classroom_access,
    require_attendance_access,
    require_sensor_access,
    require_student_management,
    require_faculty_management,
    require_classroom_management,
    require_session_control
)
from app.schemas import ApiResponse
from app.state import AppState, get_app_state

router = APIRouter(prefix="/api", tags=["RBAC Authentication & Dashboards"])


# =============================================================================
# 1. Authentication Endpoints
# =============================================================================

@router.post("/auth/login", response_model=ApiResponse[LoginResponse])
def login(
    payload: LoginRequest,
    response: Response,
    user_service = Depends(get_user_service)
):
    """
    Authenticates HOD, Faculty, or Class Advisor:
    - Verifies user ID and hashed password
    - Verifies account is active (disabled accounts rejected with 403)
    - Returns signed JWT token, profile, and target dashboard URL
    - Sets secure HTTP-only access_token cookie
    """
    user_id = payload.user_id.strip()
    user = user_service.get_user_by_id_or_email(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid User ID or password."
        )

    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid User ID or password."
        )

    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been disabled. Please contact the HOD."
        )

    # Determine redirect URL based strictly on backend role
    if user.role == UserRole.HOD:
        redirect_url = "/hod/dashboard"
    elif user.role == UserRole.FACULTY:
        redirect_url = "/faculty/dashboard"
    elif user.role == UserRole.CLASS_ADVISOR:
        redirect_url = "/advisor/dashboard"
    else:
        redirect_url = "/"

    token = create_access_token(
        user_id=user.user_id,
        role=user.role.value,
        name=user.name,
        department=user.department,
        year=user.year,
        section=user.section,
        assigned_classroom=user.assigned_classroom
    )

    # Set cookie for browser navigation
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=43200,  # 12 hours
        samesite="lax",
        secure=False
    )

    profile = UserPublicProfile(
        user_id=user.user_id,
        name=user.name,
        role=user.role.value,
        email=user.email,
        phone=user.phone,
        department=user.department,
        year=user.year,
        section=user.section,
        assigned_classroom=user.assigned_classroom,
        status=user.status.value,
        created_at=user.created_at
    )

    return ApiResponse.ok(LoginResponse(
        access_token=token,
        token_type="bearer",
        user=profile,
        redirect_url=redirect_url
    ))


@router.post("/auth/logout", response_model=ApiResponse[Dict[str, Any]])
def logout(response: Response):
    """Logs out by clearing access_token cookie."""
    response.delete_cookie(key="access_token")
    return ApiResponse.ok({"message": "Successfully logged out.", "redirect_url": "/login"})


@router.get("/auth/me", response_model=ApiResponse[UserPublicProfile])
def get_current_user_profile(user: UserModel = Depends(get_current_user)):
    """Retrieves profile of currently authenticated user."""
    profile = UserPublicProfile(
        user_id=user.user_id,
        name=user.name,
        role=user.role.value,
        email=user.email,
        phone=user.phone,
        department=user.department,
        year=user.year,
        section=user.section,
        assigned_classroom=user.assigned_classroom,
        status=user.status.value,
        created_at=user.created_at
    )
    return ApiResponse.ok(profile)


@router.post("/auth/change-password", response_model=ApiResponse[Dict[str, Any]])
def change_password(
    payload: ChangePasswordRequest,
    user: UserModel = Depends(get_current_user),
    user_service = Depends(get_user_service)
):
    """Allows authenticated user (HOD or Class Advisor) to change their own password."""
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password incorrect.")

    if payload.confirm_new_password and payload.new_password != payload.confirm_new_password:
        raise HTTPException(status_code=400, detail="New passwords do not match.")

    new_hash = hash_password(payload.new_password)
    user_service.update_password(user.user_id, new_hash)
    return ApiResponse.ok({"message": "Password changed successfully."})


# =============================================================================
# 2. HOD Advisor Management Endpoints (HOD Role Strictly Required)
# =============================================================================

@router.get("/hod/advisors", response_model=ApiResponse[List[UserPublicProfile]])
def list_advisors(
    current_hod: UserModel = Depends(require_hod),
    user_service = Depends(get_user_service)
):
    """Lists all Class Advisors (HOD only)."""
    advisors = user_service.list_advisors()
    res = [
        UserPublicProfile(
            user_id=a.user_id,
            name=a.name,
            role=a.role.value,
            email=a.email,
            phone=a.phone,
            department=a.department,
            year=a.year,
            section=a.section,
            assigned_classroom=a.assigned_classroom,
            status=a.status.value,
            created_at=a.created_at
        )
        for a in advisors
    ]
    return ApiResponse.ok(res)


@router.post("/hod/advisors", response_model=ApiResponse[UserPublicProfile], status_code=status.HTTP_201_CREATED)
def create_advisor(
    payload: CreateAdvisorRequest,
    current_hod: UserModel = Depends(require_hod),
    user_service = Depends(get_user_service)
):
    """Creates a new Class Advisor account (HOD only)."""
    existing = user_service.get_user_by_id(payload.user_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Advisor with User ID '{payload.user_id}' already exists."
        )

    if payload.confirm_password and payload.password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    hashed = hash_password(payload.password)
    now = datetime.datetime.utcnow().isoformat()
    status_enum = UserStatus.ACTIVE if (payload.status or "active").lower() == "active" else UserStatus.DISABLED

    advisor = UserModel(
        user_id=payload.user_id.strip(),
        name=payload.name.strip(),
        role=UserRole.CLASS_ADVISOR,
        password_hash=hashed,
        email=payload.email,
        phone=payload.phone,
        department=payload.department.strip(),
        year=payload.year.strip(),
        section=payload.section.strip().upper(),
        assigned_classroom=payload.assigned_classroom or f"{payload.department}-{payload.section}".strip(" -"),
        status=status_enum,
        created_at=now,
        updated_at=now
    )

    created = user_service.create_user(advisor)

    return ApiResponse.ok(UserPublicProfile(
        user_id=created.user_id,
        name=created.name,
        role=created.role.value,
        email=created.email,
        phone=created.phone,
        department=created.department,
        year=created.year,
        section=created.section,
        assigned_classroom=created.assigned_classroom,
        status=created.status.value,
        created_at=created.created_at
    ))


@router.put("/hod/advisors/{user_id}", response_model=ApiResponse[UserPublicProfile])
def update_advisor(
    user_id: str = Path(...),
    payload: UpdateAdvisorRequest = None,
    current_hod: UserModel = Depends(require_hod),
    user_service = Depends(get_user_service)
):
    """Updates an existing Advisor's details or assigned class/section (HOD only)."""
    adv = user_service.get_user_by_id(user_id)
    if not adv:
        raise HTTPException(status_code=404, detail=f"Advisor '{user_id}' not found.")
    if adv.role != UserRole.CLASS_ADVISOR:
        raise HTTPException(status_code=400, detail="Target user is not a Class Advisor.")

    updates = {}
    if payload.name is not None:
        updates["name"] = payload.name
    if payload.email is not None:
        updates["email"] = payload.email
    if payload.phone is not None:
        updates["phone"] = payload.phone
    if payload.department is not None:
        updates["department"] = payload.department
    if payload.year is not None:
        updates["year"] = payload.year
    if payload.section is not None:
        updates["section"] = payload.section.upper()
    if payload.assigned_classroom is not None:
        updates["assigned_classroom"] = payload.assigned_classroom
    if payload.status is not None:
        updates["status"] = payload.status.lower()

    updated = user_service.update_user(user_id, updates)
    return ApiResponse.ok(UserPublicProfile(
        user_id=updated.user_id,
        name=updated.name,
        role=updated.role.value,
        email=updated.email,
        phone=updated.phone,
        department=updated.department,
        year=updated.year,
        section=updated.section,
        assigned_classroom=updated.assigned_classroom,
        status=updated.status.value,
        created_at=updated.created_at
    ))


@router.patch("/hod/advisors/{user_id}/status", response_model=ApiResponse[Dict[str, Any]])
def update_advisor_status(
    user_id: str = Path(...),
    status_val: str = Query(..., alias="status", pattern="^(active|disabled)$"),
    current_hod: UserModel = Depends(require_hod),
    user_service = Depends(get_user_service)
):
    """Enables or disables an Advisor account (HOD only)."""
    adv = user_service.get_user_by_id(user_id)
    if not adv:
        raise HTTPException(status_code=404, detail=f"Advisor '{user_id}' not found.")
    if adv.role != UserRole.CLASS_ADVISOR:
        raise HTTPException(status_code=400, detail="Cannot toggle status of non-advisor account.")

    user_service.update_status(user_id, status_val)
    return ApiResponse.ok({
        "user_id": user_id,
        "status": status_val,
        "message": f"Advisor account '{user_id}' is now {status_val}."
    })


@router.post("/hod/advisors/{user_id}/reset-password", response_model=ApiResponse[Dict[str, Any]])
def reset_advisor_password(
    user_id: str = Path(...),
    payload: ResetAdvisorPasswordRequest = None,
    current_hod: UserModel = Depends(require_hod),
    user_service = Depends(get_user_service)
):
    """Resets an Advisor's password (HOD only)."""
    adv = user_service.get_user_by_id(user_id)
    if not adv:
        raise HTTPException(status_code=404, detail=f"Advisor '{user_id}' not found.")
    if adv.role != UserRole.CLASS_ADVISOR:
        raise HTTPException(status_code=400, detail="Cannot reset password of non-advisor account.")

    new_hash = hash_password(payload.new_password)
    user_service.update_password(user_id, new_hash)
    return ApiResponse.ok({
        "user_id": user_id,
        "message": f"Password for Advisor '{user_id}' has been reset successfully."
    })


# =============================================================================
# 2b. HOD Faculty Management Endpoints (HOD Role Strictly Required)
# =============================================================================

@router.get("/hod/faculty", response_model=ApiResponse[List[UserPublicProfile]])
def list_faculty(
    current_hod: UserModel = Depends(require_faculty_management),
    user_service = Depends(get_user_service)
):
    """Lists all Faculty members (HOD only)."""
    faculty = user_service.list_faculty()
    res = [
        UserPublicProfile(
            user_id=f.user_id,
            name=f.name,
            role=f.role.value,
            email=f.email,
            phone=f.phone,
            department=f.department,
            year=f.year,
            section=f.section,
            assigned_classroom=f.assigned_classroom,
            status=f.status.value,
            created_at=f.created_at
        )
        for f in faculty
    ]
    return ApiResponse.ok(res)


@router.post("/hod/faculty", response_model=ApiResponse[UserPublicProfile], status_code=status.HTTP_201_CREATED)
def create_faculty(
    payload: CreateFacultyRequest,
    current_hod: UserModel = Depends(require_faculty_management),
    user_service = Depends(get_user_service)
):
    """Creates a new Faculty account (HOD only)."""
    existing = user_service.get_user_by_id(payload.user_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Faculty with User ID '{payload.user_id}' already exists."
        )

    if payload.confirm_password and payload.password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    hashed = hash_password(payload.password)
    now = datetime.datetime.utcnow().isoformat()
    status_enum = UserStatus.ACTIVE if (payload.status or "active").lower() == "active" else UserStatus.DISABLED

    faculty_member = UserModel(
        user_id=payload.user_id.strip(),
        name=payload.name.strip(),
        role=UserRole.FACULTY,
        password_hash=hashed,
        email=payload.email,
        phone=payload.phone,
        department=payload.department.strip(),
        year=payload.year.strip() if payload.year else None,
        section=payload.section.strip().upper() if payload.section else None,
        assigned_classroom=payload.assigned_classroom,
        status=status_enum,
        created_at=now,
        updated_at=now
    )

    created = user_service.create_user(faculty_member)

    return ApiResponse.ok(UserPublicProfile(
        user_id=created.user_id,
        name=created.name,
        role=created.role.value,
        email=created.email,
        phone=created.phone,
        department=created.department,
        year=created.year,
        section=created.section,
        assigned_classroom=created.assigned_classroom,
        status=created.status.value,
        created_at=created.created_at
    ))


@router.put("/hod/faculty/{user_id}", response_model=ApiResponse[UserPublicProfile])
def update_faculty(
    user_id: str = Path(...),
    payload: UpdateFacultyRequest = None,
    current_hod: UserModel = Depends(require_faculty_management),
    user_service = Depends(get_user_service)
):
    """Updates an existing Faculty member's details or assigned classroom (HOD only)."""
    f = user_service.get_user_by_id(user_id)
    if not f:
        raise HTTPException(status_code=404, detail=f"Faculty '{user_id}' not found.")
    if f.role != UserRole.FACULTY:
        raise HTTPException(status_code=400, detail="Target user is not a Faculty member.")

    updates = {}
    if payload.name is not None:
        updates["name"] = payload.name
    if payload.email is not None:
        updates["email"] = payload.email
    if payload.phone is not None:
        updates["phone"] = payload.phone
    if payload.department is not None:
        updates["department"] = payload.department
    if payload.year is not None:
        updates["year"] = payload.year
    if payload.section is not None:
        updates["section"] = payload.section.upper()
    if payload.assigned_classroom is not None:
        updates["assigned_classroom"] = payload.assigned_classroom
    if payload.status is not None:
        updates["status"] = payload.status.lower()

    updated = user_service.update_user(user_id, updates)
    return ApiResponse.ok(UserPublicProfile(
        user_id=updated.user_id,
        name=updated.name,
        role=updated.role.value,
        email=updated.email,
        phone=updated.phone,
        department=updated.department,
        year=updated.year,
        section=updated.section,
        assigned_classroom=updated.assigned_classroom,
        status=updated.status.value,
        created_at=updated.created_at
    ))


@router.patch("/hod/faculty/{user_id}/status", response_model=ApiResponse[Dict[str, Any]])
def update_faculty_status(
    user_id: str = Path(...),
    status_val: str = Query(..., alias="status", pattern="^(active|disabled)$"),
    current_hod: UserModel = Depends(require_faculty_management),
    user_service = Depends(get_user_service)
):
    """Enables or disables a Faculty account (HOD only)."""
    f = user_service.get_user_by_id(user_id)
    if not f:
        raise HTTPException(status_code=404, detail=f"Faculty '{user_id}' not found.")
    if f.role != UserRole.FACULTY:
        raise HTTPException(status_code=400, detail="Cannot toggle status of non-faculty account.")

    user_service.update_status(user_id, status_val)
    return ApiResponse.ok({
        "user_id": user_id,
        "status": status_val,
        "message": f"Faculty account '{user_id}' is now {status_val}."
    })


@router.post("/hod/faculty/{user_id}/reset-password", response_model=ApiResponse[Dict[str, Any]])
def reset_faculty_password(
    user_id: str = Path(...),
    payload: ResetFacultyPasswordRequest = None,
    current_hod: UserModel = Depends(require_faculty_management),
    user_service = Depends(get_user_service)
):
    """Resets a Faculty member's password (HOD only)."""
    f = user_service.get_user_by_id(user_id)
    if not f:
        raise HTTPException(status_code=404, detail=f"Faculty '{user_id}' not found.")
    if f.role != UserRole.FACULTY:
        raise HTTPException(status_code=400, detail="Cannot reset password of non-faculty account.")

    new_hash = hash_password(payload.new_password)
    user_service.update_password(user_id, new_hash)
    return ApiResponse.ok({
        "user_id": user_id,
        "message": f"Password for Faculty '{user_id}' has been reset successfully."
    })


# =============================================================================
# 3. Role-Scoped Dashboard & System Status Endpoints
# =============================================================================


@router.get("/dashboard/role-summary", response_model=ApiResponse[Dict[str, Any]])
def get_role_dashboard_summary(
    user: UserModel = Depends(get_current_user),
    state: AppState = Depends(get_app_state),
    user_service = Depends(get_user_service)
):
    """
    Returns dashboard overview data strictly filtered by user's role:
    - HOD: Full system view across all classes, classrooms, advisors, devices.
    - Class Advisor: Strictly scoped to assigned Department, Year, Section.
    """
    all_students = state.db.get_all_students()
    telemetry = state.get_latest_telemetry()
    active_sess = state.session_manager.get_active_session()

    # Real Attendance from Engine
    total_present = 0
    total_late = 0
    present_student_ids = set()

    if active_sess:
        report = state.attendance_engine.generate_session_report(active_sess.session_id)
        total_present = report.present_count
        total_late = report.late_count
        for r in report.records:
            if r.status.value in ["PRESENT", "LATE"]:
                present_student_ids.add(r.student_id)

    if user.role == UserRole.HOD:
        # HOD: Full system stats
        total_enrolled = len(all_students)
        present_count = len(present_student_ids)
        absent_count = max(0, total_enrolled - present_count)
        advisors = user_service.list_advisors()
        active_advisors = sum(1 for a in advisors if a.status == UserStatus.ACTIVE)

        # Real system component status
        is_mongo_online = user_service.mongo_online or (state.db.mongo_db.is_online() if hasattr(state.db, "mongo_db") else False)
        mongo_status = "CONNECTED" if is_mongo_online else "OFFLINE (RESILIENT LOCAL DB)"
        camera_status = "CONNECTED" if telemetry["camera_online"] else "DISCONNECTED"
        ai_status = "RUNNING" if telemetry["recognition_online"] else "OFFLINE"
        faiss_vectors = state.db.get_embedding_count()

        return ApiResponse.ok({
            "role": "hod",
            "user_id": user.user_id,
            "user_name": user.name,
            "cards": {
                "total_students": total_enrolled,
                "students_present": present_count,
                "students_absent": absent_count,
                "active_advisors": active_advisors,
                "active_classrooms": 1,
                "active_cameras": 1 if telemetry["camera_online"] else 0,
                "active_esp32_devices": 1 if telemetry["camera_online"] else 0,
                "active_alerts": 0 if telemetry["camera_online"] else 1
            },
            "system_status": {
                "fastapi": "ONLINE",
                "mongodb": mongo_status,
                "camera": camera_status,
                "ai_pipeline": ai_status,
                "faiss": f"READY ({faiss_vectors} vectors)",
                "esp32": "READY",
                "websocket": "ONLINE"
            },
            "active_session": active_sess.model_dump() if active_sess else None
        })

    elif user.role == UserRole.FACULTY:
        # Faculty: Overview of active instructional session, classroom attendance, and telemetry
        total_enrolled = len(all_students)
        present_count = len(present_student_ids)
        absent_count = max(0, total_enrolled - present_count)
        att_pct = round((present_count / total_enrolled * 100), 1) if total_enrolled > 0 else 0.0

        return ApiResponse.ok({
            "role": "faculty",
            "user_id": user.user_id,
            "faculty_name": user.name,
            "department": user.department,
            "assigned_classroom": user.assigned_classroom or "Smart Classroom 1",
            "cards": {
                "total_students": total_enrolled,
                "present_today": present_count,
                "absent_today": absent_count,
                "attendance_percentage": att_pct,
                "camera_status": "CONNECTED" if telemetry["camera_online"] else "DISCONNECTED",
                "ai_status": "RUNNING" if telemetry["recognition_online"] else "OFFLINE",
                "active_alerts": 0 if telemetry["camera_online"] else 1
            },
            "system_status": {
                "fastapi": "ONLINE",
                "camera": "CONNECTED" if telemetry["camera_online"] else "DISCONNECTED",
                "ai_pipeline": "RUNNING" if telemetry["recognition_online"] else "OFFLINE",
                "sensors": "READY"
            },
            "active_session": active_sess.model_dump() if active_sess else None
        })

    else:
        # Class Advisor: Filtered strictly to assigned class & section
        advisor_dept = (user.department or "").strip().lower()
        advisor_sec = (user.section or "").strip().upper()
        advisor_year = (user.year or "").strip().lower()

        assigned_students = [
            s for s in all_students
            if (not advisor_dept or s.get("department", "").strip().lower() == advisor_dept)
            and (not advisor_sec or s.get("section", "").strip().upper() == advisor_sec)
        ]

        total_class_students = len(assigned_students)
        class_student_ids = {s["student_id"] for s in assigned_students}
        class_present = len(class_student_ids.intersection(present_student_ids))
        class_absent = max(0, total_class_students - class_present)
        att_pct = round((class_present / total_class_students * 100), 1) if total_class_students > 0 else 0.0

        return ApiResponse.ok({
            "role": "class_advisor",
            "user_id": user.user_id,
            "advisor_name": user.name,
            "department": user.department,
            "year": user.year,
            "section": user.section,
            "assigned_classroom": user.assigned_classroom or f"{user.department}-{user.section}",
            "cards": {
                "total_students": total_class_students,
                "present_today": class_present,
                "absent_today": class_absent,
                "attendance_percentage": att_pct,
                "active_camera": "CONNECTED" if telemetry["camera_online"] else "DISCONNECTED",
                "active_alerts": 0
            },
            "assigned_class_title": f"{user.year or ''} {user.department or ''} - {user.section or ''}".strip(),
            "active_session": active_sess.model_dump() if active_sess else None
        })


# =============================================================================
# 4. Role-Scoped Student Monitoring Endpoints
# =============================================================================

@router.get("/scoped/students", response_model=ApiResponse[List[Dict[str, Any]]])
def get_scoped_students(
    department: Optional[str] = Query(None),
    section: Optional[str] = Query(None),
    year: Optional[str] = Query(None),
    user: UserModel = Depends(require_student_management),
    state: AppState = Depends(get_app_state)
):
    """
    Retrieves students:
    - HOD: Full access to all students or query any class/section.
    - Class Advisor: STRICTLY constrained to advisor's assigned class & section.
      Query parameters attempting to inspect other sections are rejected or overridden.
    """
    all_students = state.db.get_all_students()

    # Determine present students in current session
    active_sess = state.session_manager.get_active_session()
    present_ids = set()
    if active_sess:
        report = state.attendance_engine.generate_session_report(active_sess.session_id)
        for r in report.records:
            if r.status.value in ["PRESENT", "LATE"]:
                present_ids.add(r.student_id)

    # Telemetry active tracks for recognition status
    telemetry = state.get_latest_telemetry()
    active_track_ids = {t.stable_student_id: t for t in telemetry["tracks"] if t.stable_student_id}

    if user.role == UserRole.CLASS_ADVISOR:
        # Backend-enforced scope restriction:
        # Even if caller passes department='ECE' or section='A', advisor's credentials prevail!
        adv_dept = (user.department or "").strip().lower()
        adv_sec = (user.section or "").strip().upper()

        if department and department.strip().lower() != adv_dept:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: You are only authorized to access department '{user.department}'."
            )
        if section and section.strip().upper() != adv_sec:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: You are only authorized to access section '{user.section}'."
            )

        filter_dept = adv_dept
        filter_sec = adv_sec
    else:
        # HOD can filter by any provided param or view all
        filter_dept = department.strip().lower() if department else None
        filter_sec = section.strip().upper() if section else None

    results = []
    for s in all_students:
        s_dept = s.get("department", "").strip().lower()
        s_sec = s.get("section", "").strip().upper()

        if filter_dept and s_dept != filter_dept:
            continue
        if filter_sec and s_sec != filter_sec:
            continue

        sid = s["student_id"]
        is_present = sid in present_ids
        track = active_track_ids.get(sid)

        rec_status = "Not Tracked"
        if track:
            rec_status = f"Recognized ({int(track.current_similarity * 100)}%)"

        att_status = "Present" if is_present else "Not Seen"

        results.append({
            "register_no": s.get("register_no", sid),
            "student_name": s["student_name"],
            "department": s.get("department", ""),
            "year": s.get("class_name") or user.year or "3rd Year",
            "section": s.get("section", ""),
            "attendance_status": att_status,
            "attendance_percentage": 100.0 if is_present else 0.0,
            "last_seen": "Today" if is_present else "Never",
            "recognition_status": rec_status
        })

    return ApiResponse.ok(results)


class BulkImportJsonRequest(BaseModel):
    students: List[Dict[str, Any]]
    default_department: Optional[str] = "AI&DS"
    default_section: Optional[str] = "B"
    default_year: Optional[str] = "3rd Year"


@router.post("/hod/students/upload-excel", response_model=ApiResponse[Dict[str, Any]])
@router.post("/students/upload-excel", response_model=ApiResponse[Dict[str, Any]])
async def upload_students_excel(
    file: UploadFile = File(...),
    default_department: str = Form("AI&DS"),
    default_section: str = Form("B"),
    default_year: str = Form("3rd Year"),
    user: UserModel = Depends(require_student_management),
    state: AppState = Depends(get_app_state)
):
    """
    Directly attaches and imports students Excel (.xlsx, .xls) or CSV into MongoDB & SQLite.
    Bypasses manual student-by-student typing.
    Accessible to HOD and Class Advisor (scoped if advisor).
    """
    filename = file.filename or "students.xlsx"
    valid_exts = (".xlsx", ".xls", ".csv")
    if not any(filename.lower().endswith(ext) for ext in valid_exts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{filename}'. Please upload an Excel (.xlsx, .xls) or CSV file."
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty."
        )

    # If caller is Class Advisor, lock default dept & section to their assignment
    if user.role == UserRole.CLASS_ADVISOR:
        default_department = user.department or default_department
        default_section = user.section or default_section
        default_year = user.year or default_year

    parsed_students, parse_errors = ExcelService.parse_students_file(
        file_bytes=file_bytes,
        filename=filename,
        default_department=default_department,
        default_section=default_section,
        default_year=default_year
    )

    if not parsed_students:
        error_msg = "; ".join(parse_errors) if parse_errors else "No valid student rows found in sheet."
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Excel parsing failed: {error_msg}"
        )

    # Class Advisor restriction: filter only to advisor cohort
    if user.role == UserRole.CLASS_ADVISOR:
        adv_dept = (user.department or "").strip().lower()
        adv_sec = (user.section or "").strip().upper()
        scoped_students = []
        for s in parsed_students:
            s_dept = s.get("department", "").strip().lower()
            s_sec = s.get("section", "").strip().upper()
            if s_dept == adv_dept and s_sec == adv_sec:
                scoped_students.append(s)
            else:
                parse_errors.append(f"Student '{s['register_no']}' omitted (outside your assigned cohort {user.department}-{user.section}).")
        parsed_students = scoped_students
        if not parsed_students:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"None of the students in the file belong to your assigned cohort ({user.department} - {user.section})."
            )

    # Commit directly to Database (MongoDB + SQLite)
    result = state.db.bulk_import_students(
        students_list=parsed_students,
        default_department=default_department,
        default_section=default_section,
        default_year=default_year
    )
    result["filename"] = filename
    result["parse_warnings"] = parse_errors

    return ApiResponse.ok(result)


@router.post("/hod/students/bulk-import-json", response_model=ApiResponse[Dict[str, Any]])
@router.post("/students/bulk-import-json", response_model=ApiResponse[Dict[str, Any]])
def bulk_import_students_json(
    payload: BulkImportJsonRequest,
    user: UserModel = Depends(require_student_management),
    state: AppState = Depends(get_app_state)
):
    """
    Directly imports student records from JSON array into MongoDB & SQLite databases.
    """
    dept = user.department if user.role == UserRole.CLASS_ADVISOR else (payload.default_department or "AI&DS")
    sec = user.section if user.role == UserRole.CLASS_ADVISOR else (payload.default_section or "B")
    year = user.year if user.role == UserRole.CLASS_ADVISOR else (payload.default_year or "3rd Year")

    result = state.db.bulk_import_students(
        students_list=payload.students,
        default_department=dept,
        default_section=sec,
        default_year=year
    )
    return ApiResponse.ok(result)


@router.get("/hod/students/template")
@router.get("/students/template")
def download_student_template(
    format: str = Query("xlsx", pattern="^(xlsx|csv)$")
):
    """Downloads an institutional student roster template (.xlsx or .csv) with sample data."""
    if format == "csv":
        data = ExcelService.generate_csv_template()
        media_type = "text/csv"
        filename = "students_roster_template.csv"
    else:
        data = ExcelService.generate_excel_template()
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = "students_roster_template.xlsx"

    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# =============================================================================
# 5. Role-Scoped Alerts Endpoints
# =============================================================================

@router.get("/alerts", response_model=ApiResponse[List[Dict[str, Any]]])
@router.get("/scoped/alerts", response_model=ApiResponse[List[Dict[str, Any]]])
def get_scoped_alerts(
    user: UserModel = Depends(get_current_user),
    state: AppState = Depends(get_app_state)
):
    """
    Returns alerts scoped to the authenticated caller:
    - HOD: System alerts + all classroom alerts.
    - Class Advisor: Strictly alerts matching assigned class and classroom.
    """
    telemetry = state.get_latest_telemetry()
    alerts: List[Dict[str, Any]] = []

    # Camera offline alert
    if not telemetry["camera_online"]:
        alerts.append({
            "id": "ALT_CAM_OFFLINE",
            "severity": "CRITICAL",
            "title": "Camera Stream Disconnected",
            "message": "Physical video capture feed is offline or unreachable.",
            "classroom_id": "ALL",
            "timestamp": datetime.datetime.utcnow().isoformat()
        })

    # Unknown face alert
    if telemetry.get("unknown_count", 0) > 0:
        alerts.append({
            "id": "ALT_UNKNOWN_FACE",
            "severity": "WARNING",
            "title": "Unknown Face Detected",
            "message": f"{telemetry['unknown_count']} unidentified face(s) visible in camera frame.",
            "classroom_id": user.assigned_classroom or "AIDS-B",
            "timestamp": datetime.datetime.utcnow().isoformat()
        })

    if user.role == UserRole.HOD:
        return ApiResponse.ok(alerts)

    # Class Advisor: Filter alerts by assigned classroom
    adv_classroom = user.assigned_classroom or "AIDS-B"
    advisor_alerts = [a for a in alerts if a["classroom_id"] in [adv_classroom, "ALL"]]
    return ApiResponse.ok(advisor_alerts)


# =============================================================================
# 6. Classroom Management & Device Mapping Endpoints
# =============================================================================

class CreateClassroomRequest(BaseModel):
    classroom_id: str
    classroom_name: Optional[str] = None
    department: str = "AI&DS"
    year: str = "3rd Year"
    section: str = "B"
    assigned_advisor_id: Optional[str] = None
    camera_source: str = "pc"
    camera_url: Optional[str] = None
    esp32_device_id: Optional[str] = None


class UpdateClassroomRequest(BaseModel):
    classroom_name: Optional[str] = None
    department: Optional[str] = None
    year: Optional[str] = None
    section: Optional[str] = None
    assigned_advisor_id: Optional[str] = None
    camera_source: Optional[str] = None
    camera_url: Optional[str] = None
    camera_status: Optional[str] = None
    esp32_device_id: Optional[str] = None
    esp32_status: Optional[str] = None
    ai_pipeline_status: Optional[str] = None
    attendance_status: Optional[str] = None


@router.get("/classrooms", response_model=ApiResponse[List[Dict[str, Any]]])
def get_classrooms(
    user: UserModel = Depends(require_any_authenticated_user),
    state: AppState = Depends(get_app_state)
):
    """Lists all registered classrooms."""
    classrooms = state.db.get_all_classrooms()
    return ApiResponse.ok(classrooms)


@router.get("/classrooms/{classroom_id}", response_model=ApiResponse[Dict[str, Any]])
def get_classroom_detail(
    classroom_id: str,
    user: UserModel = Depends(require_any_authenticated_user),
    state: AppState = Depends(get_app_state)
):
    """Retrieves details of a single classroom."""
    classroom = state.db.get_classroom_by_id(classroom_id)
    if not classroom:
        raise HTTPException(status_code=404, detail=f"Classroom '{classroom_id}' not found.")
    return ApiResponse.ok(classroom)


@router.post("/hod/classrooms", response_model=ApiResponse[Dict[str, Any]], status_code=status.HTTP_201_CREATED)
def create_classroom(
    payload: CreateClassroomRequest,
    user: UserModel = Depends(require_hod),
    state: AppState = Depends(get_app_state),
    user_service = Depends(get_user_service)
):
    """Creates a new classroom and assigns advisor/camera (HOD only)."""
    existing = state.db.get_classroom_by_id(payload.classroom_id)
    if existing:
        raise HTTPException(status_code=400, detail=f"Classroom '{payload.classroom_id}' already exists.")

    adv_name = None
    if payload.assigned_advisor_id:
        adv = user_service.get_user_by_id(payload.assigned_advisor_id)
        if adv:
            adv_name = adv.name

    data = payload.model_dump()
    data["assigned_advisor_name"] = adv_name
    created = state.db.upsert_classroom(data)
    return ApiResponse.ok(created)


@router.put("/hod/classrooms/{classroom_id}", response_model=ApiResponse[Dict[str, Any]])
def update_classroom(
    classroom_id: str,
    payload: UpdateClassroomRequest,
    user: UserModel = Depends(require_hod),
    state: AppState = Depends(get_app_state),
    user_service = Depends(get_user_service)
):
    """Updates classroom metadata, camera mapping, or advisor assignment (HOD only)."""
    existing = state.db.get_classroom_by_id(classroom_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Classroom '{classroom_id}' not found.")

    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    merged = {**existing, **updates}
    merged["classroom_id"] = classroom_id

    if "assigned_advisor_id" in updates and updates["assigned_advisor_id"]:
        adv = user_service.get_user_by_id(updates["assigned_advisor_id"])
        merged["assigned_advisor_name"] = adv.name if adv else None

    updated = state.db.upsert_classroom(merged)
    return ApiResponse.ok(updated)


@router.delete("/hod/classrooms/{classroom_id}", response_model=ApiResponse[Dict[str, Any]])
def delete_classroom(
    classroom_id: str,
    user: UserModel = Depends(require_hod),
    state: AppState = Depends(get_app_state)
):
    """Deletes a classroom entity (HOD only)."""
    success = state.db.delete_classroom(classroom_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Classroom '{classroom_id}' not found.")
    return ApiResponse.ok({"message": f"Classroom '{classroom_id}' deleted successfully."})


@router.get("/advisor/classroom-status", response_model=ApiResponse[Dict[str, Any]])
def get_advisor_classroom_status(
    user: UserModel = Depends(require_class_advisor),
    state: AppState = Depends(get_app_state)
):
    """
    Evaluates Advisor Classroom setup hierarchy:
    1. Is advisor assigned to a class?
    2. Is the class assigned to a configured classroom?
    3. Is a camera configured for that classroom?
    4. Is the camera connected & online?
    5. Is AI pipeline available?
    """
    # 1. Check Class Assignment
    has_class = bool(user.year and user.section and user.department)
    if not has_class:
        return ApiResponse.ok({
            "status_code": "NO_CLASS_ASSIGNED",
            "has_assignment": False,
            "classroom_configured": False,
            "camera_configured": False,
            "camera_online": False,
            "title": "No Class Assigned",
            "message": "You are not currently assigned to any class or section. Please contact the HOD to assign your class."
        })

    # 2. Check Classroom Assignment & Configuration
    classroom_id = user.assigned_classroom
    classroom = state.db.get_classroom_by_id(classroom_id) if classroom_id else None
    if not classroom:
        # Fallback check by advisor ID
        classroom = state.db.get_classroom_by_advisor_id(user.user_id)
        if classroom:
            classroom_id = classroom["classroom_id"]

    if not classroom:
        return ApiResponse.ok({
            "status_code": "NOT_CONFIGURED",
            "has_assignment": True,
            "class_title": f"{user.year} {user.department} - {user.section}",
            "classroom_id": classroom_id or "Unassigned",
            "classroom_configured": False,
            "camera_configured": False,
            "camera_online": False,
            "title": "CLASSROOM NOT CONFIGURED",
            "message": f"Classroom for {user.year} {user.department} - {user.section} is not configured. Camera unavailable."
        })

    # 3. Check Camera Configuration
    cam_source = classroom.get("camera_source") or "none"
    if cam_source == "none":
        return ApiResponse.ok({
            "status_code": "NOT_CONFIGURED",
            "has_assignment": True,
            "class_title": f"{classroom['year']} {classroom['department']} - {classroom['section']}",
            "classroom_id": classroom["classroom_id"],
            "classroom_name": classroom["classroom_name"],
            "classroom_configured": True,
            "camera_configured": False,
            "camera_source": "none",
            "camera_online": False,
            "title": "CLASSROOM NOT CONFIGURED",
            "message": f"Classroom {classroom['classroom_id']} camera is not configured. Camera unavailable."
        })

    # 4. Check Camera Status & AI Pipeline
    telemetry = state.get_latest_telemetry()
    cam_online = bool(telemetry["camera_online"] and (state.camera_manager is not None))

    if not cam_online:
        return ApiResponse.ok({
            "status_code": "CAMERA_OFFLINE",
            "has_assignment": True,
            "class_title": f"{classroom['year']} {classroom['department']} - {classroom['section']}",
            "classroom_id": classroom["classroom_id"],
            "classroom_name": classroom["classroom_name"],
            "classroom_configured": True,
            "camera_configured": True,
            "camera_source": cam_source,
            "camera_online": False,
            "title": "CAMERA OFFLINE",
            "message": f"Camera for {classroom['classroom_id']} is currently offline."
        })

    # 5. All checks passed: ONLINE
    return ApiResponse.ok({
        "status_code": "ONLINE",
        "has_assignment": True,
        "class_title": f"{classroom['year']} {classroom['department']} - {classroom['section']}",
        "classroom_id": classroom["classroom_id"],
        "classroom_name": classroom["classroom_name"],
        "classroom_configured": True,
        "camera_configured": True,
        "camera_source": cam_source,
        "camera_online": True,
        "ai_pipeline_online": telemetry["recognition_online"],
        "esp32_device_id": classroom.get("esp32_device_id"),
        "esp32_online": classroom.get("esp32_status") == "connected",
        "title": "Classroom Online",
        "message": "Live camera feed and real-time monitoring active."
    })


# =============================================================================
# 7. IoT & Sensor Endpoints (HOD, Faculty, Advisor Authorized)
# =============================================================================

@router.get("/sensors/status", response_model=ApiResponse[Dict[str, Any]])
def get_sensors_status(
    user: UserModel = Depends(require_sensor_access),
    state: AppState = Depends(get_app_state)
):
    """Retrieves IoT / ESP32 sensor telemetry status (HOD, Faculty, Class Advisor authorized)."""
    telemetry = state.get_latest_telemetry()
    latest_sensor = state.db.sensor_repo.get_latest() if (hasattr(state.db, "mongo_db") and state.db.mongo_db.is_online()) else None

    if latest_sensor:
        return ApiResponse.ok({
            "status": "ONLINE",
            "esp32_device_id": latest_sensor.get("device_id", "ESP32_MAIN_01"),
            "esp32_connected": True,
            "temperature_c": latest_sensor.get("temperature_c", 24.5),
            "humidity_pct": latest_sensor.get("humidity_pct", 55.0),
            "air_quality": latest_sensor.get("air_quality", "GOOD"),
            "motion_detected": latest_sensor.get("motion_detected", len(telemetry.get("tracks", [])) > 0),
            "authorized_role": user.role.value,
            "timestamp": latest_sensor.get("timestamp", datetime.datetime.utcnow().isoformat())
        })

    return ApiResponse.ok({
        "status": "ONLINE",
        "esp32_device_id": "ESP32_MAIN_01",
        "esp32_connected": telemetry.get("camera_online", False),
        "temperature_c": 24.5,
        "humidity_pct": 55.0,
        "air_quality": "GOOD",
        "motion_detected": len(telemetry.get("tracks", [])) > 0,
        "authorized_role": user.role.value,
        "timestamp": datetime.datetime.utcnow().isoformat()
    })


# =============================================================================
# 8. RBAC-Protected Attendance Endpoints (HOD, Faculty, Advisor Authorized)
# =============================================================================

@router.get("/attendance", response_model=ApiResponse[List[Dict[str, Any]]])
def list_attendance(
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    student_id: Optional[str] = Query(None, description="Filter by student ID"),
    status: Optional[str] = Query(None, description="Filter by status: PRESENT or LATE"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user: UserModel = Depends(require_attendance_access),
    state: AppState = Depends(get_app_state)
):
    """
    Lists attendance records with strict role-scoping:
    - HOD: System-wide visibility across all classrooms, sessions, and students.
    - Faculty: Visibility across their classrooms and conducted sessions.
    - Class Advisor: Strictly restricted to students in their assigned department & section.
    """
    # Role Scoping for Class Advisor
    if user.role == UserRole.CLASS_ADVISOR:
        adv_dept = (user.department or "").strip().lower()
        adv_sec = (user.section or "").strip().upper()

        if student_id:
            st = state.db.get_student(student_id)
            if not st:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Student '{student_id}' not found.")
            st_dept = (st.get("department") or "").strip().lower()
            st_sec = (st.get("section") or "").strip().upper()
            if (adv_dept and st_dept != adv_dept) or (adv_sec and st_sec != adv_sec):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access forbidden: Student is outside your assigned cohort ({user.department} - {user.section})."
                )
            records = state.db.list_attendance(session_id=session_id, student_id=student_id, status=status, limit=limit, offset=offset)
            return ApiResponse.ok(records)

        # Advisor listing without student_id: filter only to advisor's students
        all_students = state.db.get_all_students()
        advisor_student_ids = {
            s["student_id"] for s in all_students
            if (not adv_dept or (s.get("department") or "").strip().lower() == adv_dept)
            and (not adv_sec or (s.get("section") or "").strip().upper() == adv_sec)
        }
        all_recs = state.db.list_attendance(session_id=session_id, status=status, limit=500, offset=0)
        filtered = [r for r in all_recs if r.get("student_id") in advisor_student_ids]
        paged = filtered[offset:offset + limit]
        return ApiResponse.ok(paged)

    # HOD and Faculty
    records = state.db.list_attendance(
        session_id=session_id,
        student_id=student_id,
        status=status,
        limit=limit,
        offset=offset
    )
    return ApiResponse.ok(records)


@router.get("/attendance/session/{session_id}", response_model=ApiResponse[Dict[str, Any]])
def get_session_attendance(
    session_id: str = Path(...),
    user: UserModel = Depends(require_attendance_access),
    state: AppState = Depends(get_app_state)
):
    """
    Retrieves full attendance report for a specific session with RBAC scoping:
    - HOD & Faculty: Full session report and roster reconciliation.
    - Class Advisor: Scoped to their assigned cohort students within the session.
    """
    sess = state.session_manager.get_session_info(session_id)
    if not sess:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found.")

    report = state.attendance_engine.generate_session_report(session_id)
    rep_dict = report.model_dump()

    if user.role == UserRole.CLASS_ADVISOR:
        adv_dept = (user.department or "").strip().lower()
        adv_sec = (user.section or "").strip().upper()

        all_students = state.db.get_all_students()
        cohort_student_ids = {
            s["student_id"] for s in all_students
            if (not adv_dept or (s.get("department") or "").strip().lower() == adv_dept)
            and (not adv_sec or (s.get("section") or "").strip().upper() == adv_sec)
        }

        # Filter records and not_seen to advisor's cohort
        filtered_records = [r for r in rep_dict.get("records", []) if r.get("student_id") in cohort_student_ids]
        filtered_not_seen = [sid for sid in rep_dict.get("not_seen_students", []) if sid in cohort_student_ids]
        pres_cnt = sum(1 for r in filtered_records if r.get("status") == "PRESENT")
        late_cnt = sum(1 for r in filtered_records if r.get("status") == "LATE")

        rep_dict["records"] = filtered_records
        rep_dict["not_seen_students"] = filtered_not_seen
        rep_dict["total_enrolled"] = len(cohort_student_ids)
        rep_dict["present_count"] = pres_cnt
        rep_dict["late_count"] = late_cnt
        rep_dict["not_seen_count"] = len(filtered_not_seen)

    return ApiResponse.ok(rep_dict)


@router.get("/attendance/student/{student_id}", response_model=ApiResponse[Dict[str, Any]])
def get_student_attendance_history(
    student_id: str = Path(...),
    user: UserModel = Depends(require_attendance_access),
    state: AppState = Depends(get_app_state)
):
    """
    Retrieves complete attendance history for a single student:
    - HOD & Faculty: Access any student.
    - Class Advisor: Strictly restricted to their assigned cohort. Unauthorized queries return 403.
    """
    st = state.db.get_student(student_id)
    if not st:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Student '{student_id}' not found.")

    if user.role == UserRole.CLASS_ADVISOR:
        adv_dept = (user.department or "").strip().lower()
        adv_sec = (user.section or "").strip().upper()
        st_dept = (st.get("department") or "").strip().lower()
        st_sec = (st.get("section") or "").strip().upper()

        if (adv_dept and st_dept != adv_dept) or (adv_sec and st_sec != adv_sec):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: Student '{student_id}' is outside your assigned cohort ({user.department} - {user.section})."
            )

    history = state.db.get_student_attendance_history(student_id)
    tot_sessions = len(history)
    present_cnt = sum(1 for h in history if h.get("status") == "PRESENT")
    late_cnt = sum(1 for h in history if h.get("status") == "LATE")
    att_pct = round(((present_cnt + late_cnt) / tot_sessions * 100), 1) if tot_sessions > 0 else 0.0

    return ApiResponse.ok({
        "student_id": student_id,
        "student_name": st.get("student_name", student_id),
        "department": st.get("department", ""),
        "section": st.get("section", ""),
        "total_sessions": tot_sessions,
        "present_count": present_cnt,
        "late_count": late_cnt,
        "attendance_percentage": att_pct,
        "records": history
    })


@router.get("/attendance/summary", response_model=ApiResponse[Dict[str, Any]])
def get_attendance_summary(
    session_id: Optional[str] = Query(None, description="Optional session ID"),
    user: UserModel = Depends(require_attendance_access),
    state: AppState = Depends(get_app_state)
):
    """
    Returns attendance summary aggregates with role-scoping:
    - HOD & Faculty: Overall or session summary.
    - Class Advisor: Cohort-scoped attendance aggregates.
    """
    if user.role == UserRole.CLASS_ADVISOR:
        adv_dept = (user.department or "").strip().lower()
        adv_sec = (user.section or "").strip().upper()
        all_students = state.db.get_all_students()
        cohort_student_ids = {
            s["student_id"] for s in all_students
            if (not adv_dept or (s.get("department") or "").strip().lower() == adv_dept)
            and (not adv_sec or (s.get("section") or "").strip().upper() == adv_sec)
        }
        all_recs = state.db.list_attendance(session_id=session_id, limit=1000)
        cohort_recs = [r for r in all_recs if r.get("student_id") in cohort_student_ids]

        tot = len(cohort_recs)
        pres = sum(1 for r in cohort_recs if r.get("status") == "PRESENT")
        late = sum(1 for r in cohort_recs if r.get("status") == "LATE")
        uniq = len(set(r.get("student_id") for r in cohort_recs if r.get("student_id")))
        sims = [r.get("latest_similarity", 0.0) for r in cohort_recs if r.get("latest_similarity") is not None]
        avg_sim = round(float(sum(sims) / len(sims)), 4) if sims else 0.0

        return ApiResponse.ok({
            "session_id": session_id,
            "total_records": tot,
            "present_count": pres,
            "late_count": late,
            "unique_students": uniq,
            "average_similarity": avg_sim,
            "scoped_to": f"{user.department} - {user.section}"
        })

    summary = state.db.get_attendance_summary(session_id=session_id)
    return ApiResponse.ok(summary)


@router.get("/attendance/export/{session_id}")
def export_attendance(
    session_id: str = Path(...),
    format: str = Query("json", pattern="^(json|csv)$"),
    user: UserModel = Depends(require_attendance_access),
    state: AppState = Depends(get_app_state)
):
    """
    Exports session attendance in CSV or JSON format with RBAC scoping.
    """
    import io
    import csv

    sess = state.session_manager.get_session_info(session_id)
    if not sess:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found.")

    report = state.attendance_engine.generate_session_report(session_id)
    records = report.records

    if user.role == UserRole.CLASS_ADVISOR:
        adv_dept = (user.department or "").strip().lower()
        adv_sec = (user.section or "").strip().upper()
        all_students = state.db.get_all_students()
        cohort_student_ids = {
            s["student_id"] for s in all_students
            if (not adv_dept or (s.get("department") or "").strip().lower() == adv_dept)
            and (not adv_sec or (s.get("section") or "").strip().upper() == adv_sec)
        }
        records = [r for r in records if r.student_id in cohort_student_ids]

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["session_id", "student_id", "student_name", "status", "first_seen", "last_seen", "similarity"])
        for r in records:
            writer.writerow([r.session_id, r.student_id, r.student_name, r.status.value, r.first_seen, r.last_seen, r.latest_similarity])

        csv_content = output.getvalue()
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=attendance_{session_id}.csv"}
        )

    # JSON export
    return ApiResponse.ok({
        "session_id": session_id,
        "date": sess.date,
        "class_section": sess.class_section,
        "subject": sess.subject,
        "total_records": len(records),
        "records": [r.model_dump() for r in records]
    })



