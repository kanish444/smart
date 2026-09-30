import os
import sys
import pytest
import datetime
import mongomock
import numpy as np
from fastapi.testclient import TestClient

# Ensure workspace root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings, MongoSettings
from database.mongo_repository import (
    MongoDatabase,
    StudentRepository,
    ClassroomRepository,
    SessionRepository,
    AttendanceRepository,
    SensorRepository
)
from database.db_manager import DatabaseManager
from app.services.user_service import UserService
from app.models.user import UserModel, UserRole, UserStatus
from app.models.data_models import (
    StudentModel,
    FacultyModel,
    AdvisorModel,
    ClassroomModel,
    SessionModel,
    AttendanceRecordModel,
    SensorTelemetryModel
)
from app.core.security import hash_password, verify_password, create_access_token
from scripts.migrate_sqlite_to_mongo import migrate_sqlite_to_mongodb


@pytest.fixture
def mock_mongo_client():
    """Provides an isolated in-memory MongoMock client."""
    client = mongomock.MongoClient()
    return client


@pytest.fixture
def mongo_db(mock_mongo_client):
    """Provides an initialized MongoDatabase backed by MongoMock."""
    return MongoDatabase(client=mock_mongo_client, db_name="test_smartclass")


@pytest.fixture
def test_user_service(tmp_path, mock_mongo_client):
    """Provides a UserService using isolated SQLite and MongoMock."""
    db_file = str(tmp_path / "test_smartclass.sqlite")
    svc = UserService(db_path=db_file, mongo_client=mock_mongo_client)
    return svc


@pytest.fixture
def test_db_manager(tmp_path, mock_mongo_client):
    """Provides a DatabaseManager using isolated SQLite and MongoMock."""
    db_file = str(tmp_path / "test_smartclass.sqlite")
    new_db_file = str(tmp_path / "test_new_enrollment.sqlite")
    mgr = DatabaseManager(db_path=db_file, new_db_path=new_db_file, mongo_client=mock_mongo_client)
    return mgr


# =============================================================================
# 1. MongoDB Connection and Configuration
# =============================================================================

def test_01_mongodb_connection_and_configuration(mongo_db):
    """Verifies MongoDB configuration defaults, connection state, and collections."""
    settings = get_settings().mongodb
    assert isinstance(settings, MongoSettings)
    assert settings.timeout_ms > 0
    assert settings.database is not None

    assert mongo_db.is_online() is True
    assert mongo_db.users is not None
    assert mongo_db.faculty is not None
    assert mongo_db.advisors is not None
    assert mongo_db.students is not None
    assert mongo_db.classrooms is not None
    assert mongo_db.sessions is not None
    assert mongo_db.attendance_records is not None
    assert mongo_db.sensor_telemetry is not None


# =============================================================================
# 2. User Creation / Read / Update
# =============================================================================

def test_02_user_creation_read_update_mongodb(test_user_service):
    """Verifies user persistence, retrieval, and profile updates in MongoDB."""
    user = UserModel(
        user_id="TEST_USER_01",
        name="Test User",
        role=UserRole.FACULTY,
        password_hash=hash_password("testpass123"),
        email="test.user@college.edu",
        department="AI&DS",
        status=UserStatus.ACTIVE
    )
    created = test_user_service.create_user(user)
    assert created.user_id == "TEST_USER_01"

    # Read
    fetched = test_user_service.get_user_by_id("TEST_USER_01")
    assert fetched is not None
    assert fetched.name == "Test User"
    assert fetched.email == "test.user@college.edu"

    # Update
    updated = test_user_service.update_user("TEST_USER_01", {"name": "Updated Test User", "phone": "9876543210"})
    assert updated is not None
    assert updated.name == "Updated Test User"
    assert updated.phone == "9876543210"


# =============================================================================
# 3. Role Persistence (HOD, Faculty, Class Advisor)
# =============================================================================

def test_03_role_persistence_three_roles(test_user_service, mock_mongo_client):
    """Verifies all 3 roles persist correctly in users and role-specific collections."""
    db = mock_mongo_client["smartclass_vision_ai"]

    # HOD
    hod = test_user_service.get_user_by_id("HOD001")
    assert hod is not None
    assert hod.role == UserRole.HOD

    # Faculty
    fac = test_user_service.get_user_by_id("FAC001")
    assert fac is not None
    assert fac.role == UserRole.FACULTY

    # Create Advisor
    adv = UserModel(
        user_id="ADV_TEST_01",
        name="Advisor Test",
        role=UserRole.CLASS_ADVISOR,
        password_hash=hash_password("advpass123"),
        email="advisor.test@college.edu",
        department="AI&DS",
        year="3rd Year",
        section="B",
        assigned_classroom="AIDS-B",
        status=UserStatus.ACTIVE
    )
    test_user_service.create_user(adv)

    # Verify role-specific collections in MongoDB
    adv_doc = db["advisors"].find_one({"advisor_id": "ADV_TEST_01"})
    assert adv_doc is not None
    assert adv_doc["name"] == "Advisor Test"

    fac_doc = db["faculty"].find_one({"faculty_id": "FAC001"})
    assert fac_doc is not None
    assert fac_doc["name"] == "Dr. Anand Kumar"


# =============================================================================
# 4. HOD Authentication using MongoDB
# =============================================================================

def test_04_hod_authentication_using_mongodb(test_user_service):
    """Verifies HOD credential authentication against MongoDB."""
    user = test_user_service.get_user_by_id_or_email("HOD001")
    assert user is not None
    assert user.role == UserRole.HOD
    assert verify_password("admin123", user.password_hash) is True
    assert verify_password("wrongpassword", user.password_hash) is False


# =============================================================================
# 5. Faculty Authentication using MongoDB
# =============================================================================

def test_05_faculty_authentication_using_mongodb(test_user_service):
    """Verifies Faculty credential authentication against MongoDB."""
    user = test_user_service.get_user_by_id_or_email("FAC001")
    assert user is not None
    assert user.role == UserRole.FACULTY
    assert verify_password("faculty123", user.password_hash) is True


# =============================================================================
# 6. Class Advisor Authentication using MongoDB
# =============================================================================

def test_06_class_advisor_authentication_using_mongodb(test_user_service):
    """Verifies Class Advisor credential authentication against MongoDB."""
    adv = UserModel(
        user_id="ADV_AUTH_01",
        name="Advisor Auth",
        role=UserRole.CLASS_ADVISOR,
        password_hash=hash_password("advisor123"),
        email="advisor.auth@college.edu",
        status=UserStatus.ACTIVE
    )
    test_user_service.create_user(adv)

    user = test_user_service.get_user_by_id_or_email("ADV_AUTH_01")
    assert user is not None
    assert user.role == UserRole.CLASS_ADVISOR
    assert verify_password("advisor123", user.password_hash) is True


# =============================================================================
# 7. Faculty Management
# =============================================================================

def test_07_faculty_management(test_user_service):
    """Tests faculty management lifecycle: list, create, update, status toggle."""
    faculty_list = test_user_service.list_faculty()
    assert len(faculty_list) >= 1
    assert any(f.user_id == "FAC001" for f in faculty_list)

    # Disable status
    test_user_service.update_status("FAC001", "disabled")
    fac = test_user_service.get_user_by_id("FAC001")
    assert fac.status == UserStatus.DISABLED

    # Re-enable
    test_user_service.update_status("FAC001", "active")
    fac = test_user_service.get_user_by_id("FAC001")
    assert fac.status == UserStatus.ACTIVE


# =============================================================================
# 8. Advisor Management
# =============================================================================

def test_08_advisor_management(test_user_service):
    """Tests advisor management lifecycle: create, list, password reset."""
    adv = UserModel(
        user_id="ADV_MGT_01",
        name="Prof. Sharma",
        role=UserRole.CLASS_ADVISOR,
        password_hash=hash_password("sharma123"),
        email="sharma@college.edu",
        department="AI&DS",
        year="2nd Year",
        section="A",
        status=UserStatus.ACTIVE
    )
    test_user_service.create_user(adv)

    advisors = test_user_service.list_advisors()
    assert any(a.user_id == "ADV_MGT_01" for a in advisors)

    # Update password
    new_hash = hash_password("newpassword123")
    test_user_service.update_password("ADV_MGT_01", new_hash)
    updated_adv = test_user_service.get_user_by_id("ADV_MGT_01")
    assert verify_password("newpassword123", updated_adv.password_hash) is True


# =============================================================================
# 9. Student Retrieval & Scoping
# =============================================================================

def test_09_student_retrieval_and_scoping(test_db_manager):
    """Tests student creation, query, and department/section scoping in MongoDB."""
    test_db_manager.add_student(
        student_id="STU_SCOPE_01",
        student_name="Alice Smith",
        department="AI&DS",
        section="B",
        register_no="711721104001"
    )
    test_db_manager.add_student(
        student_id="STU_SCOPE_02",
        student_name="Bob Jones",
        department="CSE",
        section="A",
        register_no="711721104002"
    )

    # Test retrieval by ID
    s1 = test_db_manager.get_student("STU_SCOPE_01")
    assert s1 is not None
    assert s1["student_name"] == "Alice Smith"

    # Test retrieval by Register Number
    s2 = test_db_manager.get_student("711721104002")
    assert s2 is not None
    assert s2["student_id"] == "STU_SCOPE_02"

    # Scoped count in MongoDB
    aids_b_count = test_db_manager.student_repo.count(department="AI&DS", section="B")
    assert aids_b_count == 1
    cse_a_count = test_db_manager.student_repo.count(department="CSE", section="A")
    assert cse_a_count == 1


# =============================================================================
# 10. Classroom Persistence
# =============================================================================

def test_10_classroom_persistence(test_db_manager):
    """Tests classroom entity persistence in MongoDB."""
    room_data = {
        "classroom_id": "ROOM_TEST_101",
        "classroom_name": "Test Lab 101",
        "department": "AI&DS",
        "year": "3rd Year",
        "section": "B",
        "assigned_advisor_id": "ADV_ROOM_101",
        "assigned_advisor_name": "Prof. Ramesh",
        "camera_source": "pc",
        "camera_status": "connected",
        "esp32_device_id": "ESP32-101",
        "esp32_status": "connected",
        "ai_pipeline_status": "running",
        "faiss_status": "ready",
        "attendance_status": "active"
    }
    test_db_manager.upsert_classroom(room_data)

    fetched = test_db_manager.get_classroom_by_id("ROOM_TEST_101")
    assert fetched is not None
    assert fetched["classroom_name"] == "Test Lab 101"
    assert fetched["assigned_advisor_id"] == "ADV_ROOM_101"

    by_adv = test_db_manager.get_classroom_by_advisor_id("ADV_ROOM_101")
    assert by_adv is not None
    assert by_adv["classroom_id"] == "ROOM_TEST_101"

    # Delete
    test_db_manager.delete_classroom("ROOM_TEST_101")
    assert test_db_manager.get_classroom_by_id("ROOM_TEST_101") is None


# =============================================================================
# 11. Session Lifecycle Persistence
# =============================================================================

def test_11_session_lifecycle_persistence(test_db_manager):
    """Tests session creation, activation, and completion lifecycle in MongoDB."""
    sess_id = "SESS_TEST_2026_01"
    test_db_manager.create_session(
        session_id=sess_id,
        date="2026-09-28",
        class_section="AIDS-B",
        subject="Neural Networks",
        planned_start_time="09:00:00",
        planned_end_time="10:00:00",
        status="SCHEDULED"
    )

    s = test_db_manager.get_session(sess_id)
    assert s is not None
    assert s["status"] == "SCHEDULED"

    # Start session
    now_start = datetime.datetime.utcnow().isoformat()
    test_db_manager.update_session_status(sess_id, "ACTIVE", actual_start_time=now_start)
    s_active = test_db_manager.get_active_session("AIDS-B")
    assert s_active is not None
    assert s_active["session_id"] == sess_id

    # End session
    now_end = datetime.datetime.utcnow().isoformat()
    test_db_manager.update_session_status(sess_id, "COMPLETED", actual_end_time=now_end)
    s_done = test_db_manager.get_session(sess_id)
    assert s_done["status"] == "COMPLETED"


# =============================================================================
# 12. Attendance Persistence
# =============================================================================

def test_12_attendance_persistence(test_db_manager):
    """Tests recording attendance with first/last observations in MongoDB."""
    sess_id = "SESS_ATT_01"
    stu_id = "STU_ATT_01"

    test_db_manager.create_session(sess_id, "2026-09-28", "AIDS-B", "Data Mining", "10:00", "11:00")
    test_db_manager.add_student(stu_id, "Student One", "AI&DS", "B")

    # Record first observation
    t1 = "2026-09-28T10:02:00"
    success, is_new = test_db_manager.record_or_update_attendance(
        session_id=sess_id,
        student_id=stu_id,
        status="PRESENT",
        first_seen=t1,
        last_seen=t1,
        track_id=1,
        similarity=0.85
    )
    assert success is True
    assert is_new is True

    # Record second observation (update)
    t2 = "2026-09-28T10:15:00"
    success, is_new = test_db_manager.record_or_update_attendance(
        session_id=sess_id,
        student_id=stu_id,
        status="PRESENT",
        first_seen=t1,
        last_seen=t2,
        track_id=2,
        similarity=0.91
    )
    assert success is True
    assert is_new is False

    # Verify MongoDB record
    rec = test_db_manager.get_attendance_record(sess_id, stu_id)
    assert rec is not None
    assert rec["first_seen"] == t1
    assert rec["last_seen"] == t2
    assert rec["latest_similarity"] == 0.91


# =============================================================================
# 13. Attendance Retrieval
# =============================================================================

def test_13_attendance_retrieval(test_db_manager):
    """Tests session attendance lists and student attendance history."""
    sess_id = "SESS_RET_01"
    stu_id = "STU_RET_01"

    test_db_manager.create_session(sess_id, "2026-09-28", "AIDS-B", "Deep Learning", "11:00", "12:00")
    test_db_manager.add_student(stu_id, "Jane Doe", "AI&DS", "B")
    test_db_manager.record_or_update_attendance(
        session_id=sess_id,
        student_id=stu_id,
        status="PRESENT",
        first_seen="2026-09-28T11:05:00",
        last_seen="2026-09-28T11:20:00",
        track_id=10,
        similarity=0.88
    )

    records = test_db_manager.get_attendance_for_session(sess_id)
    assert len(records) >= 1
    assert records[0]["student_id"] == stu_id

    history = test_db_manager.get_student_attendance_history(stu_id)
    assert len(history) >= 1
    assert history[0]["session_id"] == sess_id


# =============================================================================
# 14. Existing RBAC Permissions
# =============================================================================

def test_14_existing_rbac_permissions(test_user_service):
    """Verifies RBAC rules remain enforced with MongoDB persistence."""
    from app.main import app
    from app.auth import get_user_service

    app.dependency_overrides[get_user_service] = lambda: test_user_service
    client = TestClient(app)

    # Faculty token attempting HOD endpoint -> 403 Forbidden
    fac_token = create_access_token(user_id="FAC001", role="faculty", name="Dr. Anand")
    client.cookies.clear()
    res = client.post(
        "/api/hod/faculty",
        json={"user_id": "FAC_ILLEGAL", "name": "Illegal", "password": "pass", "department": "AI&DS"},
        headers={"Authorization": f"Bearer {fac_token}"}
    )
    assert res.status_code == 403

    app.dependency_overrides.clear()


# =============================================================================
# 15. Unauthorized Access
# =============================================================================

def test_15_unauthorized_access_rejected(test_user_service):
    """Verifies unauthenticated calls to protected routes return 401."""
    from app.main import app
    from app.auth import get_user_service

    app.dependency_overrides[get_user_service] = lambda: test_user_service
    client = TestClient(app)
    client.cookies.clear()

    res = client.get("/api/auth/me")
    assert res.status_code == 401

    app.dependency_overrides.clear()


# =============================================================================
# 16. Duplicate Identifiers Rejected
# =============================================================================

def test_16_duplicate_identifiers_rejected(test_user_service):
    """Verifies that creating a duplicate user_id raises ValueError."""
    with pytest.raises(ValueError, match="already exists"):
        test_user_service.create_user(UserModel(
            user_id="HOD001",  # Already exists from bootstrap
            name="Duplicate HOD",
            role=UserRole.HOD,
            password_hash=hash_password("pass123")
        ))


# =============================================================================
# 17. Invalid Data Handling
# =============================================================================

def test_17_invalid_data_handling(test_user_service):
    """Verifies that invalid status update raises ValueError."""
    with pytest.raises(ValueError, match="Invalid status"):
        test_user_service.update_status("HOD001", "invalid_status_value")


# =============================================================================
# 18. MongoDB Unavailable Behavior
# =============================================================================

def test_18_mongodb_unavailable_behavior(tmp_path):
    """Verifies system falls back to resilient SQLite when MongoDB is unreachable."""
    db_file = str(tmp_path / "offline_smartclass.sqlite")
    # Point to invalid port to test timeout/offline handling
    offline_svc = UserService(db_path=db_file, mongo_uri="mongodb://127.0.0.1:27099")
    assert offline_svc.mongo_online is False

    # Should still succeed via SQLite fallback
    hod = offline_svc.get_user_by_id("HOD001")
    assert hod is not None
    assert hod.user_id == "HOD001"


# =============================================================================
# 19. Migration Validation
# =============================================================================

def test_19_migration_validation(mock_mongo_client):
    """Verifies non-destructive SQLite to MongoDB migration."""
    res = migrate_sqlite_to_mongodb(
        sqlite_path="database/smartclass.sqlite",
        mongo_client=mock_mongo_client
    )
    assert res["status"] == "success"
    assert res["migration_report"]["users"]["migrated"] >= 3
    assert res["migration_report"]["classrooms"]["migrated"] >= 2
    assert res["validation"]["mongo_users"] >= 3


# =============================================================================
# 20. Existing AI Startup & Import Integrity
# =============================================================================

def test_20_existing_ai_startup_import():
    """Verifies protected AI pipeline modules load without error or regression."""
    from core.scrfd_detector import SCRFDDetector
    from core.face_embedder import ArcFaceEmbedder
    from core.vector_store import FaissVectorStore
    from core.tracker import BaseTracker, get_tracker
    from core.byte_tracker import ByteTracker
    from core.recognizer import FaceRecognizer
    from core.face_alignment import FaceAligner
    from core.face_quality import FaceQualityAssessor
    from core.tracking_pipeline import TrackingPipeline
    from core.temporal_stabilizer import TemporalStabilizer

    assert SCRFDDetector is not None
    assert ArcFaceEmbedder is not None
    assert FaissVectorStore is not None
    assert BaseTracker is not None
    assert ByteTracker is not None
    assert get_tracker is not None
    assert FaceRecognizer is not None
    assert FaceAligner is not None
    assert FaceQualityAssessor is not None
    assert TrackingPipeline is not None
    assert TemporalStabilizer is not None


# =============================================================================
# 21. Existing Camera Startup & Import Integrity
# =============================================================================

def test_21_existing_camera_startup_import():
    """Verifies protected camera pipeline modules load without error or regression."""
    from camera.base_camera import BaseCamera
    from camera.smartboard_camera import SmartBoardCamera
    from camera.camera_manager import CameraManager
    from camera.droidcam_camera import DroidCamCamera
    from camera.esp32_camera import ESP32Camera

    assert BaseCamera is not None
    assert SmartBoardCamera is not None
    assert CameraManager is not None
    assert DroidCamCamera is not None
    assert ESP32Camera is not None


# =============================================================================
# 22. FAISS Index Compatibility
# =============================================================================

def test_22_faiss_index_compatibility(tmp_path):
    """Verifies FAISS vector store and index compatibility remains completely intact."""
    import faiss
    from core.vector_store import FaissVectorStore

    test_idx_path = str(tmp_path / "test_faiss_index.bin")
    store = FaissVectorStore(embedding_dim=512, index_path=test_idx_path)
    assert store.embedding_dim == 512
    assert store.index.d == 512

    # Add a sample L2-normalized 512-dim vector
    test_vec = np.random.randn(512).astype(np.float32)
    test_vec = test_vec / np.linalg.norm(test_vec)
    store.add_vector(embedding_id=1, vector=test_vec, metadata={"student_id": "STU001"})

    # Save and reload
    store.save_index(test_idx_path)
    assert os.path.exists(test_idx_path)

    loaded_index = faiss.read_index(test_idx_path)
    assert loaded_index is not None
    assert loaded_index.d == 512
    assert loaded_index.ntotal == 1
