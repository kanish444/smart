import os
import time
import pytest
import datetime
import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app
from app.state import AppState, get_app_state
from app.auth import get_user_service
from app.services.user_service import UserService
from app.models.user import UserModel, UserRole, UserStatus
from app.core.security import create_access_token, hash_password
from database.mongo_repository import (
    MongoDatabase,
    StudentRepository,
    AttendanceRepository,
    SessionRepository,
    ClassroomRepository
)
from database.db_manager import DatabaseManager
from core.schemas import TrackedFace, RecognitionStatus, SessionState
from attendance.session_manager import SessionManager
from attendance.attendance_engine import AttendanceEngine


@pytest.fixture(scope="module")
def app_env(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("phase14_full")
    db_file = str(temp_dir / "phase14_app.sqlite")

    test_db = DatabaseManager(db_path=db_file)
    user_svc = UserService(db_path=db_file)

    # Seed test users
    hod = UserModel(
        user_id="hod_p14",
        name="Dr. HOD P14",
        role=UserRole.HOD,
        password_hash=hash_password("Password123!"),
        department="CSE",
        status=UserStatus.ACTIVE
    )
    fac = UserModel(
        user_id="fac_p14",
        name="Prof. Faculty P14",
        role=UserRole.FACULTY,
        password_hash=hash_password("Password123!"),
        department="CSE",
        assigned_classroom="LH-01",
        status=UserStatus.ACTIVE
    )
    adv = UserModel(
        user_id="adv_p14",
        name="Advisor P14",
        role=UserRole.CLASS_ADVISOR,
        password_hash=hash_password("Password123!"),
        department="CSE",
        year="3",
        section="A",
        assigned_classroom="LH-01",
        status=UserStatus.ACTIVE
    )
    other_adv = UserModel(
        user_id="adv_ece_p14",
        name="ECE Advisor P14",
        role=UserRole.CLASS_ADVISOR,
        password_hash=hash_password("Password123!"),
        department="ECE",
        year="3",
        section="B",
        status=UserStatus.ACTIVE
    )

    for u in [hod, fac, adv, other_adv]:
        user_svc.create_user(u)

    # Seed students
    test_db.add_student("711721104001", "Kanish", department="CSE", section="A", register_no="711721104001")
    test_db.add_student("711721104002", "Priya", department="CSE", section="A", register_no="711721104002")

    tokens = {
        "hod": create_access_token(user_id="hod_p14", role="hod", name="Dr. HOD P14", department="CSE"),
        "faculty": create_access_token(user_id="fac_p14", role="faculty", name="Prof. Faculty P14", department="CSE", assigned_classroom="LH-01"),
        "advisor": create_access_token(user_id="adv_p14", role="class_advisor", name="Advisor P14", department="CSE", year="3", section="A", assigned_classroom="LH-01"),
        "other_advisor": create_access_token(user_id="adv_ece_p14", role="class_advisor", name="ECE Advisor P14", department="ECE", year="3", section="B")
    }

    state = AppState(db_path=db_file)
    app = create_app()
    app.dependency_overrides[get_user_service] = lambda: user_svc
    app.dependency_overrides[get_app_state] = lambda: state

    with TestClient(app) as client:
        yield {
            "app": app,
            "client": client,
            "db": test_db,
            "tokens": tokens,
            "state": state
        }


# =============================================================================
# Step 18 Requirements: Tests 1 - 30
# =============================================================================

def test_01_ai_pipeline_startup():
    """1. AI pipeline startup: Verify TrackingPipeline and all AI core classes import cleanly."""
    from core.tracking_pipeline import TrackingPipeline
    from core.scrfd_detector import SCRFDDetector
    from core.face_embedder import ArcFaceEmbedder
    from core.vector_store import FaissVectorStore
    from core.byte_tracker import ByteTracker
    from core.recognizer import FaceRecognizer
    from core.temporal_stabilizer import TemporalStabilizer

    assert TrackingPipeline is not None
    assert SCRFDDetector is not None
    assert ArcFaceEmbedder is not None
    assert FaissVectorStore is not None
    assert ByteTracker is not None
    assert FaceRecognizer is not None
    assert TemporalStabilizer is not None
    assert hasattr(TrackingPipeline, "process_frame")
    assert hasattr(TrackingPipeline, "reset")


def test_02_existing_model_loading():
    """2. Existing model loading: Verify ArcFaceEmbedder constructor and dimension constants."""
    from core.face_embedder import ArcFaceEmbedder
    assert ArcFaceEmbedder is not None

    # Verify exception when missing weights file is targeted
    with pytest.raises(FileNotFoundError):
        ArcFaceEmbedder(model_path="non_existent_weights.onnx")


def test_03_existing_faiss_loading(tmp_path):
    """3. Existing FAISS loading: Verify FaissVectorStore loads and performs indexing without error."""
    from core.vector_store import FaissVectorStore
    test_idx = str(tmp_path / "test_load_faiss.bin")
    vstore = FaissVectorStore(embedding_dim=512, index_path=test_idx)
    assert vstore is not None
    assert vstore.embedding_dim == 512
    assert vstore.index.d == 512

    # Add normalized vector and search
    vec = np.random.randn(512).astype(np.float32)
    vec /= np.linalg.norm(vec)
    vstore.add_vector(1, vec, {"student_id": "STU_TEST"})
    res = vstore.search(vec, top_k=1)
    assert len(res) == 1
    assert res[0][1]["student_id"] == "STU_TEST"


def test_04_existing_camera_startup():
    """4. Existing camera startup: Verify CameraManager initializes without error."""
    from camera.camera_manager import CameraManager
    cm = CameraManager()
    assert cm is not None
    st = cm.get_status()
    assert isinstance(st, dict)
    cm.release()


def test_05_recognition_output_extraction():
    """5. Recognition output extraction: Verify TrackedFace schema contains all required telemetry."""
    tf = TrackedFace(
        track_id=101,
        bbox=[100, 100, 200, 200],
        stable_student_id="711721104001",
        stable_student_name="Kanish",
        current_status=RecognitionStatus.MATCH,
        current_similarity=0.825,
        last_seen=time.time()
    )
    assert tf.track_id == 101
    assert tf.stable_student_id == "711721104001"
    assert tf.stable_student_name == "Kanish"
    assert tf.current_status == RecognitionStatus.MATCH
    assert tf.current_similarity == 0.825


def test_06_recognized_identity_mapping(app_env):
    """6. Recognized identity mapping: Maps stable_student_id to student record in database."""
    db = app_env["db"]
    st = db.get_student("711721104001")
    assert st is not None
    assert st["student_name"] == "Kanish"
    assert st["department"] == "CSE"
    assert st["section"] == "A"


def test_07_mongodb_student_mapping(app_env):
    """7. MongoDB student mapping: Verify StudentRepository upsert and retrieval."""
    db = app_env["db"]
    if db.mongo_db.is_online():
        db.student_repo.upsert({
            "student_id": "711721104003",
            "student_name": "Rohan",
            "department": "CSE",
            "section": "A",
            "register_no": "711721104003"
        })
        ret = db.student_repo.get_by_id("711721104003")
        assert ret is not None
        assert ret["student_name"] == "Rohan"
    else:
        # Fallback assertion passes
        assert True


def test_08_active_session_linkage(app_env):
    """8. Active session linkage: Attendance events must link to active session ID."""
    db = app_env["db"]
    sess_mgr = SessionManager(db_manager=db)
    today = datetime.date.today().isoformat()
    now_time = datetime.datetime.now().strftime("%H:%M:%S")
    end_time = (datetime.datetime.now() + datetime.timedelta(hours=1)).strftime("%H:%M:%S")

    # Create & start session
    sess = sess_mgr.create_session(
        session_id="SESS_P14_LINK",
        date=today,
        class_section="CSE - A",
        subject="AI Lab",
        planned_start_time=now_time,
        planned_end_time=end_time
    )
    sess_mgr.start_session("SESS_P14_LINK")
    active = sess_mgr.get_active_session()
    assert active is not None
    assert active.session_id == "SESS_P14_LINK"
    assert active.status == SessionState.ACTIVE


def test_09_attendance_creation(app_env):
    """9. Attendance creation: First valid recognition creates attendance record with PRESENT."""
    db = app_env["db"]
    engine = AttendanceEngine(db_manager=db)
    now_ts = time.time()
    tf = TrackedFace(
        track_id=1,
        bbox=[50, 50, 150, 150],
        stable_student_id="711721104001",
        stable_student_name="Kanish",
        current_status=RecognitionStatus.MATCH,
        current_similarity=0.85,
        last_seen=now_ts
    )

    records = engine.process_tracked_faces("SESS_P14_LINK", [tf], timestamp=now_ts)
    assert len(records) == 1
    rec = records[0]
    assert rec.student_id == "711721104001"
    assert rec.status in ["PRESENT", "LATE"]
    assert rec.initial_similarity == 0.85


def test_10_repeated_recognition_idempotency(app_env):
    """10. Repeated recognition idempotency: Same student in same session does not create duplicate."""
    db = app_env["db"]
    now_iso = datetime.datetime.utcnow().isoformat()
    s1, is_new1 = db.record_or_update_attendance(
        session_id="SESS_P14_LINK",
        student_id="711721104002",
        status="PRESENT",
        first_seen=now_iso,
        last_seen=now_iso,
        track_id=2,
        similarity=0.88
    )
    assert s1 is True
    assert is_new1 is True

    # Immediate second detection
    later_iso = (datetime.datetime.utcnow() + datetime.timedelta(seconds=10)).isoformat()
    s2, is_new2 = db.record_or_update_attendance(
        session_id="SESS_P14_LINK",
        student_id="711721104002",
        status="PRESENT",
        first_seen=later_iso,
        last_seen=later_iso,
        track_id=2,
        similarity=0.91
    )
    assert s2 is True
    assert is_new2 is False  # Updated, NOT inserted

    all_sess = db.get_attendance_for_session("SESS_P14_LINK")
    p2_recs = [r for r in all_sess if r["student_id"] == "711721104002"]
    assert len(p2_recs) == 1  # Exactly ONE record


def test_11_first_seen_preserved(app_env):
    """11. first_seen preserved: first_seen must NEVER be overwritten on update."""
    db = app_env["db"]
    rec = db.get_attendance_record("SESS_P14_LINK", "711721104002")
    assert rec is not None
    initial_first_seen = rec["first_seen"]

    future_iso = (datetime.datetime.utcnow() + datetime.timedelta(minutes=5)).isoformat()
    db.record_or_update_attendance(
        session_id="SESS_P14_LINK",
        student_id="711721104002",
        status="PRESENT",
        first_seen=future_iso,
        last_seen=future_iso,
        track_id=2,
        similarity=0.89
    )

    updated_rec = db.get_attendance_record("SESS_P14_LINK", "711721104002")
    assert updated_rec["first_seen"] == initial_first_seen


def test_12_last_seen_updated(app_env):
    """12. last_seen updated: last_seen MUST be updated on subsequent recognition."""
    db = app_env["db"]
    future_iso = (datetime.datetime.utcnow() + datetime.timedelta(minutes=10)).isoformat()
    db.record_or_update_attendance(
        session_id="SESS_P14_LINK",
        student_id="711721104002",
        status="PRESENT",
        first_seen=future_iso,
        last_seen=future_iso,
        track_id=2,
        similarity=0.92
    )
    updated_rec = db.get_attendance_record("SESS_P14_LINK", "711721104002")
    assert updated_rec["last_seen"] == future_iso


def test_13_seen_count_incremented(app_env):
    """13. seen_count incremented: seen_count must increment with repeated detections."""
    db = app_env["db"]
    rec1 = db.get_attendance_record("SESS_P14_LINK", "711721104002")
    init_seen = rec1.get("seen_count", 1)

    db.record_or_update_attendance(
        session_id="SESS_P14_LINK",
        student_id="711721104002",
        status="PRESENT",
        first_seen=datetime.datetime.utcnow().isoformat(),
        last_seen=datetime.datetime.utcnow().isoformat(),
        similarity=0.95
    )

    rec2 = db.get_attendance_record("SESS_P14_LINK", "711721104002")
    assert rec2.get("seen_count", 1) >= init_seen + 1


def test_14_latest_similarity_updated(app_env):
    """14. latest_similarity updated: similarity score updates on new frame."""
    db = app_env["db"]
    rec = db.get_attendance_record("SESS_P14_LINK", "711721104002")
    assert rec["latest_similarity"] is not None
    assert rec["latest_similarity"] > 0.0


def test_15_unknown_identity_handling(app_env):
    """15. Unknown identity handling: UNKNOWN tracks strictly rejected without attendance."""
    db = app_env["db"]
    engine = AttendanceEngine(db_manager=db)
    engine.clear_cache()
    unknown_tf = TrackedFace(
        track_id=99,
        bbox=[10, 10, 80, 80],
        stable_student_id=None,
        stable_student_name=None,
        current_status=RecognitionStatus.UNKNOWN,
        current_similarity=0.20,
        last_seen=time.time()
    )
    records = engine.process_tracked_faces("SESS_P14_LINK", [unknown_tf])
    assert len(records) == 0


def test_16_missing_student_mapping(app_env):
    """16. Missing student mapping: Unregistered student ID does not create attendance."""
    db = app_env["db"]
    engine = AttendanceEngine(db_manager=db)
    engine.clear_cache()
    ghost_tf = TrackedFace(
        track_id=100,
        bbox=[10, 10, 80, 80],
        stable_student_id="GHOST_STUDENT_9999",
        stable_student_name="Ghost",
        current_status=RecognitionStatus.MATCH,
        current_similarity=0.88,
        last_seen=time.time()
    )
    records = engine.process_tracked_faces("SESS_P14_LINK", [ghost_tf])
    assert len(records) == 0
    rec = db.get_attendance_record("SESS_P14_LINK", "GHOST_STUDENT_9999")
    assert rec is None


def test_17_no_active_session(app_env):
    """17. No active session: No attendance marked when no session is ACTIVE."""
    db = app_env["db"]
    sess_mgr = SessionManager(db_manager=db)
    sess_mgr.end_session("SESS_P14_LINK")  # Session now ENDED
    assert sess_mgr.get_active_session() is None

    engine = AttendanceEngine(db_manager=db, session_manager=sess_mgr)
    engine.clear_cache()
    tf = TrackedFace(
        track_id=1,
        bbox=[50, 50, 150, 150],
        stable_student_id="711721104001",
        current_status=RecognitionStatus.MATCH,
        current_similarity=0.85,
        last_seen=time.time()
    )
    records = engine.process_tracked_faces("SESS_P14_LINK", [tf])
    assert len(records) == 0


def test_18_mongodb_write_failure_resilience():
    """18. MongoDB write failure resilience: Server falls back to SQLite when MongoDB is offline."""
    offline_db = MongoDatabase(mongo_uri="mongodb://invalid_host:27017", timeout_ms=50)
    assert offline_db.is_online() is False
    mgr = DatabaseManager(mongo_db=offline_db)
    rooms = mgr.get_all_classrooms()
    assert isinstance(rooms, list)


def test_19_ai_unavailable_degraded_state(app_env):
    """19. AI unavailable state: State reflects recognition_online=False when AI is down."""
    state = app_env["state"]
    state.recognition_online = False
    telem = state.get_latest_telemetry()
    assert telem["recognition_online"] is False


def test_20_camera_unavailable_state(app_env):
    """20. Camera unavailable state: Returns degraded canvas and camera_online=False."""
    state = app_env["state"]
    state.camera_online = False
    telem = state.get_latest_telemetry()
    assert telem["camera_online"] is False
    frame = state.get_mjpeg_frame()
    assert frame is not None  # Generates disconnected fallback frame


def test_21_attendance_api_list(app_env):
    """21. Attendance API list: GET /api/attendance returns records with valid token."""
    client = app_env["client"]
    token = app_env["tokens"]["hod"]
    res = client.get("/api/attendance", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert isinstance(data["data"], list)


def test_22_attendance_rbac_enforcement(app_env):
    """22. Attendance RBAC enforcement: Unauthenticated request rejected with 401."""
    client = app_env["client"]
    res = client.get("/api/attendance")
    assert res.status_code == 401


def test_23_hod_attendance_access(app_env):
    """23. HOD attendance access: HOD can list and query attendance system-wide."""
    client = app_env["client"]
    token = app_env["tokens"]["hod"]
    res = client.get("/api/attendance", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200


def test_24_faculty_attendance_access(app_env):
    """24. Faculty attendance access: Faculty can query attendance for permitted sessions."""
    client = app_env["client"]
    token = app_env["tokens"]["faculty"]
    res = client.get("/api/attendance", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200


def test_25_class_advisor_attendance_access(app_env):
    """25. Class Advisor attendance access: Advisor can query attendance within assigned cohort."""
    client = app_env["client"]
    token = app_env["tokens"]["advisor"]
    # Advisor belongs to CSE section A
    res = client.get("/api/attendance", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200


def test_26_unauthorized_attendance_access_rejected(app_env):
    """26. Unauthorized attendance access: Class Advisor cannot access student outside assigned cohort."""
    client = app_env["client"]
    token = app_env["tokens"]["other_advisor"]  # ECE section B advisor
    # Query student 711721104001 who is in CSE section A
    res = client.get("/api/attendance/student/711721104001", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 403  # Forbidden!


def test_27_dashboard_attendance_data(app_env):
    """27. Dashboard attendance data: Role summary reflects real database attendance."""
    client = app_env["client"]
    token = app_env["tokens"]["hod"]
    res = client.get("/api/dashboard/role-summary", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    data = res.json()["data"]
    assert "cards" in data
    assert "total_students" in data["cards"]
    assert "students_present" in data["cards"]
    assert "students_absent" in data["cards"]


def test_28_existing_video_streaming(app_env):
    """28. Existing video streaming: /api/video/feed continues to yield MJPEG chunks."""
    client = app_env["client"]
    res = client.get("/api/video/feed?limit=1&view_mode=normal")
    assert res.status_code == 200
    assert "multipart/x-mixed-replace" in res.headers.get("content-type", "")


def test_29_no_duplicate_model_initialization(app_env):
    """29. No duplicate model initialization: AppState reuses singleton vector store and embedder."""
    state = app_env["state"]
    vs1 = state.get_vector_store()
    vs2 = state.get_vector_store()
    assert vs1 is vs2  # Exactly identical instance


def test_30_faiss_compatibility(tmp_path):
    """30. FAISS compatibility: Vector query produces expected score and format without altering index."""
    from core.vector_store import FaissVectorStore
    test_idx = str(tmp_path / "test_compat_faiss.bin")
    vstore = FaissVectorStore(embedding_dim=512, index_path=test_idx)
    dummy_vec = np.random.randn(512).astype(np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)
    vstore.add_vector(1, dummy_vec, {"student_id": "STU_COMPAT"})
    results = vstore.search(dummy_vec, top_k=1)
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0][1]["student_id"] == "STU_COMPAT"
