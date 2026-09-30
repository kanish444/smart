import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.user_service import UserService
from app.models.user import UserModel, UserRole, UserStatus
from app.core.security import hash_password, create_access_token
from app.services.excel_service import ExcelService
from database.db_manager import DatabaseManager


@pytest.fixture(scope="module")
def excel_client(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("excel_test")
    test_db = str(temp_dir / "smartclass.sqlite")
    new_db = str(temp_dir / "new_enrollment.sqlite")

    user_svc = UserService(db_path=test_db, mongo_uri="mongodb://localhost:27017")

    # ADV001 (if not already created)
    if not user_svc.get_user_by_id("ADV001"):
        adv = UserModel(
            user_id="ADV001",
            name="Prof. Advisor",
            role=UserRole.CLASS_ADVISOR,
            password_hash=hash_password("adv123"),
            email="advisor@college.edu",
            department="AI&DS",
            year="3rd Year",
            section="B",
            assigned_classroom="AIDS-B",
            status=UserStatus.ACTIVE
        )
        user_svc.create_user(adv)

    app = create_app()
    from app.auth import get_user_service
    from app.state import get_app_state
    state = get_app_state()
    # Inject isolated test db
    test_db_manager = DatabaseManager(db_path=test_db, new_db_path=new_db, mongo_uri="mongodb://localhost:27017")
    state.db = test_db_manager
    app.dependency_overrides[get_user_service] = lambda: user_svc
    app.dependency_overrides[get_app_state] = lambda: state

    with TestClient(app) as client:
        yield {
            "client": client,
            "hod_token": create_access_token("HOD001", "hod", "Dr. S. K. HOD"),
            "fac_token": create_access_token("FAC001", "faculty", "Dr. Faculty"),
            "adv_token": create_access_token("ADV001", "class_advisor", "Prof. Advisor", department="AI&DS", section="B"),
            "db": test_db_manager
        }


def test_01_excel_service_parse_xlsx():
    """Validates that ExcelService correctly parses xlsx bytes into student dicts."""
    df = pd.DataFrame([
        {"Register Number": "7376222AD201", "Student Name": "Rohan Verma", "Department": "AI&DS", "Section": "B", "Year": "3rd Year"},
        {"Register Number": "7376222AD202", "Student Name": "Sneha Roy", "Department": "AI&DS", "Section": "B", "Year": "3rd Year"},
    ])
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    raw_bytes = buf.getvalue()

    records, errors = ExcelService.parse_students_file(raw_bytes, "students.xlsx")
    assert len(records) == 2
    assert records[0]["register_no"] == "7376222AD201"
    assert records[0]["student_name"] == "Rohan Verma"
    assert records[1]["register_no"] == "7376222AD202"
    assert records[1]["student_name"] == "Sneha Roy"


def test_02_excel_service_parse_csv():
    """Validates that ExcelService correctly parses CSV bytes."""
    csv_text = "Roll No,Student Name,Dept,Sec\n7376222AD301,Manoj Kumar,AI&DS,B\n7376222AD302,Kavitha R,AI&DS,B\n"
    records, errors = ExcelService.parse_students_file(csv_text.encode("utf-8"), "roster.csv")
    assert len(records) == 2
    assert records[0]["register_no"] == "7376222AD301"
    assert records[0]["student_name"] == "Manoj Kumar"
    assert records[1]["register_no"] == "7376222AD302"


def test_03_hod_can_upload_excel_directly_to_database(excel_client):
    """
    Validates HOD can attach and upload an Excel file directly to database without manual entry:
    - Writes to SQLite and MongoDB
    - Students immediately appear in get_all_students() and /api/scoped/students
    """
    client = excel_client["client"]
    token = excel_client["hod_token"]

    df = pd.DataFrame([
        {"Register Number": "7376222AD401", "Student Name": "Pooja Hegde", "Department": "AI&DS", "Section": "B", "Year": "3rd Year"},
        {"Register Number": "7376222AD402", "Student Name": "Karan Singhania", "Department": "AI&DS", "Section": "B", "Year": "3rd Year"},
        {"Register Number": "7376222AD403", "Student Name": "Deepika Padukone", "Department": "AI&DS", "Section": "B", "Year": "3rd Year"}
    ])
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)

    files = {"file": ("class_roster.xlsx", buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    data = {"default_department": "AI&DS", "default_section": "B", "default_year": "3rd Year"}

    resp = client.post(
        "/api/hod/students/upload-excel",
        headers={"Authorization": f"Bearer {token}"},
        files=files,
        data=data
    )
    assert resp.status_code == 200
    res_data = resp.json()["data"]
    assert res_data["total"] == 3
    assert res_data["inserted"] == 3
    assert len(res_data["students"]) == 3

    # Verify students directly in database
    db = excel_client["db"]
    all_students = db.get_all_students()
    all_regs = [s["register_no"] for s in all_students]
    assert "7376222AD401" in all_regs
    assert "7376222AD402" in all_regs
    assert "7376222AD403" in all_regs

    # Verify via API /api/scoped/students
    resp_scoped = client.get("/api/scoped/students", headers={"Authorization": f"Bearer {token}"})
    assert resp_scoped.status_code == 200
    scoped_regs = [s["register_no"] for s in resp_scoped.json()["data"]]
    assert "7376222AD401" in scoped_regs


def test_04_faculty_cannot_upload_excel(excel_client):
    """Verifies that unauthorized faculty cannot upload student rosters (RBAC enforcement)."""
    client = excel_client["client"]
    token = excel_client["fac_token"]

    buf = io.BytesIO(b"fake excel content")
    files = {"file": ("students.xlsx", buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}

    resp = client.post(
        "/api/hod/students/upload-excel",
        headers={"Authorization": f"Bearer {token}"},
        files=files
    )
    assert resp.status_code in (401, 403)


def test_05_download_template(excel_client):
    """Verifies downloading the Excel and CSV template."""
    client = excel_client["client"]

    # Excel template
    resp_xlsx = client.get("/api/hod/students/template?format=xlsx")
    assert resp_xlsx.status_code == 200
    assert len(resp_xlsx.content) > 1000

    # CSV template
    resp_csv = client.get("/api/hod/students/template?format=csv")
    assert resp_csv.status_code == 200
    assert "Register Number,Student Name" in resp_csv.text
