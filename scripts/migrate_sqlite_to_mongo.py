import os
import sys
import sqlite3
import argparse
from typing import Dict, Any, List
from loguru import logger

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings
from database.mongo_repository import MongoDatabase


def migrate_sqlite_to_mongodb(
    sqlite_path: str = "database/smartclass.sqlite",
    new_enrollment_path: str = "database/new_enrollment.sqlite",
    mongo_uri: str = None,
    database_name: str = None,
    dry_run: bool = False,
    mongo_client: Any = None
) -> Dict[str, Any]:
    """
    Controlled, idempotent, non-destructive migration from SQLite to MongoDB:
    - Users (HOD, Faculty, Class Advisors) -> users, faculty, advisors
    - Classrooms -> classrooms
    - Students (from both smartclass.sqlite & new_enrollment.sqlite) -> students
    - Sessions -> sessions
    - Attendance -> attendance_records
    - Validates migration counts post-migration
    - Preserves SQLite database unchanged
    """
    settings = get_settings().mongodb
    uri = mongo_uri or settings.uri
    dbname = database_name or settings.database

    logger.info(f"Starting SQLite to MongoDB migration. Target DB: '{dbname}', DryRun: {dry_run}")

    if not os.path.exists(sqlite_path):
        logger.error(f"Source SQLite database not found at '{sqlite_path}'.")
        return {"status": "error", "message": f"SQLite database '{sqlite_path}' not found."}

    # Initialize MongoDB connection
    mongo_db = MongoDatabase(mongo_uri=uri, db_name=dbname, client=mongo_client)
    if not mongo_db.is_online():
        logger.error(f"Cannot connect to MongoDB at '{uri}'. Migration aborted.")
        return {"status": "error", "message": f"Cannot connect to MongoDB at '{uri}'."}

    report = {
        "users": {"scanned": 0, "migrated": 0, "errors": 0},
        "faculty": {"scanned": 0, "migrated": 0, "errors": 0},
        "advisors": {"scanned": 0, "migrated": 0, "errors": 0},
        "classrooms": {"scanned": 0, "migrated": 0, "errors": 0},
        "students": {"scanned": 0, "migrated": 0, "errors": 0},
        "sessions": {"scanned": 0, "migrated": 0, "errors": 0},
        "attendance": {"scanned": 0, "migrated": 0, "errors": 0},
    }

    # Connect to SQLite
    sconn = sqlite3.connect(sqlite_path)
    sconn.row_factory = sqlite3.Row
    scur = sconn.cursor()

    # 1. Migrate Users
    try:
        scur.execute("SELECT * FROM users;")
        users = [dict(r) for r in scur.fetchall()]
        report["users"]["scanned"] = len(users)
        for u in users:
            uid = u["user_id"]
            if not dry_run:
                mongo_db.users.update_one({"user_id": uid}, {"$set": u}, upsert=True)
            report["users"]["migrated"] += 1

            # Sync faculty / advisor projections
            if u.get("role") == "faculty":
                report["faculty"]["scanned"] += 1
                fac_doc = {
                    "faculty_id": uid,
                    "name": u.get("name"),
                    "email": u.get("email"),
                    "phone": u.get("phone"),
                    "department": u.get("department") or "AI&DS",
                    "assigned_classroom": u.get("assigned_classroom"),
                    "status": u.get("status") or "active",
                    "created_at": u.get("created_at"),
                    "updated_at": u.get("updated_at")
                }
                if not dry_run:
                    mongo_db.faculty.update_one({"faculty_id": uid}, {"$set": fac_doc}, upsert=True)
                report["faculty"]["migrated"] += 1

            elif u.get("role") == "class_advisor":
                report["advisors"]["scanned"] += 1
                adv_doc = {
                    "advisor_id": uid,
                    "name": u.get("name"),
                    "email": u.get("email"),
                    "phone": u.get("phone"),
                    "department": u.get("department") or "AI&DS",
                    "year": u.get("year") or "3rd Year",
                    "section": u.get("section") or "B",
                    "assigned_classroom": u.get("assigned_classroom"),
                    "status": u.get("status") or "active",
                    "created_at": u.get("created_at"),
                    "updated_at": u.get("updated_at")
                }
                if not dry_run:
                    mongo_db.advisors.update_one({"advisor_id": uid}, {"$set": adv_doc}, upsert=True)
                report["advisors"]["migrated"] += 1
    except Exception as e:
        logger.error(f"Error migrating users: {e}")
        report["users"]["errors"] += 1

    # 2. Migrate Classrooms
    try:
        scur.execute("SELECT * FROM classrooms;")
        classrooms = [dict(r) for r in scur.fetchall()]
        report["classrooms"]["scanned"] = len(classrooms)
        for c in classrooms:
            cid = c["classroom_id"]
            if not dry_run:
                mongo_db.classrooms.update_one({"classroom_id": cid}, {"$set": c}, upsert=True)
            report["classrooms"]["migrated"] += 1
    except Exception as e:
        logger.error(f"Error migrating classrooms: {e}")
        report["classrooms"]["errors"] += 1

    # 3. Migrate Students (smartclass.sqlite + new_enrollment.sqlite)
    seen_student_ids = set()
    try:
        # Check authoritative new_enrollment.sqlite first
        if os.path.exists(new_enrollment_path):
            with sqlite3.connect(new_enrollment_path) as nconn:
                nconn.row_factory = sqlite3.Row
                ncur = nconn.cursor()
                ncur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='enrolled_students';")
                if ncur.fetchone():
                    ncur.execute("SELECT * FROM enrolled_students;")
                    enrolled = [dict(r) for r in ncur.fetchall()]
                    for s in enrolled:
                        sid = s["student_id"]
                        seen_student_ids.add(sid)
                        stu_doc = {
                            "student_id": sid,
                            "register_no": s.get("register_number"),
                            "student_name": s.get("name"),
                            "department": s.get("department") or "AI&DS",
                            "section": s.get("section") or "B",
                            "class_name": s.get("class") or f"{s.get('department', '')} - {s.get('section', '')}".strip(" -"),
                            "status": "active",
                            "sample_count": 0,
                            "created_at": s.get("created_at"),
                            "updated_at": s.get("updated_at")
                        }
                        report["students"]["scanned"] += 1
                        if not dry_run:
                            mongo_db.students.update_one({"student_id": sid}, {"$set": stu_doc}, upsert=True)
                        report["students"]["migrated"] += 1

        # Check smartclass.sqlite students table
        scur.execute("SELECT * FROM students;")
        stu_rows = [dict(r) for r in scur.fetchall()]
        for s in stu_rows:
            sid = s["student_id"]
            if sid not in seen_student_ids:
                report["students"]["scanned"] += 1
                if not dry_run:
                    mongo_db.students.update_one({"student_id": sid}, {"$set": s}, upsert=True)
                report["students"]["migrated"] += 1
    except Exception as e:
        logger.error(f"Error migrating students: {e}")
        report["students"]["errors"] += 1

    # 4. Migrate Sessions
    try:
        scur.execute("SELECT * FROM sessions;")
        sessions = [dict(r) for r in scur.fetchall()]
        report["sessions"]["scanned"] = len(sessions)
        for sess in sessions:
            sid = sess["session_id"]
            if not dry_run:
                mongo_db.sessions.update_one({"session_id": sid}, {"$set": sess}, upsert=True)
            report["sessions"]["migrated"] += 1
    except Exception as e:
        logger.error(f"Error migrating sessions: {e}")
        report["sessions"]["errors"] += 1

    # 5. Migrate Attendance
    try:
        scur.execute("SELECT * FROM attendance;")
        attendance_rows = [dict(r) for r in scur.fetchall()]
        report["attendance"]["scanned"] = len(attendance_rows)
        for att in attendance_rows:
            sid = att["session_id"]
            stid = att["student_id"]
            if not dry_run:
                mongo_db.attendance_records.update_one(
                    {"session_id": sid, "student_id": stid},
                    {"$set": att},
                    upsert=True
                )
            report["attendance"]["migrated"] += 1
    except Exception as e:
        logger.error(f"Error migrating attendance: {e}")
        report["attendance"]["errors"] += 1

    sconn.close()

    # Validation check
    validation = {}
    if not dry_run:
        validation = {
            "mongo_users": mongo_db.users.count_documents({}),
            "mongo_faculty": mongo_db.faculty.count_documents({}),
            "mongo_advisors": mongo_db.advisors.count_documents({}),
            "mongo_classrooms": mongo_db.classrooms.count_documents({}),
            "mongo_students": mongo_db.students.count_documents({}),
            "mongo_sessions": mongo_db.sessions.count_documents({}),
            "mongo_attendance_records": mongo_db.attendance_records.count_documents({}),
        }
        logger.info(f"Migration completed successfully. MongoDB Counts: {validation}")

    return {
        "status": "success",
        "dry_run": dry_run,
        "migration_report": report,
        "validation": validation
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate SQLite data to MongoDB")
    parser.add_argument("--sqlite", default="database/smartclass.sqlite", help="Path to smartclass.sqlite")
    parser.add_argument("--uri", default=None, help="MongoDB connection URI")
    parser.add_argument("--db", default=None, help="MongoDB database name")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without writing to MongoDB")
    args = parser.parse_args()

    res = migrate_sqlite_to_mongodb(
        sqlite_path=args.sqlite,
        mongo_uri=args.uri,
        database_name=args.db,
        dry_run=args.dry_run
    )
    print(res)
