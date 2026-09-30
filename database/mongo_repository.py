import os
import datetime
from typing import Optional, List, Dict, Any, Union
from loguru import logger
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError, DuplicateKeyError, ServerSelectionTimeoutError

from config.settings import get_settings
from app.models.data_models import (
    StudentModel,
    FacultyModel,
    AdvisorModel,
    ClassroomModel,
    SessionModel,
    AttendanceRecordModel,
    SensorTelemetryModel
)


class MongoUnavailableError(Exception):
    """Raised when MongoDB is unreachable and an operation strictly requires it."""
    pass


class MongoDatabase:
    """
    Centralized MongoDB Connection and Collection Access Layer:
    - Maintains MongoClient connection with configurable timeout
    - Manages lifecycle of collections: users, faculty, advisors, students, classrooms, sessions, attendance_records, sensor_telemetry
    - Automatically builds unique and compound indexes
    - Safe error containment (never leaks passwords or connection strings)
    """

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        db_name: Optional[str] = None,
        client: Optional[Any] = None,
        timeout_ms: Optional[int] = None
    ):
        settings = get_settings().mongodb
        self.mongo_uri = mongo_uri or settings.uri
        self.db_name = db_name or settings.database
        self.timeout_ms = timeout_ms or settings.timeout_ms

        self.client: Optional[MongoClient] = client
        self.db: Optional[Any] = None
        self.is_connected = False

        if client is not None:
            self.db = client[self.db_name]
            self.is_connected = True
            self.init_indexes()
        else:
            self.connect()

    def connect(self) -> bool:
        """Attempts connection to MongoDB with timeout."""
        try:
            self.client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=self.timeout_ms,
                connectTimeoutMS=self.timeout_ms
            )
            # Ping to verify server responsiveness
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.is_connected = True
            logger.info(f"MongoDB connected successfully to '{self.db_name}'.")
            self.init_indexes()
            return True
        except Exception as e:
            self.is_connected = False
            self.client = None
            self.db = None
            logger.warning(f"MongoDB unavailable ({type(e).__name__}: {e}). Operating in resilient mode.")
            return False

    def is_online(self) -> bool:
        """Checks if MongoDB connection is active."""
        if not self.is_connected or self.client is None or self.db is None:
            return False
        try:
            self.client.admin.command('ping')
            return True
        except Exception:
            self.is_connected = False
            return False

    def get_collection(self, name: str):
        """Retrieves a MongoDB collection or None if offline."""
        if not self.is_connected or self.db is None:
            return None
        return self.db[name]

    @property
    def users(self):
        return self.get_collection("users")

    @property
    def faculty(self):
        return self.get_collection("faculty")

    @property
    def advisors(self):
        return self.get_collection("advisors")

    @property
    def students(self):
        return self.get_collection("students")

    @property
    def classrooms(self):
        return self.get_collection("classrooms")

    @property
    def sessions(self):
        return self.get_collection("sessions")

    @property
    def attendance_records(self):
        return self.get_collection("attendance_records")

    @property
    def sensor_telemetry(self):
        return self.get_collection("sensor_telemetry")

    def init_indexes(self):
        """Builds all required uniqueness, lookup, and composite indexes."""
        if self.db is None:
            return
        try:
            # 1. users
            self.db["users"].create_index([("user_id", ASCENDING)], unique=True)
            self.db["users"].create_index([("email", ASCENDING)], sparse=True)
            self.db["users"].create_index([("role", ASCENDING)])

            # 2. faculty
            self.db["faculty"].create_index([("faculty_id", ASCENDING)], unique=True)
            self.db["faculty"].create_index([("department", ASCENDING)])

            # 3. advisors
            self.db["advisors"].create_index([("advisor_id", ASCENDING)], unique=True)
            self.db["advisors"].create_index([("assigned_classroom", ASCENDING)])
            self.db["advisors"].create_index([("department", ASCENDING), ("year", ASCENDING), ("section", ASCENDING)])

            # 4. students
            self.db["students"].create_index([("student_id", ASCENDING)], unique=True)
            self.db["students"].create_index([("register_no", ASCENDING)], sparse=True)
            self.db["students"].create_index([("department", ASCENDING), ("section", ASCENDING)])

            # 5. classrooms
            self.db["classrooms"].create_index([("classroom_id", ASCENDING)], unique=True)
            self.db["classrooms"].create_index([("assigned_advisor_id", ASCENDING)])
            self.db["classrooms"].create_index([("department", ASCENDING), ("section", ASCENDING)])

            # 6. sessions
            self.db["sessions"].create_index([("session_id", ASCENDING)], unique=True)
            self.db["sessions"].create_index([("date", ASCENDING)])
            self.db["sessions"].create_index([("status", ASCENDING)])
            self.db["sessions"].create_index([("class_section", ASCENDING), ("status", ASCENDING)])

            # 7. attendance_records
            self.db["attendance_records"].create_index(
                [("session_id", ASCENDING), ("student_id", ASCENDING)],
                unique=True
            )
            self.db["attendance_records"].create_index([("session_id", ASCENDING)])
            self.db["attendance_records"].create_index([("student_id", ASCENDING)])
            self.db["attendance_records"].create_index([("marked_at", DESCENDING)])

            # 8. sensor_telemetry
            self.db["sensor_telemetry"].create_index([("classroom_id", ASCENDING), ("timestamp", DESCENDING)])
            self.db["sensor_telemetry"].create_index([("device_id", ASCENDING)])

            logger.info("MongoDB indexes verified on all collections.")
        except Exception as e:
            logger.warning(f"Error creating MongoDB indexes: {e}")


# =============================================================================
# REPOSITORY IMPLEMENTATIONS
# =============================================================================

class StudentRepository:
    """Data-access repository for students collection."""

    def __init__(self, db: MongoDatabase):
        self.db = db

    def get_by_id(self, student_id: str) -> Optional[Dict[str, Any]]:
        col = self.db.students
        if col is None:
            return None
        doc = col.find_one({"student_id": student_id.strip()})
        if doc:
            doc.pop("_id", None)
        return doc

    def get_by_register_no(self, register_no: str) -> Optional[Dict[str, Any]]:
        col = self.db.students
        if col is None:
            return None
        doc = col.find_one({"register_no": register_no.strip()})
        if doc:
            doc.pop("_id", None)
        return doc

    def list(
        self,
        department: Optional[str] = None,
        section: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        col = self.db.students
        if col is None:
            return []
        query: Dict[str, Any] = {}
        if department:
            query["department"] = department
        if section:
            query["section"] = section

        cursor = col.find(query).sort("student_id", ASCENDING).skip(offset).limit(limit)
        results = []
        for d in cursor:
            d.pop("_id", None)
            results.append(d)
        return results

    def count(self, department: Optional[str] = None, section: Optional[str] = None) -> int:
        col = self.db.students
        if col is None:
            return 0
        query: Dict[str, Any] = {}
        if department:
            query["department"] = department
        if section:
            query["section"] = section
        return col.count_documents(query)

    def upsert(self, student: Union[StudentModel, Dict[str, Any]]) -> Dict[str, Any]:
        col = self.db.students
        if col is None:
            raise MongoUnavailableError("MongoDB is offline.")
        data = student.model_dump() if isinstance(student, StudentModel) else dict(student)
        sid = data["student_id"].strip()
        data["updated_at"] = datetime.datetime.utcnow().isoformat()
        col.update_one({"student_id": sid}, {"$set": data}, upsert=True)
        return self.get_by_id(sid) or data

    def delete(self, student_id: str) -> bool:
        col = self.db.students
        if col is None:
            return False
        res = col.delete_one({"student_id": student_id.strip()})
        return res.deleted_count > 0


class FacultyRepository:
    """Data-access repository for faculty collection."""

    def __init__(self, db: MongoDatabase):
        self.db = db

    def get_by_id(self, faculty_id: str) -> Optional[Dict[str, Any]]:
        col = self.db.faculty
        if col is None:
            return None
        doc = col.find_one({"faculty_id": faculty_id.strip()})
        if doc:
            doc.pop("_id", None)
        return doc

    def list(self, department: Optional[str] = None) -> List[Dict[str, Any]]:
        col = self.db.faculty
        if col is None:
            return []
        query = {"department": department} if department else {}
        cursor = col.find(query).sort("name", ASCENDING)
        results = []
        for d in cursor:
            d.pop("_id", None)
            results.append(d)
        return results

    def upsert(self, faculty: Union[FacultyModel, Dict[str, Any]]) -> Dict[str, Any]:
        col = self.db.faculty
        if col is None:
            raise MongoUnavailableError("MongoDB is offline.")
        data = faculty.model_dump() if isinstance(faculty, FacultyModel) else dict(faculty)
        fid = data["faculty_id"].strip()
        data["updated_at"] = datetime.datetime.utcnow().isoformat()
        col.update_one({"faculty_id": fid}, {"$set": data}, upsert=True)
        return self.get_by_id(fid) or data

    def delete(self, faculty_id: str) -> bool:
        col = self.db.faculty
        if col is None:
            return False
        res = col.delete_one({"faculty_id": faculty_id.strip()})
        return res.deleted_count > 0


class AdvisorRepository:
    """Data-access repository for advisors collection."""

    def __init__(self, db: MongoDatabase):
        self.db = db

    def get_by_id(self, advisor_id: str) -> Optional[Dict[str, Any]]:
        col = self.db.advisors
        if col is None:
            return None
        doc = col.find_one({"advisor_id": advisor_id.strip()})
        if doc:
            doc.pop("_id", None)
        return doc

    def list(self, department: Optional[str] = None) -> List[Dict[str, Any]]:
        col = self.db.advisors
        if col is None:
            return []
        query = {"department": department} if department else {}
        cursor = col.find(query).sort("name", ASCENDING)
        results = []
        for d in cursor:
            d.pop("_id", None)
            results.append(d)
        return results

    def upsert(self, advisor: Union[AdvisorModel, Dict[str, Any]]) -> Dict[str, Any]:
        col = self.db.advisors
        if col is None:
            raise MongoUnavailableError("MongoDB is offline.")
        data = advisor.model_dump() if isinstance(advisor, AdvisorModel) else dict(advisor)
        aid = data["advisor_id"].strip()
        data["updated_at"] = datetime.datetime.utcnow().isoformat()
        col.update_one({"advisor_id": aid}, {"$set": data}, upsert=True)
        return self.get_by_id(aid) or data

    def delete(self, advisor_id: str) -> bool:
        col = self.db.advisors
        if col is None:
            return False
        res = col.delete_one({"advisor_id": advisor_id.strip()})
        return res.deleted_count > 0


class ClassroomRepository:
    """Data-access repository for classrooms collection."""

    def __init__(self, db: MongoDatabase):
        self.db = db

    def get_by_id(self, classroom_id: str) -> Optional[Dict[str, Any]]:
        col = self.db.classrooms
        if col is None:
            return None
        doc = col.find_one({"classroom_id": classroom_id.strip()})
        if doc:
            doc.pop("_id", None)
        return doc

    def get_by_advisor_id(self, advisor_id: str) -> Optional[Dict[str, Any]]:
        col = self.db.classrooms
        if col is None:
            return None
        doc = col.find_one({"assigned_advisor_id": advisor_id.strip()})
        if doc:
            doc.pop("_id", None)
        return doc

    def list(self) -> List[Dict[str, Any]]:
        col = self.db.classrooms
        if col is None:
            return []
        cursor = col.find({}).sort("classroom_id", ASCENDING)
        results = []
        for d in cursor:
            d.pop("_id", None)
            results.append(d)
        return results

    def upsert(self, classroom: Union[ClassroomModel, Dict[str, Any]]) -> Dict[str, Any]:
        col = self.db.classrooms
        if col is None:
            raise MongoUnavailableError("MongoDB is offline.")
        data = classroom.model_dump() if isinstance(classroom, ClassroomModel) else dict(classroom)
        cid = data["classroom_id"].strip()
        data["updated_at"] = datetime.datetime.utcnow().isoformat()
        col.update_one({"classroom_id": cid}, {"$set": data}, upsert=True)
        return self.get_by_id(cid) or data

    def delete(self, classroom_id: str) -> bool:
        col = self.db.classrooms
        if col is None:
            return False
        res = col.delete_one({"classroom_id": classroom_id.strip()})
        return res.deleted_count > 0


class SessionRepository:
    """Data-access repository for sessions collection."""

    def __init__(self, db: MongoDatabase):
        self.db = db

    def get_by_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        col = self.db.sessions
        if col is None:
            return None
        doc = col.find_one({"session_id": session_id.strip()})
        if doc:
            doc.pop("_id", None)
        return doc

    def get_active(self, class_section: Optional[str] = None) -> Optional[Dict[str, Any]]:
        col = self.db.sessions
        if col is None:
            return None
        query: Dict[str, Any] = {"status": {"$in": ["ACTIVE", "PAUSED"]}}
        if class_section:
            query["class_section"] = class_section.strip()
        doc = col.find_one(query)
        if doc:
            doc.pop("_id", None)
        return doc

    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        date: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        col = self.db.sessions
        if col is None:
            return []
        query: Dict[str, Any] = {}
        if date:
            query["date"] = date
        if status:
            query["status"] = status
        cursor = col.find(query).sort("created_at", DESCENDING).skip(offset).limit(limit)
        results = []
        for d in cursor:
            d.pop("_id", None)
            results.append(d)
        return results

    def create(self, session: Union[SessionModel, Dict[str, Any]]) -> Dict[str, Any]:
        col = self.db.sessions
        if col is None:
            raise MongoUnavailableError("MongoDB is offline.")
        data = session.model_dump() if isinstance(session, SessionModel) else dict(session)
        sid = data["session_id"].strip()
        existing = col.find_one({"session_id": sid})
        if existing:
            raise ValueError(f"Session '{sid}' already exists.")
        col.insert_one(data.copy())
        return self.get_by_id(sid) or data

    def update_status(
        self,
        session_id: str,
        status: str,
        actual_start_time: Optional[str] = None,
        actual_end_time: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        col = self.db.sessions
        if col is None:
            return None
        sid = session_id.strip()
        updates: Dict[str, Any] = {
            "status": status,
            "updated_at": datetime.datetime.utcnow().isoformat()
        }
        if actual_start_time:
            updates["actual_start_time"] = actual_start_time
        if actual_end_time:
            updates["actual_end_time"] = actual_end_time

        col.update_one({"session_id": sid}, {"$set": updates})
        return self.get_by_id(sid)


class AttendanceRepository:
    """Data-access repository for attendance_records collection."""

    def __init__(self, db: MongoDatabase):
        self.db = db

    def get_record(self, session_id: str, student_id: str) -> Optional[Dict[str, Any]]:
        col = self.db.attendance_records
        if col is None:
            return None
        doc = col.find_one({"session_id": session_id.strip(), "student_id": student_id.strip()})
        if doc:
            doc.pop("_id", None)
        return doc

    def get_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        col = self.db.attendance_records
        if col is None:
            return []
        cursor = col.find({"session_id": session_id.strip()}).sort("marked_at", ASCENDING)
        results = []
        for d in cursor:
            d.pop("_id", None)
            results.append(d)
        return results

    def get_student_history(self, student_id: str) -> List[Dict[str, Any]]:
        col = self.db.attendance_records
        if col is None:
            return []
        cursor = col.find({"student_id": student_id.strip()}).sort("marked_at", DESCENDING)
        results = []
        for d in cursor:
            d.pop("_id", None)
            results.append(d)
        return results

    def record_or_update(
        self,
        session_id: str,
        student_id: str,
        status: str,
        first_seen: str,
        last_seen: str,
        first_track_id: Optional[int] = None,
        last_track_id: Optional[int] = None,
        initial_similarity: Optional[float] = None,
        latest_similarity: Optional[float] = None,
        student_name: Optional[str] = None,
        department: Optional[str] = None
    ) -> Dict[str, Any]:
        col = self.db.attendance_records
        if col is None:
            raise MongoUnavailableError("MongoDB is offline.")
        sid = session_id.strip()
        stid = student_id.strip()
        now = datetime.datetime.utcnow().isoformat()

        existing = col.find_one({"session_id": sid, "student_id": stid})
        if existing:
            update_fields: Dict[str, Any] = {
                "last_seen": last_seen,
                "updated_at": now
            }
            if last_track_id is not None:
                update_fields["last_track_id"] = last_track_id
            if latest_similarity is not None:
                update_fields["latest_similarity"] = latest_similarity
            if student_name and not existing.get("student_name"):
                update_fields["student_name"] = student_name
            if department and not existing.get("department"):
                update_fields["department"] = department
            # Preserve PRESENT: if already PRESENT, do not demote to LATE
            current_status = existing.get("status")
            if current_status != "PRESENT" and status == "PRESENT":
                update_fields["status"] = "PRESENT"

            col.update_one(
                {"session_id": sid, "student_id": stid},
                {"$set": update_fields, "$inc": {"seen_count": 1}}
            )
            res = self.get_record(sid, stid)
            return res or {}
        else:
            record_data = {
                "session_id": sid,
                "student_id": stid,
                "student_name": student_name,
                "department": department,
                "status": status,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "first_track_id": first_track_id,
                "last_track_id": last_track_id,
                "initial_similarity": initial_similarity,
                "latest_similarity": latest_similarity,
                "seen_count": 1,
                "marked_at": now,
                "updated_at": now
            }
            col.insert_one(record_data.copy())
            res = self.get_record(sid, stid)
            return res or record_data

    def list(
        self,
        session_id: Optional[str] = None,
        student_id: Optional[str] = None,
        status: Optional[str] = None,
        department: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        col = self.db.attendance_records
        if col is None:
            return []
        query: Dict[str, Any] = {}
        if session_id:
            query["session_id"] = session_id.strip()
        if student_id:
            query["student_id"] = student_id.strip()
        if status:
            query["status"] = status.strip().upper()
        if department:
            query["department"] = department.strip()
        cursor = col.find(query).sort("marked_at", DESCENDING).skip(offset).limit(limit)
        results = []
        for d in cursor:
            d.pop("_id", None)
            results.append(d)
        return results

    def get_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        col = self.db.attendance_records
        if col is None:
            return {
                "total_records": 0,
                "present_count": 0,
                "late_count": 0,
                "unique_students": 0,
                "average_similarity": 0.0
            }
        query: Dict[str, Any] = {}
        if session_id:
            query["session_id"] = session_id.strip()

        cursor = list(col.find(query))
        total_records = len(cursor)
        present_count = sum(1 for r in cursor if r.get("status") == "PRESENT")
        late_count = sum(1 for r in cursor if r.get("status") == "LATE")
        unique_students = len(set(r.get("student_id") for r in cursor if r.get("student_id")))
        sims = [r.get("latest_similarity", 0.0) for r in cursor if r.get("latest_similarity") is not None]
        avg_sim = round(float(sum(sims) / len(sims)), 4) if sims else 0.0

        return {
            "session_id": session_id,
            "total_records": total_records,
            "present_count": present_count,
            "late_count": late_count,
            "unique_students": unique_students,
            "average_similarity": avg_sim
        }


class SensorRepository:
    """Data-access repository for sensor_telemetry collection (Phase 15 preparation)."""

    def __init__(self, db: MongoDatabase):
        self.db = db

    def insert(self, telemetry: Union[SensorTelemetryModel, Dict[str, Any]]) -> Dict[str, Any]:
        col = self.db.sensor_telemetry
        if col is None:
            raise MongoUnavailableError("MongoDB is offline.")
        data = telemetry.model_dump() if isinstance(telemetry, SensorTelemetryModel) else dict(telemetry)
        if not data.get("telemetry_id"):
            data["telemetry_id"] = f"TEL-{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
        col.insert_one(data.copy())
        data.pop("_id", None)
        return data

    def get_latest(self, classroom_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        col = self.db.sensor_telemetry
        if col is None:
            return None
        query = {"classroom_id": classroom_id} if classroom_id else {}
        doc = col.find_one(query, sort=[("timestamp", DESCENDING)])
        if doc:
            doc.pop("_id", None)
        return doc

    def query(self, classroom_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        col = self.db.sensor_telemetry
        if col is None:
            return []
        cursor = col.find({"classroom_id": classroom_id}).sort("timestamp", DESCENDING).limit(limit)
        results = []
        for d in cursor:
            d.pop("_id", None)
            results.append(d)
        return results


# Global singleton instance holder
_global_mongo_db: Optional[MongoDatabase] = None


def get_mongo_db(mongo_uri: Optional[str] = None, db_name: Optional[str] = None) -> MongoDatabase:
    """Returns or initializes the global singleton MongoDatabase instance."""
    global _global_mongo_db
    if _global_mongo_db is None:
        _global_mongo_db = MongoDatabase(mongo_uri=mongo_uri, db_name=db_name)
    return _global_mongo_db
