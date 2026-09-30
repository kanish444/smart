from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class StudentModel(BaseModel):
    """MongoDB Student Profile record."""
    student_id: str = Field(..., description="Unique Student Identifier")
    register_no: Optional[str] = Field(None, description="Official College Register Number")
    student_name: str = Field(..., description="Full Name of Student")
    department: str = Field(default="AI&DS", description="Academic Department")
    section: str = Field(default="B", description="Class Section")
    class_name: Optional[str] = Field(None, description="Class Identifier e.g. AI&DS - B")
    status: str = Field(default="active", description="Enrollment status: active or inactive")
    sample_count: int = Field(default=0, description="Number of face samples enrolled")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class FacultyModel(BaseModel):
    """MongoDB Faculty Profile record."""
    faculty_id: str = Field(..., description="Unique Faculty Identifier (maps to user_id)")
    name: str = Field(..., description="Faculty Full Name")
    email: Optional[str] = Field(None, description="Faculty Email")
    phone: Optional[str] = Field(None, description="Faculty Phone")
    department: str = Field(default="AI&DS", description="Department")
    assigned_classroom: Optional[str] = Field(None, description="Primary Assigned Classroom")
    status: str = Field(default="active", description="Account status: active or disabled")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AdvisorModel(BaseModel):
    """MongoDB Class Advisor Profile record."""
    advisor_id: str = Field(..., description="Unique Advisor Identifier (maps to user_id)")
    name: str = Field(..., description="Advisor Full Name")
    email: Optional[str] = Field(None, description="Advisor Email")
    phone: Optional[str] = Field(None, description="Advisor Phone")
    department: str = Field(default="AI&DS", description="Department")
    year: str = Field(default="3rd Year", description="Class Year")
    section: str = Field(default="B", description="Section")
    assigned_classroom: Optional[str] = Field(None, description="Assigned Classroom ID")
    status: str = Field(default="active", description="Account status: active or disabled")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ClassroomModel(BaseModel):
    """MongoDB Classroom Entity and hardware configuration record."""
    classroom_id: str = Field(..., description="Unique Classroom Identifier e.g. AIDS-B")
    classroom_name: str = Field(..., description="Classroom Display Name")
    department: str = Field(default="AI&DS", description="Department")
    year: str = Field(default="3rd Year", description="Year")
    section: str = Field(default="B", description="Section")
    assigned_advisor_id: Optional[str] = Field(None, description="Assigned Class Advisor User ID")
    assigned_advisor_name: Optional[str] = Field(None, description="Assigned Class Advisor Full Name")
    camera_source: str = Field(default="pc", description="Camera source type: pc, droidcam, esp32, rtsp")
    camera_url: Optional[str] = Field(None, description="Camera stream URL if remote")
    camera_status: str = Field(default="connected", description="Camera status: connected, disconnected")
    esp32_device_id: Optional[str] = Field(None, description="ESP32 hardware device identifier")
    esp32_status: str = Field(default="disconnected", description="ESP32 status: connected, disconnected")
    ai_pipeline_status: str = Field(default="running", description="Vision AI pipeline status")
    faiss_status: str = Field(default="ready", description="FAISS runtime index status")
    attendance_status: str = Field(default="active", description="Attendance engine status")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class SessionModel(BaseModel):
    """MongoDB Classroom Session lifecycle record."""
    session_id: str = Field(..., description="Unique Session Identifier")
    date: str = Field(..., description="Session Date (YYYY-MM-DD)")
    class_section: str = Field(..., description="Target Class Section e.g. AIDS-B")
    subject: str = Field(..., description="Course/Subject Name")
    planned_start_time: str = Field(..., description="Planned Start Time (HH:MM or ISO)")
    planned_end_time: str = Field(..., description="Planned End Time (HH:MM or ISO)")
    actual_start_time: Optional[str] = Field(None, description="Actual Start Time ISO")
    actual_end_time: Optional[str] = Field(None, description="Actual End Time ISO")
    status: str = Field(default="SCHEDULED", description="Session state: SCHEDULED, ACTIVE, PAUSED, COMPLETED")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AttendanceRecordModel(BaseModel):
    """MongoDB Student Attendance Record for a specific lecture session."""
    attendance_id: Optional[str] = Field(None, description="Attendance record ID")
    session_id: str = Field(..., description="Associated Session Identifier")
    student_id: str = Field(..., description="Student Identifier")
    student_name: Optional[str] = Field(None, description="Student Name cached for fast reporting")
    department: Optional[str] = Field(None, description="Student academic department")
    status: str = Field(default="PRESENT", description="Status: PRESENT, LATE, ABSENT")
    first_seen: str = Field(..., description="First detection timestamp ISO")
    last_seen: str = Field(..., description="Most recent detection timestamp ISO")
    first_track_id: Optional[int] = Field(None, description="ByteTrack ID at first identification")
    last_track_id: Optional[int] = Field(None, description="ByteTrack ID at most recent observation")
    initial_similarity: Optional[float] = Field(None, description="Cosine similarity at initial match")
    latest_similarity: Optional[float] = Field(None, description="Cosine similarity at latest observation")
    seen_count: int = Field(default=1, description="Number of valid observations in session")
    marked_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class SensorTelemetryModel(BaseModel):
    """MongoDB Sensor Telemetry schema (prepared for Phase 15 ESP32 ingestion)."""
    telemetry_id: Optional[str] = Field(None, description="Unique Telemetry Event ID")
    device_id: str = Field(..., description="ESP32 hardware device identifier")
    classroom_id: str = Field(..., description="Classroom where device is deployed")
    temperature_c: Optional[float] = Field(None, description="Ambient temperature in Celsius")
    humidity_pct: Optional[float] = Field(None, description="Relative humidity percentage")
    air_quality: Optional[str] = Field(None, description="Air quality indicator (GOOD, MODERATE, POOR)")
    motion_detected: Optional[bool] = Field(None, description="Passive infrared motion detection")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
