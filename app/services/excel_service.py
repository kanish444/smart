import io
import re
import csv
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger
import pandas as pd


class ExcelService:
    """
    Institutional Student Excel/CSV Parsing and Generation Service:
    - Parses Excel (.xlsx, .xls) and CSV rosters into structured student records.
    - Intelligently maps flexible column headers without requiring rigid column order.
    - Generates downloadable pre-formatted templates for institutional staff.
    """

    COLUMN_ALIASES = {
        "register_no": [
            "register number", "register no", "register_no", "reg no", "reg_no", "regno",
            "roll no", "roll_no", "rollno", "student_id", "student id", "id", "reg num",
            "registration number", "reg_number", "roll number"
        ],
        "student_name": [
            "student name", "student_name", "studentname", "name", "full name", "full_name",
            "candidate name", "candidate_name", "student"
        ],
        "department": [
            "department", "dept", "dept_name", "branch", "course", "degree", "stream"
        ],
        "section": [
            "section", "sec", "sec.", "section_name", "class_section", "sec_name"
        ],
        "year": [
            "year", "class_year", "academic_year", "study_year", "batch"
        ],
        "class_name": [
            "class", "class_name", "class name", "classroom", "cohort"
        ]
    }

    @classmethod
    def _normalize_str(cls, val: Any) -> str:
        if val is None or pd.isna(val):
            return ""
        s = str(val).strip()
        # Clean potential floats like 7376222.0 from Excel numeric formats
        if s.endswith(".0") and s[:-2].isdigit():
            s = s[:-2]
        return s

    @classmethod
    def _match_header(cls, col_name: str) -> Optional[str]:
        norm = re.sub(r"[^\w\s]", " ", col_name.lower())
        norm = " ".join(norm.split())
        for canonical, aliases in cls.COLUMN_ALIASES.items():
            if norm == canonical or norm in aliases:
                return canonical
            for alias in aliases:
                if alias == norm:
                    return canonical
        return None

    @classmethod
    def parse_students_file(
        cls,
        file_bytes: bytes,
        filename: str,
        default_department: str = "AI&DS",
        default_section: str = "B",
        default_year: str = "3rd Year"
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Parses uploaded file bytes (.xlsx, .xls, .csv).
        Returns: (parsed_students, warnings_and_errors)
        """
        fname_lower = filename.lower()
        records: List[Dict[str, Any]] = []
        errors: List[str] = []

        try:
            if fname_lower.endswith(".csv"):
                # Try UTF-8-sig first (handles Excel BOM), then Latin-1
                try:
                    text_content = file_bytes.decode("utf-8-sig")
                except UnicodeDecodeError:
                    text_content = file_bytes.decode("latin-1", errors="replace")

                stream = io.StringIO(text_content)
                reader = csv.reader(stream)
                rows = list(reader)
                if not rows:
                    return [], ["The uploaded CSV file is empty."]

                raw_headers = [cls._normalize_str(c) for c in rows[0]]
                data_rows = rows[1:]
                df = pd.DataFrame(data_rows, columns=raw_headers)
            else:
                # Excel file (.xlsx, .xls)
                df = pd.read_excel(io.BytesIO(file_bytes), dtype=str)
                df = df.fillna("")

        except Exception as e:
            logger.error(f"Failed to read student file '{filename}': {e}")
            return [], [f"Failed to read file format: {str(e)}"]

        if df.empty:
            return [], ["The uploaded sheet contains no data rows."]

        # Map column headers to canonical names
        col_mapping: Dict[str, str] = {}
        for col in df.columns:
            canonical = cls._match_header(str(col))
            if canonical and canonical not in col_mapping.values():
                col_mapping[col] = canonical

        # Check required columns
        mapped_canonicals = set(col_mapping.values())
        if "register_no" not in mapped_canonicals and "student_name" not in mapped_canonicals:
            return [], [
                f"Could not identify required columns (Register Number, Student Name). "
                f"Detected columns: {list(df.columns)}. Please check header labels."
            ]

        # Process each row
        seen_regs = set()
        for idx, (_, row) in enumerate(df.iterrows(), start=2):
            raw_reg = ""
            raw_name = ""
            raw_dept = ""
            raw_sec = ""
            raw_year = ""
            raw_cls = ""

            for col, canonical in col_mapping.items():
                val = cls._normalize_str(row.get(col))
                if canonical == "register_no":
                    raw_reg = val
                elif canonical == "student_name":
                    raw_name = val
                elif canonical == "department":
                    raw_dept = val
                elif canonical == "section":
                    raw_sec = val
                elif canonical == "year":
                    raw_year = val
                elif canonical == "class_name":
                    raw_cls = val

            # Skip completely empty rows
            if not raw_reg and not raw_name:
                continue

            if not raw_reg:
                errors.append(f"Row {idx}: Missing Register Number for '{raw_name or 'Unknown'}'. Skipped.")
                continue

            if not raw_name:
                errors.append(f"Row {idx}: Missing Student Name for Register No '{raw_reg}'. Skipped.")
                continue

            if raw_reg in seen_regs:
                errors.append(f"Row {idx}: Duplicate Register Number '{raw_reg}' in file. Using latest row.")
            seen_regs.add(raw_reg)

            # Apply defaults
            dept = raw_dept or default_department or "AI&DS"
            sec = (raw_sec or default_section or "B").strip().upper()
            year = raw_year or default_year or "3rd Year"
            cls_name = raw_cls or f"{year} {dept} - {sec}".strip()

            records.append({
                "student_id": raw_reg,
                "register_no": raw_reg,
                "student_name": raw_name,
                "department": dept,
                "section": sec,
                "year": year,
                "class_name": cls_name,
                "status": "active"
            })

        logger.info(f"ExcelService: Parsed {len(records)} students from '{filename}' with {len(errors)} notices.")
        return records, errors

    @classmethod
    def generate_excel_template(cls) -> bytes:
        """
        Generates a professionally formatted Excel (.xlsx) roster template with sample rows.
        """
        sample_data = [
            {
                "Register Number": "7376222AD101",
                "Student Name": "Aarav Sharma",
                "Department": "AI&DS",
                "Year": "3rd Year",
                "Section": "B",
                "Class": "3rd Year AI&DS - B"
            },
            {
                "Register Number": "7376222AD102",
                "Student Name": "Ananya Patel",
                "Department": "AI&DS",
                "Year": "3rd Year",
                "Section": "B",
                "Class": "3rd Year AI&DS - B"
            },
            {
                "Register Number": "7376222AD103",
                "Student Name": "Bhavya Sri",
                "Department": "AI&DS",
                "Year": "3rd Year",
                "Section": "B",
                "Class": "3rd Year AI&DS - B"
            },
            {
                "Register Number": "7376222AD104",
                "Student Name": "Dinesh Kumar",
                "Department": "AI&DS",
                "Year": "3rd Year",
                "Section": "B",
                "Class": "3rd Year AI&DS - B"
            },
            {
                "Register Number": "7376222AD105",
                "Student Name": "Harish V",
                "Department": "AI&DS",
                "Year": "3rd Year",
                "Section": "B",
                "Class": "3rd Year AI&DS - B"
            }
        ]

        df = pd.DataFrame(sample_data)
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Students_Roster")
            worksheet = writer.sheets["Students_Roster"]
            # Auto-fit column widths
            for col in worksheet.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                col_letter = col[0].column_letter
                worksheet.column_dimensions[col_letter].width = max(max_len + 4, 15)

        return out.getvalue()

    @classmethod
    def generate_csv_template(cls) -> bytes:
        """
        Generates a standard CSV roster template.
        """
        sample_rows = [
            ["Register Number", "Student Name", "Department", "Year", "Section", "Class"],
            ["7376222AD101", "Aarav Sharma", "AI&DS", "3rd Year", "B", "3rd Year AI&DS - B"],
            ["7376222AD102", "Ananya Patel", "AI&DS", "3rd Year", "B", "3rd Year AI&DS - B"],
            ["7376222AD103", "Bhavya Sri", "AI&DS", "3rd Year", "B", "3rd Year AI&DS - B"]
        ]
        out = io.StringIO()
        writer = csv.writer(out)
        for r in sample_rows:
            writer.writerow(r)
        return out.getvalue().encode("utf-8-sig")
