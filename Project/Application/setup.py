from cx_Freeze import setup, Executable
import sys
import os

# dependencies
build_exe_options = {
    "packages": [
        "os", "tkinter", "cv2", "numpy", "PIL", "tensorflow", "gdown", "dotenv", "tempfile"
    ],
    "includes": [
        "idna.idnadata"  # Sometimes required by urllib/gdown
    ],
    "include_files": [
        ("pap_smear_model.keras", "pap_smear_model.keras"),  # if local model used
        (".env", ".env")  # if you rely on env vars
    ],
    "excludes": ["unittest", "email", "http", "html", "xml", "pydoc"],
    "include_msvcr": True,  # Include Microsoft C runtime libs
}

base = None
if sys.platform == "win32":
    base = "Win32GUI"  # Prevents a terminal window from opening with the app


setup(
    name="CytoplasmDetection",
    version="2.1",
    description="Cell Cytoplasm Detection App",
    author="Damunza3SmartMadMan",
    author_email="damunzandm@outlook.com",  # optional
    company_name="DanseApps",
    copyright="© 2025 Damunza3SmartMadMan",
    executables=[Executable("cytoplasm.py")]
)