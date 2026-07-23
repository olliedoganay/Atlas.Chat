from __future__ import annotations

import atexit
import ast
import json
import os
import queue
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .run_contract import DEFAULT_SUBSCRIBER_QUEUE_SIZE, put_bounded_queue


@dataclass
class LanguageSpec:
    image: str
    filename: str
    command: list[str]
    needs_compile: bool = False
    requires_network: bool = False


@dataclass
class RunPlan:
    image: str
    filename: str
    command: list[str]
    ports: dict[int, int] = field(default_factory=dict)  # host:container
    gui: bool = False
    web_container_port: int | None = None
    requires_network: bool = False
    uses_apt: bool = False


PYTHON_GUI_IMAGE = "docker.io/library/python:3.12-slim"
LEGACY_PYTHON_GUI_IMAGES = (
    "atlas-python-gui:workspace2",
    "atlas-python-gui:workspace1",
    "atlas-python-gui:latest",
)
PYTHON_GUI_BASE_APT_PACKAGES = (
    "xvfb",
    "x11vnc",
    "fluxbox",
    "novnc",
    "websockify",
    "fonts-dejavu",
    "fontconfig",
    "ca-certificates",
)
PYTHON_TERMINAL_APT_PACKAGES = (
    "xterm",
    "ncurses-term",
    "libncursesw6",
)
PYTHON_SYSTEM_PACKAGE_RULES: dict[str, tuple[str, ...]] = {
    "tkinter": ("tcl8.6", "tk8.6", "tk"),
    "turtle": ("tcl8.6", "tk8.6", "tk"),
    "customtkinter": ("tcl8.6", "tk8.6", "tk"),
    "ttkbootstrap": ("tcl8.6", "tk8.6", "tk"),
    "pygame": (
        "libsdl2-2.0-0",
        "libsdl2-image-2.0-0",
        "libsdl2-mixer-2.0-0",
        "libsdl2-ttf-2.0-0",
        "libfreetype6",
        "libportmidi0",
        "libasound2",
    ),
    "cv2": ("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1"),
    "matplotlib": ("fontconfig", "libfreetype6", "tcl8.6", "tk8.6", "tk"),
    "PIL": ("libjpeg62-turbo", "zlib1g"),
    "PyQt5": (
        "libgl1",
        "libglib2.0-0",
        "libdbus-1-3",
        "libx11-xcb1",
        "libxkbcommon-x11-0",
        "libxcb-cursor0",
        "libxcb-icccm4",
        "libxcb-image0",
        "libxcb-keysyms1",
        "libxcb-randr0",
        "libxcb-render-util0",
        "libxcb-shape0",
        "libxcb-xinerama0",
        "libsm6",
        "libxext6",
        "libxrender1",
    ),
    "PyQt6": (
        "libgl1",
        "libglib2.0-0",
        "libdbus-1-3",
        "libx11-xcb1",
        "libxkbcommon-x11-0",
        "libxcb-cursor0",
        "libxcb-icccm4",
        "libxcb-image0",
        "libxcb-keysyms1",
        "libxcb-randr0",
        "libxcb-render-util0",
        "libxcb-shape0",
        "libxcb-xinerama0",
        "libsm6",
        "libxext6",
        "libxrender1",
    ),
    "PySide2": (
        "libgl1",
        "libglib2.0-0",
        "libdbus-1-3",
        "libx11-xcb1",
        "libxkbcommon-x11-0",
        "libxcb-cursor0",
        "libxcb-icccm4",
        "libxcb-image0",
        "libxcb-keysyms1",
        "libxcb-randr0",
        "libxcb-render-util0",
        "libxcb-shape0",
        "libxcb-xinerama0",
        "libsm6",
        "libxext6",
        "libxrender1",
    ),
    "PySide6": (
        "libgl1",
        "libglib2.0-0",
        "libdbus-1-3",
        "libx11-xcb1",
        "libxkbcommon-x11-0",
        "libxcb-cursor0",
        "libxcb-icccm4",
        "libxcb-image0",
        "libxcb-keysyms1",
        "libxcb-randr0",
        "libxcb-render-util0",
        "libxcb-shape0",
        "libxcb-xinerama0",
        "libsm6",
        "libxext6",
        "libxrender1",
    ),
    "PySimpleGUI": ("tcl8.6", "tk8.6", "tk"),
    "wx": ("libgtk-3-0", "libgl1", "libglib2.0-0"),
    "kivy": (
        "libgl1",
        "libglib2.0-0",
        "libsdl2-2.0-0",
        "libsdl2-image-2.0-0",
        "libsdl2-mixer-2.0-0",
        "libsdl2-ttf-2.0-0",
    ),
    "dearpygui": ("libgl1", "libglib2.0-0", "libx11-6"),
    "pyglet": ("libgl1", "libglib2.0-0", "libx11-6"),
    "arcade": ("libgl1", "libglib2.0-0", "libx11-6"),
    "ursina": ("libgl1", "libglib2.0-0", "libx11-6"),
    "sounddevice": ("libportaudio2",),
    "pyaudio": ("portaudio19-dev", "build-essential"),
    "soundfile": ("libsndfile1",),
    "librosa": ("libsndfile1", "ffmpeg"),
    "pydub": ("ffmpeg",),
    "moviepy": ("ffmpeg",),
    "imageio_ffmpeg": ("ffmpeg",),
    "weasyprint": (
        "libcairo2",
        "libffi-dev",
        "libpango-1.0-0",
        "libpangoft2-1.0-0",
        "shared-mime-info",
    ),
    "cairosvg": ("libcairo2", "libffi-dev"),
    "lxml": ("libxml2", "libxslt1.1"),
    "psycopg2": ("libpq5",),
    "MySQLdb": ("default-libmysqlclient-dev", "build-essential"),
    "mysql": ("libmariadb3",),
    "pyodbc": ("unixodbc", "unixodbc-dev"),
    "ldap": ("libldap2-dev", "libsasl2-dev"),
    "magic": ("libmagic1",),
    "fitz": ("libmupdf-dev",),
}
PYTHON_PIP_ALIASES = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "bs4": "beautifulsoup4",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "dotenv": "python-dotenv",
    "serial": "pyserial",
    "Crypto": "pycryptodome",
    "dateutil": "python-dateutil",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "fitz": "PyMuPDF",
    "OpenGL": "PyOpenGL",
    "wx": "wxPython",
    "psycopg2": "psycopg2-binary",
    "MySQLdb": "mysqlclient",
    "mysql": "mysql-connector-python",
    "flask_cors": "flask-cors",
    "flask_socketio": "flask-socketio",
    "socketio": "python-socketio",
    "jwt": "PyJWT",
    "jose": "python-jose",
    "multipart": "python-multipart",
    "telegram": "python-telegram-bot",
    "discord": "discord.py",
    "googleapiclient": "google-api-python-client",
    "google_auth_oauthlib": "google-auth-oauthlib",
    "google.generativeai": "google-generativeai",
    "google.cloud.storage": "google-cloud-storage",
    "google.cloud.bigquery": "google-cloud-bigquery",
    "google.cloud.translate": "google-cloud-translate",
    "google.cloud.speech": "google-cloud-speech",
    "google.cloud.texttospeech": "google-cloud-texttospeech",
    "google.cloud.vision": "google-cloud-vision",
    "google.cloud.aiplatform": "google-cloud-aiplatform",
    "azure.storage.blob": "azure-storage-blob",
    "azure.identity": "azure-identity",
    "azure.ai.openai": "azure-ai-openai",
    "mpl_toolkits": "matplotlib",
    "sentence_transformers": "sentence-transformers",
    "qdrant_client": "qdrant-client",
    "chromadb": "chromadb",
    "langchain_core": "langchain-core",
    "langchain_openai": "langchain-openai",
    "langchain_community": "langchain-community",
    "langchain_text_splitters": "langchain-text-splitters",
    "llama_index": "llama-index",
    "tiktoken": "tiktoken",
    "soundfile": "soundfile",
    "speech_recognition": "SpeechRecognition",
    "Levenshtein": "python-Levenshtein",
    "magic": "python-magic",
    "slugify": "python-slugify",
    "tkinterdnd2": "tkinterdnd2",
}
PYTHON_NAMESPACE_IMPORTS = {"azure", "google", "zope"}
PYTHON_SAFE_PACKAGE_SPEC_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*"
    r"(?:\[[A-Za-z0-9_,.-]+\])?"
    r"(?:(?:===|==|!=|~=|>=|<=|>|<)[A-Za-z0-9][A-Za-z0-9_.!+*:-]*"
    r"(?:,(?:===|==|!=|~=|>=|<=|>|<)[A-Za-z0-9][A-Za-z0-9_.!+*:-]*)*)?$"
)
PYTHON_PIP_INSTALL_HINT_RE = re.compile(
    r"(?i)(?:^|[^\w.-])(?:python(?:3)?\s+-m\s+pip|pip(?:3)?|%\s*pip)\s+install\s+(.+)$"
)
PYTHON_REQUIREMENTS_HINT_RE = re.compile(
    r"(?i)^\s*(?:[#/]+\s*)?(?:requirements?|dependencies?|packages?|requires?)\b\s*:?\s*(.+)$"
)
PYTHON_PIP_OPTION_VALUE_FLAGS = {
    "-C",
    "-c",
    "-e",
    "-f",
    "-i",
    "-r",
    "-t",
    "--abi",
    "--cache-dir",
    "--config-settings",
    "--constraint",
    "--editable",
    "--extra-index-url",
    "--find-links",
    "--implementation",
    "--index-url",
    "--platform",
    "--prefix",
    "--python-version",
    "--requirement",
    "--root",
    "--src",
    "--target",
    "--trusted-host",
    "--upgrade-strategy",
}
PYTHON_PIP_HINT_STOP_WORDS = {
    "and",
    "before",
    "first",
    "for",
    "install",
    "library",
    "none",
    "package",
    "packages",
    "pip",
    "python",
    "python3",
    "required",
    "requires",
    "run",
    "standard",
    "stdlib",
    "then",
    "to",
    "use",
    "using",
}
PYTHON_GUI_IMPORTS = {
    "tkinter",
    "turtle",
    "customtkinter",
    "ttkbootstrap",
    "pygame",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "PySimpleGUI",
    "wx",
    "kivy",
    "dearpygui",
    "pyglet",
    "arcade",
    "ursina",
}
PYTHON_TERMINAL_IMPORTS = {
    "asciimatics",
    "blessed",
    "click",
    "cmd",
    "code",
    "curses",
    "getpass",
    "inquirer",
    "npyscreen",
    "prompt_toolkit",
    "questionary",
    "readchar",
    "readline",
    "rich",
    "termios",
    "textual",
    "typer",
    "tty",
    "urwid",
}
PYTHON_GUI_MARKERS = (
    r"^\s*import\s+(tkinter|customtkinter|ttkbootstrap)\b",
    r"^\s*from\s+(tkinter|customtkinter|ttkbootstrap)\b",
    r"\bpygame\.display\.set_mode\s*\(",
    r"\bTk\s*\(",
    r"\btk\.Tk\s*\(",
    r"\btkinter\.Tk\s*\(",
    r"\bttkbootstrap\.Window\s*\(",
    r"\bturtle\.(Screen|done|mainloop)\s*\(",
    r"\bQApplication\s*\(",
    r"\bwx\.App\s*\(",
    r"\bApp\s*\(\)\.run\s*\(",
    r"\bplt\.show\s*\(",
    r"\bmatplotlib\.pyplot\.show\s*\(",
    r"\bImageTk\b",
    r"\bTkinterDnD\b",
    r"\bdearpygui\.",
    r"\bpyglet\.app\.run\s*\(",
    r"\barcade\.run\s*\(",
)
PYTHON_TERMINAL_MARKERS = (
    r"\binput\s*\(",
    r"\bgetpass\.getpass\s*\(",
    r"\bcurses\.(wrapper|initscr)\s*\(",
    r"\binitscr\s*\(",
    r"\.cmdloop\s*\(",
    r"\bPrompt\.ask\s*\(",
    r"\bConfirm\.ask\s*\(",
    r"\bclick\.prompt\s*\(",
    r"\btyper\.prompt\s*\(",
    r"\bquestionary\.",
    r"\binquirer\.",
    r"\bLive\s*\(",
    r"\bProgress\s*\(",
    r"\bConsole\s*\(",
)
PYTHON_DYNAMIC_IMPORT_MARKERS = (
    r"\bimportlib\.import_module\s*\(",
    r"\b__import__\s*\(",
)
NOVNC_CONTAINER_PORT = 6080
PYTHON_WEB_DEFAULT_PORTS = {
    "flask": 5000,
    "fastapi": 8000,
    "uvicorn": 8000,
    "django": 8000,
    "streamlit": 8501,
    "gradio": 7860,
    "dash": 8050,
    "nicegui": 8080,
    "bottle": 8080,
    "panel": 5006,
    "bokeh": 5006,
    "socketserver": 8000,
}
PYTHON_WEB_MARKERS = (
    r"\bFlask\s*\(",
    r"\bFastAPI\s*\(",
    r"\buvicorn\.run\s*\(",
    r"\bapp\.run\s*\(",
    r"\brun_server\s*\(",
    r"\bHTTPServer\s*\(",
    r"\bsocketserver\.TCPServer\s*\(",
    r"\bhttp\.server\.",
)


def _python_gui_detected(code: str, imports: set[str] | None = None) -> bool:
    if imports and imports.intersection(PYTHON_GUI_IMPORTS):
        return True
    if _python_gui_args(code):
        return True
    for marker in PYTHON_GUI_MARKERS:
        if re.search(marker, code, re.MULTILINE):
            return True
    return False


def _python_terminal_detected(code: str, imports: set[str]) -> bool:
    if imports.intersection(PYTHON_TERMINAL_IMPORTS):
        return True
    for marker in PYTHON_TERMINAL_MARKERS:
        if re.search(marker, code, re.MULTILINE):
            return True
    return False


def _python_gui_args(code: str) -> list[str]:
    if re.search(r"add_argument\s*\([^)]*['\"]--gui['\"]", code, re.DOTALL):
        return ["--gui"]
    return []


def _python_web_port(code: str, imports: set[str]) -> int | None:
    has_web_import = bool(imports.intersection(PYTHON_WEB_DEFAULT_PORTS))
    has_web_marker = any(re.search(marker, code, re.MULTILINE) for marker in PYTHON_WEB_MARKERS)
    if not has_web_import and not has_web_marker:
        return None

    explicit_port = re.search(r"\bport\s*=\s*(\d{2,5})", code)
    if explicit_port:
        port = int(explicit_port.group(1))
        if 1 <= port <= 65535:
            return port

    if re.search(r"\bHTTPServer\s*\(|\bsocketserver\.TCPServer\s*\(|\bhttp\.server\.", code):
        return 8000

    for name, port in PYTHON_WEB_DEFAULT_PORTS.items():
        if name in imports:
            return port
    return 5000


def _python_web_command(code: str, imports: set[str], web_port: int | None) -> tuple[str | None, set[str]]:
    if not web_port:
        return None, set()
    if "streamlit" in imports:
        return (
            f"streamlit run /tmp/main.py --server.address 0.0.0.0 --server.port {web_port}",
            set(),
        )
    if "fastapi" in imports:
        app_name = _python_web_app_name(code, "FastAPI") or "app"
        return (
            f"uvicorn main:{app_name} --host 0.0.0.0 --port {web_port}",
            {"uvicorn"},
        )
    if "flask" in imports:
        app_name = _python_web_app_name(code, "Flask") or "app"
        return (
            f"flask --app main:{app_name} run --host 0.0.0.0 --port {web_port}",
            set(),
        )
    return None, set()


def _python_web_app_name(code: str, factory_name: str) -> str | None:
    match = re.search(
        rf"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*{re.escape(factory_name)}\s*\(",
        code,
        re.MULTILINE,
    )
    return match.group(1) if match else None


def _python_imports(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _python_imports_by_regex(code)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                root = name.name.split(".", 1)[0]
                if root:
                    imports.add(root)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            root = node.module.split(".", 1)[0]
            if root:
                imports.add(root)
    return imports


def _python_import_modules(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _python_import_modules_by_regex(code)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name:
                    modules.add(name.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
            for name in node.names:
                if name.name != "*":
                    modules.add(f"{node.module}.{name.name}")
        elif isinstance(node, ast.Call):
            module = _python_dynamic_import_module_from_call(node)
            if module:
                modules.add(module)
    return modules


def _python_dynamic_import_module_from_call(node: ast.Call) -> str | None:
    if not node.args:
        return None
    func = node.func
    is_importlib_call = (
        isinstance(func, ast.Attribute)
        and func.attr == "import_module"
        and isinstance(func.value, ast.Name)
        and func.value.id == "importlib"
    )
    is_dunder_import = isinstance(func, ast.Name) and func.id == "__import__"
    if not is_importlib_call and not is_dunder_import:
        return None
    first_arg = node.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        module = first_arg.value.strip()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$", module):
            return module
    return None


def _python_imports_by_regex(code: str) -> set[str]:
    imports: set[str] = set()
    for match in re.finditer(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", code, re.MULTILINE):
        imports.add(match.group(1))
    return imports


def _python_import_modules_by_regex(code: str) -> set[str]:
    modules: set[str] = set()
    for match in re.finditer(
        r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)",
        code,
        re.MULTILINE,
    ):
        modules.add(match.group(1))
    for match in re.finditer(
        r"\b(?:importlib\.import_module|__import__)\s*\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)['\"]",
        code,
    ):
        modules.add(match.group(1))
    return modules


def _python_alias_for_import(name: str) -> str | None:
    parts = name.split(".")
    for index in range(len(parts), 0, -1):
        alias = PYTHON_PIP_ALIASES.get(".".join(parts[:index]))
        if alias:
            return alias
    return None


def _python_pip_packages(imports: set[str], modules: set[str] | None = None) -> list[str]:
    stdlib = getattr(sys, "stdlib_module_names", set())
    modules = modules or set()
    packages: set[str] = set()
    aliased_roots: set[str] = set()
    for module in modules:
        alias = _python_alias_for_import(module)
        if alias:
            packages.add(alias)
            aliased_roots.add(module.split(".", 1)[0])

    for name in imports:
        if not name or name in stdlib:
            continue
        alias = PYTHON_PIP_ALIASES.get(name)
        if alias:
            packages.add(alias)
        elif name in PYTHON_NAMESPACE_IMPORTS and name in aliased_roots:
            continue
        elif name not in PYTHON_NAMESPACE_IMPORTS:
            packages.add(name)
    return sorted(packages)


def _python_declared_pip_packages(code: str) -> list[str]:
    packages: list[str] = []
    for raw_line in code.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = re.sub(r"^\s*(?:#|//)+\s*", "", line)
        for match in PYTHON_PIP_INSTALL_HINT_RE.finditer(normalized):
            packages.extend(_python_package_tokens_from_hint(match.group(1)))

        requirements_match = PYTHON_REQUIREMENTS_HINT_RE.match(normalized)
        if requirements_match:
            hint = requirements_match.group(1)
            if _python_hint_declares_no_packages(hint):
                continue
            pip_match = PYTHON_PIP_INSTALL_HINT_RE.search(hint)
            if pip_match:
                hint = pip_match.group(1)
            packages.extend(_python_package_tokens_from_hint(hint.replace(",", " ")))
    return sorted(dict.fromkeys(packages))


def _python_hint_declares_no_packages(hint: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", hint.lower()).strip()
    return (
        normalized in {"none", "no", "na", "n a", "stdlib", "standard library"}
        or normalized.startswith("none ")
        or normalized.startswith("no external")
        or normalized.startswith("standard library")
        or normalized.startswith("python standard library")
    )


def _python_package_tokens_from_hint(hint: str) -> list[str]:
    cleaned = re.split(r"\s*(?:&&|\|\||;|\|)\s*", hint, maxsplit=1)[0]
    cleaned = re.split(r"\s+(?:and\s+then|then|before|after|to\s+run)\b", cleaned, maxsplit=1)[0]
    try:
        tokens = shlex.split(cleaned, comments=True)
    except ValueError:
        tokens = cleaned.split()

    packages: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        token = token.strip().strip("`'\"()[]{}.,")
        if not token:
            continue
        lowered = token.lower()
        if lowered in PYTHON_PIP_HINT_STOP_WORDS:
            continue
        if lowered in {"pip", "pip3", "install", "-m"}:
            continue
        if token.startswith("-"):
            option = token.split("=", 1)[0]
            if option in PYTHON_PIP_OPTION_VALUE_FLAGS and "=" not in token:
                skip_next = True
            continue
        if token.endswith((".txt", ".in")) or "/" in token or "\\" in token:
            continue
        if token.startswith((".", "http:", "https:", "git+", "hg+", "svn+", "bzr+")):
            continue
        if PYTHON_SAFE_PACKAGE_SPEC_RE.match(token):
            packages.append(token)
    return packages


def _python_dependency_repair_may_need_network(code: str) -> bool:
    return any(re.search(marker, code, re.MULTILINE) for marker in PYTHON_DYNAMIC_IMPORT_MARKERS)


def _python_repair_package_map(imports: set[str], modules: set[str]) -> dict[str, list[str]]:
    stdlib = getattr(sys, "stdlib_module_names", set())
    package_map: dict[str, list[str]] = {}

    def add(key: str, package: str) -> None:
        if not key or not package:
            return
        package_map.setdefault(key, [])
        if package not in package_map[key]:
            package_map[key].append(package)

    for name, package in PYTHON_PIP_ALIASES.items():
        add(name, package)
        root = name.split(".", 1)[0]
        if "." not in name and root not in stdlib:
            add(root, package)

    for module in modules:
        alias = _python_alias_for_import(module)
        if not alias:
            continue
        parts = module.split(".")
        for index in range(len(parts), 0, -1):
            add(".".join(parts[:index]), alias)
        add(parts[0], alias)

    for name in imports:
        if not name or name in stdlib:
            continue
        package = PYTHON_PIP_ALIASES.get(name, name)
        add(name, package)

    return {key: sorted(packages) for key, packages in sorted(package_map.items())}


def _python_dependency_repair_script(imports: set[str], modules: set[str]) -> str:
    repair_map = json.dumps(_python_repair_package_map(imports, modules), sort_keys=True)
    safe_spec = PYTHON_SAFE_PACKAGE_SPEC_RE.pattern
    return "\n".join(
        [
            "cat > /tmp/atlas_python_repair.py <<'PY'",
            "import re",
            "import subprocess",
            "import sys",
            "",
            f"REPAIR_MAP = {repair_map}",
            f"SAFE_PACKAGE_SPEC_RE = re.compile({safe_spec!r})",
            "STDLIB = set(getattr(sys, 'stdlib_module_names', set()))",
            "NAMESPACE_IMPORTS = {'azure', 'google', 'zope'}",
            "MISSING_PATTERNS = (",
            "    re.compile(r\"(?:ModuleNotFoundError|ImportError): No module named ['\\\"]([^'\\\"]+)['\\\"]\"),",
            "    re.compile(r\"No module named ['\\\"]([^'\\\"]+)['\\\"]\"),",
            "    re.compile(r\"DistributionNotFound: The ['\\\"]([^'\\\"]+)['\\\"] distribution was not found\"),",
            ")",
            "",
            "def run_command(command):",
            "    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')",
            "    output = []",
            "    assert process.stdout is not None",
            "    for chunk in process.stdout:",
            "        print(chunk, end='', flush=True)",
            "        output.append(chunk)",
            "        if sum(len(item) for item in output) > 60000:",
            "            output = [''.join(output)[-60000:]]",
            "    return process.wait(), ''.join(output)",
            "",
            "def missing_modules(output):",
            "    modules = []",
            "    for pattern in MISSING_PATTERNS:",
            "        for match in pattern.finditer(output):",
            "            name = match.group(1).strip()",
            "            if name and name not in modules:",
            "                modules.append(name)",
            "    return modules",
            "",
            "def package_candidates(module):",
            "    packages = []",
            "    parts = module.split('.')",
            "    for index in range(len(parts), 0, -1):",
            "        for package in REPAIR_MAP.get('.'.join(parts[:index]), []):",
            "            if package not in packages:",
            "                packages.append(package)",
            "    root = parts[0]",
            "    if not packages and root not in STDLIB and root not in NAMESPACE_IMPORTS:",
            "        packages.append(root)",
            "    return [package for package in packages if SAFE_PACKAGE_SPEC_RE.match(package)]",
            "",
            "def repair_packages(modules):",
            "    packages = []",
            "    for module in modules:",
            "        for package in package_candidates(module):",
            "            if package not in packages:",
            "                packages.append(package)",
            "    return packages",
            "",
            "def install_packages(packages):",
            "    print('[atlas-runner] detected missing Python module; installing repair packages: ' + ' '.join(packages), flush=True)",
            "    command = [sys.executable, '-m', 'pip', 'install', '--quiet', '--no-input', '--disable-pip-version-check', '--root-user-action=ignore', *packages]",
            "    return subprocess.call(command)",
            "",
            "def main():",
            "    command = sys.argv[1:]",
            "    if not command:",
            "        print('[atlas-runner] no Python command supplied for dependency repair', file=sys.stderr)",
            "        return 2",
            "    code, output = run_command(command)",
            "    if code == 0:",
            "        return 0",
            "    modules = missing_modules(output)",
            "    packages = repair_packages(modules)",
            "    if not packages:",
            "        return code",
            "    if install_packages(packages) != 0:",
            "        print('[atlas-runner] dependency repair install failed', flush=True)",
            "        return code",
            "    print('[atlas-runner] retrying after dependency repair', flush=True)",
            "    retry_code, _ = run_command(command)",
            "    return retry_code",
            "",
            "if __name__ == '__main__':",
            "    raise SystemExit(main())",
            "PY",
        ]
    )


def _python_apt_packages(imports: set[str], modules: set[str], gui: bool, terminal: bool) -> list[str]:
    packages: list[str] = []
    if gui:
        packages.extend(PYTHON_GUI_BASE_APT_PACKAGES)
    if terminal:
        packages.extend(PYTHON_TERMINAL_APT_PACKAGES)
    package_names = set(imports)
    for module in modules:
        package_names.add(module)
        package_names.add(module.split(".", 1)[0])
    for name in sorted(package_names):
        packages.extend(PYTHON_SYSTEM_PACKAGE_RULES.get(name, ()))
    return sorted(dict.fromkeys(packages))


def _reserve_host_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


LANGUAGES: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        image=PYTHON_GUI_IMAGE,
        filename="main.py",
        command=[
            "sh",
            "-c",
            (
                "set -e; cp /work/main.py /tmp/main.py; cd /tmp; "
                "PY_IMPORTS=$(python - <<'PY'\n"
                "import ast, sys\n"
                "STDLIB = set(sys.stdlib_module_names)\n"
                "ALIAS = {'cv2':'opencv-python','sklearn':'scikit-learn','PIL':'Pillow','bs4':'beautifulsoup4','yaml':'PyYAML','skimage':'scikit-image'}\n"
                "src = open('/tmp/main.py').read()\n"
                "try:\n"
                "    tree = ast.parse(src)\n"
                "except SyntaxError:\n"
                "    sys.exit(0)\n"
                "pkgs = set()\n"
                "for node in ast.walk(tree):\n"
                "    if isinstance(node, ast.Import):\n"
                "        for n in node.names:\n"
                "            pkgs.add(n.name.split('.')[0])\n"
                "    elif isinstance(node, ast.ImportFrom):\n"
                "        if node.module and node.level == 0:\n"
                "            pkgs.add(node.module.split('.')[0])\n"
                "needed = sorted({ALIAS.get(p, p) for p in pkgs if p and p not in STDLIB})\n"
                "print(' '.join(needed))\n"
                "PY\n"
                "); "
                "if [ -n \"$PY_IMPORTS\" ]; then echo \"[atlas-runner] installing: $PY_IMPORTS\"; pip install --quiet --no-input --disable-pip-version-check --root-user-action=ignore $PY_IMPORTS || true; fi; "
                "python /tmp/main.py"
            ),
        ],
    ),
    "javascript": LanguageSpec(
        image="docker.io/library/node:20-alpine",
        filename="main.js",
        command=[
            "sh",
            "-c",
            (
                "set -e; mkdir -p /tmp/app; cp /work/main.js /tmp/app/main.js; cd /tmp/app; "
                "DEPS=$(node -e \"const fs=require('fs');const src=fs.readFileSync('/tmp/app/main.js','utf8');const core=new Set(require('module').builtinModules);const s=new Set();const re=/require\\(['\\\"]([^'\\\"]+)['\\\"]\\)|from ['\\\"]([^'\\\"]+)['\\\"]|import ['\\\"]([^'\\\"]+)['\\\"]/g;let m;while((m=re.exec(src))){let p=m[1]||m[2]||m[3];if(!p||p.startsWith('.')||p.startsWith('/')||p.startsWith('node:'))continue;if(p.startsWith('@')){p=p.split('/').slice(0,2).join('/')}else{p=p.split('/')[0]}if(!core.has(p))s.add(p)}process.stdout.write([...s].join(' '))\"); "
                "if [ -n \"$DEPS\" ]; then echo \"[atlas-runner] installing: $DEPS\"; npm init -y >/dev/null 2>&1; npm install --silent --no-audit --no-fund $DEPS >/dev/null 2>&1 || true; fi; "
                "node /tmp/app/main.js"
            ),
        ],
        requires_network=True,
    ),
    "typescript": LanguageSpec(
        image="docker.io/library/node:20-alpine",
        filename="main.ts",
        command=[
            "sh",
            "-c",
            (
                "set -e; mkdir -p /tmp/app; cp /work/main.ts /tmp/app/main.ts; cd /tmp/app; "
                "DEPS=$(node -e \"const fs=require('fs');const src=fs.readFileSync('/tmp/app/main.ts','utf8');const core=new Set(require('module').builtinModules);const s=new Set();const re=/require\\(['\\\"]([^'\\\"]+)['\\\"]\\)|from ['\\\"]([^'\\\"]+)['\\\"]|import ['\\\"]([^'\\\"]+)['\\\"]/g;let m;while((m=re.exec(src))){let p=m[1]||m[2]||m[3];if(!p||p.startsWith('.')||p.startsWith('/')||p.startsWith('node:'))continue;if(p.startsWith('@')){p=p.split('/').slice(0,2).join('/')}else{p=p.split('/')[0]}if(!core.has(p))s.add(p)}process.stdout.write([...s].join(' '))\"); "
                "npm init -y >/dev/null 2>&1; "
                "if [ -n \"$DEPS\" ]; then echo \"[atlas-runner] installing: $DEPS\"; npm install --silent --no-audit --no-fund $DEPS >/dev/null 2>&1 || true; fi; "
                "npx --yes tsx /tmp/app/main.ts"
            ),
        ],
        requires_network=True,
    ),
    "go": LanguageSpec(
        image="docker.io/library/golang:1.22-alpine",
        filename="main.go",
        command=[
            "sh",
            "-c",
            (
                "set -e; mkdir -p /tmp/app; cp /work/main.go /tmp/app/main.go; cd /tmp/app; "
                "go mod init atlasrun >/dev/null 2>&1 || true; "
                "echo '[atlas-runner] resolving modules'; "
                "go mod tidy >/dev/null 2>&1 || true; "
                "go run ."
            ),
        ],
        requires_network=True,
    ),
    "rust": LanguageSpec(
        image="docker.io/library/rust:1-slim",
        filename="main.rs",
        command=[
            "sh",
            "-c",
            (
                "set -e; mkdir -p /tmp/app/src; cp /work/main.rs /tmp/app/src/main.rs; cd /tmp/app; "
                "printf '[package]\\nname=\"atlasrun\"\\nversion=\"0.1.0\"\\nedition=\"2021\"\\n' > Cargo.toml; "
                "CRATES=$(grep -E '^[[:space:]]*(use |extern crate )' src/main.rs | sed -E 's/^[[:space:]]*use |^[[:space:]]*extern crate //' | awk '{print $1}' | sed -E 's/([A-Za-z0-9_]+).*/\\1/' | sort -u | grep -Ev '^(std|core|alloc|crate|self|super)$' || true); "
                "if [ -n \"$CRATES\" ]; then echo \"[atlas-runner] installing: $CRATES\"; which cargo-add >/dev/null 2>&1 || true; for c in $CRATES; do cargo add $c >/dev/null 2>&1 || true; done; fi; "
                "cargo run --quiet 2>&1"
            ),
        ],
        requires_network=True,
    ),
    "c": LanguageSpec(
        image="docker.io/library/gcc:14.2",
        filename="main.c",
        command=["sh", "-c", "cp /work/main.c /tmp/main.c && gcc /tmp/main.c -o /tmp/app -lm && /tmp/app"],
    ),
    "cpp": LanguageSpec(
        image="docker.io/library/gcc:14.2",
        filename="main.cpp",
        command=["sh", "-c", "cp /work/main.cpp /tmp/main.cpp && g++ /tmp/main.cpp -o /tmp/app -lm && /tmp/app"],
    ),
    "java": LanguageSpec(
        image="docker.io/library/eclipse-temurin:21-jdk",
        filename="Main.java",
        command=["sh", "-c", "cp /work/Main.java /tmp/Main.java && cd /tmp && javac Main.java && java Main"],
    ),
    "ruby": LanguageSpec(
        image="docker.io/library/ruby:3-alpine",
        filename="main.rb",
        command=[
            "sh",
            "-c",
            (
                "set -e; apk add --no-cache build-base >/dev/null 2>&1 || true; "
                "cp /work/main.rb /tmp/main.rb; cd /tmp; "
                "GEMS=$(ruby -e \"src=File.read('/tmp/main.rb'); core=%w[date time json yaml fileutils pathname set open-uri securerandom digest base64 csv open3 tempfile timeout uri net/http net/https stringio strscan logger optparse ostruct singleton thread]; gs=src.scan(/^\\s*require\\s+['\\\"]([^'\\\"]+)['\\\"]/).flatten.map{|s|s.split('/').first}.uniq - core; puts gs.join(' ')\"); "
                "if [ -n \"$GEMS\" ]; then echo \"[atlas-runner] installing: $GEMS\"; gem install --silent --no-document $GEMS >/dev/null 2>&1 || true; fi; "
                "ruby /tmp/main.rb"
            ),
        ],
        requires_network=True,
    ),
    "php": LanguageSpec(
        image="docker.io/library/composer:2",
        filename="main.php",
        command=[
            "sh",
            "-c",
            "set -e; mkdir -p /tmp/app; cp /work/main.php /tmp/app/main.php; cd /tmp/app; php /tmp/app/main.php",
        ],
    ),
    "bash": LanguageSpec(
        image="docker.io/library/bash:5.2",
        filename="main.sh",
        command=["bash", "/work/main.sh"],
    ),
    "csharp": LanguageSpec(
        image="mcr.microsoft.com/dotnet/sdk:8.0",
        filename="Program.cs",
        command=[
            "sh",
            "-c",
            "mkdir -p /tmp/app && cp /work/Program.cs /tmp/app/Program.cs && cd /tmp/app && dotnet new console --force -o . >/dev/null && cp /work/Program.cs ./Program.cs && dotnet run --nologo",
        ],
    ),
    "kotlin": LanguageSpec(
        image="docker.io/zenika/kotlin:1.4.20",
        filename="main.kts",
        command=["sh", "-c", "cp /work/main.kts /tmp/main.kts && kotlinc -script /tmp/main.kts"],
    ),
    "swift": LanguageSpec(
        image="docker.io/library/swift:5.9",
        filename="main.swift",
        command=["sh", "-c", "cp /work/main.swift /tmp/main.swift && swift /tmp/main.swift"],
    ),
    "perl": LanguageSpec(
        image="docker.io/library/perl:5",
        filename="main.pl",
        command=[
            "sh",
            "-c",
            (
                "set -e; cp /work/main.pl /tmp/main.pl; "
                "MODS=$(grep -Eo '^\\s*use\\s+[A-Za-z0-9_:]+' /tmp/main.pl | awk '{print $2}' | grep -Ev '^(strict|warnings|utf8|lib|feature|constant|vars|parent|base|overload|Exporter|Carp)$' | sort -u || true); "
                "if [ -n \"$MODS\" ]; then echo \"[atlas-runner] installing: $MODS\"; cpanm --quiet --notest $MODS >/dev/null 2>&1 || true; fi; "
                "perl /tmp/main.pl"
            ),
        ],
        requires_network=True,
    ),
    "lua": LanguageSpec(
        image="docker.io/nickblah/lua:5.4-alpine",
        filename="main.lua",
        command=["lua", "/work/main.lua"],
    ),
    "r": LanguageSpec(
        image="docker.io/library/r-base:4.4.3",
        filename="main.R",
        command=[
            "sh",
            "-c",
            (
                "set -e; cp /work/main.R /tmp/main.R; "
                "PKGS=$(grep -Eo '(library|require)\\([A-Za-z0-9._]+' /tmp/main.R | sed -E 's/(library|require)\\(//' | sort -u || true); "
                "if [ -n \"$PKGS\" ]; then echo \"[atlas-runner] installing: $PKGS\"; for p in $PKGS; do Rscript -e \"if(!require('$p',quietly=TRUE))install.packages('$p',repos='https://cloud.r-project.org')\" >/dev/null 2>&1 || true; done; fi; "
                "Rscript /tmp/main.R"
            ),
        ],
        requires_network=True,
    ),
    "elixir": LanguageSpec(
        image="docker.io/library/elixir:1.16-alpine",
        filename="main.exs",
        command=["elixir", "/work/main.exs"],
    ),
    "dart": LanguageSpec(
        image="docker.io/library/dart:3.6.2",
        filename="main.dart",
        command=[
            "sh",
            "-c",
            (
                "set -e; mkdir -p /tmp/app/bin; cp /work/main.dart /tmp/app/bin/main.dart; cd /tmp/app; "
                "dart create -q -t console --force . >/dev/null 2>&1 || true; "
                "cp /work/main.dart bin/main.dart; "
                "PKGS=$(grep -Eo \"package:[A-Za-z0-9_]+\" bin/main.dart | sed 's/package://' | sort -u || true); "
                "if [ -n \"$PKGS\" ]; then echo \"[atlas-runner] installing: $PKGS\"; for p in $PKGS; do dart pub add $p >/dev/null 2>&1 || true; done; fi; "
                "dart run bin/main.dart"
            ),
        ],
        requires_network=True,
    ),
}


LANGUAGE_ALIASES: dict[str, str] = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "node": "javascript",
    "ts": "typescript",
    "golang": "go",
    "rs": "rust",
    "c++": "cpp",
    "cxx": "cpp",
    "cc": "cpp",
    "rb": "ruby",
    "sh": "bash",
    "shell": "bash",
    "zsh": "bash",
    "cs": "csharp",
    "c#": "csharp",
    "kt": "kotlin",
    "kts": "kotlin",
    "pl": "perl",
    "ex": "elixir",
    "exs": "elixir",
}


CLIENT_LANGUAGES = {"html", "htm"}
DEFAULT_RUNNER_TIMEOUT_SECONDS = 120
DEFAULT_GUI_RUNNER_TIMEOUT_SECONDS = 900
RUNNER_NETWORK_ENV = "ATLAS_RUNNER_NETWORK"
RUNNER_TIMEOUT_ENV = "ATLAS_RUNNER_TIMEOUT_SECONDS"
RUNNER_GUI_TIMEOUT_ENV = "ATLAS_RUNNER_GUI_TIMEOUT_SECONDS"
RUNNER_MAX_CONCURRENT_ENV = "ATLAS_RUNNER_MAX_CONCURRENT"
RUNNER_HISTORY_LIMIT_ENV = "ATLAS_RUNNER_HISTORY_LIMIT"
RUNNER_SUBSCRIBER_QUEUE_SIZE_ENV = "ATLAS_RUNNER_SUBSCRIBER_QUEUE_SIZE"
RUNNER_MAX_OUTPUT_BYTES_ENV = "ATLAS_RUNNER_MAX_OUTPUT_BYTES"
RUNNER_STORAGE_LIMIT_ENV = "ATLAS_RUNNER_STORAGE_LIMIT"
RUNNER_ALLOWED_NETWORKS = {"none", "bridge"}
RUNNER_INTERNAL_NETWORK = "atlas-runner-internal"
DEFAULT_RUNNER_MAX_CONCURRENT = 2
MAX_RUNNER_MAX_CONCURRENT = 8
DEFAULT_RUNNER_HISTORY_LIMIT = 2_000
MAX_RUNNER_HISTORY_LIMIT = 20_000
MAX_RUNNER_SUBSCRIBER_QUEUE_SIZE = 2_048
DEFAULT_RUNNER_MAX_OUTPUT_BYTES = 1_048_576
MAX_RUNNER_MAX_OUTPUT_BYTES = 16_777_216
DEFAULT_RUNNER_TMPFS_SIZE = "512m"
DEFAULT_RUNNER_MAX_FILE_BYTES = 536_870_912
RUNNER_STREAM_CHUNK_SIZE = 4_096
RUNNER_MAX_CODE_BYTES = 1_048_576


def resolve_language(language: str) -> str | None:
    normalized = (language or "").strip().lower()
    if not normalized:
        return None
    if normalized in LANGUAGES:
        return normalized
    if normalized in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[normalized]
    return None


def supported_languages() -> list[str]:
    return sorted(set(LANGUAGES.keys()) | set(LANGUAGE_ALIASES.keys()) | CLIENT_LANGUAGES)


def _docker_binary() -> str | None:
    return shutil.which("docker")


def _remove_legacy_python_gui_images(binary: str | None = None) -> None:
    binary = binary or _docker_binary()
    if not binary:
        return
    for image in LEGACY_PYTHON_GUI_IMAGES:
        subprocess.run(
            [binary, "image", "rm", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _python_plan(code: str) -> RunPlan:
    imports = _python_imports(code)
    modules = _python_import_modules(code)
    web_port = _python_web_port(code, imports)
    terminal = bool(not web_port and _python_terminal_detected(code, imports))
    gui = terminal or _python_gui_detected(code, imports)
    web_command, web_packages = _python_web_command(code, imports, web_port)
    declared_packages = set(_python_declared_pip_packages(code))
    pip_packages = sorted(set(_python_pip_packages(imports, modules)) | web_packages | declared_packages)
    apt_packages = _python_apt_packages(imports, modules, gui, terminal)
    repair_may_need_network = _python_dependency_repair_may_need_network(code)
    gui_args = " ".join(_python_gui_args(code))
    gui_args_suffix = f" {gui_args}" if gui_args else ""
    script_parts = [
        "set -e",
        "cp /work/main.py /tmp/main.py",
        "cd /tmp",
        (
            "export DEBIAN_FRONTEND=noninteractive DISPLAY=:99 SCREEN_GEOMETRY=1280x800x24 "
            "VNC_PORT=5900 NOVNC_PORT=6080 PYTHONUNBUFFERED=1 SDL_AUDIODRIVER=dummy "
            "PYGAME_HIDE_SUPPORT_PROMPT=1 ALSA_CONFIG_PATH=/dev/null"
        ),
        _python_dependency_repair_script(imports, modules),
    ]
    if apt_packages:
        apt_args = shlex.join(apt_packages)
        script_parts.extend(
            [
                f"echo {shlex.quote('[atlas-runner] installing system dependencies: ' + ' '.join(apt_packages))}",
                "apt-get update >/dev/null",
                f"apt-get install -y --no-install-recommends {apt_args} >/dev/null",
                "rm -rf /var/lib/apt/lists/*",
            ],
        )
    if gui:
        script_parts.extend(
            [
                "ln -sf /usr/share/novnc/vnc.html /usr/share/novnc/index.html",
                "mkdir -p /root/.fluxbox",
                (
                    "printf '%s\\n' 'session.screen0.workspaces: 1' 'session.screen0.workspaceNames: Main' "
                    "'session.screen0.toolbar.tools: workspacename, iconbar, systemtray, clock' > /root/.fluxbox/init"
                ),
                "rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 2>/dev/null || true",
                'Xvfb :99 -screen 0 "$SCREEN_GEOMETRY" -ac +extension GLX +render -noreset >/tmp/xvfb.log 2>&1 & sleep 0.6',
                (
                    "fluxbox -rc /root/.fluxbox/init >/tmp/fluxbox.log 2>&1 & "
                    'x11vnc -display :99 -nopw -forever -shared -rfbport "$VNC_PORT" -quiet >/tmp/x11vnc.log 2>&1 & '
                    'websockify --web /usr/share/novnc "$NOVNC_PORT" "localhost:$VNC_PORT" >/tmp/novnc.log 2>&1 & '
                    "sleep 0.4"
                ),
                "echo '[atlas-runner] GUI ready on port 6080'",
            ],
        )
    if pip_packages:
        pip_args = shlex.join(pip_packages)
        script_parts.append(f"echo {shlex.quote('[atlas-runner] installing Python packages: ' + ' '.join(pip_packages))}")
        script_parts.append(
            f"pip install --quiet --no-input --disable-pip-version-check --root-user-action=ignore {pip_args} "
            "|| echo '[atlas-runner] initial package install failed; continuing to run and repair imports if possible'"
        )
    if web_port:
        script_parts.append(f"echo {shlex.quote(f'[atlas-runner] web preview will use container port {web_port}')}")
        script_parts.append(
            f"export PORT={web_port} HOST=0.0.0.0 GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT={web_port}"
        )
    if terminal:
        terminal_command = (
            f"cd /tmp; export TERM=xterm-256color; python -u /tmp/main.py{gui_args_suffix}; "
            "status=$?; echo; echo \"[atlas-runner] terminal app exited with status $status\"; "
            "sleep 2; exit $status"
        )
        script_parts.append("echo '[atlas-runner] starting terminal UI in virtual display'")
        script_parts.append(
            "xterm -geometry 120x34 -fa 'DejaVu Sans Mono' -fs 12 -title 'Atlas Terminal' "
            f"-e sh -c {shlex.quote(terminal_command)}"
        )
    elif web_command:
        script_parts.append(f"python /tmp/atlas_python_repair.py sh -c {shlex.quote(web_command)}")
    else:
        script_parts.append(f"python /tmp/atlas_python_repair.py python -u /tmp/main.py{gui_args_suffix}")
    ports: dict[int, int] = {}
    if gui:
        ports[_reserve_host_port()] = NOVNC_CONTAINER_PORT
    if web_port:
        ports[_reserve_host_port()] = web_port
    return RunPlan(
        image=PYTHON_GUI_IMAGE,
        filename="main.py",
        command=["sh", "-c", "\n".join(script_parts)],
        ports=ports,
        gui=gui,
        web_container_port=web_port,
        requires_network=bool(apt_packages or pip_packages or repair_may_need_network),
        uses_apt=bool(apt_packages),
    )


def resolve_plan(language: str, code: str, progress: "Any | None" = None) -> RunPlan:
    if language == "python":
        return _python_plan(code)
    spec = LANGUAGES[language]
    return RunPlan(
        image=spec.image,
        filename=spec.filename,
        command=list(spec.command),
        requires_network=spec.requires_network,
    )


def _configured_runner_network() -> str:
    configured = os.environ.get(RUNNER_NETWORK_ENV, "none").strip().lower() or "none"
    if configured not in RUNNER_ALLOWED_NETWORKS:
        configured = "none"
    return configured


def _runner_network_policy(plan: RunPlan) -> str:
    configured = _configured_runner_network()
    if configured == "none" and plan.ports:
        # Docker's built-in "none" network cannot publish preview ports. A
        # private --internal bridge keeps loopback previews working without
        # silently granting outbound access.
        return RUNNER_INTERNAL_NETWORK
    return configured


def _runner_outbound_network_enabled(network: str) -> bool:
    return network == "bridge"


def _bounded_positive_int_env(key: str, default: int, maximum: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return min(maximum, max(1, value))


def _runner_max_concurrent() -> int:
    return _bounded_positive_int_env(
        RUNNER_MAX_CONCURRENT_ENV,
        DEFAULT_RUNNER_MAX_CONCURRENT,
        MAX_RUNNER_MAX_CONCURRENT,
    )


def _runner_history_limit() -> int:
    return _bounded_positive_int_env(
        RUNNER_HISTORY_LIMIT_ENV,
        DEFAULT_RUNNER_HISTORY_LIMIT,
        MAX_RUNNER_HISTORY_LIMIT,
    )


def _runner_subscriber_queue_size() -> int:
    return _bounded_positive_int_env(
        RUNNER_SUBSCRIBER_QUEUE_SIZE_ENV,
        DEFAULT_SUBSCRIBER_QUEUE_SIZE,
        MAX_RUNNER_SUBSCRIBER_QUEUE_SIZE,
    )


def _runner_max_output_bytes() -> int:
    return _bounded_positive_int_env(
        RUNNER_MAX_OUTPUT_BYTES_ENV,
        DEFAULT_RUNNER_MAX_OUTPUT_BYTES,
        MAX_RUNNER_MAX_OUTPUT_BYTES,
    )


def _runner_storage_limit() -> str | None:
    raw = os.environ.get(RUNNER_STORAGE_LIMIT_ENV, "").strip()
    if not raw:
        return None
    if not re.fullmatch(r"[1-9][0-9]*(?:[kmgtKMGT](?:i?[bB])?|[bB])?", raw):
        raise RuntimeError(
            f"{RUNNER_STORAGE_LIMIT_ENV} must be a positive byte count "
            "with an optional K, M, G, T, KiB, MiB, GiB, or TiB suffix."
        )
    return raw


def _runner_storage_limit_supported(binary: str) -> bool:
    try:
        completed = subprocess.run(
            [binary, "run", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and "--storage-opt" in completed.stdout


def _inspect_internal_network(binary: str) -> bool | None:
    try:
        completed = subprocess.run(
            [
                binary,
                "network",
                "inspect",
                RUNNER_INTERNAL_NETWORK,
                "--format",
                "{{.Internal}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip().lower() == "true"


def _ensure_internal_network(binary: str) -> None:
    internal = _inspect_internal_network(binary)
    if internal is True:
        return
    if internal is False:
        raise RuntimeError(
            f"Docker network '{RUNNER_INTERNAL_NETWORK}' already exists but is not internal; "
            "remove or rename it before running an isolated preview."
        )

    try:
        completed = subprocess.run(
            [
                binary,
                "network",
                "create",
                "--driver",
                "bridge",
                "--internal",
                "--label",
                "atlas.runner.network=1",
                RUNNER_INTERNAL_NETWORK,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Docker timed out while creating the isolated preview network.") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to create the isolated preview network: {exc}") from exc

    if completed.returncode == 0 or _inspect_internal_network(binary) is True:
        return
    detail = (completed.stderr or completed.stdout or "").strip()
    suffix = f" Details: {detail.splitlines()[-1]}" if detail else ""
    raise RuntimeError(f"Failed to create the isolated preview network.{suffix}")


def _runner_timeout_seconds(plan: RunPlan) -> int:
    env_key = RUNNER_GUI_TIMEOUT_ENV if plan.gui else RUNNER_TIMEOUT_ENV
    default_value = DEFAULT_GUI_RUNNER_TIMEOUT_SECONDS if plan.gui else DEFAULT_RUNNER_TIMEOUT_SECONDS
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return default_value
    try:
        return max(1, int(raw))
    except ValueError:
        return default_value


def docker_status() -> dict[str, Any]:
    binary = _docker_binary()
    if not binary:
        return {
            "available": False,
            "reason": "Docker CLI was not found on PATH. Install Docker Desktop and try again.",
        }
    try:
        completed = subprocess.run(
            [binary, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": "Docker is installed but the daemon did not respond in time."}
    except OSError as exc:
        return {"available": False, "reason": f"Failed to invoke docker: {exc}"}

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        reason = "Docker Desktop is installed but not running. Start Docker Desktop and try again."
        if detail:
            reason = f"{reason} Details: {detail.splitlines()[-1]}"
        return {"available": False, "reason": reason}
    version = completed.stdout.strip() or "unknown"
    return {"available": True, "server_version": version}


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@dataclass
class RunnerProcess:
    run_id: str
    language: str
    container_name: str
    work_dir: Path
    started_at: float
    timeout_seconds: int
    network: str
    process: subprocess.Popen
    history_limit: int
    subscriber_queue_size: int
    max_output_bytes: int
    history: deque[dict[str, Any]] = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock)
    finished: bool = False
    exit_code: int | None = None
    output_bytes: int = 0
    output_truncated: bool = False
    subscribers: list[queue.Queue[dict[str, Any]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.history = deque(maxlen=max(1, self.history_limit))


class CodeRunner:
    def __init__(
        self,
        *,
        max_concurrent: int | None = None,
        history_limit: int | None = None,
        subscriber_queue_size: int | None = None,
        max_output_bytes: int | None = None,
    ) -> None:
        self._runs: dict[str, RunnerProcess] = {}
        self._starting_names: set[str] = set()
        self._lock = threading.Lock()
        self._shutting_down = False
        self._owner_id = uuid.uuid4().hex
        self._max_concurrent = min(
            MAX_RUNNER_MAX_CONCURRENT,
            max(1, int(max_concurrent if max_concurrent is not None else _runner_max_concurrent())),
        )
        self._history_limit = min(
            MAX_RUNNER_HISTORY_LIMIT,
            max(1, int(history_limit if history_limit is not None else _runner_history_limit())),
        )
        self._subscriber_queue_size = min(
            MAX_RUNNER_SUBSCRIBER_QUEUE_SIZE,
            max(
                1,
                int(
                    subscriber_queue_size
                    if subscriber_queue_size is not None
                    else _runner_subscriber_queue_size()
                ),
            ),
        )
        self._max_output_bytes = min(
            MAX_RUNNER_MAX_OUTPUT_BYTES,
            max(1, int(max_output_bytes if max_output_bytes is not None else _runner_max_output_bytes())),
        )

    def start(self, language: str, code: str) -> dict[str, Any]:
        resolved = resolve_language(language)
        if not resolved:
            raise RuntimeError(f"Language '{language}' is not supported.")
        if resolved in CLIENT_LANGUAGES:
            raise RuntimeError("HTML is rendered in the client sandbox, not via Docker.")
        code_bytes = len(code.encode("utf-8"))
        if code_bytes > RUNNER_MAX_CODE_BYTES:
            raise RuntimeError(
                f"Code is too large to run ({code_bytes} bytes); "
                f"the limit is {RUNNER_MAX_CODE_BYTES} bytes."
            )

        binary = _docker_binary()
        if not binary:
            raise RuntimeError("Docker CLI was not found on PATH.")

        self._cleanup_finished_runs()
        plan = resolve_plan(resolved, code)
        run_id = uuid.uuid4().hex[:16]
        container_name = f"atlas-run-{run_id}"
        configured_network = _configured_runner_network()
        network = _runner_network_policy(plan)
        timeout_seconds = _runner_timeout_seconds(plan)
        storage_limit = _runner_storage_limit()
        if storage_limit and not _runner_storage_limit_supported(binary):
            raise RuntimeError(
                f"{RUNNER_STORAGE_LIMIT_ENV} is configured, but this container engine "
                "does not support the docker run --storage-opt flag."
            )
        outbound_network = _runner_outbound_network_enabled(network)
        dependency_network_unmet = plan.requires_network and not outbound_network
        restricted_rootfs = not plan.uses_apt and not plan.requires_network
        self._reserve_start(container_name)

        work_dir: Path | None = None
        process: subprocess.Popen | None = None
        registered = False
        try:
            self._cleanup_stale_containers(binary)
            _remove_legacy_python_gui_images(binary)
            if network == RUNNER_INTERNAL_NETWORK:
                _ensure_internal_network(binary)

            work_dir = Path(tempfile.mkdtemp(prefix=f"atlas-run-{run_id}-"))
            source_path = work_dir / plan.filename
            source_path.write_text(code, encoding="utf-8")
            if restricted_rootfs and os.name != "nt":
                # The non-root container user must be able to traverse the bind
                # mount and read the known source file. Keep directory listing
                # disabled so other local users cannot browse submitted code.
                work_dir.chmod(0o711)
                source_path.chmod(0o444)

            docker_args: list[str] = [
                binary,
                "run",
                "--rm",
                "-i",
                "--init",
                "--name",
                container_name,
                "--memory",
                "2g",
                "--memory-swap",
                "2g",
                "--cpus",
                "2",
                "--pids-limit",
                "512",
                "--ulimit",
                "nofile=1024:1024",
                "--ulimit",
                f"fsize={DEFAULT_RUNNER_MAX_FILE_BYTES}:{DEFAULT_RUNNER_MAX_FILE_BYTES}",
                "--stop-timeout",
                "5",
                "--tmpfs",
                f"/tmp:rw,exec,nosuid,nodev,size={DEFAULT_RUNNER_TMPFS_SIZE},mode=1777",
                "--network",
                network,
                "--security-opt",
                "no-new-privileges",
                "--cap-drop",
                "ALL",
                "--label",
                "atlas.runner=1",
                "--label",
                f"atlas.runner.run_id={run_id}",
                "--label",
                f"atlas.runner.owner_pid={os.getpid()}",
                "--label",
                f"atlas.runner.owner_id={self._owner_id}",
                "-v",
                f"{work_dir}:/work:ro",
                "-w",
                "/work",
            ]
            if storage_limit:
                docker_args.extend(["--storage-opt", f"size={storage_limit}"])
            if restricted_rootfs:
                docker_args.extend(["--read-only", "--user", "65534:65534", "--env", "HOME=/tmp"])
            if plan.uses_apt:
                for capability in ("CHOWN", "DAC_OVERRIDE", "FOWNER", "SETGID", "SETUID"):
                    docker_args.extend(["--cap-add", capability])
            for host_port, container_port in plan.ports.items():
                docker_args.extend(["-p", f"127.0.0.1:{host_port}:{container_port}"])
            docker_args.append(plan.image)
            docker_args.extend(plan.command)

            creation_flags = 0
            if sys.platform == "win32":
                creation_flags = 0x08000000  # CREATE_NO_WINDOW

            process = subprocess.Popen(
                docker_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                bufsize=1,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creation_flags if sys.platform == "win32" else 0,
            )

            runner = RunnerProcess(
                run_id=run_id,
                language=resolved,
                container_name=container_name,
                work_dir=work_dir,
                started_at=time.time(),
                timeout_seconds=timeout_seconds,
                network=network,
                process=process,
                history_limit=self._history_limit,
                subscriber_queue_size=self._subscriber_queue_size,
                max_output_bytes=self._max_output_bytes,
            )
            with self._lock:
                if self._shutting_down:
                    raise RuntimeError("The code runner is shutting down and cannot start a new run.")
                self._runs[run_id] = runner
                self._starting_names.discard(container_name)
                registered = True

            if dependency_network_unmet:
                self._emit(
                    runner,
                    {
                        "type": "output",
                        "stream": "stderr",
                        "chunk": (
                            "[atlas-runner] outbound network is disabled by "
                            f"{RUNNER_NETWORK_ENV}=none; dependency installation may fail. "
                            f"Set {RUNNER_NETWORK_ENV}=bridge to explicitly allow outbound access.\n"
                        ),
                    },
                    enforce_output_limit=False,
                )

            stdout_thread = threading.Thread(
                target=self._pump_stream,
                args=(runner, process.stdout, "stdout"),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=self._pump_stream,
                args=(runner, process.stderr, "stderr"),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()
            threading.Thread(
                target=self._wait_for_exit,
                args=(runner, (stdout_thread, stderr_thread)),
                daemon=True,
            ).start()
            threading.Thread(target=self._enforce_timeout, args=(runner,), daemon=True).start()

            response: dict[str, Any] = {
                "run_id": run_id,
                "language": resolved,
                "container": container_name,
                "configured_network": configured_network,
                "network": network,
                "outbound_network": outbound_network,
                "dependency_network_required": plan.requires_network,
                "network_requirement_unmet": dependency_network_unmet,
                "timeout_seconds": timeout_seconds,
                "filesystem_mode": "read-only" if restricted_rootfs else "writable",
            }
            if storage_limit:
                response["storage_limit"] = storage_limit
            if dependency_network_unmet:
                response["network_warning"] = (
                    f"Dependencies may require outbound access; explicitly set "
                    f"{RUNNER_NETWORK_ENV}=bridge to allow it."
                )
            if plan.gui:
                host_port = next(iter(plan.ports.keys()))
                response["vnc_url"] = (
                    f"http://127.0.0.1:{host_port}/vnc.html?autoconnect=1&resize=remote&reconnect=1"
                )
            if plan.web_container_port:
                for host_port, container_port in plan.ports.items():
                    if container_port == plan.web_container_port:
                        response["web_url"] = f"http://127.0.0.1:{host_port}/"
                        break
            return response
        except OSError as exc:
            raise RuntimeError(f"Failed to start Docker: {exc}") from exc
        finally:
            if not registered:
                with self._lock:
                    self._starting_names.discard(container_name)
                if process is not None:
                    self._remove_container_name(binary, container_name)
                    try:
                        process.kill()
                    except OSError:
                        pass
                if work_dir is not None:
                    shutil.rmtree(work_dir, ignore_errors=True)

    def subscribe(
        self,
        run_id: str,
    ) -> tuple[list[dict[str, Any]], queue.Queue[dict[str, Any]], bool]:
        runner = self._require(run_id)
        with runner.lock:
            history = list(runner.history)
            if runner.finished:
                return history, queue.Queue(maxsize=runner.subscriber_queue_size), True
            subscriber: queue.Queue[dict[str, Any]] = queue.Queue(
                maxsize=runner.subscriber_queue_size
            )
            runner.subscribers.append(subscriber)
            return history, subscriber, False

    def unsubscribe(self, run_id: str, subscriber: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            runner = self._runs.get(run_id)
        if not runner:
            return
        with runner.lock:
            if subscriber in runner.subscribers:
                runner.subscribers.remove(subscriber)

    def stop(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            runner = self._runs.get(run_id)
        if not runner:
            return {"run_id": run_id, "status": "unknown"}
        self._kill_container(runner)
        try:
            runner.process.kill()
        except OSError:
            pass
        return {"run_id": run_id, "status": "stopping"}

    def _require(self, run_id: str) -> RunnerProcess:
        with self._lock:
            runner = self._runs.get(run_id)
        if not runner:
            raise RuntimeError(f"Runner '{run_id}' is not known.")
        return runner

    def _reserve_start(self, container_name: str) -> None:
        with self._lock:
            if self._shutting_down:
                raise RuntimeError("The code runner is shutting down and cannot start a new run.")
            active_count = len(self._starting_names) + sum(
                1 for runner in self._runs.values() if not runner.finished
            )
            if active_count >= self._max_concurrent:
                raise RuntimeError(
                    f"Runner concurrency limit reached ({self._max_concurrent} active runs). "
                    "Stop or wait for a run to finish before starting another."
                )
            self._starting_names.add(container_name)

    def _emit(
        self,
        runner: RunnerProcess,
        event: dict[str, Any],
        *,
        enforce_output_limit: bool = True,
    ) -> None:
        with runner.lock:
            events = self._bounded_events_locked(runner, event, enforce_output_limit)
            for queued_event in events:
                runner.history.append(queued_event)
            subscribers = list(runner.subscribers)
        for queued_event in events:
            for subscriber in subscribers:
                put_bounded_queue(subscriber, queued_event)

    @staticmethod
    def _bounded_events_locked(
        runner: RunnerProcess,
        event: dict[str, Any],
        enforce_output_limit: bool,
    ) -> list[dict[str, Any]]:
        if not enforce_output_limit or event.get("type") != "output":
            return [event]

        chunk = str(event.get("chunk", ""))
        encoded = chunk.encode("utf-8")
        remaining = max(0, runner.max_output_bytes - runner.output_bytes)
        if len(encoded) <= remaining:
            runner.output_bytes += len(encoded)
            return [event]

        bounded: list[dict[str, Any]] = []
        if remaining:
            partial = encoded[:remaining].decode("utf-8", errors="ignore")
            runner.output_bytes += len(partial.encode("utf-8"))
            if partial:
                partial_event = dict(event)
                partial_event["chunk"] = partial
                bounded.append(partial_event)
        if not runner.output_truncated:
            runner.output_truncated = True
            bounded.append(
                {
                    "type": "output",
                    "stream": "stderr",
                    "chunk": (
                        f"[atlas-runner] output truncated after "
                        f"{runner.max_output_bytes} bytes\n"
                    ),
                }
            )
        return bounded

    def _pump_stream(self, runner: RunnerProcess, stream: Iterable[str] | None, channel: str) -> None:
        if stream is None:
            return
        try:
            readline = getattr(stream, "readline", None)
            if callable(readline):
                while True:
                    chunk = readline(RUNNER_STREAM_CHUNK_SIZE)
                    if not chunk:
                        break
                    self._emit(runner, {"type": "output", "stream": channel, "chunk": chunk})
            else:
                for chunk in stream:
                    if chunk is None:
                        break
                    self._emit(runner, {"type": "output", "stream": channel, "chunk": chunk})
        except Exception as exc:  # pragma: no cover - defensive
            self._emit(
                runner,
                {
                    "type": "output",
                    "stream": "stderr",
                    "chunk": f"[atlas-runner] stream error: {exc}\n",
                },
                enforce_output_limit=False,
            )

    def _wait_for_exit(
        self,
        runner: RunnerProcess,
        stream_threads: tuple[threading.Thread, ...] = (),
    ) -> None:
        try:
            exit_code = runner.process.wait()
        except Exception as exc:  # pragma: no cover - defensive
            exit_code = -1
            self._emit(
                runner,
                {
                    "type": "output",
                    "stream": "stderr",
                    "chunk": f"[atlas-runner] wait error: {exc}\n",
                },
                enforce_output_limit=False,
            )
        for stream_thread in stream_threads:
            stream_thread.join(timeout=1)
        duration_ms = int((time.time() - runner.started_at) * 1000)
        event = {
            "type": "exit",
            "code": exit_code,
            "duration_ms": duration_ms,
        }
        with runner.lock:
            runner.finished = True
            runner.exit_code = exit_code
            runner.history.append(event)
            subscribers = list(runner.subscribers)
            runner.subscribers.clear()
        for subscriber in subscribers:
            put_bounded_queue(subscriber, event)
        shutil.rmtree(runner.work_dir, ignore_errors=True)

    def _enforce_timeout(self, runner: RunnerProcess) -> None:
        deadline = runner.started_at + max(1, runner.timeout_seconds)
        while time.time() < deadline:
            with runner.lock:
                if runner.finished:
                    return
            time.sleep(min(1.0, max(0.05, deadline - time.time())))
        with runner.lock:
            if runner.finished:
                return
        self._emit(
            runner,
            {
                "type": "output",
                "stream": "stderr",
                "chunk": f"[atlas-runner] stopped after {runner.timeout_seconds}s timeout\n",
            },
            enforce_output_limit=False,
        )
        self._kill_container(runner)
        try:
            runner.process.kill()
        except OSError:
            pass

    def _kill_container(self, runner: RunnerProcess) -> None:
        binary = _docker_binary()
        if not binary:
            return
        self._remove_container_name(binary, runner.container_name)

    @staticmethod
    def _remove_container_name(binary: str, container_name: str) -> None:
        try:
            subprocess.run(
                [binary, "rm", "-f", container_name],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _cleanup_finished_runs(self) -> None:
        with self._lock:
            finished = [run_id for run_id, runner in self._runs.items() if runner.finished]
            for run_id in finished:
                self._runs.pop(run_id, None)

    def _cleanup_stale_containers(self, binary: str, *, preserve_active: bool = True) -> None:
        with self._lock:
            active_names = (
                {
                    runner.container_name
                    for runner in self._runs.values()
                    if not runner.finished
                }
                | set(self._starting_names)
                if preserve_active
                else set()
            )
        try:
            completed = subprocess.run(
                [
                    binary,
                    "ps",
                    "-a",
                    "--filter",
                    "label=atlas.runner=1",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return
        if completed.returncode != 0:
            return
        stale_names = [
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip() and line.strip() not in active_names
        ]
        for name in stale_names:
            labels = self._container_labels(binary, name)
            if labels is None:
                continue
            owner_id = labels.get("atlas.runner.owner_id", "").strip()
            owner_pid_raw = labels.get("atlas.runner.owner_pid", "").strip()
            try:
                owner_pid = int(owner_pid_raw)
            except ValueError:
                owner_pid = None

            owned_by_this_runner = owner_id == self._owner_id
            legacy_owned_by_this_process = not owner_id and owner_pid == os.getpid()
            stale_owner = owner_pid is None or not _process_is_alive(owner_pid)
            if owned_by_this_runner or legacy_owned_by_this_process or stale_owner:
                self._remove_container_name(binary, name)

    @staticmethod
    def _container_labels(binary: str, container_name: str) -> dict[str, str] | None:
        try:
            completed = subprocess.run(
                [
                    binary,
                    "inspect",
                    "--format",
                    "{{json .Config.Labels}}",
                    container_name,
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
        try:
            labels = json.loads(completed.stdout.strip() or "{}")
        except json.JSONDecodeError:
            return None
        if not isinstance(labels, dict):
            return None
        return {
            str(key): str(value)
            for key, value in labels.items()
            if isinstance(key, str) and value is not None
        }

    def shutdown(self) -> None:
        with self._lock:
            if self._shutting_down:
                return
            self._shutting_down = True
            runners = list(self._runs.values())
            self._runs.clear()

        binary = _docker_binary()
        for runner in runners:
            self._emit(
                runner,
                {
                    "type": "output",
                    "stream": "stderr",
                    "chunk": "[atlas-runner] shutting down\n",
                },
                enforce_output_limit=False,
            )
            if binary:
                self._remove_container_name(binary, runner.container_name)
            try:
                runner.process.kill()
            except OSError:
                pass
            shutil.rmtree(runner.work_dir, ignore_errors=True)

        if binary:
            self._cleanup_stale_containers(binary, preserve_active=False)

    close = shutdown


_runner_singleton: CodeRunner | None = None
_runner_lock = threading.Lock()


def get_runner() -> CodeRunner:
    global _runner_singleton
    with _runner_lock:
        if _runner_singleton is None:
            _runner_singleton = CodeRunner()
        return _runner_singleton


def shutdown_runner() -> None:
    with _runner_lock:
        runner = _runner_singleton
    if runner is not None:
        runner.shutdown()


def _shutdown_runner_at_exit() -> None:
    shutdown_runner()


atexit.register(_shutdown_runner_at_exit)
