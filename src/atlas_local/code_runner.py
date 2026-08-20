from __future__ import annotations

import atexit
import ast
import hashlib
import io
import json
import logging
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
import tokenize
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .run_contract import DEFAULT_SUBSCRIBER_QUEUE_SIZE, put_bounded_queue


logger = logging.getLogger(__name__)


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
    isolated_gui_preview: bool = False
    web_container_port: int | None = None
    requires_network: bool = False
    uses_apt: bool = False
    runtime: str | None = None
    requested_packages: tuple[str, ...] = ()
    unsupported_packages: tuple[str, ...] = ()


PYTHON_BASE_IMAGE = "docker.io/library/python:3.12-slim"
PYTHON_GUI_RUNTIME_NAME = "python-gui"
PYTHON_GUI_RUNTIME_VERSION = "1.0.1"
PYTHON_GUI_IMAGE = f"localhost/atlas-python-gui-runtime:{PYTHON_GUI_RUNTIME_VERSION}"
PYTHON_GUI_RUNTIME_LABEL = "com.atlas.runner.runtime"
PYTHON_GUI_RUNTIME_VERSION_LABEL = "com.atlas.runner.runtime.version"
PYTHON_GUI_RUNTIME_DEFINITION_LABEL = "com.atlas.runner.runtime.definition-sha256"
PYTHON_GUI_RUNTIME_CONTEXT = Path(__file__).resolve().parent / "runner_images" / "python_gui"
PYTHON_GUI_RUNTIME_ALLOWED_PACKAGES = {
    "numpy": "2.5.2",
    "pygame": "2.6.1",
}
LEGACY_PYTHON_GUI_IMAGES = (
    "localhost/atlas-python-gui-runtime:1.0.0",
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
    "util-linux",
    "fonts-dejavu",
    "fontconfig",
    "ca-certificates",
)
PYTHON_RUNNER_UID = 65534
PYTHON_PREVIEW_DISPLAY_UID = PYTHON_RUNNER_UID
PYTHON_PREVIEW_WEB_UID = PYTHON_RUNNER_UID
PYTHON_RUNNER_HOME = "/tmp/atlas-user"
PYTHON_RUNNER_SITE = "/tmp/atlas-python-site"
PYTHON_PREVIEW_DISPLAY_HOME = "/tmp/atlas-display-home"
PYTHON_PREVIEW_WEB_HOME = "/tmp/atlas-web-home"
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
PYTHON_REQUIREMENTS_HINT_LABELS = (
    "requirements",
    "requirement",
    "dependencies",
    "dependency",
    "packages",
    "package",
    "requires",
    "require",
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
    waiting_for_call = False
    call_depth = 0
    ignored_types = {
        tokenize.COMMENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.INDENT,
        tokenize.NL,
        tokenize.NEWLINE,
    }
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for token in tokens:
            if call_depth:
                if token.type == tokenize.OP:
                    if token.string == "(":
                        call_depth += 1
                    elif token.string == ")":
                        call_depth -= 1
                elif token.type == tokenize.STRING:
                    try:
                        value = ast.literal_eval(token.string)
                    except (SyntaxError, ValueError):
                        continue
                    if value == "--gui":
                        return ["--gui"]
                continue

            if waiting_for_call:
                if token.type in ignored_types:
                    continue
                if token.type == tokenize.OP and token.string == "(":
                    call_depth = 1
                    waiting_for_call = False
                    continue
                waiting_for_call = False

            if token.type == tokenize.NAME and token.string == "add_argument":
                waiting_for_call = True
    except (IndentationError, tokenize.TokenError):
        pass
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
        normalized = _strip_python_hint_comment(line)
        pip_hint = _python_pip_install_hint(normalized)
        if pip_hint is not None:
            packages.extend(_python_package_tokens_from_hint(pip_hint))

        requirements_hint = _python_requirements_hint(normalized)
        if requirements_hint is not None:
            hint = requirements_hint
            if _python_hint_declares_no_packages(hint):
                continue
            nested_pip_hint = _python_pip_install_hint(hint)
            if nested_pip_hint is not None:
                hint = nested_pip_hint
            packages.extend(_python_package_tokens_from_hint(hint.replace(",", " ")))
    return sorted(dict.fromkeys(packages))


def _strip_python_hint_comment(line: str) -> str:
    normalized = line.lstrip()
    while normalized:
        if normalized.startswith("//"):
            normalized = normalized[2:].lstrip()
        elif normalized.startswith("#"):
            normalized = normalized[1:].lstrip()
        else:
            break
    return normalized


def _skip_python_hint_whitespace(value: str, index: int) -> int:
    while index < len(value) and value[index].isspace():
        index += 1
    return index


def _require_python_hint_whitespace(value: str, index: int) -> int | None:
    end = _skip_python_hint_whitespace(value, index)
    return end if end > index else None


def _python_pip_install_hint(value: str) -> str | None:
    lowered = value.lower()
    for start in range(len(value)):
        if start and (value[start - 1].isalnum() or value[start - 1] in "_.-"):
            continue
        payload_start = _python_pip_install_payload_start(value, lowered, start)
        if payload_start is not None:
            return value[payload_start:]
    return None


def _python_pip_install_payload_start(value: str, lowered: str, start: int) -> int | None:
    index = start
    if lowered.startswith("%", index):
        index = _skip_python_hint_whitespace(value, index + 1)
        if not lowered.startswith("pip", index):
            return None
        index += len("pip")
    elif lowered.startswith("python3", index):
        index += len("python3")
        index = _require_python_hint_whitespace(value, index)
        if index is None or not lowered.startswith("-m", index):
            return None
        index = _require_python_hint_whitespace(value, index + len("-m"))
        if index is None or not lowered.startswith("pip", index):
            return None
        index += len("pip")
    elif lowered.startswith("python", index):
        index += len("python")
        index = _require_python_hint_whitespace(value, index)
        if index is None or not lowered.startswith("-m", index):
            return None
        index = _require_python_hint_whitespace(value, index + len("-m"))
        if index is None or not lowered.startswith("pip", index):
            return None
        index += len("pip")
    elif lowered.startswith("pip3", index):
        index += len("pip3")
    elif lowered.startswith("pip", index):
        index += len("pip")
    else:
        return None

    index = _require_python_hint_whitespace(value, index)
    if index is None or not lowered.startswith("install", index):
        return None
    index = _require_python_hint_whitespace(value, index + len("install"))
    if index is None or index >= len(value):
        return None
    return index


def _python_requirements_hint(value: str) -> str | None:
    normalized = value.lstrip()
    lowered = normalized.lower()
    for label in PYTHON_REQUIREMENTS_HINT_LABELS:
        if not lowered.startswith(label):
            continue
        index = len(label)
        if index < len(normalized) and (normalized[index].isalnum() or normalized[index] == "_"):
            continue
        index = _skip_python_hint_whitespace(normalized, index)
        if index < len(normalized) and normalized[index] == ":":
            index = _skip_python_hint_whitespace(normalized, index + 1)
        if index < len(normalized):
            return normalized[index:]
    return None


def _python_hint_declares_no_packages(hint: str) -> bool:
    normalized = " ".join(
        "".join(
            character if character.isascii() and character.isalnum() else " "
            for character in hint.lower()
        ).split()
    )
    return (
        normalized in {"none", "no", "na", "n a", "stdlib", "standard library"}
        or normalized.startswith("none ")
        or normalized.startswith("no external")
        or normalized.startswith("standard library")
        or normalized.startswith("python standard library")
    )


def _python_package_tokens_from_hint(hint: str) -> list[str]:
    cleaned = _truncate_python_package_hint(hint)
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


def _truncate_python_package_hint(hint: str) -> str:
    end = len(hint)
    for separator in ("&&", "||", ";", "|"):
        position = hint.find(separator)
        if position >= 0:
            end = min(end, position)
    instruction_start = _python_hint_instruction_start(hint[:end])
    if instruction_start is not None:
        end = min(end, instruction_start)
    return hint[:end]


def _python_hint_instruction_start(hint: str) -> int | None:
    index = 0
    while index < len(hint):
        if not hint[index].isspace():
            index += 1
            continue
        whitespace_start = index
        index = _skip_python_hint_whitespace(hint, index)
        word_start = index
        while index < len(hint) and hint[index].isalpha():
            index += 1
        if index == word_start:
            index += 1
            continue
        first_word = hint[word_start:index].lower()
        if index < len(hint) and (hint[index].isalnum() or hint[index] == "_"):
            continue
        if first_word in {"then", "before", "after"}:
            return whitespace_start
        if first_word not in {"and", "to"}:
            continue
        second_start = _require_python_hint_whitespace(hint, index)
        if second_start is None:
            continue
        second_end = second_start
        while second_end < len(hint) and hint[second_end].isalpha():
            second_end += 1
        second_word = hint[second_start:second_end].lower()
        if second_end < len(hint) and (hint[second_end].isalnum() or hint[second_end] == "_"):
            continue
        if (first_word, second_word) in {("and", "then"), ("to", "run")}:
            return whitespace_start
    return None


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


def _python_dependency_repair_script(
    imports: set[str],
    modules: set[str],
    *,
    install_target: str | None = None,
    offline_runtime: bool = False,
) -> str:
    repair_map = json.dumps(_python_repair_package_map(imports, modules), sort_keys=True)
    safe_spec = PYTHON_SAFE_PACKAGE_SPEC_RE.pattern
    bundled_runtime_packages = ", ".join(
        f"{name}=={version}"
        for name, version in sorted(PYTHON_GUI_RUNTIME_ALLOWED_PACKAGES.items())
    )
    offline_runtime_detail = (
        "Atlas did not grant submitted code network access. Bundled dependencies: "
        f"{bundled_runtime_packages}. Update the trusted runtime image to add more."
    )
    install_target_args = (
        f", '--target', {install_target!r}, '--upgrade'"
        if install_target
        else ", '--root-user-action=ignore'"
    )
    install_lines = (
        [
            "def install_packages(packages):",
            (
                "    print('[atlas-runner] offline GUI runtime does not include: ' + "
                "' '.join(packages) + '. ' "
                f"+ {offline_runtime_detail!r}, "
                "file=sys.stderr, flush=True)"
            ),
            "    return 1",
        ]
        if offline_runtime
        else [
            "def install_packages(packages):",
            "    print('[atlas-runner] detected missing Python module; installing repair packages: ' + ' '.join(packages), flush=True)",
            (
                "    command = [sys.executable, '-m', 'pip', 'install', '--quiet', "
                "'--no-input', '--disable-pip-version-check'"
                f"{install_target_args}, *packages]"
            ),
            "    return subprocess.call(command)",
        ]
    )
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
            *install_lines,
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


def _python_unprivileged_command(
    command: list[str],
    *,
    uid: int,
    home: str,
    environment: dict[str, str] | None = None,
) -> str:
    command_environment = {
        "HOME": home,
        "USER": str(uid),
        "LOGNAME": str(uid),
        **(environment or {}),
    }
    if uid == PYTHON_RUNNER_UID:
        return shlex.join(
            [
                "env",
                *(f"{key}={value}" for key, value in command_environment.items()),
                *command,
            ]
        )
    return shlex.join(
        [
            "setpriv",
            f"--reuid={uid}",
            f"--regid={uid}",
            "--clear-groups",
            "--no-new-privs",
            "--inh-caps=-all",
            "--ambient-caps=-all",
            "--pdeathsig=TERM",
            "env",
            *(f"{key}={value}" for key, value in command_environment.items()),
            *command,
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


def _canonical_python_package_name(package_spec: str) -> tuple[str, str]:
    match = re.match(
        r"^([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[A-Za-z0-9_,.-]+\])?(.*)$",
        package_spec.strip(),
    )
    if not match:
        return "", package_spec.strip()
    name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
    return name, match.group(2).strip()


def _python_gui_runtime_unsupported_packages(packages: Iterable[str]) -> tuple[str, ...]:
    unsupported: list[str] = []
    for package in packages:
        name, constraint = _canonical_python_package_name(package)
        included_version = PYTHON_GUI_RUNTIME_ALLOWED_PACKAGES.get(name)
        if included_version is None or constraint not in {"", f"=={included_version}", f"==={included_version}"}:
            unsupported.append(package)
    return tuple(sorted(dict.fromkeys(unsupported)))


def _reserve_host_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


LANGUAGES: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        image=PYTHON_BASE_IMAGE,
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
RUNNER_INTERNAL_NETWORK_PATTERN = re.compile(
    rf"^{re.escape(RUNNER_INTERNAL_NETWORK)}-[0-9a-f]{{16}}$"
)
MAX_STALE_NETWORK_CLEANUP = 64
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
        try:
            subprocess.run(
                [binary, "image", "rm", image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue


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
    unsupported_packages = _python_gui_runtime_unsupported_packages(pip_packages) if gui else ()
    gui_args = " ".join(_python_gui_args(code))
    gui_args_suffix = f" {gui_args}" if gui_args else ""
    runner_environment = {
        "DISPLAY": ":99",
        "PYTHONPATH": PYTHON_RUNNER_SITE,
        "PYTHONPYCACHEPREFIX": f"{PYTHON_RUNNER_HOME}/pycache",
        "XDG_RUNTIME_DIR": PYTHON_RUNNER_HOME,
    }
    script_parts = [
        "set -e",
        "umask 022",
        "cp /work/main.py /tmp/main.py",
        "chmod 0444 /tmp/main.py",
        "cd /tmp",
        (
            "export DEBIAN_FRONTEND=noninteractive DISPLAY=:99 SCREEN_GEOMETRY=1280x800x24 "
            "VNC_PORT=5900 NOVNC_PORT=6080 PYTHONUNBUFFERED=1 SDL_AUDIODRIVER=dummy "
            "PYGAME_HIDE_SUPPORT_PROMPT=1 ALSA_CONFIG_PATH=/dev/null"
        ),
        _python_dependency_repair_script(
            imports,
            modules,
            install_target=PYTHON_RUNNER_SITE if gui else None,
            offline_runtime=gui,
        ),
        "chmod 0444 /tmp/atlas_python_repair.py",
    ]
    if gui:
        script_parts.append(
            f"echo '[atlas-runner] using prepared offline Python GUI runtime {PYTHON_GUI_RUNTIME_VERSION}'"
        )
    if apt_packages and not gui:
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
        display_environment = {
            "DISPLAY": ":99",
            "XDG_RUNTIME_DIR": PYTHON_PREVIEW_DISPLAY_HOME,
        }
        display_commands = {
            "xvfb": _python_unprivileged_command(
                [
                    "Xvfb",
                    ":99",
                    "-screen",
                    "0",
                    "1280x800x24",
                    "-ac",
                    "+extension",
                    "GLX",
                    "+render",
                    "-noreset",
                ],
                uid=PYTHON_PREVIEW_DISPLAY_UID,
                home=PYTHON_PREVIEW_DISPLAY_HOME,
                environment=display_environment,
            ),
            "fluxbox": _python_unprivileged_command(
                ["fluxbox", "-rc", "/tmp/atlas-fluxbox-init"],
                uid=PYTHON_PREVIEW_DISPLAY_UID,
                home=PYTHON_PREVIEW_DISPLAY_HOME,
                environment=display_environment,
            ),
            "x11vnc": _python_unprivileged_command(
                [
                    "x11vnc",
                    "-display",
                    ":99",
                    "-nopw",
                    "-forever",
                    "-shared",
                    "-rfbport",
                    "5900",
                    "-quiet",
                ],
                uid=PYTHON_PREVIEW_DISPLAY_UID,
                home=PYTHON_PREVIEW_DISPLAY_HOME,
                environment=display_environment,
            ),
            "websockify": _python_unprivileged_command(
                [
                    "websockify",
                    "--web",
                    "/usr/share/novnc",
                    "6080",
                    "127.0.0.1:5900",
                ],
                uid=PYTHON_PREVIEW_WEB_UID,
                home=PYTHON_PREVIEW_WEB_HOME,
            ),
        }
        script_parts.extend(
            [
                (
                    "install -d -m 0700 "
                    f"{PYTHON_RUNNER_HOME} {PYTHON_RUNNER_SITE}"
                ),
                (
                    f"install -d -m 0700 {PYTHON_PREVIEW_DISPLAY_HOME}"
                ),
                (
                    f"install -d -m 0700 {PYTHON_PREVIEW_WEB_HOME}"
                ),
                (
                    "printf '%s\\n' 'session.screen0.workspaces: 1' 'session.screen0.workspaceNames: Main' "
                    "'session.screen0.toolbar.tools: workspacename, iconbar, systemtray, clock' "
                    "> /tmp/atlas-fluxbox-init"
                ),
                "chmod 0444 /tmp/atlas-fluxbox-init",
                "rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 2>/dev/null || true",
                f"{display_commands['xvfb']} >/tmp/xvfb.log 2>&1 & sleep 0.6",
                (
                    f"{display_commands['fluxbox']} >/tmp/fluxbox.log 2>&1 & "
                    f"{display_commands['x11vnc']} >/tmp/x11vnc.log 2>&1 & "
                    f"{display_commands['websockify']} >/tmp/novnc.log 2>&1 & "
                    "sleep 0.4"
                ),
                "echo '[atlas-runner] GUI ready on port 6080'",
            ],
        )
    if pip_packages and not gui:
        pip_args = shlex.join(pip_packages)
        script_parts.append(f"echo {shlex.quote('[atlas-runner] installing Python packages: ' + ' '.join(pip_packages))}")
        pip_command = [
            "python",
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-input",
            "--disable-pip-version-check",
        ]
        if gui:
            pip_command.extend(["--target", PYTHON_RUNNER_SITE, "--upgrade"])
            pip_install = _python_unprivileged_command(
                [*pip_command, *pip_packages],
                uid=PYTHON_RUNNER_UID,
                home=PYTHON_RUNNER_HOME,
                environment=runner_environment,
            )
        else:
            pip_install = f"{shlex.join(pip_command)} --root-user-action=ignore {pip_args}"
        script_parts.append(
            f"{pip_install} || echo "
            "'[atlas-runner] initial package install failed; continuing to run and repair imports if possible'"
        )
    if web_port:
        script_parts.append(f"echo {shlex.quote(f'[atlas-runner] web preview will use container port {web_port}')}")
        script_parts.append(
            f"export PORT={web_port} HOST=0.0.0.0 GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT={web_port}"
        )
    if terminal:
        terminal_command = (
            "cd /tmp; export TERM=xterm-256color; "
            f"python /tmp/atlas_python_repair.py python -u /tmp/main.py{gui_args_suffix}; "
            "status=$?; echo; echo \"[atlas-runner] terminal app exited with status $status\"; "
            "sleep 2; exit $status"
        )
        script_parts.append("echo '[atlas-runner] starting terminal UI in virtual display'")
        script_parts.append(
            _python_unprivileged_command(
                [
                    "xterm",
                    "-geometry",
                    "120x34",
                    "-fa",
                    "DejaVu Sans Mono",
                    "-fs",
                    "12",
                    "-title",
                    "Atlas Terminal",
                    "-e",
                    "sh",
                    "-c",
                    terminal_command,
                ],
                uid=PYTHON_RUNNER_UID,
                home=PYTHON_RUNNER_HOME,
                environment=runner_environment,
            )
        )
    elif web_command:
        command = ["python", "/tmp/atlas_python_repair.py", "sh", "-c", web_command]
        if gui:
            script_parts.append(
                _python_unprivileged_command(
                    command,
                    uid=PYTHON_RUNNER_UID,
                    home=PYTHON_RUNNER_HOME,
                    environment=runner_environment,
                )
            )
        else:
            script_parts.append(shlex.join(command))
    else:
        command = [
            "python",
            "/tmp/atlas_python_repair.py",
            "python",
            "-u",
            "/tmp/main.py",
            *_python_gui_args(code),
        ]
        if gui:
            script_parts.append(
                _python_unprivileged_command(
                    command,
                    uid=PYTHON_RUNNER_UID,
                    home=PYTHON_RUNNER_HOME,
                    environment=runner_environment,
                )
            )
        else:
            script_parts.append(shlex.join(command))
    ports: dict[int, int] = {}
    if gui:
        ports[_reserve_host_port()] = NOVNC_CONTAINER_PORT
    if web_port:
        ports[_reserve_host_port()] = web_port
    return RunPlan(
        image=PYTHON_GUI_IMAGE if gui else PYTHON_BASE_IMAGE,
        filename="main.py",
        command=["sh", "-c", "\n".join(script_parts)],
        ports=ports,
        gui=gui,
        isolated_gui_preview=gui,
        web_container_port=web_port,
        requires_network=bool(not gui and (apt_packages or pip_packages or repair_may_need_network)),
        uses_apt=bool(apt_packages and not gui),
        runtime=PYTHON_GUI_RUNTIME_NAME if gui else None,
        requested_packages=tuple(pip_packages),
        unsupported_packages=unsupported_packages,
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


def _inspect_internal_network(
    binary: str,
    network_name: str = RUNNER_INTERNAL_NETWORK,
) -> bool | None:
    try:
        completed = subprocess.run(
            [
                binary,
                "network",
                "inspect",
                network_name,
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


def _ensure_internal_network(
    binary: str,
    network_name: str = RUNNER_INTERNAL_NETWORK,
    *,
    labels: dict[str, str] | None = None,
) -> None:
    internal = _inspect_internal_network(binary, network_name)
    if internal is True:
        existing_labels = _network_labels(binary, network_name)
        expected_labels = {"atlas.runner.network": "1", **(labels or {})}
        if existing_labels is None or any(
            existing_labels.get(key) != value
            for key, value in expected_labels.items()
        ):
            raise RuntimeError(
                f"Docker network '{network_name}' already exists but is not owned "
                "by this Atlas runner."
            )
        return
    if internal is False:
        raise RuntimeError(
            f"Docker network '{network_name}' already exists but is not internal; "
            "remove or rename it before running an isolated preview."
        )

    label_args: list[str] = []
    for key, value in (labels or {}).items():
        label_args.extend(["--label", f"{key}={value}"])
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
                *label_args,
                network_name,
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

    if completed.returncode == 0:
        return
    if _inspect_internal_network(binary, network_name) is True:
        existing_labels = _network_labels(binary, network_name)
        expected_labels = {"atlas.runner.network": "1", **(labels or {})}
        if existing_labels is not None and all(
            existing_labels.get(key) == value
            for key, value in expected_labels.items()
        ):
            return
    detail = (completed.stderr or completed.stdout or "").strip()
    suffix = f" Details: {detail.splitlines()[-1]}" if detail else ""
    raise RuntimeError(f"Failed to create the isolated preview network.{suffix}")


def _remove_internal_network(binary: str, network_name: str) -> None:
    if (
        network_name != RUNNER_INTERNAL_NETWORK
        and not RUNNER_INTERNAL_NETWORK_PATTERN.fullmatch(network_name)
    ):
        return
    labels = _network_labels(binary, network_name)
    if labels is None or labels.get("atlas.runner.network") != "1":
        return
    try:
        subprocess.run(
            [binary, "network", "rm", network_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def _network_labels(binary: str, network_name: str) -> dict[str, str] | None:
    try:
        completed = subprocess.run(
            [
                binary,
                "network",
                "inspect",
                network_name,
                "--format",
                "{{json .Labels}}",
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


def _python_gui_runtime_definition_hash() -> str:
    digest = hashlib.sha256()
    for name in ("Dockerfile", ".dockerignore", "runtime-requirements.txt"):
        path = PYTHON_GUI_RUNTIME_CONTEXT / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _inspect_python_gui_runtime(binary: str) -> tuple[bool, int | None, str | None]:
    try:
        completed = subprocess.run(
            [binary, "image", "inspect", PYTHON_GUI_IMAGE],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.warning("Could not inspect the trusted Python GUI runtime image.", exc_info=True)
        return False, None, "Docker could not inspect the Python GUI runtime image."
    if completed.returncode != 0:
        return False, None, None
    try:
        payload = json.loads(completed.stdout)
        image = payload[0]
        labels = image.get("Config", {}).get("Labels") or image.get("Labels") or {}
        size = int(image.get("Size")) if image.get("Size") is not None else None
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False, None, "The local runtime image metadata is unreadable."
    expected_hash = _python_gui_runtime_definition_hash()
    ready = (
        labels.get(PYTHON_GUI_RUNTIME_LABEL) == PYTHON_GUI_RUNTIME_NAME
        and labels.get(PYTHON_GUI_RUNTIME_VERSION_LABEL) == PYTHON_GUI_RUNTIME_VERSION
        and labels.get(PYTHON_GUI_RUNTIME_DEFINITION_LABEL) == expected_hash
    )
    if not ready:
        return False, size, "The local runtime image is stale and must be prepared again."
    return True, size, None


class PythonGuiRuntimeManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "missing"
        self._message = "Python GUI runtime is not prepared."
        self._error: str | None = None
        self._progress = 0.0
        self._logs: deque[str] = deque(maxlen=80)
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._started_at: float | None = None
        self._completed_at: float | None = None
        self._image_size_bytes: int | None = None

    def status(self, binary: str | None = None) -> dict[str, Any]:
        binary = binary or _docker_binary()
        if not binary:
            return self._snapshot(
                state="unavailable",
                message="Docker CLI was not found on PATH.",
            )
        with self._lock:
            if self._state == "preparing":
                return self._snapshot_locked()
        ready, size, inspect_error = _inspect_python_gui_runtime(binary)
        with self._lock:
            if ready:
                self._state = "ready"
                self._message = "Secure offline Python GUI runtime is ready."
                self._error = None
                self._progress = 1.0
                self._image_size_bytes = size
            elif self._state != "failed":
                self._state = "missing"
                self._message = inspect_error or "Python GUI runtime is not prepared."
                self._progress = 0.0
                self._image_size_bytes = size
            return self._snapshot_locked()

    def prepare(self) -> dict[str, Any]:
        binary = _docker_binary()
        if not binary:
            return {
                **self.status(binary),
                "started": False,
            }
        current = self.status(binary)
        if current["state"] == "ready":
            return {**current, "started": False}
        with self._lock:
            if self._state == "preparing":
                return {**self._snapshot_locked(), "started": False}
            self._state = "preparing"
            self._message = (
                "Starting trusted runtime preparation. Submitted code is not mounted or executed."
            )
            self._error = None
            self._progress = 0.02
            self._logs.clear()
            self._started_at = time.time()
            self._completed_at = None
            thread = threading.Thread(
                target=self._build,
                args=(binary,),
                name="atlas-python-gui-runtime-build",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return {**self._snapshot_locked(), "started": True}

    def shutdown(self) -> None:
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    def is_preparing(self) -> bool:
        with self._lock:
            return self._state == "preparing"

    def _build(self, binary: str) -> None:
        definition_hash = _python_gui_runtime_definition_hash()
        command = [
            binary,
            "build",
            "--pull",
            "--file",
            str(PYTHON_GUI_RUNTIME_CONTEXT / "Dockerfile"),
            "--tag",
            PYTHON_GUI_IMAGE,
            "--build-arg",
            f"ATLAS_RUNTIME_DEFINITION_SHA256={definition_hash}",
            str(PYTHON_GUI_RUNTIME_CONTEXT),
        ]
        creation_flags = 0x08000000 if sys.platform == "win32" else 0
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                bufsize=1,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creation_flags,
            )
            with self._lock:
                self._process = process
            if process.stdout is not None:
                for raw_line in process.stdout:
                    self._record_build_line(raw_line)
            return_code = process.wait()
            ready, size, inspect_error = _inspect_python_gui_runtime(binary)
            with self._lock:
                self._process = None
                self._completed_at = time.time()
                self._image_size_bytes = size
                if return_code == 0 and ready:
                    self._state = "ready"
                    self._progress = 1.0
                    self._message = "Secure offline Python GUI runtime is ready."
                    self._error = None
                else:
                    self._state = "failed"
                    self._progress = 0.0
                    detail = inspect_error or self._last_log_locked() or f"container build exited {return_code}"
                    self._error = (
                        "Python GUI runtime preparation failed. Check Docker connectivity and retry. "
                        f"Details: {detail}"
                    )
                    self._message = "Runtime preparation failed."
        except OSError:
            logger.exception("Could not start the trusted Python GUI runtime build.")
            with self._lock:
                self._process = None
                self._completed_at = time.time()
                self._state = "failed"
                self._progress = 0.0
                self._message = "Runtime preparation failed."
                self._error = (
                    "Could not start the trusted container build. "
                    "Check Docker availability and retry."
                )

    def _record_build_line(self, raw_line: str) -> None:
        line = raw_line.strip()
        if not line:
            return
        line = line[-500:]
        progress = None
        step_match = re.search(r"\bSTEP\s+(\d+)/(\d+)\b", line, re.IGNORECASE)
        if step_match:
            current, total = (int(value) for value in step_match.groups())
            if total > 0:
                progress = min(0.95, max(0.05, current / total * 0.9))
        with self._lock:
            self._logs.append(line)
            self._message = line
            if progress is not None:
                self._progress = progress

    def _snapshot(self, *, state: str, message: str) -> dict[str, Any]:
        with self._lock:
            payload = self._snapshot_locked()
        payload["state"] = state
        payload["message"] = message
        return payload

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "name": PYTHON_GUI_RUNTIME_NAME,
            "version": PYTHON_GUI_RUNTIME_VERSION,
            "image": PYTHON_GUI_IMAGE,
            "state": self._state,
            "message": self._message,
            "error": self._error,
            "progress": self._progress,
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "image_size_bytes": self._image_size_bytes,
            "bundled_packages": [
                f"{name}=={version}"
                for name, version in sorted(PYTHON_GUI_RUNTIME_ALLOWED_PACKAGES.items())
            ],
            "execution_network": "internal-preview-only",
            "submitted_code_used_during_preparation": False,
            "log_tail": list(self._logs)[-20:],
        }

    def _last_log_locked(self) -> str | None:
        return self._logs[-1] if self._logs else None


_python_gui_runtime_manager = PythonGuiRuntimeManager()


def python_gui_runtime_status() -> dict[str, Any]:
    return _python_gui_runtime_manager.status()


def prepare_python_gui_runtime() -> dict[str, Any]:
    return _python_gui_runtime_manager.prepare()


def _unsupported_python_gui_runtime_message(plan: RunPlan) -> str:
    requested = ", ".join(plan.unsupported_packages)
    bundled = ", ".join(
        f"{name}=={version}"
        for name, version in sorted(PYTHON_GUI_RUNTIME_ALLOWED_PACKAGES.items())
    )
    return (
        "The secure offline Python GUI runtime does not include these requested dependencies: "
        f"{requested}. Bundled third-party packages: {bundled}. Atlas will not grant submitted "
        "code network access or install arbitrary packages at run time. Remove the dependency or "
        "add a reviewed, pinned package to the trusted runtime image and bump its version."
    )


def prepare_runner_runtime(language: str, code: str) -> dict[str, Any]:
    resolved = resolve_language(language)
    if not resolved:
        raise RuntimeError(f"Language '{language}' is not supported.")
    if resolved in CLIENT_LANGUAGES:
        return {"required": False, "runtime": None, "started": False}
    plan = resolve_plan(resolved, code)
    if plan.unsupported_packages:
        raise RuntimeError(_unsupported_python_gui_runtime_message(plan))
    if plan.runtime != PYTHON_GUI_RUNTIME_NAME:
        return {"required": False, "runtime": None, "started": False}
    runtime = prepare_python_gui_runtime()
    return {
        "required": True,
        "runtime": runtime,
        "started": bool(runtime.get("started", False)),
    }


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
        logger.warning("Docker status probe timed out.")
        return {"available": False, "reason": "Docker is installed but the daemon did not respond in time."}
    except OSError:
        logger.warning("Docker status probe could not invoke Docker.", exc_info=True)
        return {
            "available": False,
            "reason": "Docker could not be checked. Verify Docker Desktop is installed and try again.",
        }

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if detail:
            logger.warning(
                "Docker status probe exited with code %s: %r",
                completed.returncode,
                detail[-2048:],
            )
        else:
            logger.warning(
                "Docker status probe exited with code %s without diagnostic output.",
                completed.returncode,
            )
        return {
            "available": False,
            "reason": "Docker Desktop is installed but not running. Start Docker Desktop and try again.",
        }
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
        if plan.unsupported_packages:
            raise RuntimeError(_unsupported_python_gui_runtime_message(plan))
        if plan.runtime == PYTHON_GUI_RUNTIME_NAME:
            runtime = _python_gui_runtime_manager.status(binary)
            if runtime.get("state") != "ready":
                raise RuntimeError(
                    "The secure offline Python GUI runtime is not ready. "
                    "Prepare it from the runner and wait for preparation to finish before retrying."
                )
        if plan.gui and not plan.isolated_gui_preview:
            raise RuntimeError(
                "Refusing to publish a GUI preview without the runner privilege boundary."
            )
        run_id = uuid.uuid4().hex[:16]
        container_name = f"atlas-run-{run_id}"
        configured_network = _configured_runner_network()
        network_policy = _runner_network_policy(plan)
        network = (
            f"{RUNNER_INTERNAL_NETWORK}-{run_id}"
            if network_policy == RUNNER_INTERNAL_NETWORK
            else network_policy
        )
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
        internal_network_created = False
        try:
            self._cleanup_stale_containers(binary)
            self._cleanup_stale_networks(binary)
            _remove_legacy_python_gui_images(binary)
            if network_policy == RUNNER_INTERNAL_NETWORK:
                _ensure_internal_network(
                    binary,
                    network,
                    labels={
                        "atlas.runner.owner_id": self._owner_id,
                        "atlas.runner.owner_pid": str(os.getpid()),
                        "atlas.runner.run_id": run_id,
                    },
                )
                internal_network_created = True

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
            if plan.web_container_port and not outbound_network:
                self._emit(
                    runner,
                    {
                        "type": "output",
                        "stream": "stderr",
                        "chunk": (
                            "[atlas-runner] browser web preview is disabled while outbound "
                            f"network access is blocked. Set {RUNNER_NETWORK_ENV}=bridge only "
                            "if you explicitly want the preview and its browser-side network access.\n"
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
            if plan.web_container_port and outbound_network:
                for host_port, container_port in plan.ports.items():
                    if container_port == plan.web_container_port:
                        response["web_url"] = f"http://127.0.0.1:{host_port}/"
                        break
            elif plan.web_container_port:
                response["web_preview_disabled"] = True
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
                if internal_network_created:
                    _remove_internal_network(binary, network)

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

    def active_run_count(self) -> int:
        with self._lock:
            return len(self._starting_names) + sum(
                1 for runner in self._runs.values() if not runner.finished
            )

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
        binary = _docker_binary()
        if binary:
            _remove_internal_network(binary, runner.network)

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

    def _cleanup_stale_networks(
        self,
        binary: str,
        *,
        preserve_active: bool = True,
    ) -> None:
        with self._lock:
            if preserve_active:
                active_networks = {
                    runner.network
                    for runner in self._runs.values()
                    if not runner.finished
                    and RUNNER_INTERNAL_NETWORK_PATTERN.fullmatch(runner.network)
                }
                for container_name in self._starting_names:
                    if not container_name.startswith("atlas-run-"):
                        continue
                    run_id = container_name.removeprefix("atlas-run-")
                    network_name = f"{RUNNER_INTERNAL_NETWORK}-{run_id}"
                    if RUNNER_INTERNAL_NETWORK_PATTERN.fullmatch(network_name):
                        active_networks.add(network_name)
            else:
                active_networks = set()
        try:
            completed = subprocess.run(
                [
                    binary,
                    "network",
                    "ls",
                    "--filter",
                    "label=atlas.runner.network=1",
                    "--format",
                    "{{.Name}}",
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

        candidates: list[str] = []
        for line in completed.stdout.splitlines():
            name = line.strip()
            if (
                (
                    name == RUNNER_INTERNAL_NETWORK
                    or RUNNER_INTERNAL_NETWORK_PATTERN.fullmatch(name)
                )
                and name not in active_networks
            ):
                candidates.append(name)
                if len(candidates) >= MAX_STALE_NETWORK_CLEANUP:
                    break

        for name in candidates:
            labels = _network_labels(binary, name)
            if labels is None or labels.get("atlas.runner.network") != "1":
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
                _remove_internal_network(binary, name)

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
        _python_gui_runtime_manager.shutdown()
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
                _remove_internal_network(binary, runner.network)
            try:
                runner.process.kill()
            except OSError:
                pass
            shutil.rmtree(runner.work_dir, ignore_errors=True)

        if binary:
            self._cleanup_stale_containers(binary, preserve_active=False)
            self._cleanup_stale_networks(binary, preserve_active=False)

    close = shutdown


_runner_singleton: CodeRunner | None = None
_runner_lock = threading.Lock()


def get_runner() -> CodeRunner:
    global _runner_singleton
    with _runner_lock:
        if _runner_singleton is None:
            _runner_singleton = CodeRunner()
        return _runner_singleton


def runner_activity_status() -> dict[str, int | bool]:
    with _runner_lock:
        runner = _runner_singleton
    active_runs = runner.active_run_count() if runner is not None else 0
    runtime_preparing = _python_gui_runtime_manager.is_preparing()
    return {
        "busy": active_runs > 0 or runtime_preparing,
        "active_runs": active_runs,
        "runtime_preparing": runtime_preparing,
    }


def shutdown_runner() -> None:
    _python_gui_runtime_manager.shutdown()
    with _runner_lock:
        runner = _runner_singleton
    if runner is not None:
        runner.shutdown()


def _shutdown_runner_at_exit() -> None:
    shutdown_runner()


atexit.register(_shutdown_runner_at_exit)
