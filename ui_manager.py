"""Manage the NeMo Agent Toolkit web UI as a background process from a notebook.

The UI is a separate Next.js app (NVIDIA/NeMo-Agent-Toolkit-UI). This module
clones it on first use, installs dependencies, and runs the dev server against
the local `nat serve` backend.

Usage:
    from ui_manager import ui_manager

    ui_manager.start()
    ui_manager.show_ui_link()
    ...
    ui_manager.stop()
"""

import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

REPO_URL = "https://github.com/NVIDIA/NeMo-Agent-Toolkit-UI.git"
UI_DIR = Path(__file__).parent / "NeMo-Agent-Toolkit-UI"
UI_PORT = 3000
BACKEND_PORT = 8000

# Display-only strings. These name the assistant in the UI chrome; they do not
# affect the workflow, which is defined entirely by config.yml.
DEFAULT_BRANDING = {
    "NEXT_PUBLIC_NAT_WORKFLOW": "Climate Science Assistant",
    "NEXT_PUBLIC_NAT_GREETING_TITLE": "Hi, I'm your Climate Science Assistant",
    "NEXT_PUBLIC_NAT_GREETING_SUBTITLE": (
        "Ask me about climate data, weather patterns, and temperature trends."
    ),
    "NEXT_PUBLIC_NAT_INPUT_PLACEHOLDER": "Ask about climate science...",
    "NEXT_PUBLIC_NAT_WELCOME_MESSAGE_ON": "true",
}


def _port_open(port, host="127.0.0.1", timeout=1.0):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


class UIManager:
    def __init__(
        self,
        ui_dir=UI_DIR,
        port=UI_PORT,
        backend_port=BACKEND_PORT,
        branding=None,
    ):
        self.ui_dir = Path(ui_dir)
        self.port = port
        self.backend_port = backend_port
        self.branding = DEFAULT_BRANDING if branding is None else branding
        self.process = None
        self.log_path = self.ui_dir.parent / "ui_server.log"

    # -- setup ---------------------------------------------------------

    MIN_NODE = (20, 9, 0)

    def _check_node(self):
        if shutil.which("npm") is None:
            raise RuntimeError(
                "npm not found on PATH. Install Node.js 20.9+ (e.g. `nvm install 22`) "
                "and restart the Jupyter kernel so it picks up the new PATH."
            )

        raw = subprocess.run(
            ["node", "-v"], capture_output=True, text=True, check=True
        ).stdout.strip()
        version = tuple(int(part) for part in raw.lstrip("v").split(".")[:3])
        if version < self.MIN_NODE:
            want = ".".join(str(part) for part in self.MIN_NODE)
            raise RuntimeError(
                f"Node {raw} is too old; the UI's dependencies require >= {want}.\n"
                f"Fix with:  nvm install 22 && nvm alias default 22\n"
                f"Then restart the Jupyter kernel so it picks up the new version."
            )

    def _clone(self):
        if self.ui_dir.exists():
            return
        print(f"Cloning UI from {REPO_URL} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(self.ui_dir)],
            check=True,
        )

    def _install(self, timeout=900):
        # npm writes this marker only after a complete install, so it is a
        # reliable signal where a bare node_modules/ check is not: an install
        # that died midway leaves a populated but unusable tree.
        if (self.ui_dir / "node_modules" / ".package-lock.json").exists():
            return

        if (self.ui_dir / "node_modules").exists():
            print("Found an incomplete node_modules; removing it first ...")
            shutil.rmtree(self.ui_dir / "node_modules", ignore_errors=True)

        print("Installing UI dependencies (several minutes the first time) ...")
        try:
            subprocess.run(
                ["npm", "install"], cwd=self.ui_dir, check=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"npm install exceeded {timeout}s and was stopped. Re-run the cell, "
                "or run `npm install` in the UI directory from a terminal to watch it."
            )

    def _write_env(self):
        """Point the UI at the local NAT server.

        The repo ships a `.env` with sensible defaults; `.env.local` takes
        precedence and is where we override the backend address and port.
        The UI runs a gateway proxy on PORT that forwards to Next.js on
        NEXT_INTERNAL_URL, so PORT is the one you open in a browser.
        """
        lines = [
            f"NAT_BACKEND_URL=http://127.0.0.1:{self.backend_port}",
            f"PORT={self.port}",
            "",
        ]
        for key, value in self.branding.items():
            # Quote anything with spaces/apostrophes so dotenv keeps it intact.
            lines.append(f'{key}="{value}"' if " " in value else f"{key}={value}")

        content = "\n".join(lines) + "\n"
        env_file = self.ui_dir / ".env.local"
        changed = (
            not env_file.exists() or env_file.read_text() != content
        )
        env_file.write_text(content)
        return changed

    # -- lifecycle -----------------------------------------------------

    def start(self, wait=120, restart=False):
        self._check_node()
        self._clone()
        self._install()
        # NEXT_PUBLIC_* values are inlined when Next.js compiles, so a config
        # change only takes effect on a fresh server.
        config_changed = self._write_env()

        if _port_open(self.port):
            if not (restart or config_changed):
                print(f"UI already running on port {self.port}.")
                return
            reason = "config changed" if config_changed else "restart requested"
            print(f"Restarting UI on port {self.port} ({reason}) ...")
            self.stop(quiet=True)

        if not _port_open(self.backend_port):
            print(
                f"Warning: nothing is listening on port {self.backend_port}. "
                "Start the NAT server cell first, or the UI will have no backend."
            )

        env = os.environ.copy()
        env["PORT"] = str(self.port)

        log = open(self.log_path, "w")
        print(f"Starting UI on port {self.port} (log: {self.log_path}) ...")
        self.process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=self.ui_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

        deadline = time.time() + wait
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"UI process exited with code {self.process.returncode}. "
                    f"See {self.log_path} for details."
                )
            if _port_open(self.port):
                print("UI is up.")
                return
            time.sleep(1)

        raise TimeoutError(
            f"UI did not start within {wait}s. Check {self.log_path}."
        )

    def show_ui_link(self):
        url = f"http://localhost:{self.port}"
        name = self.branding.get("NEXT_PUBLIC_NAT_WORKFLOW", "NeMo Agent Toolkit")
        try:
            from IPython.display import HTML, display

            display(
                HTML(
                    f'<p>{name}: '
                    f'<a href="{url}" target="_blank">{url}</a></p>'
                )
            )
        except ImportError:
            print(f"{name}: {url}")
        return url

    def logs(self, lines=40):
        if not self.log_path.exists():
            print("No log file yet.")
            return
        with open(self.log_path) as handle:
            print("".join(handle.readlines()[-lines:]))

    def stop(self, quiet=False):
        if self.process is not None and self.process.poll() is None:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            self.process = None
        elif _port_open(self.port):
            # Started by an earlier kernel session; fall back to the port.
            subprocess.run(["pkill", "-f", "proxy/server.js"], check=False)
            subprocess.run(["pkill", "-f", "next dev -p 3099"], check=False)
            self.process = None
        else:
            if not quiet:
                print("UI is not running.")
            return

        for _ in range(15):
            if not _port_open(self.port):
                break
            time.sleep(1)

        if not quiet:
            print("UI stopped.")


ui_manager = UIManager()
