"""Ollama and dependency bootstrap."""
import shutil, subprocess, sys, time
from . import config

def ensure_packages():
    required = {"imagehash": "imagehash", "ollama": "ollama", "PIL": "Pillow"}
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   package, "--break-system-packages", "-q"])

def ollama_running() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False

def start_ollama_server(log) -> bool:
    log.info("Ollama: starting server...")
    subprocess.Popen(["ollama", "serve"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(config.SERVER_RESTART_WAIT * 2):
        time.sleep(0.5)
        if ollama_running():
            log.info("Ollama: server ready")
            return True
    log.error("Ollama: server failed to start")
    return False

def restart_ollama(log) -> bool:
    log.warning("Attempting Ollama server restart...")
    subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
    time.sleep(2)
    return start_ollama_server(log)

def ensure_ollama(log) -> str:
    """Ensure Ollama is installed, running, and model is pulled.
    Returns the model name that is available."""
    model = config.MODEL

    if shutil.which("ollama") is None:
        log.info("Ollama not found — installing via Homebrew...")
        subprocess.check_call(["brew", "install", "ollama"])

    if not ollama_running():
        if not start_ollama_server(log):
            sys.exit("Cannot start Ollama server. Aborting.")

    pulled = subprocess.run(["ollama", "list"],
                             capture_output=True, text=True).stdout
    if model in pulled:
        log.info(f"Model : {model}")
    elif config.MODEL_FALLBACK in pulled:
        model = config.MODEL_FALLBACK
        log.info(f"Model : {model} (q8_0 not found, using fallback)")
    else:
        log.info(f"Pulling {model} (~10GB)...")
        subprocess.check_call(["ollama", "pull", model])
        log.info(f"Model : {model} ready")

    return model
