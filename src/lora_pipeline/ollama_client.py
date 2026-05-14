"""Qwen via Ollama — streaming, thinking log, watchdog, exponential backoff."""
import json, logging, threading, time
import ollama as _ollama
from . import config
from .bootstrap import ollama_running, restart_ollama

log = logging.getLogger("processor")


class TimeoutError(Exception):
    pass


def call_qwen(img_b64: str, stem: str, model: str) -> dict:
    result, exc_holder = {}, []

    def _call():
        try:
            tokens = []
            in_think = False
            seen_open = False
            think_partial = ""

            stream = _ollama.chat(
                model=model,
                messages=[{"role": "user", "content": config.PROMPT,
                           "images": [img_b64]}],
                options={"temperature": 0, "num_predict": config.NUM_PREDICT},
                stream=True,
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                tokens.append(token)

                if not seen_open:
                    joined = "".join(tokens)
                    if "<think>" in joined:
                        seen_open = True
                        in_think = True
                        think_partial = joined[joined.index("<think>") + len("<think>"):]
                elif in_think:
                    think_partial += token
                    if "</think>" in think_partial:
                        before = think_partial[:think_partial.index("</think>")]
                        for line in before.split("\n"):
                            line = line.strip()
                            if line:
                                log.info(f"  [{stem}] 💭 {line}")
                        in_think = False
                        think_partial = ""
                    else:
                        while "\n" in think_partial:
                            line, think_partial = think_partial.split("\n", 1)
                            line = line.strip()
                            if line:
                                log.info(f"  [{stem}] 💭 {line}")

            raw = "".join(tokens).strip()
            if "</think>" in raw:
                raw = raw[raw.rfind("</think>") + len("</think>"):].strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            result["raw"] = raw
        except Exception as e:
            exc_holder.append(e)

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=config.CALL_TIMEOUT)

    if t.is_alive():
        raise TimeoutError(f"Qwen call timed out after {config.CALL_TIMEOUT}s")
    if exc_holder:
        raise exc_holder[0]

    return json.loads(result.get("raw", ""))


def analyse_with_backoff(img_b64: str, stem: str, model: str) -> dict:
    last_error = None
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            return call_qwen(img_b64, stem, model)

        except json.JSONDecodeError as e:
            wait = config.BACKOFF_BASE ** attempt
            log.warning(f"  [{stem}] JSON parse error (attempt {attempt+1}): {e} — retry in {wait}s")
            last_error = {"_parse_error": str(e)}
            time.sleep(wait)

        except (ConnectionRefusedError, TimeoutError,
                _ollama.ResponseError) as e:
            wait = config.BACKOFF_BASE ** attempt
            log.warning(f"  [{stem}] Connection/timeout (attempt {attempt+1}): {e}")
            last_error = {"_error": str(e)}
            if attempt >= 1 and not ollama_running():
                log.warning("  Ollama not responding — restarting...")
                if restart_ollama(log):
                    time.sleep(2)
                    continue
            time.sleep(wait)

        except Exception as e:
            wait = config.BACKOFF_BASE ** attempt
            log.warning(f"  [{stem}] Unexpected error (attempt {attempt+1}): {e} — retry in {wait}s")
            last_error = {"_error": str(e)}
            time.sleep(wait)

    log.error(f"  [{stem}] All {config.MAX_RETRIES+1} attempts failed")
    return last_error or {"_error": "unknown"}
