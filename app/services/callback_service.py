from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict

from app.config import settings


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get_callback_url(payload_dict: Dict[str, Any]) -> str:
    parametros = payload_dict.get("parametros", {}) if isinstance(payload_dict, dict) else {}
    callback_url_payload = _safe_text(parametros.get("callback_url"))
    callback_url_env = _safe_text(settings.callback_url)

    return callback_url_payload or callback_url_env


def get_callback_secret() -> str:
    return _safe_text(settings.callback_secret)


def should_send_callback(payload_dict: Dict[str, Any]) -> bool:
    if not settings.callback_enabled:
        return False

    callback_url = get_callback_url(payload_dict)
    return bool(callback_url)


def send_callback(payload_dict: Dict[str, Any], contrato_m8: Dict[str, Any]) -> Dict[str, Any]:
    callback_url = get_callback_url(payload_dict)
    callback_secret = get_callback_secret()

    if not callback_url:
        return {
            "callback_enviado": False,
            "callback_status": "ignorado",
            "callback_http_status": None,
            "callback_url": "",
            "callback_mensagem": "Callback URL não configurada.",
        }

    body = json.dumps(contrato_m8).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "rec-motor-python/1.0",
    }

    if callback_secret:
        headers["X-REC-CALLBACK-SECRET"] = callback_secret

    req = urllib.request.Request(
        url=callback_url,
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=settings.callback_timeout_seconds) as resp:
            response_body = resp.read().decode("utf-8", errors="ignore")
            return {
                "callback_enviado": True,
                "callback_status": "ok",
                "callback_http_status": getattr(resp, "status", 200),
                "callback_url": callback_url,
                "callback_mensagem": response_body[:1000],
            }
    except urllib.error.HTTPError as e:
        try:
            error_body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            error_body = str(e)

        return {
            "callback_enviado": False,
            "callback_status": "http_error",
            "callback_http_status": getattr(e, "code", None),
            "callback_url": callback_url,
            "callback_mensagem": error_body[:1000],
        }
    except Exception as e:
        return {
            "callback_enviado": False,
            "callback_status": "exception",
            "callback_http_status": None,
            "callback_url": callback_url,
            "callback_mensagem": str(e),
        }
