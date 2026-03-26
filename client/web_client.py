from __future__ import annotations

import html
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import threading
import time
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
import requests

from client.client_models import API_VERSION, BiometricClientError
from client.main_client import BiometricClient
from client.ml_extractor import extract_embedding


def _require_model_path() -> str:
    model_path = os.getenv("MODEL_PATH", "").strip()
    if not model_path:
        raise RuntimeError("MODEL_PATH environment variable is required for web client")
    return model_path


SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8000")
DEFAULT_THRESHOLD = float(os.getenv("CLIENT_THRESHOLD", "800"))
MODEL_PATH = _require_model_path()
LSH_BITS = int(os.getenv("LSH_BITS", "9"))
LFW_ROOT_ENV = os.getenv("LFW_ROOT", "").strip()
LFW_MAX_PER_IDENTITY = max(1, int(os.getenv("LFW_MAX_PER_IDENTITY", "10")))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

client = BiometricClient(server_url=SERVER_URL, model_path=MODEL_PATH, threshold=DEFAULT_THRESHOLD, lsh_bits=LSH_BITS)
app = FastAPI(title="Biometric Client Web UI")

_populate_status_lock = threading.Lock()
_populate_thread: threading.Thread | None = None
_populate_pause_event = threading.Event()
_populate_pause_event.set()
_populate_status: dict[str, object] = {
    "state": "idle",
    "message": "Not started",
    "done": 0,
    "total": 0,
    "percent": 0.0,
    "enrolled": 0,
    "failed": 0,
    "elapsed": "0.0s",
    "lfw_root": "",
    "max_per_identity": LFW_MAX_PER_IDENTITY,
    "sample_failures": [],
}


def _extract_payload_data(payload: dict) -> dict:
    data = payload.get("data")
    if isinstance(data, dict):
        merged = dict(payload)
        merged.update(data)
        return merged
    return payload


def _parse_api_error(response: requests.Response, fallback_prefix: str = "Request failed") -> str:
    request_id = response.headers.get("X-Request-ID", "")
    try:
        payload = response.json()
    except Exception:
        suffix = f" (request_id={request_id})" if request_id else ""
        return f"{fallback_prefix}: HTTP {response.status_code}{suffix}"

    if not isinstance(payload, dict):
        suffix = f" (request_id={request_id})" if request_id else ""
        return f"{fallback_prefix}: HTTP {response.status_code}{suffix}"

    detail = payload.get("detail", {})
    if not isinstance(detail, dict):
        suffix = f" (request_id={request_id})" if request_id else ""
        return f"{fallback_prefix}: HTTP {response.status_code}{suffix}"

    code = str(detail.get("code", "SERVER_ERROR"))
    message = str(detail.get("message", "Request failed"))
    detail_request_id = str(detail.get("request_id", "")).strip()
    effective_request_id = detail_request_id or request_id
    rid_suffix = f" (request_id={effective_request_id})" if effective_request_id else ""
    return f"{fallback_prefix}: {code}: {message}{rid_suffix}"


def _populate_status_snapshot() -> dict[str, object]:
    with _populate_status_lock:
        return dict(_populate_status)


def _populate_status_update(**updates: object) -> None:
    with _populate_status_lock:
        _populate_status.update(updates)


def _resolve_lfw_root() -> Path:
    candidates: list[Path] = []
    if LFW_ROOT_ENV:
        candidates.append(Path(LFW_ROOT_ENV))
    candidates.extend(
        [
            Path("/models/lfw-deepfunneled/lfw-deepfunneled"),
            Path("/models/lfw-funneled/lfw-funneled"),
            Path("/models/lfw-deepfunneled"),
            Path("/models/lfw-funneled"),
            Path("models/lfw-deepfunneled/lfw-deepfunneled"),
            Path("models/lfw-funneled/lfw-funneled"),
        ]
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise RuntimeError(
        "LFW root directory not found. Set LFW_ROOT env var or mount dataset under /models/lfw-deepfunneled/lfw-deepfunneled"
    )


def _collect_lfw_images(lfw_root: Path, max_per_identity: int | None = None) -> list[tuple[str, list[Path]]]:
    rows: list[tuple[str, list[Path]]] = []
    for identity_dir in sorted([path for path in lfw_root.iterdir() if path.is_dir()]):
        label = identity_dir.name
        collected = 0
        images: list[Path] = []
        for file_path in sorted(identity_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(file_path)
                collected += 1
                if max_per_identity is not None and collected >= max_per_identity:
                    break
        if images:
            rows.append((label, images))
    return rows


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"


def _populate_db_from_lfw_worker() -> None:
    try:
        lfw_root = _resolve_lfw_root()
        rows = _collect_lfw_images(lfw_root, max_per_identity=LFW_MAX_PER_IDENTITY)
        if not rows:
            raise RuntimeError(f"No images found under LFW root: {lfw_root}")

        started = time.perf_counter()
        enrolled = 0
        failed = 0
        failures: list[dict] = []
        total = len(rows)
        progress_every = 25

        _populate_status_update(
            state="running",
            message="LFW population is running",
            done=0,
            total=total,
            percent=0.0,
            enrolled=0,
            failed=0,
            elapsed="0.0s",
            lfw_root=str(lfw_root),
            max_per_identity=LFW_MAX_PER_IDENTITY,
            sample_failures=[],
        )

        for index, (label, image_paths) in enumerate(rows, start=1):
            while not _populate_pause_event.is_set():
                elapsed = time.perf_counter() - started
                _populate_status_update(
                    state="paused",
                    message="LFW population is paused",
                    done=index - 1,
                    total=total,
                    percent=round(((index - 1) / max(1, total)) * 100.0, 2),
                    enrolled=enrolled,
                    failed=failed,
                    elapsed=_format_duration(elapsed),
                    sample_failures=failures,
                )
                time.sleep(0.5)

            snapshot_state = str(_populate_status_snapshot().get("state", ""))
            if snapshot_state == "paused":
                _populate_status_update(state="running", message="LFW population is running")

            try:
                enroll_result = client.enroll_user_images([str(path) for path in image_paths], label)
                enrolled += int(enroll_result.get("enrollment_calls", 0))
            except Exception as exc:
                failed += 1
                # Log exception type only; never expose full details to UI
                print(f"[LFW Enroll Error] label={label} error_type={type(exc).__name__}", flush=True)
                if len(failures) < 20:
                    failures.append({"label": label, "image_count": len(image_paths), "error": f"{type(exc).__name__}: enrollment failed"})

            if index == 1 or index == total or index % progress_every == 0:
                elapsed = time.perf_counter() - started
                percent = (index / max(1, total)) * 100.0
                _populate_status_update(
                    done=index,
                    total=total,
                    percent=round(percent, 2),
                    enrolled=enrolled,
                    failed=failed,
                    elapsed=_format_duration(elapsed),
                    sample_failures=failures,
                )
                print(
                    {
                        "event": "lfw_populate_progress",
                        "done": index,
                        "total": total,
                        "enrolled": enrolled,
                        "failed": failed,
                        "elapsed": _format_duration(elapsed),
                    },
                    flush=True,
                )

        elapsed = time.perf_counter() - started
        final_state = "completed" if failed == 0 else "completed_with_failures"
        final_message = "LFW population completed successfully" if failed == 0 else "LFW population completed with failures"
        _populate_status_update(
            state=final_state,
            message=final_message,
            done=total,
            total=total,
            percent=100.0,
            enrolled=enrolled,
            failed=failed,
            elapsed=_format_duration(elapsed),
            max_per_identity=LFW_MAX_PER_IDENTITY,
            sample_failures=failures,
        )
    except Exception as exc:
        _populate_status_update(
            state="failed",
            message="LFW populate failed: an error occurred",
        )


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "upload.jpg").suffix or ".jpg"
    with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        data = upload.file.read()
        handle.write(data)
        return Path(handle.name)


def _render_page(
    message: str = "",
    details: dict | None = None,
    advanced_details: dict | None = None,
) -> HTMLResponse:
    detail_html = ""
    advanced_detail_html = ""
    if details is not None:
        rows = "".join(
            f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
            for key, value in details.items()
        )
        detail_html = f"<table>{rows}</table>"
    if advanced_details is not None:
        rows = "".join(
            f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
            for key, value in advanced_details.items()
        )
        advanced_detail_html = (
            '<details class="card">'
            '<summary><b>Advanced details</b></summary>'
            f"<table>{rows}</table>"
            "</details>"
        )

    body = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset=\"utf-8\" />
      <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
      <title>Biometric Client UI</title>
      <style>
        body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 24px auto; padding: 0 12px; }}
        .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 14px; margin-bottom: 14px; }}
        input, button {{ margin: 6px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .msg {{ background: #f5f7ff; border: 1px solid #cfd8ff; padding: 10px; border-radius: 8px; margin-bottom: 12px; }}
        .hint {{ color: #555; font-size: 0.92rem; }}
      </style>
    </head>
    <body>
      <h1>Biometric Client Web UI</h1>
      <p class=\"hint\">Model: <b>{html.escape(MODEL_PATH)}</b> | API: <b>{html.escape(SERVER_URL)}</b></p>
      <p class=\"hint\">Tip: If you previously benchmarked with a different client context, re-enroll users via this UI before authenticating.</p>
        <p><a href="/admin">Advanced/Admin</a></p>
      {f'<div class="msg">{html.escape(message)}</div>' if message else ''}
            {detail_html}
            {advanced_detail_html}

      <div class=\"card\">
        <h2>Enroll User</h2>
        <form action=\"/enroll\" method=\"post\" enctype=\"multipart/form-data\">
          <label>User UUID/Label</label><br />
          <input type=\"text\" name=\"user_uuid\" required /> <br />
          <label>Face Image(s)</label><br />
          <input type=\"file\" name=\"images\" accept=\".jpg,.jpeg,.png,.bmp,.webp\" multiple required /> <br />
          <span class=\"hint\">You can upload one image or multiple images for the same user.</span><br />
          <button type=\"submit\">Enroll</button>
        </form>
      </div>

      <div class=\"card\">
        <h2>Authenticate User</h2>
        <form action=\"/authenticate\" method=\"post\" enctype=\"multipart/form-data\">
          <label>Probe Image</label><br />
          <input type=\"file\" name=\"image\" accept=\".jpg,.jpeg,.png,.bmp,.webp\" required /> <br />
          <label>Threshold (optional, default {DEFAULT_THRESHOLD})</label><br />
          <input type=\"number\" step=\"0.0001\" name=\"threshold\" /> <br />
          <button type=\"submit\">Authenticate</button>
        </form>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(body)


def _render_database_page(
    message: str = "",
    summary: dict | None = None,
    rows: list[dict] | None = None,
    page: int = 1,
    page_size: int = 5,
    total_pages: int = 1,
    label_query: str = "",
) -> HTMLResponse:
    summary_html = ""
    if summary is not None:
        summary_html = (
            "<div class=\"card\">"
            f"<p><b>Total Chunks:</b> {html.escape(str(summary.get('total_chunks', 0)))}</p>"
            f"<p><b>Total Labels:</b> {html.escape(str(summary.get('total_labels', 0)))}</p>"
            "</div>"
        )

    rows_html = ""
    if rows is not None:
        if rows:
            table_rows = []
            for row in rows:
                labels = row.get("labels", [])
                labels_text = ", ".join(str(label) for label in labels) if labels else ""
                table_rows.append(
                    "<tr>"
                    f"<td>{html.escape(str(row.get('chunk_key', '')))}</td>"
                    f"<td>{html.escape(str(row.get('bucket_id', '')))}</td>"
                    f"<td>{html.escape(labels_text)}</td>"
                    "</tr>"
                )
            rows_html = (
                "<div class=\"card\">"
                "<h2>Database Records</h2>"
                "<p class=\"hint\">Shows chunk record metadata without exposing raw encrypted payloads.</p>"
                "<table><thead><tr><th>Chunk Key</th><th>Bucket</th><th>Labels</th></tr></thead>"
                f"<tbody>{''.join(table_rows)}</tbody></table>"
                "</div>"
            )
        else:
            rows_html = '<div class="card"><p>No database records found.</p></div>'

    prev_query = urlencode({"page": max(1, page - 1), "page_size": page_size, "label": label_query})
    next_query = urlencode({"page": min(total_pages, page + 1), "page_size": page_size, "label": label_query})
    search_form = (
        "<div class=\"card\">"
        "<h2>Search & Pagination</h2>"
        "<form action=\"/database\" method=\"get\">"
        "<label>Label contains</label><br />"
        f"<input type=\"text\" name=\"label\" value=\"{html.escape(label_query)}\" /> <br />"
        "<label>Page size</label><br />"
        f"<input type=\"number\" name=\"page_size\" min=\"1\" max=\"200\" value=\"{page_size}\" />"
        "<input type=\"hidden\" name=\"page\" value=\"1\" /> <br />"
        "<button type=\"submit\">Apply</button>"
        "</form>"
        f"<p class=\"hint\"><b>Page:</b> {page}/{total_pages} | <b>Page size:</b> {page_size}</p>"
        f"<p><a href=\"/database?{prev_query}\">Previous</a> | <a href=\"/database?{next_query}\">Next</a></p>"
        "</div>"
    )

    body = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset=\"utf-8\" />
      <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
      <title>Biometric Database Viewer</title>
      <style>
        body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 24px auto; padding: 0 12px; }}
        .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 14px; margin-bottom: 14px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
        pre {{ margin: 10px 0 0 0; background: #f7f7f7; padding: 10px; border-radius: 8px; }}
        .msg {{ background: #f5f7ff; border: 1px solid #cfd8ff; padding: 10px; border-radius: 8px; margin-bottom: 12px; }}
        .hint {{ color: #555; font-size: 0.92rem; }}
      </style>
    </head>
    <body>
      <h1>Database Viewer</h1>
            <p class=\"hint\">Shows chunk metadata for safe admin inspection.</p>
            <p><a href=\"/\">Back to Client UI</a> | <a href=\"/database\">Refresh</a></p>
      {f'<div class="msg">{html.escape(message)}</div>' if message else ''}
      {summary_html}
      {search_form}
      {rows_html}
    </body>
    </html>
    """
    return HTMLResponse(body)


def _fetch_db_records(page: int = 1, page_size: int = 5, label_query: str = "") -> tuple[dict | None, list[dict] | None, str | None]:
    try:
        response = requests.get(
            f"{SERVER_URL}/db/records",
            params={"page": page, "page_size": page_size, "label": label_query},
            timeout=30,
        )
        if response.status_code >= 400:
            return None, None, _parse_api_error(response, fallback_prefix="Failed to load database records")

        payload = response.json()
        if not isinstance(payload, dict):
            return None, None, "Failed to load database records: invalid payload format"

        payload_data = _extract_payload_data(payload)
        summary = {
            "total_chunks": int(payload_data.get("total_chunks", 0)),
            "total_labels": int(payload_data.get("total_labels", 0)),
            "page": int(payload_data.get("page", page)),
            "page_size": int(payload_data.get("page_size", page_size)),
            "total_pages": int(payload_data.get("total_pages", 1)),
            "label_query": str(payload_data.get("label_query", label_query)),
        }
        rows = payload_data.get("rows", [])
        if not isinstance(rows, list):
            return None, None, "Invalid server response format"
        return summary, rows, None
    except requests.Timeout:
        return None, None, "Request timed out (>30s). Server may be busy."
    except requests.ConnectionError:
        return None, None, "Failed to connect to server. Please check the connection."
    except Exception:
        # Generic error; never expose exception details to UI
        return None, None, "Failed to load database records. Please try again."


def _render_admin_page(message: str = "") -> HTMLResponse:
    status = _populate_status_snapshot()
    state = str(status.get("state", "idle"))
    running_or_paused = state in {"running", "paused", "starting"}
    running = state == "running"
    paused = state == "paused"
    auto_refresh = '<meta http-equiv="refresh" content="3" />' if running_or_paused else ""
    percent = float(status.get("percent", 0.0) or 0.0)
    percent = max(0.0, min(100.0, percent))
    done = int(status.get("done", 0) or 0)
    total = int(status.get("total", 0) or 0)
    enrolled = int(status.get("enrolled", 0) or 0)
    failed = int(status.get("failed", 0) or 0)
    elapsed = str(status.get("elapsed", "0.0s"))
    lfw_root = str(status.get("lfw_root", ""))
    max_per_identity = int(status.get("max_per_identity", LFW_MAX_PER_IDENTITY) or LFW_MAX_PER_IDENTITY)
    status_message = str(status.get("message", ""))
    button_disabled = "disabled" if running_or_paused else ""
    pause_button_disabled = "" if running else "disabled"
    resume_button_disabled = "" if paused else "disabled"
    button_text = "Populate Running..." if running_or_paused else "Populate DB from LFW (All Images)"

    body = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
        {auto_refresh}
        <title>Biometric Admin</title>
        <style>
            body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 24px auto; padding: 0 12px; }}
            .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 14px; margin-bottom: 14px; }}
            input, button {{ margin: 6px 0; }}
            .msg {{ background: #f5f7ff; border: 1px solid #cfd8ff; padding: 10px; border-radius: 8px; margin-bottom: 12px; }}
            .hint {{ color: #555; font-size: 0.92rem; }}
            .warn {{ color: #8a5500; }}
            progress {{ width: 100%; height: 18px; }}
        </style>
    </head>
    <body>
        <h1>Admin</h1>
        <p><a href=\"/\">Back to Client UI</a> | <a href=\"/database\">Database Labels</a></p>
        {f'<div class="msg">{html.escape(message)}</div>' if message else ''}

        <div class=\"card\">
            <h2>LFW Populate Status</h2>
            <p><b>State:</b> {html.escape(state)} | <b>Message:</b> {html.escape(status_message)}</p>
            <progress value=\"{percent:.2f}\" max=\"100\"></progress>
            <p><b>Progress:</b> {percent:.1f}% ({done}/{total})</p>
            <p class=\"hint\"><b>Enrolled:</b> {enrolled} | <b>Failed:</b> {failed} | <b>Elapsed:</b> {html.escape(elapsed)}</p>
            <p class=\"hint\"><b>LFW Root:</b> {html.escape(lfw_root)}</p>
            <p class=\"hint\"><b>Max images per identity:</b> {max_per_identity}</p>
            <p class=\"hint\">This page auto-refreshes every 3s while populate is running.</p>
        </div>

        <div class=\"card\">
            <h2>Delete Label</h2>
            <p class=\"hint\">Deletes all occurrences of a label when the packed chunk can be rebuilt safely.</p>
            <p class=\"hint warn\">Legacy multi-face chunks created before per-face ciphertext tracking may be blocked from partial deletion.</p>
            <form action=\"/admin/delete-label\" method=\"post\">
                <label>Label</label><br />
                <input type=\"text\" name=\"user_uuid\" required /> <br />
                <button type=\"submit\">Delete Label</button>
            </form>
        </div>

        <div class=\"card\">
            <h2>Reset Database</h2>
            <p class=\"hint warn\">This removes all enrolled chunks from the server database.</p>
            <form action=\"/admin/reset\" method=\"post\">
                <label>Type RESET_DB to confirm</label><br />
                <input type=\"text\" name=\"confirm_text\" required /> <br />
                <button type=\"submit\">Reset DB</button>
            </form>
        </div>

        <div class="card">
            <h2>Populate DB from LFW</h2>
            <p class="hint">Enrolls up to {LFW_MAX_PER_IDENTITY} images per identity from LFW into the server DB.</p>
            <p class="hint warn">This may take a long time for full LFW and should be run when server load is low.</p>
            <form action="/admin/populate-lfw" method="post" onsubmit="return confirm('Populate DB with ALL LFW images for ALL identities? This may take a long time.');">
                <button type="submit" {button_disabled}>{button_text}</button>
            </form>
            <form action="/admin/populate-lfw/pause" method="post" style="display:inline;">
                <button type="submit" {pause_button_disabled}>Pause</button>
            </form>
            <form action="/admin/populate-lfw/resume" method="post" style="display:inline;">
                <button type="submit" {resume_button_disabled}>Resume</button>
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(body)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return _render_page()


@app.get("/database", response_class=HTMLResponse)
def database_page(page: int = 1, page_size: int = 5, label: str = "") -> HTMLResponse:
    effective_page = max(1, int(page))
    effective_page_size = max(1, min(200, int(page_size)))
    summary, rows, error = _fetch_db_records(page=effective_page, page_size=effective_page_size, label_query=label)
    if error:
        return _render_database_page(message=error)
    return _render_database_page(
        summary=summary,
        rows=rows,
        page=int(summary.get("page", effective_page)),
        page_size=int(summary.get("page_size", effective_page_size)),
        total_pages=int(summary.get("total_pages", 1)),
        label_query=str(summary.get("label_query", label)),
    )


@app.get("/admin", response_class=HTMLResponse)
def admin_page() -> HTMLResponse:
    return _render_admin_page()


@app.post("/admin/delete-label", response_class=HTMLResponse)
def admin_delete_label(user_uuid: str = Form(...)) -> HTMLResponse:
    try:
        response = requests.post(
            f"{SERVER_URL}/admin/delete-label",
            json={
                "api_version": API_VERSION,
                "user_uuid": user_uuid,
            },
            timeout=60,
        )
        if response.status_code >= 400:
            return _render_admin_page(message=_parse_api_error(response, fallback_prefix="Delete failed"))

        payload = response.json()
        if not isinstance(payload, dict):
            return _render_admin_page(message="Delete failed: invalid payload format")
        payload_data = _extract_payload_data(payload)

        blocked = payload_data.get("blocked_chunks", [])
        blocked_text = f" blocked_chunks={blocked}" if blocked else ""
        return _render_admin_page(
            message=(
                f"Deleted {payload_data.get('deleted_count', 0)} entries for '{user_uuid}' "
                f"across {payload_data.get('affected_chunks', 0)} chunks.{blocked_text}"
            )
        )
    except requests.Timeout:
        return _render_admin_page(message="Delete failed: request timed out")
    except requests.ConnectionError:
        return _render_admin_page(message="Delete failed: could not connect to API")
    except Exception:
        return _render_admin_page(message="Delete failed: unexpected client error")


@app.post("/admin/reset", response_class=HTMLResponse)
def admin_reset(confirm_text: str = Form(...)) -> HTMLResponse:
    try:
        response = requests.post(
            f"{SERVER_URL}/admin/reset",
            json={
                "api_version": API_VERSION,
                "confirm_text": confirm_text,
            },
            timeout=60,
        )
        if response.status_code >= 400:
            return _render_admin_page(message=_parse_api_error(response, fallback_prefix="Reset failed"))

        payload = response.json()
        if not isinstance(payload, dict):
            return _render_admin_page(message="Reset failed: invalid payload format")
        payload_data = _extract_payload_data(payload)

        return _render_admin_page(message=f"Database reset complete. Deleted chunks: {payload_data.get('deleted_chunks', 0)}")
    except requests.Timeout:
        return _render_admin_page(message="Reset failed: request timed out")
    except requests.ConnectionError:
        return _render_admin_page(message="Reset failed: could not connect to API")
    except Exception:
        return _render_admin_page(message="Reset failed: unexpected client error")


@app.post("/admin/populate-lfw", response_class=HTMLResponse)
def admin_populate_lfw() -> HTMLResponse:
    global _populate_thread
    try:
        if _populate_thread is not None and _populate_thread.is_alive():
            return _render_admin_page(message="LFW populate is already running")

        _populate_pause_event.set()
        _populate_status_update(
            state="starting",
            message="Starting LFW population...",
            done=0,
            total=0,
            percent=0.0,
            enrolled=0,
            failed=0,
            elapsed="0.0s",
            max_per_identity=LFW_MAX_PER_IDENTITY,
            sample_failures=[],
        )
        _populate_thread = threading.Thread(target=_populate_db_from_lfw_worker, daemon=True)
        _populate_thread.start()
        return _render_admin_page(message="LFW populate started. Monitor progress below.")
    except Exception:
        return _render_admin_page(message="LFW populate failed: unexpected client error")


@app.post("/admin/populate-lfw/pause", response_class=HTMLResponse)
def admin_populate_lfw_pause() -> HTMLResponse:
    try:
        if _populate_thread is None or not _populate_thread.is_alive():
            return _render_admin_page(message="LFW populate is not running")
        _populate_pause_event.clear()
        _populate_status_update(state="paused", message="LFW population is paused")
        return _render_admin_page(message="Pause requested. Population will pause shortly.")
    except Exception:
        return _render_admin_page(message="Pause failed: unexpected client error")


@app.post("/admin/populate-lfw/resume", response_class=HTMLResponse)
def admin_populate_lfw_resume() -> HTMLResponse:
    try:
        if _populate_thread is None or not _populate_thread.is_alive():
            return _render_admin_page(message="LFW populate is not running")
        _populate_pause_event.set()
        _populate_status_update(state="running", message="LFW population is running")
        return _render_admin_page(message="Population resumed.")
    except Exception:
        return _render_admin_page(message="Resume failed: unexpected client error")


@app.post("/enroll", response_class=HTMLResponse)
def enroll(
    user_uuid: str = Form(...),
    image: UploadFile | None = File(default=None),
    images: list[UploadFile] | None = File(default=None),
) -> HTMLResponse:
    temp_paths: list[Path] = []
    try:
        uploads: list[UploadFile] = []
        if images:
            uploads.extend(images)
        if image is not None and image.filename:
            uploads.append(image)
        uploads = [upload for upload in uploads if upload and upload.filename]
        if not uploads:
            return _render_page(message="Enroll failed: please upload at least one image")

        for upload in uploads:
            temp_paths.append(_save_upload(upload))

        result = client.enroll_user_images([str(path) for path in temp_paths], user_uuid)
        details = dict(result)
        details["enroll_image_count"] = len(temp_paths)
        return _render_page(message=f"Enrolled '{user_uuid}' successfully", details=details)
    except BiometricClientError as exc:
        return _render_page(message=f"Enroll failed: {exc}")
    except Exception:
        return _render_page(message="Enroll failed: unexpected client error")
    finally:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)


@app.post("/authenticate", response_class=HTMLResponse)
def authenticate(
    image: UploadFile = File(...),
    threshold: str = Form(default=""),
) -> HTMLResponse:
    temp_path: Path | None = None
    try:
        temp_path = _save_upload(image)
        embedding = extract_embedding(MODEL_PATH, str(temp_path))
        effective_threshold = float(threshold) if threshold.strip() else None
        diagnostics = client.authenticate_embedding_diagnostics(embedding, threshold=effective_threshold)
        winner = diagnostics.get("winner_uuid")
        message = f"Authentication result: winner={winner}" if winner else "Authentication result: no match"
        details = {
            "winner_uuid": winner,
            "best_distance": diagnostics.get("best_distance"),
            "candidate_count": diagnostics.get("candidate_count"),
            "timings_ms_total": diagnostics.get("timings_ms", {}).get("total"),
        }
        advanced_details = {
            "fallback_used": diagnostics.get("fallback_used"),
            "fallback_bucket_count": diagnostics.get("fallback_bucket_count"),
            "nearby_requested_bucket_count": diagnostics.get("nearby_requested_bucket_count"),
            "nearby_requested_buckets": diagnostics.get("nearby_requested_buckets"),
            "nearby_existing_bucket_count": diagnostics.get("nearby_existing_bucket_count"),
            "nearby_existing_buckets": diagnostics.get("nearby_existing_buckets"),
            "miss_retry_used": diagnostics.get("miss_retry_used"),
            "miss_retry_bucket_count": diagnostics.get("miss_retry_bucket_count"),
            "miss_retry_requested_bucket_count": diagnostics.get("miss_retry_requested_bucket_count"),
            "miss_retry_requested_buckets": diagnostics.get("miss_retry_requested_buckets"),
            "miss_retry_existing_bucket_count": diagnostics.get("miss_retry_existing_bucket_count"),
            "miss_retry_existing_buckets": diagnostics.get("miss_retry_existing_buckets"),
            "threshold": diagnostics.get("threshold"),
            "timings_ms_request": diagnostics.get("timings_ms", {}).get("request_roundtrip"),
        }
        return _render_page(message=message, details=details, advanced_details=advanced_details)
    except BiometricClientError as exc:
        return _render_page(message=f"Authenticate failed: {exc}")
    except Exception:
        return _render_page(message="Authenticate failed: unexpected client error")
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
