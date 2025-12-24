"""
My Chess Notebook 
"""

import json
import time
import random
import uuid
from pathlib import Path
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, Tuple, List

import streamlit as st
import streamlit.components.v1 as components
import chess
import chess.svg as chess_svg

import hashlib
import chess.pgn
import io
import re

import zipfile



# ===================== App Config & Styling =====================
st.set_page_config(page_title="My Chess Notebook (White)", page_icon="♟️", layout="wide")

ACCENT = "#0ea5e9"
DANGER = "#ef4444"
MUTED = "#6b7280"

st.markdown(
    f"""
    <style>
    .small-muted {{ color: {MUTED}; font-size: 0.85rem; }}
    .coverage-map {{ font-size: 0.9rem; line-height: 1.4; }}
    .moves-bubbles > span {{
        display:inline-block; padding:.25rem .5rem; border-radius:.5rem;
        margin:.18rem .18rem; border:1px solid rgba(148,163,184,.25);
    }}
    .moves-bubbles > span.w {{ background: rgba(34,197,94,.10); }}
    .moves-bubbles > span.b {{ background: rgba(239,68,68,.10); }}

    .moves-bubbles .comment-marker {{
        padding: 0;
        margin: 0 0 0 0.15rem;
        border: none;
        background: transparent;
        display: inline;
        font-size: 0.75rem;
        vertical-align: text-top;
    }}
    .board-context-title {{
        font-size: 1.05rem; 
        font-weight: 650; 
        margin: 0 0 .35rem 0; 
    }}
    .comment-card {{
        border:1px solid rgba(148,163,184,.25);
        border-radius:.7rem;
        padding:.35rem .6rem;
        margin-bottom:.2rem;
        background: rgba(148,163,184,.05);
    }}
    .comment-card p {{ margin-top: 0; margin-bottom: 0; }}
    .comment-card button {{
        padding: 0.1rem 0.35rem;
        min-height: 0;
        font-size: 0.7rem;
    }}
    .soft-panel {{
        border-radius: .9rem;
        padding: .9rem;
        border:1px solid rgba(148,163,184,.25);
        background: rgba(148,163,184,.06);
    }}
    .flash {{
        padding:.6rem .8rem;
        border-left: 4px solid {DANGER};
        background: rgba(239,68,68,.08);
        border-radius:.25rem;
        margin:.25rem 0 .75rem 0;
    }}
    .board-wrap {{
        border-radius: .9rem;
        overflow: hidden;
        border:1px solid rgba(148,163,184,.25);
        background: #111318;
    }}
    .tag-pills span {{
        display:inline-block;
        padding:.18rem .5rem;
        border-radius:999px;
        margin:.15rem .15rem;
        border:1px solid rgba(148,163,184,.25);
        background: rgba(14,165,233,.10);
        font-size: .75rem;
    }}

    .block-container {{ padding-top: .9rem; }}
    h1 {{ margin-bottom: .35rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("My Chess Notebook — White")
st.caption("Add moves, get next-move suggestions from your saved lines, and attach comments.")


# ===================== Storage =====================
DATA_DIR = Path("data/games")
DATA_DIR.mkdir(parents=True, exist_ok=True)

TACTICS_DIR = Path("data/tactics")
TACTICS_DIR.mkdir(parents=True, exist_ok=True)


# ===================== Session State =====================
def _init_state() -> None:
    ss = st.session_state

    defaults = {
        "board": chess.Board(),
        "moves_san": [],
        "comments": [],
        "flipped": False,
        "view_mode": "edit",
        "delete_pending": None,
        "current_file": None,
        "current_ply_idx": 0,
        "templated_from": None,
        "loaded_snapshot": None,
        "move_input": "",
        "flash": "",
        "source_text": "",
        "url_text": "",
        "new_comment_text": "",
        "edit_comment_idx": None,
        "coverage_slider_synced": True,
        "coverage_last_attach_ply": 0,
        "walkthrough_mode": False,
        "walkthrough_revealed": [],
        "tactic_mode": False,
        "tactic_current_file": None,
        "tactic_payload": None,
        "tactic_board": chess.Board(),
        "tactic_offset": 0,
        "tactic_input": "",
        "tactic_feedback": "",
        "tactic_complete": False,
        "tactic_start_ply": 0,
        "tactic_end_ply": 0,
        "tactic_start_selected": False,
        "tactic_end_selected": False,
        "tactic_start_black_ply": None,
        "tactic_note_draft": "",
        "tactic_note_flash": "",
        "tactic_new_note": "",
        "tactic_trigger_text": "",
        "tactic_filter_tags": [],
        "tactic_filter_match_all": False,
        "tactic_reveal_next": False,
        "tactic_show_notes": False,
        "tactic_show_solution": False,
        "tactic_show_tags": False, 
        "pgn_fingerprint": None,
        "pgn_games": [],
        "pgn_parse_errors": [],
        "pgn_selected_idx": 0,
        "pgn_uploader_nonce": 0,
        "pgn_auto_import_single": True,
        "pgn_pending_import": False,
        "tactic_delete_pending": None, 
        "tactic_manage_query": "",       
        "tactic_manage_limit": 50,       
        "pgn_paste_text": "",
  
    }

    for key, value in defaults.items():
        ss.setdefault(key, value)

_init_state()


# ===================== Small Helpers =====================
def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _normalize_source(s: str) -> str:
    return (s or "").strip().lower()

def _norm_san(s: str) -> str:
    return san_match_key(s)

def normalize_castling_san(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s

    suffix = ""
    if s[-1] in {"+", "#"}:
        suffix = s[-1]
        core = s[:-1].strip()
    else:
        core = s

    core_u = core.upper()
    if core_u in {"0-0", "O-O"}:
        return "O-O" + suffix
    if core_u in {"0-0-0", "O-O-O"}:
        return "O-O-O" + suffix

    return core + suffix

def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def _write_json(path: Path, payload: dict) -> bool:
    try:
        path.write_text(json.dumps(payload, indent=2))
        return True
    except Exception:
        return False

def _resolve_path_str(path: str | Path | None) -> str:
    """Best-effort absolute path string; returns empty string when missing/invalid."""
    if path is None:
        return ""
    try:
        return str(Path(path).resolve())
    except Exception:
        return str(path)

def _paths_equal(a: str | Path | None, b: str | Path | None) -> bool:
    """Compare two paths after resolving; tolerant of falsy/invalid inputs."""
    return _resolve_path_str(a) == _resolve_path_str(b)

def san_match_key(s: str) -> str:
    """
    Normalized key for *matching only*.
    Ignores capture/check/mate decoration so users can type with or without them.
    """
    s = normalize_castling_san(s or "").strip()

    # ignore check/mate suffix (and any accidental trailing whitespace)
    s = s.rstrip().rstrip("+#")

    # ignore capture marker
    s = s.replace("x", "")

    # optional: tolerate promotion notation differences: "e8Q" vs "e8=Q"
    s = s.replace("=", "")

    return s


def _game_file_mtimes() -> Tuple[int, ...]:
    """Sorted mtimes for all saved games, used as cache keys."""
    return tuple(sorted(fp.stat().st_mtime_ns for fp in DATA_DIR.glob("game_*.json")))


def parse_san_lenient(board: chess.Board, user_text: str) -> tuple[chess.Move | None, list[str]]:
    """
    Resolve a user-entered move in a lenient way:
      - user may omit '+/#'
      - user may omit 'x'
    Returns (move, options). If move is None and options non-empty, it was ambiguous.
    """
    raw = normalize_castling_san((user_text or "").strip())
    if not raw:
        return None, []

    # Fast path: try direct parsing (allow missing +/#)
    for attempt in (raw, raw.rstrip("+#")):
        try:
            return board.parse_san(attempt), []
        except Exception:
            pass

    # Slow path: compare against legal moves using a stripped match key
    key = san_match_key(raw)
    matches: list[chess.Move] = []
    for mv in board.legal_moves:
        try:
            if san_match_key(board.san(mv)) == key:
                matches.append(mv)
        except Exception:
            continue

    if len(matches) == 1:
        return matches[0], []
    if len(matches) > 1:
        # Return canonical SAN options so the UI can prompt disambiguation
        opts = []
        for mv in matches:
            try:
                opts.append(board.san(mv))
            except Exception:
                pass
        return None, sorted(set(opts))

    return None, []

def _pgn_fingerprint(name: str, b: bytes) -> str:
    h = hashlib.md5()
    h.update((name or "").encode("utf-8", errors="ignore"))
    h.update(b or b"")
    return h.hexdigest()


def _label_from_headers(headers: dict, game_num: int) -> str:
    w = (headers.get("White") or "").strip() or "White"
    b = (headers.get("Black") or "").strip() or "Black"
    date = (headers.get("UTCDate") or headers.get("Date") or "").strip()
    result = (headers.get("Result") or "").strip()
    parts = [f"{game_num}. {w} vs {b}"]
    if date:
        parts.append(date)
    if result and result != "*":
        parts.append(result)
    return " — ".join(parts)


def _default_source_from_headers(headers: dict) -> str:
    w = (headers.get("White") or "").strip()
    b = (headers.get("Black") or "").strip()
    date = (headers.get("UTCDate") or headers.get("Date") or "").strip()
    event = (headers.get("Event") or "").strip()

    core = ""
    if w or b:
        core = f"{w or 'White'} vs {b or 'Black'}"
    else:
        core = event or "PGN Import"

    if date and date != "????.??.??":
        core += f" ({date})"
    if event and event.lower() not in core.lower():
        core = f"{event} — {core}"

    return f"PGN: {core}"


_RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}

def _strip_pgn_variations(text: str) -> str:
    """Remove (...) variations (best-effort, supports nesting)."""
    out = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out)

def _clean_movetext(text: str) -> str:
    """Remove comments, variations, NAGs; keep mostly SAN tokens."""
    t = text or ""
    # { ... } comments
    t = re.sub(r"\{[^}]*\}", " ", t, flags=re.S)
    # ; to end-of-line comments
    t = re.sub(r";[^\n]*", " ", t)
    # ( ... ) variations
    t = _strip_pgn_variations(t)
    # $N NAGs
    t = re.sub(r"\$\d+", " ", t)
    return t

def _parse_movetext_line_to_san(movetext: str) -> tuple[list[str], list[str]]:
    """
    Parse a bare PGN movetext line (no headers) into canonical SAN list.
    Tolerates move numbers, results, and annotation suffixes like !?!!??.
    """
    errors: list[str] = []
    board = chess.Board()
    moves_san: list[str] = []

    cleaned = _clean_movetext(movetext)
    tokens = cleaned.replace("\r", " ").replace("\n", " ").split()

    for raw in tokens:
        tok = raw.strip()
        if not tok:
            continue

        # stop at result marker
        if tok in _RESULT_TOKENS:
            break

        # remove leading move numbers like "12." or "12..." or "12...Qxd4"
        tok = re.sub(r"^\d+\.(\.\.)?", "", tok).strip()
        if not tok or tok in ("...", ".."):
            continue

        # strip common annotation glyphs at end: !, ?, !!, ??, !?, ?!, etc
        tok = re.sub(r"[!\?]+$", "", tok).strip()
        if not tok:
            continue

        tok = normalize_castling_san(tok)

        mv, options = parse_san_lenient(board, tok)
        if mv is None:
            if options:
                errors.append(
                    f"Ambiguous token '{raw}' at ply {len(moves_san)+1}. Try one of: {', '.join(options[:8])}"
                )
            else:
                errors.append(f"Could not parse token '{raw}' at ply {len(moves_san)+1}.")
            return [], errors

        san = board.san(mv)
        board.push(mv)
        moves_san.append(san)

    if not moves_san:
        return [], ["No moves found in pasted line/PGN."]

    return moves_san, []

def parse_pgn_text(pgn_text: str) -> tuple[list[dict], list[str]]:
    """
    Parse 1+ PGN games from text. If no PGN game structure is found, fall back to
    parsing a bare movetext line (e.g. '1. d4 d5 2. Bf4 ...').
    Returns ([game_dict...], errors).
    """
    errors: list[str] = []
    games: list[dict] = []

    text = (pgn_text or "").strip()
    if not text:
        return [], ["Empty PGN input."]

    # --- Try standard PGN parser for 1+ games ---
    pgn_io = io.StringIO(text)
    game_num = 0
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break

        game_num += 1
        headers = {k: str(v) for k, v in dict(game.headers).items()}

        # Supports FEN/SetUp when present
        try:
            board = game.board()
        except Exception:
            board = chess.Board()

        moves_san: list[str] = []
        try:
            for mv in game.mainline_moves():
                san = board.san(mv)
                board.push(mv)
                moves_san.append(san)
        except Exception as e:
            errors.append(f"Game {game_num}: could not parse moves ({e}).")
            continue

        if not moves_san:
            continue

        site = (headers.get("Site") or "").strip()
        url = site if site.startswith("http") else ""

        source = _default_source_from_headers(headers)
        label = _label_from_headers(headers, game_num)

        games.append(
            {
                "label": label,
                "headers": headers,
                "moves_san": moves_san,
                "comments": [],
                "source": source,
                "url": url,
            }
        )

    if games:
        return games, errors

    # --- Fallback: bare movetext line ---
    moves_san, errs = _parse_movetext_line_to_san(text)
    if errs:
        return [], errs

    games.append(
        {
            "label": "1. Pasted PGN line",
            "headers": {},
            "moves_san": moves_san,
            "comments": [],
            "source": "PGN: Pasted line",
            "url": "",
        }
    )
    return games, []

def parse_pgn_upload(pgn_bytes: bytes) -> tuple[list[dict], list[str]]:
    text = (pgn_bytes or b"").decode("utf-8", errors="replace")
    return parse_pgn_text(text)



def on_import_selected_pgn(do_rerun: bool = False) -> None:
    games = st.session_state.get("pgn_games") or []
    if not games:
        st.session_state.flash = "No parsed PGN games to import."
        return

    idx = _safe_int(st.session_state.get("pgn_selected_idx", 0), 0)
    idx = max(0, min(idx, len(games) - 1))
    g = games[idx]

    # Clear current state (prevents mixing)
    reset_all()

    st.session_state.moves_san = list(g.get("moves_san") or [])
    st.session_state.comments = list(g.get("comments") or [])
    st.session_state.source_text = str(g.get("source") or "").strip()
    st.session_state.url_text = str(g.get("url") or "").strip()

    st.session_state.current_file = None
    st.session_state.view_mode = "edit"
    st.session_state.templated_from = None
    st.session_state.loaded_snapshot = None

    st.session_state.current_ply_idx = len(st.session_state.moves_san)
    _reposition_board()

    st.session_state.flash = (
        f"Imported PGN: {st.session_state.source_text} ({len(st.session_state.moves_san)} plies)."
    )

    reset_pgn_import_ui()

    # Optionally rerun when triggered outside widget callbacks
    if do_rerun:
        st.rerun()



def reset_pgn_import_ui() -> None:
    ss = st.session_state
    ss.pgn_fingerprint = None
    ss.pgn_games = []
    ss.pgn_parse_errors = []
    ss.pgn_selected_idx = 0
    ss.pgn_paste_text = ""
    ss.pgn_uploader_nonce = int(ss.get("pgn_uploader_nonce", 0)) + 1  # forces file_uploader to clear

def on_parse_pgn_paste() -> None:
    text = (st.session_state.get("pgn_paste_text") or "").strip()
    if not text:
        st.session_state.pgn_games = []
        st.session_state.pgn_parse_errors = ["Paste PGN text first."]
        st.session_state.pgn_fingerprint = None
        return

    fp = _pgn_fingerprint("PASTE_PGN", text.encode("utf-8", errors="ignore"))
    if st.session_state.get("pgn_fingerprint") == fp:
        return

    games, errs = parse_pgn_text(text)
    st.session_state.pgn_fingerprint = fp
    st.session_state.pgn_games = games
    st.session_state.pgn_parse_errors = errs
    st.session_state.pgn_selected_idx = 0

    if games and not errs and st.session_state.get("pgn_auto_import_single", True) and len(games) == 1:
        st.session_state.pgn_pending_import = True

def on_mark_pgn_pending_import() -> None:
    st.session_state.pgn_pending_import = True


def _unique_source_name(base: str) -> str:
    base = (base or "").strip() or "Imported game"
    if not _is_source_taken(base):
        return base
    i = 2
    while True:
        cand = f"{base} (import {i})"
        if not _is_source_taken(cand):
            return cand
        i += 1


def _unique_game_path(ts_hint: str | None = None) -> Path:
    ts = (ts_hint or "").strip() or time.strftime("%Y%m%d_%H%M%S")
    # Avoid collisions by suffixing a short uuid chunk.
    return DATA_DIR / f"game_{ts}_{uuid.uuid4().hex[:6]}.json"


def _unique_tactic_path(ts_hint: str | None = None) -> Path:
    ts = (ts_hint or "").strip() or time.strftime("%Y%m%d_%H%M%S")
    return TACTICS_DIR / f"tactic_{ts}_{uuid.uuid4().hex[:6]}.json"


def _read_zip_json(zf: zipfile.ZipFile, name: str) -> dict | None:
    try:
        raw = zf.read(name)
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None


def import_mcn_upload(file_name: str, file_bytes: bytes) -> dict:
    """
    Import either:
      - a single game JSON file, or
      - a ZIP exported from this app (games + optional tactics)
    Returns a report dict to show in the UI.
    """
    report = {
        "imported_games": [],     # list[Path]
        "imported_tactics": 0,
        "warnings": [],
        "errors": [],
    }

    if not file_bytes:
        report["errors"].append("Empty upload.")
        return report

    name_lower = (file_name or "").lower()

    # ---- ZIP PACKAGE ----
    if name_lower.endswith(".zip"):
        try:
            zf = zipfile.ZipFile(io.BytesIO(file_bytes), "r")
        except Exception as e:
            report["errors"].append(f"Could not open zip: {e}")
            return report

        # Accept both:
        # - games/*.json (your exporter below)
        # - loose game_*.json at root (more forgiving)
        game_members = [
            n for n in zf.namelist()
            if n.lower().endswith(".json") and (
                n.startswith("games/") or Path(n).name.startswith("game_")
            )
        ]
        tactic_members = [
            n for n in zf.namelist()
            if n.lower().endswith(".json") and (
                n.startswith("tactics/") or Path(n).name.startswith("tactic_")
            )
        ]

        if not game_members and not tactic_members:
            report["errors"].append("Zip did not contain any games/ or tactics/ JSON files.")
            return report

        # 1) Plan new destinations for all games first, so we can rewrite references
        rel_to_new_abs: dict[str, str] = {}
        staged_games: list[tuple[str, dict]] = []

        for member in sorted(game_members):
            payload = _read_zip_json(zf, member)
            if not payload:
                report["warnings"].append(f"Could not parse: {member}")
                continue

            moves = payload.get("moves_san") or []
            if not isinstance(moves, list) or not moves:
                report["warnings"].append(f"Skipping (no moves): {member}")
                continue

            old_source = str(payload.get("source") or "").strip()
            payload["source"] = _unique_source_name(old_source or Path(member).stem)

            out_path = _unique_game_path(payload.get("timestamp"))
            new_abs = _resolve_path_str(out_path)

            # Key by the archive-relative name AND also bare filename (for forgiving imports)
            rel_to_new_abs[member] = new_abs
            rel_to_new_abs[Path(member).name] = new_abs

            staged_games.append((member, payload))

        # 2) Write games, rewriting comment source_file pointers to new abs paths
        for member, payload in staged_games:
            out_path = Path(rel_to_new_abs[member])

            # Rewrite comments
            fixed_comments: list[dict] = []
            for c in (payload.get("comments") or []):
                c = dict(c)
                origin = c.get("origin") or "original"

                if origin == "original":
                    # Ensure originals point to *this* imported game file
                    c["origin"] = "original"
                    c["source_file"] = _resolve_path_str(out_path)
                    if not c.get("id"):
                        c["id"] = str(uuid.uuid4())
                    if not c.get("source_label"):
                        c["source_label"] = payload.get("source") or out_path.name
                    fixed_comments.append(c)
                    continue

                # Imported comment: try to rewrite source_file if it references a bundled game
                if origin == "imported":
                    src = str(c.get("source_file") or "").strip()
                    if src:
                        # If exporter wrote "games/<file>.json", map that too
                        src_norm = src.replace("\\", "/")
                        src_norm = src_norm[2:] if src_norm.startswith("./") else src_norm
                        if src_norm.startswith("games/"):
                            src_norm = src_norm
                        # Map by exact member, or by basename
                        new_src = rel_to_new_abs.get(src_norm) or rel_to_new_abs.get(Path(src_norm).name)

                        if new_src:
                            c["source_file"] = new_src
                        else:
                            # Keep but warn: it won’t resolve locally
                            report["warnings"].append(
                                f"Imported comment references a source not in this package: {src}"
                            )
                    fixed_comments.append(c)
                    continue

                fixed_comments.append(c)

            payload["comments"] = fixed_comments

            try:
                out_path.write_text(json.dumps(payload, indent=2))
                report["imported_games"].append(out_path)
            except Exception as e:
                report["errors"].append(f"Failed writing {out_path.name}: {e}")

        # 3) Import tactics (optional), rewriting source_game_file if possible
        imported_tactics = 0
        for member in sorted(tactic_members):
            payload = _read_zip_json(zf, member)
            if not payload:
                report["warnings"].append(f"Could not parse: {member}")
                continue

            src = str(payload.get("source_game_file") or "").strip()
            if src:
                src_norm = src.replace("\\", "/")
                src_norm = src_norm[2:] if src_norm.startswith("./") else src_norm
                if src_norm.startswith("games/"):
                    mapped = rel_to_new_abs.get(src_norm) or rel_to_new_abs.get(Path(src_norm).name)
                else:
                    mapped = rel_to_new_abs.get(src_norm) or rel_to_new_abs.get(Path(src_norm).name)

                if mapped:
                    payload["source_game_file"] = mapped
                else:
                    report["warnings"].append(f"Tactic references a game not imported: {src}")

            outp = _unique_tactic_path(payload.get("timestamp"))
            try:
                outp.write_text(json.dumps(payload, indent=2))
                imported_tactics += 1
            except Exception as e:
                report["errors"].append(f"Failed writing tactic {outp.name}: {e}")

        report["imported_tactics"] = imported_tactics
        return report

    # ---- SINGLE GAME JSON ----
    try:
        payload = json.loads(file_bytes.decode("utf-8", errors="replace"))
    except Exception as e:
        report["errors"].append(f"Could not parse JSON: {e}")
        return report

    moves = payload.get("moves_san") or []
    if not isinstance(moves, list) or not moves:
        report["errors"].append("JSON did not look like a saved game (missing moves_san).")
        return report

    payload["source"] = _unique_source_name(str(payload.get("source") or "").strip() or "Imported game")
    out_path = _unique_game_path(payload.get("timestamp"))

    # Rewrite originals to point at the new file
    fixed_comments: list[dict] = []
    for c in (payload.get("comments") or []):
        c = dict(c)
        if (c.get("origin") or "original") == "original":
            c["origin"] = "original"
            c["source_file"] = _resolve_path_str(out_path)
            if not c.get("id"):
                c["id"] = str(uuid.uuid4())
            if not c.get("source_label"):
                c["source_label"] = payload.get("source") or out_path.name
        fixed_comments.append(c)
    payload["comments"] = fixed_comments

    try:
        out_path.write_text(json.dumps(payload, indent=2))
        report["imported_games"].append(out_path)
    except Exception as e:
        report["errors"].append(f"Failed writing {out_path.name}: {e}")

    return report


def export_mcn_zip(
    selected_game_paths: list[Path],
    include_tactics: bool = True,
    include_source_games_for_imported_comments: bool = True,
) -> tuple[bytes, dict]:
    """
    Create a ZIP that can be re-imported via import_mcn_upload().
    ZIP layout:
      games/<game_file>.json
      tactics/<tactic_file>.json   (optional)
      manifest.json
    We rewrite comment source_file pointers to relative 'games/<file>.json' when possible.
    """
    meta = {"warnings": [], "games": 0, "tactics": 0}

    # Build closure of games to include (optional)
    to_include: dict[str, Path] = {}
    queue: list[Path] = [Path(p) for p in (selected_game_paths or [])]

    def add_game(p: Path):
        ap = _resolve_path_str(p)
        if not ap or ap in to_include:
            return
        to_include[ap] = p

    for p in queue:
        if p.exists():
            add_game(p)

    if include_source_games_for_imported_comments:
        # BFS: include games referenced by imported comments
        added = True
        while added:
            added = False
            current = list(to_include.values())
            for gp in current:
                payload = _read_json(gp) or {}
                for c in (payload.get("comments") or []):
                    if (c.get("origin") or "") != "imported":
                        continue
                    src = str(c.get("source_file") or "").strip()
                    if not src:
                        continue
                    srcp = Path(src)
                    if srcp.exists() and str(srcp.resolve()) not in to_include:
                        add_game(srcp)
                        added = True

    # Map abs->zip relative
    abs_to_rel: dict[str, str] = {}
    for ap, p in to_include.items():
        abs_to_rel[ap] = f"games/{Path(p).name}"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Games
        for ap, p in to_include.items():
            payload = _read_json(p)
            if not payload:
                meta["warnings"].append(f"Unreadable game skipped: {Path(p).name}")
                continue

            rel = abs_to_rel.get(_resolve_path_str(p))
            if not rel:
                continue

            # Rewrite comment pointers
            rewritten_comments: list[dict] = []
            for c in (payload.get("comments") or []):
                c = dict(c)
                origin = c.get("origin") or "original"

                if origin == "original":
                    c["source_file"] = rel
                    rewritten_comments.append(c)
                    continue

                if origin == "imported":
                    src = str(c.get("source_file") or "").strip()
                    src_abs = _resolve_path_str(src)
                    if src_abs and src_abs in abs_to_rel:
                        c["source_file"] = abs_to_rel[src_abs]
                    else:
                        # Leave as-is (may not resolve after import)
                        meta["warnings"].append(
                            f"Imported comment source not included in export: {Path(src).name if src else '(missing)'}"
                        )
                    rewritten_comments.append(c)
                    continue

                rewritten_comments.append(c)

            payload["comments"] = rewritten_comments
            zf.writestr(rel, json.dumps(payload, indent=2))
            meta["games"] += 1

        # Tactics (optional)
        if include_tactics:
            # include tactics whose source_game_file is among included games
            included_abs = set(abs_to_rel.keys())
            for tfp in list_tactics():
                tp = _read_json(tfp)
                if not tp:
                    continue
                src = _resolve_path_str(tp.get("source_game_file"))
                if src and src in included_abs:
                    tp = dict(tp)
                    tp["source_game_file"] = abs_to_rel[src]
                    zf.writestr(f"tactics/{Path(tfp).name}", json.dumps(tp, indent=2))
                    meta["tactics"] += 1

        manifest = {
            "format": "MyChessNotebookExport",
            "version": 1,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "games": meta["games"],
            "tactics": meta["tactics"],
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    return buf.getvalue(), meta


# ===================== Editability / Mode =====================
def is_read_only() -> bool:
    return st.session_state.get("view_mode") == "explore"

def _is_comment_editable_here(c: dict) -> bool:
    cur = st.session_state.get("current_file")
    src = c.get("source_file")
    return c.get("origin") == "original" and ((src == cur) if cur else (not src))

def _is_comment_deletable_here(c: dict) -> bool:
    return _is_comment_editable_here(c)

def is_comment_visible_at_ply(c: dict, ply_idx: int) -> bool:
    s = _safe_int(c.get("start", 0), 0)
    return s < ply_idx

def board_context_title() -> str:
    ss = st.session_state

    if ss.get("tactic_mode"):
        payload = ss.get("tactic_payload") or {}
        label = (payload.get("source_label") or "").strip()
        return f"Tactic mode — {label}" if label else "Tactic mode"

    if ss.get("current_file"):
        name = (ss.get("source_text") or "").strip()
        return name or "Loaded game"

    if ss.get("templated_from"):
        return "Templated game"

    return "New game"

# ===================== Move Annotation & Rendering =====================
@lru_cache(maxsize=256)
def _compute_move_annotations_cached(moves_tuple: Tuple[str, ...]) -> List[str]:
    """Return per-move suffix annotations ('+', '#', or '') based on resulting position."""
    board = chess.Board()
    anns: List[str] = []

    for san in moves_tuple:
        ann = ""
        try:
            mv = board.parse_san(san)
            board.push(mv)
            if board.is_checkmate():
                ann = "#"
            elif board.is_check():
                ann = "+"
        except Exception:
            ann = ""
        anns.append(ann)

    return anns


def compute_move_annotations(moves: List[str]) -> List[str]:
    return _compute_move_annotations_cached(tuple(moves))


def ply_label(i: int, moves: List[str]) -> str:
    """Human label for a ply index within a move list."""
    if not moves:
        return f"{(i // 2) + 1}."

    i = max(0, min(i, len(moves) - 1))
    annotations = compute_move_annotations(moves)

    base = moves[i].rstrip("+#")
    ann = annotations[i] if i < len(annotations) else ""
    san_here = base + ann

    move_num = (i // 2) + 1
    is_white = (i % 2 == 0)
    return f"{move_num}. {san_here}" if is_white else f"{move_num}. ... {san_here}"


def format_span(s: int, moves: List[str]) -> str:
    if not moves:
        return f"{(s // 2) + 1}."
    s = max(0, min(s, len(moves) - 1))
    return ply_label(s, moves)


def render_moves_bubbles(moves: List[str]) -> None:
    """Compact move list with current position highlight and comment markers in Explore/Walkthrough."""
    if not moves:
        st.markdown('<span class="small-muted">No moves yet.</span>', unsafe_allow_html=True)
        return

    annotations = compute_move_annotations(moves)

    commented_plies = set()
    for c in st.session_state.get("comments", []):
        s = _safe_int(c.get("start", -1), -1)
        if 0 <= s < len(moves):
            commented_plies.add(s)

    out = ['<div class="moves-bubbles">']
    for i, san in enumerate(moves):
        cls = "w" if i % 2 == 0 else "b"
        num = i // 2 + 1

        is_current = (i == st.session_state.current_ply_idx - 1)
        style = f' style="border: 2px solid {ACCENT};"' if is_current else ""

        base_san = san.rstrip("+#")
        ann = annotations[i] if i < len(annotations) else ""
        display_san = base_san + ann

        label = f"{num}. {display_san}" if cls == "w" else f"{display_san}"

        if i in commented_plies:
            label += '<span class="comment-marker small-muted">*</span>'

        out.append(f'<span class="{cls}"{style}>{label}</span>')

    out.append("</div>")
    st.markdown("".join(out), unsafe_allow_html=True)


# ===================== Source Name Uniqueness =====================
def _existing_sources_map() -> dict[str, List[Path]]:
    out: dict[str, List[Path]] = defaultdict(list)
    for fp in sorted(DATA_DIR.glob("game_*.json")):
        payload = _read_json(fp)
        if not payload:
            continue
        key = _normalize_source(payload.get("source", ""))
        if key:
            out[key].append(fp)
    return out


def _is_source_taken(source: str, exclude_path: Path | None = None) -> bool:
    norm = _normalize_source(source)
    if not norm:
        return False

    mp = _existing_sources_map()
    if norm not in mp:
        return False

    if exclude_path is None:
        return True

    return any(not _paths_equal(p, exclude_path) for p in mp[norm])


# ===================== Next-move Suggestion Index =====================
def _build_next_move_index_raw() -> Dict[Tuple[str, ...], Counter]:
    """
    Build an index mapping prefix(tuple of SAN moves) -> Counter(next SAN move).
    """
    index: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
    for fp in sorted(DATA_DIR.glob("game_*.json")):
        payload = _read_json(fp)
        if not payload:
            continue
        moves: List[str] = payload.get("moves_san", []) or []
        for i in range(len(moves)):
            prefix = tuple(moves[:i])
            nxt = moves[i]
            index[prefix][nxt] += 1
    return index


@st.cache_data
def build_next_move_index_cached(file_mtimes: Tuple[int, ...]) -> Dict[Tuple[str, ...], Counter]:
    """Cached wrapper. file_mtimes is only used as a cache key."""
    return _build_next_move_index_raw()


def get_suggestions(prefix: Tuple[str, ...], index: Dict[Tuple[str, ...], Counter], limit: int = 10):
    counts = index.get(prefix, Counter())
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


# ===================== Snapshot & Dirty Checks =====================
def _canonical_comments_for_snapshot(comments: List[dict]) -> List[dict]:
    """Canonicalize comments so snapshots are stable across ordering/paths."""
    def canon(c: dict) -> dict:
        origin = str(c.get("origin") or "").strip()
        start = _safe_int(c.get("start", 0), 0)
        text = str(c.get("text", "")).strip()

        src = c.get("source_file")
        src_abs = _resolve_path_str(src)

        return {
            "origin": origin,
            "start": start,
            "text": text,
            "id": str(c.get("id") or ""),
            "source_file": src_abs,
            "source_label": str(c.get("source_label") or "").strip(),
            "source_comment_id": str(c.get("source_comment_id") or ""),
        }

    items = [canon(c) for c in (comments or [])]
    items.sort(key=lambda d: (d["origin"], d["start"], d["text"], d["source_file"], d["id"], d["source_comment_id"]))
    return items


def _snapshot_current_game_state() -> str:
    payload = {
        "moves_san": list(st.session_state.get("moves_san", [])),
        "source": str(st.session_state.get("source_text") or "").strip(),
        "url": str(st.session_state.get("url_text") or "").strip(),
        "comments": _canonical_comments_for_snapshot(st.session_state.get("comments", [])),
    }
    return json.dumps(payload, sort_keys=True)


def moves_dirty_vs_loaded() -> bool:
    """
    True if the current game is a saved game and the move list differs from the
    baseline snapshot captured at open/overwrite time. (Comment-only changes ignored.)
    """
    cur_file = st.session_state.get("current_file")
    baseline = st.session_state.get("loaded_snapshot")
    if not cur_file or not baseline:
        return False

    try:
        base = json.loads(baseline)
        return list(base.get("moves_san", [])) != list(st.session_state.get("moves_san", []))
    except Exception:
        return False


# ===================== Board Positioning =====================
def _reposition_board(ply_idx: int | None = None) -> None:
    """Set board to the position after ply_idx and update current_ply_idx."""
    if ply_idx is None:
        ply_idx = int(st.session_state.current_ply_idx)

    ply_idx = max(0, min(ply_idx, len(st.session_state.moves_san)))
    st.session_state.current_ply_idx = ply_idx

    board = chess.Board()
    for i, san in enumerate(st.session_state.moves_san):
        if i >= ply_idx:
            break
        try:
            mv = board.parse_san(san)
            board.push(mv)
        except Exception:
            break

    st.session_state.board = board
    st.session_state.flash = ""


def _current_attach_ply() -> int:
    """
    Attach comments/selection to the last fully played move:
      attach_ply = current_ply_idx - 1 (clamped).
    """
    moves = st.session_state.get("moves_san", [])
    if not moves:
        return 0

    cur = int(st.session_state.get("current_ply_idx", 0))
    max_ply = len(moves) - 1
    if cur <= 0:
        return 0
    return min(cur - 1, max_ply)


# ===================== Persistence: Comments Serialization =====================
def _serialize_comments_for_save(
    comments: List[dict],
    max_ply: int,
    current_path: Path | None,
    default_source_label: str | None,
) -> List[dict]:
    """
    Normalize comments for saving:
    - clamp start to [0, max_ply]
    - imported comments keep source pointers
    - original comments get stable ids + source metadata
    """
    out: List[dict] = []
    cur_abs = _resolve_path_str(current_path)

    for c in comments:
        s = _safe_int(c.get("start", 0), 0)
        s = min(max(0, s), max_ply)
        text = str(c.get("text", ""))

        if c.get("origin") == "imported" and c.get("source_file"):
            src_file = _resolve_path_str(c.get("source_file"))
            base = {
                "origin": "imported",
                "text": text,
                "start": s,
                "source_file": src_file,
            }
            if c.get("source_label"):
                base["source_label"] = str(c["source_label"])
            if c.get("source_comment_id"):
                base["source_comment_id"] = str(c["source_comment_id"])
        else:
            cid = c.get("id") or str(uuid.uuid4())
            base = {
                "origin": "original",
                "id": cid,
                "text": text,
                "start": s,
            }

            source_file = cur_abs or _resolve_path_str(c.get("source_file"))
            if source_file:
                base["source_file"] = source_file

            label = default_source_label or c.get("source_label")
            if not label and cur_abs:
                label = Path(cur_abs).name
            if label:
                base["source_label"] = label

        out.append(base)

    return out


# ===================== Persistence: Save / Load Games =====================
def save_game_json(
    moves_san: List[str],
    source: str,
    url: str = "",
    comments: List[dict] | None = None,
) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")

    max_ply = max(0, len(moves_san) - 1)
    comments_payload = _serialize_comments_for_save(
        comments or [],
        max_ply=max_ply,
        current_path=None,
        default_source_label=source,
    )

    payload = {
        "timestamp": ts,
        "moves_san": moves_san,
        "source": source,
        "url": url,
        "comments": comments_payload,
    }

    out_path = DATA_DIR / f"game_{ts}.json"

    # Ensure original comments point to the final saved file.
    resolved = _resolve_path_str(out_path)
    for c in payload["comments"]:
        if c.get("origin") == "original":
            c["source_file"] = resolved
            if not c.get("source_label"):
                c["source_label"] = source

    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def _refresh_imported_comments_in_state() -> None:
    """
    If imported comments reference originals by source_comment_id, refresh their
    text/start from the current state of the source file.
    """
    refreshed: List[dict] = []

    for c in st.session_state.comments:
        if c.get("origin") != "imported":
            refreshed.append(c)
            continue

        src = _resolve_path_str(c.get("source_file"))
        cid = c.get("source_comment_id")
        if not src:
            refreshed.append(c)
            continue

        payload = _read_json(Path(src))
        if not payload:
            refreshed.append(c)
            continue

        found = None
        for sc in payload.get("comments", []):
            if cid and sc.get("id") == cid:
                found = sc
                break

        # Fallback for legacy imported comments without ids
        if not found and cid is None:
            for sc in payload.get("comments", []):
                if (
                    sc.get("origin") == "original"
                    and str(sc.get("text", "")).strip() == str(c.get("text", "")).strip()
                    and _safe_int(sc.get("start", -1), -1) == _safe_int(c.get("start", -2), -2)
                ):
                    found = sc
                    break

        if found:
            nc = dict(c)
            nc["text"] = found.get("text", c.get("text"))
            nc["start"] = _safe_int(found.get("start", c.get("start")), _safe_int(c.get("start", 0), 0))
            if not nc.get("source_comment_id") and found.get("id"):
                nc["source_comment_id"] = found.get("id")
            refreshed.append(nc)
        else:
            refreshed.append(c)

    st.session_state.comments = refreshed


def load_game_from_path(path: Path) -> None:
    if st.session_state.get("tactic_mode", False) or st.session_state.get("tactic_payload"):
        exit_tactic_mode_hard()

    payload = _read_json(path)
    if not payload:
        st.session_state.flash = f"Could not read {path.name}."
        return

    moves = payload.get("moves_san", []) or []
    source = payload.get("source", "") or ""
    url = payload.get("url", "") or ""
    comments = payload.get("comments", []) or []

    temp = chess.Board()
    applied: List[str] = []
    for raw_san in moves:
        mv, options = parse_san_lenient(temp, raw_san)
        if mv is None:
            break
        applied.append(temp.san(mv))  # canonicalize on read
        temp.push(mv)


    st.session_state.moves_san = applied
    st.session_state.source_text = source
    st.session_state.url_text = url
    st.session_state.move_input = ""
    st.session_state.current_file = _resolve_path_str(path)

    st.session_state.current_ply_idx = len(applied)
    _reposition_board()

    # Normalize comments to valid plies and unify metadata
    max_ply = max(0, len(applied) - 1)
    fixed: List[dict] = []

    for c in comments:
        s = min(max(0, _safe_int(c.get("start", 0), 0)), max_ply)
        origin = c.get("origin")

        if origin == "imported":
            src = c.get("source_file")
            src_label = c.get("source_label") or (Path(src).name if src else None)
            fixed.append(
                {
                    "text": str(c.get("text", "")),
                    "start": s,
                    "source_file": _resolve_path_str(src) or None,
                    "source_label": src_label,
                    "origin": "imported",
                    "source_comment_id": c.get("source_comment_id"),
                }
            )
        else:
            fixed.append(
                {
                    "id": c.get("id"),
                    "text": str(c.get("text", "")),
                    "start": s,
                    "source_file": st.session_state.current_file,
                    "source_label": c.get("source_label") or source or Path(path).name,
                    "origin": "original",
                }
            )

    st.session_state.comments = fixed
    _refresh_imported_comments_in_state()
    import_missing_comments_for_all_plies()

    st.session_state.view_mode = "explore"
    st.session_state.delete_pending = None
    st.session_state.flash = f"Opened {source} ({len(applied)} moves). Explore mode by default — click Edit to modify moves."

    # Coverage slider auto-follow
    st.session_state.pop("coverage_slider", None)
    st.session_state.coverage_slider_synced = True
    st.session_state.coverage_last_attach_ply = len(applied) - 1 if applied else 0

    # Walkthrough resets on open
    st.session_state.walkthrough_mode = False
    st.session_state.walkthrough_revealed = []
    st.session_state.templated_from = None

    # Snapshot baseline for dirty state
    st.session_state.loaded_snapshot = _snapshot_current_game_state()


# ===================== Reset State =====================
def reset_all() -> None:
    exit_tactic_mode_hard()
    ss = st.session_state
    ss.board = chess.Board()
    ss.moves_san = []
    ss.comments = []
    ss.move_input = ""
    ss.source_text = ""
    ss.url_text = ""
    ss.flash = ""
    ss.new_comment_text = ""
    ss.current_file = None
    ss.delete_pending = None
    ss.view_mode = "edit"
    ss.current_ply_idx = 0

    ss.pop("coverage_slider", None)
    ss.coverage_slider_synced = True
    ss.coverage_last_attach_ply = 0

    ss.walkthrough_mode = False
    ss.walkthrough_revealed = []

    ss.templated_from = None
    ss.loaded_snapshot = None

    # Tactic selection reset (creation tab)
    ss.tactic_start_selected = False
    ss.tactic_end_selected = False
    ss.tactic_start_black_ply = None
    ss.tactic_start_ply = 0
    ss.tactic_end_ply = 0


# ===================== Search: Find Games by Move =====================
def game_has_move_anywhere(moves_san: List[str], target_san: str, side: str = "white") -> bool:
    t = _norm_san(target_san)
    if not t:
        return False

    for i, san in enumerate(moves_san):
        if _norm_san(san) != t:
            continue
        if side == "white" and (i % 2 != 0):
            continue
        if side == "black" and (i % 2 != 1):
            continue
        return True

    return False


def find_games_with_move(target_san: str, side: str = "white") -> List[Path]:
    hits: List[Path] = []
    for fp in sorted(DATA_DIR.glob("game_*.json"), reverse=True):
        payload = _read_json(fp)
        if not payload:
            continue
        moves = payload.get("moves_san", []) or []
        if game_has_move_anywhere(moves, target_san, side=side):
            hits.append(fp)
    return hits


# ===================== Comment Import =====================
def import_comments_for_ply(ply: int) -> None:
    """
    Import any comments from saved games that match the *entire current line*
    (i.e., same move list) and have a comment attached at `ply`.
    """
    if ply < 0:
        return

    cur_moves = st.session_state.moves_san
    cur_len = len(cur_moves)
    if cur_len == 0 or ply >= cur_len:
        return

    existing = {(c.get("start"), c.get("text")) for c in st.session_state.comments}

    for fp in sorted(DATA_DIR.glob("game_*.json")):
        payload = _read_json(fp)
        if not payload:
            continue

        saved_moves: List[str] = payload.get("moves_san", []) or []
        if len(saved_moves) <= ply:
            continue
        if saved_moves[:cur_len] != cur_moves:
            continue

        for c in payload.get("comments", []) or []:
            s = _safe_int(c.get("start", -1), -1)
            text = str(c.get("text", "")).strip()
            if s == ply and text:
                key = (s, text)
                if key in existing:
                    continue

                src_label = payload.get("source") or Path(fp).name
                st.session_state.comments.append(
                    {
                        "text": text,
                        "start": s,
                        "source_file": _resolve_path_str(fp),
                        "source_label": src_label,
                        "origin": "imported",
                        "source_comment_id": c.get("id"),
                    }
                )
                existing.add(key)


def import_missing_comments_for_all_plies() -> None:
    """
    For each prefix of the current move list, import original comments from any other
    saved game with the same prefix at that ply.
    """
    cur_moves = st.session_state.moves_san
    if not cur_moves:
        return

    have_key = set()
    for c in st.session_state.comments:
        s = _safe_int(c.get("start", -1), -1)
        txt = str(c.get("text", "")).strip()
        have_key.add((s, txt))

    cur_file_abs = _resolve_path_str(st.session_state.get("current_file"))

    for ply in range(len(cur_moves)):
        prefix = cur_moves[: ply + 1]

        for fp in sorted(DATA_DIR.glob("game_*.json")):
            if cur_file_abs and _paths_equal(fp, cur_file_abs):
                continue

            payload = _read_json(fp)
            if not payload:
                continue

            saved_moves: List[str] = payload.get("moves_san", []) or []
            if len(saved_moves) <= ply:
                continue
            if saved_moves[: ply + 1] != prefix:
                continue

            src_label = payload.get("source") or Path(fp).name
            for sc in payload.get("comments", []) or []:
                if sc.get("origin") != "original":
                    continue

                s = _safe_int(sc.get("start", -1), -1)
                if s != ply:
                    continue

                txt = str(sc.get("text", "")).strip()
                if not txt:
                    continue

                if (s, txt) in have_key:
                    continue

                st.session_state.comments.append(
                    {
                        "text": txt,
                        "start": s,
                        "source_file": _resolve_path_str(fp),
                        "source_label": src_label,
                        "origin": "imported",
                        "source_comment_id": sc.get("id"),
                    }
                )
                have_key.add((s, txt))


def _propagate_comment_edit_to_imports(source_file: str, source_comment_id: str, new_text: str, new_start: int) -> None:
    """
    Update imported copies of a source comment across all saved games.
    """
    for fp in sorted(DATA_DIR.glob("game_*.json")):
        if _paths_equal(fp, source_file):
            continue

        payload = _read_json(fp)
        if not payload:
            continue

        changed = False
        for c in payload.get("comments", []) or []:
            if c.get("origin") != "imported":
                continue
            if not _paths_equal(c.get("source_file", ""), source_file):
                continue

            if c.get("source_comment_id") and c.get("source_comment_id") == source_comment_id:
                c["text"] = new_text
                c["start"] = int(new_start)
                changed = True

        if changed:
            _write_json(fp, payload)


# ===================== Sync Imported Comments (All Games) =====================
def _sync_all_imported_comments() -> tuple[int, int]:
    """
    Rescan all saved games and refresh imported comments from their source originals.
    Returns (files_touched, comments_synced).
    """
    originals_by_id: dict[tuple[str, str], dict] = {}
    originals_fallback: dict[tuple[str, int, str], dict] = {}

    # Pass 1: index original comments
    for fp in sorted(DATA_DIR.glob("game_*.json")):
        payload = _read_json(fp)
        if not payload:
            continue

        src_abs = _resolve_path_str(fp)
        src_label = (payload.get("source") or "").strip() or fp.name

        for c in payload.get("comments", []) or []:
            if c.get("origin") != "original":
                continue

            cid = c.get("id")
            start = _safe_int(c.get("start", 0), 0)
            text = str(c.get("text", "")).strip()

            entry = {
                "source_file": src_abs,
                "source_label": src_label,
                "id": cid,
                "start": start,
                "text": text,
            }

            if cid:
                originals_by_id[(src_abs, cid)] = entry
            originals_fallback[(src_abs, start, text)] = entry

    # Pass 2: rewrite imported comments
    files_touched = 0
    comments_synced = 0

    for fp in sorted(DATA_DIR.glob("game_*.json")):
        payload = _read_json(fp)
        if not payload:
            continue

        changed = False
        for c in payload.get("comments", []) or []:
            if c.get("origin") != "imported":
                continue

            src = c.get("source_file") or ""
            src_abs = _resolve_path_str(src)

            cid = c.get("source_comment_id")
            start = _safe_int(c.get("start", 0), 0)
            text = str(c.get("text", "")).strip()

            updated_from = None
            if src_abs and cid and (src_abs, cid) in originals_by_id:
                updated_from = originals_by_id[(src_abs, cid)]
            elif src_abs and (src_abs, start, text) in originals_fallback:
                updated_from = originals_fallback[(src_abs, start, text)]

            if updated_from:
                c["text"] = updated_from["text"]
                c["start"] = updated_from["start"]
                c["source_file"] = updated_from["source_file"]
                c["source_label"] = updated_from["source_label"]
                if updated_from.get("id"):
                    c["source_comment_id"] = updated_from["id"]
                changed = True
                comments_synced += 1

        if changed:
            if _write_json(fp, payload):
                files_touched += 1

    return files_touched, comments_synced


# ===================== Delete Propagation =====================
def _purge_imports_of_source(deleted_path: Path) -> tuple[int, int]:
    """
    Remove imported comments that reference deleted_path from all other games.
    Returns (files_touched, comments_removed).
    """
    deleted_abs = _resolve_path_str(deleted_path)
    files_touched = 0
    comments_removed = 0

    for fp in sorted(DATA_DIR.glob("game_*.json")):
        if _paths_equal(fp, deleted_abs):
            continue

        payload = _read_json(fp)
        if not payload:
            continue

        old_comments = payload.get("comments", []) or []
        new_comments: List[dict] = []
        removed_here = 0

        for c in old_comments:
            if c.get("origin") == "imported" and _paths_equal(c.get("source_file", ""), deleted_abs):
                removed_here += 1
                continue
            new_comments.append(c)

        if removed_here:
            payload["comments"] = new_comments
            if _write_json(fp, payload):
                files_touched += 1
                comments_removed += removed_here

    return files_touched, comments_removed


# ===================== Similarity =====================
def _find_similar_games(game_path: Path, limit: int = 5) -> List[dict]:
    """
    Find saved games with the longest common prefix (by SAN sequence).
    """
    payload = _read_json(game_path)
    if not payload:
        return []

    current_moves: List[str] = payload.get("moves_san", []) or []
    cur_abs = _resolve_path_str(game_path)
    out: List[dict] = []

    def divergence_label(other_moves: List[str], shared: int) -> str:
        if shared == 0:
            return "start position"
        last_idx = shared - 1
        try:
            return ply_label(last_idx, current_moves or other_moves)
        except Exception:
            return f"ply {last_idx + 1}"

    for fp in sorted(DATA_DIR.glob("game_*.json")):
        if _paths_equal(fp, cur_abs):
            continue

        other_payload = _read_json(fp)
        if not other_payload:
            continue

        other_moves: List[str] = other_payload.get("moves_san", []) or []

        lcp = 0
        for a, b in zip(current_moves, other_moves):
            if a != b:
                break
            lcp += 1

        ts = other_payload.get("timestamp", fp.stem.replace("game_", ""))
        label = (other_payload.get("source") or "").strip() or "(untitled)"

        out.append(
            {
                "path": fp,
                "label": label,
                "timestamp": ts,
                "shared_moves": lcp,
                "diverge_label": divergence_label(other_moves, lcp),
            }
        )

    out.sort(key=lambda d: (-d["shared_moves"], d["timestamp"]))
    return out[:limit]


# ===================== New-game Matching (Prefix -> Games) =====================
@st.cache_data
def build_games_index_cached(file_mtimes: Tuple[int, ...]) -> List[dict]:
    """
    Cached list of saved games with minimal fields for fast prefix filtering.
    file_mtimes is used only as a cache key.
    """
    out: List[dict] = []
    for fp in sorted(DATA_DIR.glob("game_*.json"), reverse=True):
        payload = _read_json(fp)
        if not payload:
            continue
        moves = payload.get("moves_san", []) or []
        ts = payload.get("timestamp", fp.stem.replace("game_", ""))
        label = (payload.get("source") or "").strip() or "(untitled)"
        out.append(
            {
                "path": fp,
                "timestamp": ts,
                "label": label,
                "moves": moves,
            }
        )
    return out


def find_games_matching_prefix(prefix_moves: List[str], games_index: List[dict]) -> List[dict]:
    """
    Return games whose moves start with prefix_moves.
    Adds 'next_san' (the next ply SAN) and 'remaining_plies'.
    """
    plen = len(prefix_moves)
    matches: List[dict] = []

    for g in games_index:
        moves = g.get("moves") or []
        if moves[:plen] != prefix_moves:
            continue

        next_san = moves[plen] if plen < len(moves) else None
        matches.append(
            {
                "path": g["path"],
                "label": g["label"],
                "timestamp": g["timestamp"],
                "next_san": next_san,
                "remaining_plies": max(0, len(moves) - plen),
                "moves": moves,
            }
        )

    # Prefer lines that continue (have a next move), then newest-ish timestamp as tie-breaker, then label.
    matches.sort(key=lambda d: (d["next_san"] is None, d["timestamp"], d["label"]), reverse=False)
    return matches


# ===================== Coverage Map =====================
def _build_coverage_tree_raw() -> dict:
    """
    Build a prefix tree over all saved games.

    Each node:
      - count: number of games that reach this position
      - children: {SAN -> node}
      - game_label: if exactly one game reaches this node, that game's label; otherwise None
    """
    tree: dict = {"count": 0, "children": {}, "game_label": None}

    def bump(node: dict, label: str) -> None:
        node["count"] = node.get("count", 0) + 1
        if node["count"] == 1:
            node["game_label"] = label
        else:
            if node.get("game_label") not in (None, label):
                node["game_label"] = None

    for fp in sorted(DATA_DIR.glob("game_*.json")):
        payload = _read_json(fp)
        if not payload:
            continue

        moves: List[str] = payload.get("moves_san", []) or []
        label = (payload.get("source") or "").strip() or fp.stem.replace("game_", "") or fp.name

        bump(tree, label)
        node = tree
        for san in moves:
            children = node.setdefault("children", {})
            if san not in children:
                children[san] = {"count": 0, "children": {}, "game_label": None}
            node = children[san]
            bump(node, label)

    return tree


@st.cache_data
def build_coverage_tree_cached(file_mtimes: Tuple[int, ...]) -> dict:
    """Cached wrapper. file_mtimes is only used as a cache key."""
    return _build_coverage_tree_raw()


def _render_coverage_html(tree: dict, max_depth: int = 20) -> str:
    """
    Render the coverage tree as indented HTML.

    max_depth is in plies (half-moves). When a node is uniquely reached by one game
    (count == 1) and has a game_label, render that label and stop recursing.
    """
    if not tree or tree.get("count", 0) == 0:
        return "<span class='small-muted'>No saved games yet.</span>"

    lines: List[str] = []
    total_games = tree.get("count", 0)
    lines.append(f"<div class='small-muted'>Total games in database: {total_games}</div>")

    def visit(children: dict, prefix_moves: List[str], depth: int) -> None:
        if depth >= max_depth:
            return

        sorted_items = sorted(
            children.items(),
            key=lambda kv: (-kv[1].get("count", 0), kv[0]),
        )

        for san, node in sorted_items:
            moves = prefix_moves + [san]
            ply_idx = len(moves) - 1

            try:
                label = ply_label(ply_idx, moves)
            except Exception:
                label = f"{ply_idx + 1}. {san}"

            indent = "&nbsp;" * 4 * depth
            count = node.get("count", 0)
            game_label = node.get("game_label")

            if count == 1 and game_label:
                suffix = f"<span class='small-muted'>({game_label})</span>"
                lines.append(f"{indent}{label} {suffix}<br>")
                continue

            suffix = f"<span class='small-muted'>({count})</span>"
            lines.append(f"{indent}{label} {suffix}<br>")

            visit(node.get("children", {}), moves, depth + 1)

    visit(tree.get("children", {}), [], 0)
    return "<div class='coverage-map'>" + "".join(lines) + "</div>"


# ===================== Tactics: Helpers =====================
def _apply_moves_to_board(moves: List[str]) -> chess.Board:
    b = chess.Board()
    for san in moves:
        b.push_san(san)
    return b


def normalize_tactic_window(moves_len: int, start_ply: int, end_ply: int) -> tuple[int, int]:
    """
    Normalize (start_ply, end_ply) so both are White plies (even indices), clamped,
    and end_ply >= start_ply.
    """
    if moves_len <= 0:
        return 0, 0

    start_ply = max(0, min(start_ply, moves_len - 1))
    end_ply = max(0, min(end_ply, moves_len - 1))

    if start_ply % 2 == 1:
        start_ply = min(start_ply + 1, moves_len - 1)

    if end_ply % 2 == 1:
        end_ply = max(end_ply - 1, 0)

    end_ply = max(start_ply, end_ply)
    return start_ply, end_ply


def save_tactic_json(
    source_game_file: str,
    source_label: str,
    url: str,
    start_ply: int,
    end_ply: int,
    game_moves: List[str],
    note: str = "",
    tags: List[str] | None = None, 
) -> Path:
    """
    Save a tactic as:
    - start_fen: position after moves[:start_ply]
    - moves_san: moves from start_ply..end_ply inclusive (tactic line)
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    tid = str(uuid.uuid4())

    start_moves = game_moves[:start_ply]
    tactic_moves = game_moves[start_ply : end_ply + 1]

    start_board = _apply_moves_to_board(start_moves)
    trigger_black_ply = max(0, int(start_ply) - 1)
    trigger_black_move = ""
    trigger_black_move_num = None
    if 0 <= trigger_black_ply < len(game_moves):
        trigger_black_move = str(game_moves[trigger_black_ply])
        trigger_black_move_num = (trigger_black_ply // 2) + 1

    payload = {
        "id": tid,
        "timestamp": ts,
        "source_game_file": _resolve_path_str(source_game_file),
        "source_label": source_label,
        "url": url or "",
        "start_ply": int(start_ply),
        "end_ply": int(end_ply),
        "start_fen": start_board.fen(),
        "moves_san": tactic_moves,
        "note": str(note or "").rstrip(),
        "tags": sorted(set(tags or [])),
        "trigger_black_ply": trigger_black_ply,
        "trigger_black_move": trigger_black_move,
        "trigger_black_move_num": trigger_black_move_num,
    }

    out_path = TACTICS_DIR / f"tactic_{ts}_{tid[:8]}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def list_tactics() -> List[Path]:
    return sorted(TACTICS_DIR.glob("tactic_*.json"), reverse=True)


def load_tactic(path: Path) -> dict | None:
    return _read_json(path)


def load_random_tactic(
    selected_tags: List[str] | None = None,
    match_all: bool = False,
) -> tuple[Path | None, dict | None]:
    """
    If selected_tags is provided, pick a random tactic whose payload['tags']
    matches the filter.
    """
    file_mtimes = _tactic_file_mtimes()
    catalog = build_tactics_catalog_cached(file_mtimes)

    if not catalog:
        return None, None

    selected = [str(t).strip() for t in (selected_tags or []) if str(t).strip()]
    if selected:
        catalog = [it for it in catalog if _tactic_matches_selected_tags(it.get("tags") or [], selected, match_all)]

    if not catalog:
        return None, None

    item = random.choice(catalog)
    return Path(item["path"]), item["payload"]

def list_tactics_for_game(game_file: str) -> List[tuple[Path, dict]]:
    """
    Return [(path, payload), ...] for tactics whose source_game_file matches game_file.
    """
    if not game_file:
        return []

    game_abs = _resolve_path_str(game_file)

    out: List[tuple[Path, dict]] = []
    for fp in list_tactics():  # already reverse-sorted newest first
        payload = load_tactic(fp)
        if not payload:
            continue

        src = payload.get("source_game_file") or ""
        src_abs = _resolve_path_str(src)

        if src_abs and _paths_equal(src_abs, game_abs):
            out.append((fp, payload))

    return out

def _tactic_reset_runtime() -> None:
    st.session_state.tactic_board = chess.Board()
    st.session_state.tactic_offset = 0
    st.session_state.tactic_input = ""
    st.session_state.tactic_feedback = ""
    st.session_state.tactic_complete = False
    st.session_state.tactic_note_flash = ""
    st.session_state.tactic_reveal_next = False
    st.session_state.tactic_show_notes = False
    st.session_state.tactic_show_solution = False
    st.session_state.tactic_show_tags = False  # optional



def _tactic_load_into_state(fp: Path, payload: dict) -> None:
    st.session_state.tactic_current_file = _resolve_path_str(fp)
    st.session_state.tactic_payload = payload
    st.session_state.tactic_show_notes = False
    _tactic_reset_runtime()

    fen = payload.get("start_fen") or chess.Board().fen()
    try:
        st.session_state.tactic_board = chess.Board(fen)
    except Exception:
        st.session_state.tactic_board = chess.Board()

    st.session_state.tactic_note_draft = str(payload.get("note", "") or "")

    # Precompute trigger label for the UI
    st.session_state.tactic_trigger_text = _tactic_trigger_text(payload)



def _tactic_autoplay_black_if_needed() -> None:
    """
    After a correct White move, if a Black reply exists in the tactic line, autoplay it.
    Tactic offset counts plies within payload['moves_san'].
    """
    payload = st.session_state.get("tactic_payload") or {}
    moves = payload.get("moves_san") or []
    off = int(st.session_state.get("tactic_offset", 0))

    # If it's Black's ply within the tactic, autoplay.
    if off < len(moves) and (off % 2 == 1):
        try:
            st.session_state.tactic_board.push_san(moves[off])
            st.session_state.tactic_offset = off + 1
        except Exception:
            st.session_state.tactic_feedback = "Tactic data error (could not play Black reply)."
            st.session_state.tactic_complete = True

FILES = "abcdefgh"

def _is_light_square(sq: chess.Square) -> bool:
    # a1 is dark. (file+rank) odd => light
    return (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1


def _tag_for_piece_at_square(piece: chess.Piece, sq: chess.Square,
                             pawn_id_by_square: dict[chess.Square, str]) -> str:
    color = "white" if piece.color == chess.WHITE else "black"

    if piece.piece_type == chess.PAWN:
        # Prefer identity from tactic-start file mapping; fallback to file at capture square
        return pawn_id_by_square.get(sq) or f"{color} {FILES[chess.square_file(sq)]}-pawn"

    if piece.piece_type == chess.BISHOP:
        sq_color = "light-squared" if _is_light_square(sq) else "dark-squared"
        return f"{color} {sq_color} bishop"

    return f"{color} {chess.piece_name(piece.piece_type)}"


def compute_tactic_piece_tags(start_fen: str, tactic_moves_san: List[str]) -> List[str]:
    """
    Tags every piece that:
      - moves in the tactic window, OR
      - gets captured during the tactic window (including en passant)

    Pawn tags use FILE at tactic start (carried forward as pawns move):
      'white a-pawn', 'black c-pawn', etc.
    Bishops are light/dark-squared.
    """
    try:
        board = chess.Board(start_fen)
    except Exception:
        board = chess.Board()

    # Pawn identity anchored to file at tactic start; carried forward as pawns move.
    pawn_id_by_square: dict[chess.Square, str] = {}
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.PAWN:
            color = "white" if piece.color == chess.WHITE else "black"
            pawn_id_by_square[sq] = f"{color} {FILES[chess.square_file(sq)]}-pawn"

    tags: set[str] = set()

    for san in (tactic_moves_san or []):
        try:
            move = board.parse_san(san)
        except Exception:
            break

        mover = board.piece_at(move.from_square)

        # --- TAG CAPTURED PIECE (victim) BEFORE PUSH ---
        if board.is_capture(move):
            if board.is_en_passant(move):
                # Captured pawn is not on to_square.
                # If White captures e.p., captured pawn is one rank behind to_square (to-8).
                # If Black captures e.p., captured pawn is one rank ahead to_square (to+8).
                cap_sq = move.to_square - 8 if (mover and mover.color == chess.WHITE) else move.to_square + 8
            else:
                cap_sq = move.to_square

            victim = board.piece_at(cap_sq)
            if victim:
                tags.add(_tag_for_piece_at_square(victim, cap_sq, pawn_id_by_square))

            # If a pawn was captured, remove its identity mapping (if present)
            pawn_id_by_square.pop(cap_sq, None)

        # --- TAG MOVING PIECE ---
        if mover:
            tags.add(_tag_for_piece_at_square(mover, move.from_square, pawn_id_by_square))

            # Track pawn identity movement
            if mover.piece_type == chess.PAWN:
                pid = pawn_id_by_square.get(move.from_square) or f"{('white' if mover.color else 'black')} {FILES[chess.square_file(move.from_square)]}-pawn"
                pawn_id_by_square.pop(move.from_square, None)
                pawn_id_by_square[move.to_square] = pid

            # Castling also moves a rook
            if board.is_castling(move):
                color = "white" if mover.color == chess.WHITE else "black"
                tags.add(f"{color} rook")

            # Promotion (optional extra tag)
            if move.promotion:
                color = "white" if mover.color == chess.WHITE else "black"
                tags.add(f"{color} promotion to {chess.piece_name(move.promotion)}")

        board.push(move)

    def _sort_key(t: str):
        return (0 if t.startswith("white ") else 1, t)

    return sorted(tags, key=_sort_key)


def _tactic_trigger_text(payload: dict) -> str:
    """
    Returns something like: '7... c5' (the last Black move before the tactic starts).
    Uses stored fields if present; otherwise tries to read the source game.
    """
    if not payload:
        return ""

    # Prefer stored fields (new tactics)
    san = (payload.get("trigger_black_move") or "").strip()
    move_num = payload.get("trigger_black_move_num")

    start_ply = _safe_int(payload.get("start_ply", 0), 0)
    if move_num is None and start_ply > 0:
        move_num = ((start_ply - 1) // 2) + 1

    # Fallback for older tactics: read from source game file
    if not san:
        src = _resolve_path_str(payload.get("source_game_file"))
        if src:
            gp = _read_json(Path(src))
            if gp:
                moves = gp.get("moves_san", []) or []
                trigger_black_ply = max(0, start_ply - 1)
                if 0 <= trigger_black_ply < len(moves):
                    san = str(moves[trigger_black_ply]).strip()
                    if move_num is None:
                        move_num = (trigger_black_ply // 2) + 1

    if not san or not move_num:
        return ""

    return f"{move_num}... {san}"

def _tactic_file_mtimes() -> Tuple[int, ...]:
    return tuple(sorted(fp.stat().st_mtime_ns for fp in TACTICS_DIR.glob("tactic_*.json")))

@st.cache_data
def build_tactics_catalog_cached(file_mtimes: Tuple[int, ...]) -> List[dict]:
    """
    Cached list of tactics with payload + tags for fast filtering.
    file_mtimes is used only as a cache key.
    """
    out: List[dict] = []
    for fp in sorted(TACTICS_DIR.glob("tactic_*.json"), reverse=True):
        payload = _read_json(fp)
        if not payload:
            continue
        out.append(
            {
                "path": _resolve_path_str(fp),
                "payload": payload,
                "tags": payload.get("tags") or [],
            }
        )
    return out

def _tactic_matches_selected_tags(tactic_tags: List[str], selected: List[str], match_all: bool) -> bool:
    if not selected:
        return True
    stags = set(tactic_tags or [])
    sel = set(selected or [])
    return sel.issubset(stags) if match_all else bool(stags.intersection(sel))

def _all_tactic_tags_from_catalog(catalog: List[dict]) -> List[str]:
    tags: set[str] = set()
    for it in catalog:
        for t in it.get("tags") or []:
            if t:
                tags.add(str(t))

    def _sort_key(t: str):
        # white tags first, then black, then alphabetical
        return (0 if t.startswith("white ") else 1, t)

    return sorted(tags, key=_sort_key)


def format_tactic_solution(payload: dict) -> str:
    """
    Format tactic solution as chess-style lines:
      6. c5 Qxb3
      7. axb3

    Uses payload['start_ply'] (0-based ply in source game) to compute the first move number.
    Assumes start_ply is a White ply (even), but tolerates odd just in case.
    """
    if not payload:
        return "(empty)"

    moves = payload.get("moves_san") or []
    if not moves:
        return "(empty)"

    start_ply = _safe_int(payload.get("start_ply", 0), 0)

    # Move number in chess notation is 1-based and increments after White's move.
    # If start_ply is even => White to play at move ((start_ply // 2) + 1).
    # If start_ply is odd  => Black to play at move ((start_ply // 2) + 1).
    move_num = (start_ply // 2) + 1

    lines: list[str] = []
    i = 0

    # Rare edge case: tactic begins with a Black move.
    if start_ply % 2 == 1:
        lines.append(f"{move_num}... {moves[0]}")
        i = 1
        move_num += 1

    # Normal: group as (White, Black) per move number
    while i < len(moves):
        w = moves[i]
        b = moves[i + 1] if (i + 1) < len(moves) else None

        if b:
            lines.append(f"{move_num}. {w} {b}")
        else:
            lines.append(f"{move_num}. {w}")

        move_num += 1
        i += 2

    return "\n".join(lines)

def exit_tactic_mode_hard() -> None:
    """Reset ALL tactic-mode UI/runtime so opening a game feels like tactic mode never happened."""
    ss = st.session_state

    ss.tactic_mode = False

    ss.tactic_current_file = None
    ss.tactic_payload = None
    ss.tactic_board = chess.Board()
    ss.tactic_offset = 0
    ss.tactic_input = ""
    ss.tactic_feedback = ""
    ss.tactic_complete = False
    ss.tactic_reveal_next = False

    ss.tactic_note_draft = ""
    ss.tactic_note_flash = ""
    ss.tactic_trigger_text = ""
    ss.tactic_show_notes = False

def make_load_tactic_from_manage(path_str: str, payload: dict):
    def _cb():
        st.session_state.tactic_mode = True
        _tactic_load_into_state(Path(path_str), payload)
    return _cb


def on_mark_delete_tactic(path_str: str) -> None:
    st.session_state.tactic_delete_pending = str(path_str)


def on_cancel_delete_tactic() -> None:
    st.session_state.tactic_delete_pending = None


def make_delete_tactic_confirm(path_str: str):
    def _cb():
        p = Path(path_str)
        try:
            # If deleting the currently loaded tactic, clear it first
            cur = st.session_state.get("tactic_current_file")
            deleting_current = _paths_equal(cur, path_str)

            if p.exists():
                p.unlink()

            st.session_state.tactic_delete_pending = None
            st.session_state.tactic_note_flash = ""
            st.session_state.tactic_feedback = f"Deleted tactic: {p.name}"

            if deleting_current:
                st.session_state.tactic_current_file = None
                st.session_state.tactic_payload = None
                _tactic_reset_runtime()

                # Optional: auto-load the next tactic (respects your existing filter)
                fp, payload = load_random_tactic(
                    selected_tags=st.session_state.get("tactic_filter_tags", []),
                    match_all=bool(st.session_state.get("tactic_filter_match_all", False)),
                )
                if payload:
                    _tactic_load_into_state(fp, payload)
                else:
                    st.session_state.tactic_feedback = "Deleted. No remaining tactics match your filter."

        except Exception as e:
            st.session_state.tactic_delete_pending = None
            st.session_state.tactic_feedback = f"Could not delete tactic: {e}"
    return _cb




# ===================== Callbacks: Navigation =====================
def on_step_forward() -> None:
    _reposition_board(st.session_state.current_ply_idx + 1)


def on_step_backward() -> None:
    _reposition_board(st.session_state.current_ply_idx - 1)


def on_go_to_start() -> None:
    _reposition_board(0)


def on_go_to_end() -> None:
    _reposition_board(len(st.session_state.moves_san))


def _jump_to_allowed_ply(allowed: List[int], direction: int) -> None:
    """Jump among a discrete list of allowed current_ply_idx values."""
    if not allowed:
        return
    cur = int(st.session_state.get("current_ply_idx", 0))
    allowed_sorted = sorted(allowed)

    if direction > 0:
        nxt = next((x for x in allowed_sorted if x > cur), allowed_sorted[-1])
        _reposition_board(nxt)
    else:
        prv = next((x for x in reversed(allowed_sorted) if x < cur), allowed_sorted[0])
        _reposition_board(prv)


def _allowed_ply_idxs_after_black(moves_len: int) -> List[int]:
    """
    Positions after a Black move => current_ply_idx is even and >= 2.
    (Because last played move index is current_ply_idx-1; odd means Black.)
    """
    return [i for i in range(2, moves_len + 1) if i % 2 == 0]


def _allowed_ply_idxs_after_white(moves_len: int, start_white_ply: int) -> List[int]:
    """
    Positions after a White move => current_ply_idx is odd.
    End selection begins after the first White move in the window.
    """
    lo = max(1, start_white_ply + 1)
    return [i for i in range(lo, moves_len + 1) if i % 2 == 1]


# ===================== Callbacks: Moves =====================
def _try_add_move(move_text: str) -> None:
    move_text = normalize_castling_san(move_text)
    if not move_text:
        st.session_state.flash = "Type a move or click a suggestion."
        return

    if st.session_state.current_ply_idx != len(st.session_state.moves_san):
        st.session_state.flash = "Can only add a move when at the end of the line. Use the navigation controls to get there."
        return

    try:
        temp = chess.Board()
        for san in st.session_state.moves_san:
            # These should be canonical after this change, but still safe.
            temp.push(temp.parse_san(san))

        mv, options = parse_san_lenient(temp, move_text)
        if mv is None:
            if options:
                st.session_state.flash = "Ambiguous move. Try one of: " + ", ".join(options[:8])
            else:
                st.session_state.flash = f"Illegal or unrecognized move: {move_text}"
            return

        canonical = temp.san(mv)   # canonicalizes SAN (adds captures/check markers)
        temp.push(mv)

        st.session_state.moves_san.append(canonical)
        st.session_state.move_input = ""
        st.session_state.flash = ""

        import_comments_for_ply(len(st.session_state.moves_san) - 1)
        _reposition_board(len(st.session_state.moves_san))

    except Exception:
        st.session_state.flash = f"Illegal or unrecognized move: {move_text}"



def on_add_from_text() -> None:
    move_text = (st.session_state.move_input or "").strip()
    _try_add_move(move_text)


def make_on_add_suggested(move_san: str):
    def _cb():
        _try_add_move(move_san)
    return _cb


def on_undo() -> None:
    if not st.session_state.moves_san:
        st.session_state.flash = ""
        return

    if st.session_state.current_ply_idx != len(st.session_state.moves_san):
        st.session_state.flash = "First navigate to the end of the line to undo."
        return

    st.session_state.moves_san.pop()

    # Clamp comments to new end
    max_ply = max(0, len(st.session_state.moves_san) - 1)
    new_comments: List[dict] = []
    for c in st.session_state.comments:
        s = _safe_int(c.get("start", 0), 0)
        if s <= max_ply:
            c["start"] = min(s, max_ply)
            new_comments.append(c)
    st.session_state.comments = new_comments

    _reposition_board(len(st.session_state.moves_san))
    st.session_state.flash = ""


def on_clear() -> None:
    reset_all()


def on_flip() -> None:
    st.session_state.flipped = not st.session_state.flipped


# ===================== Callbacks: Modes / Templates =====================
def on_set_explore_mode() -> None:
    """Switch current saved game to Explore mode (moves locked)."""
    if not st.session_state.get("current_file"):
        return
    st.session_state.view_mode = "explore"
    st.session_state.flash = "Explore mode: moves locked (comments still editable in their source game)."

def on_set_edit_mode() -> None:
    """Switch current saved game to Edit mode (moves unlocked)."""
    if not st.session_state.get("current_file"):
        return
    st.session_state.view_mode = "edit"
    st.session_state.flash = "Edit mode: moves unlocked. Remember to Save changes when done."


def on_use_as_template() -> None:
    """
    Create a new, unsaved game from the currently opened saved game:
    - moves and URL are copied
    - this game's original comments become imported comments
    """
    cur_file = st.session_state.get("current_file")
    if not cur_file:
        return

    cur_abs = _resolve_path_str(cur_file)
    cur_name = Path(cur_file).name

    rebased: List[dict] = []
    for c in st.session_state.comments:
        s = _safe_int(c.get("start", 0), 0)
        origin = c.get("origin")
        src = c.get("source_file")
        src_abs = _resolve_path_str(src) or None

        if origin == "original" and src_abs and _paths_equal(src_abs, cur_abs):
            rebased.append(
                {
                    "text": str(c.get("text", "")),
                    "start": s,
                    "source_file": cur_abs,
                    "source_label": c.get("source_label") or st.session_state.get("source_text") or cur_name,
                    "origin": "imported",
                    "source_comment_id": c.get("id"),
                }
            )
        else:
            nc = dict(c)
            nc["start"] = s
            rebased.append(nc)

    st.session_state.comments = rebased

    st.session_state.current_file = None
    st.session_state.delete_pending = None
    st.session_state.view_mode = "edit"

    st.session_state.source_text = ""
    st.session_state.walkthrough_mode = False
    st.session_state.walkthrough_revealed = []

    st.session_state.flash = "Using this game as a template. Comments are imported from the original game."
    st.session_state.templated_from = cur_abs
    st.session_state.loaded_snapshot = None


# ===================== Callbacks: Comments =====================
def on_add_comment() -> None:
    text = (st.session_state.new_comment_text or "").strip()
    if not text:
        st.session_state.flash = "Enter comment text."
        return
    if not st.session_state.moves_san:
        st.session_state.flash = "Add at least one move before attaching a comment."
        return

    attach_ply = _current_attach_ply()
    cur_file = st.session_state.get("current_file")

    st.session_state.comments.append(
        {
            "id": str(uuid.uuid4()),
            "text": text,
            "start": attach_ply,
            "source_file": _resolve_path_str(cur_file) or None,
            "source_label": st.session_state.get("source_text") or (Path(cur_file).name if cur_file else None),
            "origin": "original",
        }
    )

    st.session_state.new_comment_text = ""
    st.session_state.flash = ""


def make_delete_comment(idx: int):
    def _cb():
        if 0 <= idx < len(st.session_state.comments):
            c = st.session_state.comments[idx]
            if not _is_comment_deletable_here(c):
                st.session_state.flash = "This comment isn’t deletable here."
                return

            del st.session_state.comments[idx]

            edit_idx = st.session_state.get("edit_comment_idx")
            if edit_idx is not None:
                if idx == edit_idx:
                    st.session_state.edit_comment_idx = None
                    st.session_state.new_comment_text = ""
                elif idx < edit_idx:
                    st.session_state.edit_comment_idx = edit_idx - 1
    return _cb


def make_start_edit_comment(idx: int):
    def _cb():
        if 0 <= idx < len(st.session_state.comments):
            c = st.session_state.comments[idx]
            if _is_comment_editable_here(c):
                st.session_state.edit_comment_idx = idx
                st.session_state.new_comment_text = c.get("text", "")
            else:
                st.session_state.flash = "This comment isn’t editable here."
    return _cb


def on_confirm_edit_comment() -> None:
    idx = st.session_state.get("edit_comment_idx")
    if idx is None or not (0 <= idx < len(st.session_state.comments)):
        st.session_state.flash = "No comment selected for editing."
        st.session_state.edit_comment_idx = None
        return

    text = (st.session_state.new_comment_text or "").strip()
    if not text:
        st.session_state.flash = "Comment text cannot be empty."
        return

    old = st.session_state.comments[idx]
    if not _is_comment_editable_here(old):
        st.session_state.flash = "This comment isn’t editable here."
        return

    s = _safe_int(old.get("start", 0), 0)
    new_id = old.get("id") or str(uuid.uuid4())

    st.session_state.comments[idx] = {
        "id": new_id,
        "text": text,
        "start": s,
        "source_file": old.get("source_file"),
        "source_label": old.get("source_label"),
        "origin": "original",
    }

    st.session_state.edit_comment_idx = None
    st.session_state.new_comment_text = ""
    st.session_state.flash = ""

    cur_file = st.session_state.get("current_file")
    if cur_file:
        on_save_comments_to_current()
        try:
            _propagate_comment_edit_to_imports(
                source_file=old.get("source_file"),
                source_comment_id=new_id,
                new_text=text,
                new_start=s,
            )
        except Exception:
            pass


def on_cancel_edit_comment() -> None:
    st.session_state.edit_comment_idx = None
    st.session_state.new_comment_text = ""
    st.session_state.flash = ""


def make_reveal_comment(idx: int):
    """In walkthrough mode, reveal a comment's text."""
    def _cb():
        revealed = list(st.session_state.get("walkthrough_revealed", []))
        if idx not in revealed:
            revealed.append(idx)
            st.session_state.walkthrough_revealed = revealed
    return _cb


# ===================== Callbacks: Persist Current Game =====================
def on_update_all_games() -> None:
    try:
        files_touched, comments_synced = _sync_all_imported_comments()
        st.session_state.flash = (
            f"Updated imported comments in {files_touched} games "
            f"({comments_synced} comments synchronized)."
        )
    except Exception as e:
        st.session_state.flash = f"Update-all failed: {e}"


def on_save_comments_to_current() -> None:
    path_str = st.session_state.get("current_file")
    if not path_str:
        st.session_state.flash = "Open a saved game first (no current file to update)."
        return

    p = Path(path_str)
    payload = _read_json(p)
    if not payload:
        st.session_state.flash = f"Couldn't read {p.name}."
        return

    max_ply = max(0, len(st.session_state.moves_san) - 1)
    payload["comments"] = _serialize_comments_for_save(
        st.session_state.comments,
        max_ply=max_ply,
        current_path=p,
        default_source_label=payload.get("source") or None,
    )

    if _write_json(p, payload):
        st.session_state.flash = f"Updated comments in {p.name}"
    else:
        st.session_state.flash = f"Couldn't write {p.name}"


def on_save() -> None:
    if not st.session_state.moves_san:
        st.session_state.flash = "Add at least one move before saving."
        return

    source = (st.session_state.source_text or "").strip()
    if not source:
        st.session_state.flash = "Please enter a Source (this is the game's name)."
        return

    if _is_source_taken(source):
        st.session_state.flash = f'A game with Source "{source}" already exists. Choose a different Source.'
        return

    path = save_game_json(
        st.session_state.moves_san,
        source,
        (st.session_state.url_text or "").strip(),
        st.session_state.comments,
    )
    st.session_state.flash = f"Saved: {path}"
    reset_all()


def on_save_overwrite_current() -> None:
    path_str = st.session_state.get("current_file")
    if not path_str:
        st.session_state.flash = "Open a saved game first."
        return

    p = Path(path_str)
    payload = _read_json(p)
    if not payload:
        st.session_state.flash = f"Couldn't read {p.name}."
        return

    new_source = (st.session_state.source_text or "").strip()
    new_url = (st.session_state.url_text or "").strip()

    if not new_source:
        st.session_state.flash = "Source (game name) cannot be empty."
        return

    if _is_source_taken(new_source, exclude_path=p):
        st.session_state.flash = f'A different game already uses Source "{new_source}". Pick another name.'
        return

    payload["source"] = new_source
    payload["url"] = new_url
    payload["moves_san"] = list(st.session_state.moves_san)

    max_ply = max(0, len(st.session_state.moves_san) - 1)
    payload["comments"] = _serialize_comments_for_save(
        st.session_state.comments,
        max_ply=max_ply,
        current_path=p,
        default_source_label=new_source,
    )

    if _write_json(p, payload):
        st.session_state.flash = f"Saved changes to {p.name}"
        st.session_state.loaded_snapshot = _snapshot_current_game_state()
    else:
        st.session_state.flash = f"Couldn't write {p.name}"


# ===================== Callbacks: Delete Current Game =====================
def on_mark_delete_current() -> None:
    cur = st.session_state.get("current_file")
    if not cur:
        st.session_state.flash = "Open a saved game first."
        return
    st.session_state.delete_pending = cur
    st.session_state.flash = f"Confirm delete: {Path(cur).name}"


def on_cancel_delete_current() -> None:
    st.session_state.delete_pending = None
    st.session_state.flash = ""


def make_delete_confirm(path: Path):
    def _cb():
        try:
            files_touched, comments_removed = _purge_imports_of_source(path)
            path.unlink()
            st.session_state.flash = (
                f"Deleted {path.name}. Removed {comments_removed} imported comments from {files_touched} files."
            )
            cur = st.session_state.get("current_file")
            if cur and _paths_equal(cur, path):
                reset_all()
                st.session_state.current_file = None
        except Exception as e:
            st.session_state.flash = f"Could not delete {path.name}: {e}"
        finally:
            st.session_state.delete_pending = None
    return _cb


# ===================== Callbacks: Tactic Mode (Training) =====================
def on_enter_tactic_mode() -> None:
    st.session_state.tactic_mode = True
    st.session_state.walkthrough_mode = False
    st.session_state.walkthrough_revealed = []

    fp, payload = load_random_tactic(
        selected_tags=st.session_state.get("tactic_filter_tags", []),
        match_all=bool(st.session_state.get("tactic_filter_match_all", False)),
    )

    if not payload:
        if st.session_state.get("tactic_filter_tags"):
            st.session_state.tactic_feedback = "No tactics match your current filter. Clear or adjust the filter."
        else:
            st.session_state.tactic_feedback = "No tactics saved yet. Create one in the Tactics tab first."
        st.session_state.tactic_payload = None
        st.session_state.tactic_current_file = None
        st.session_state.tactic_complete = False
        st.session_state.tactic_board = chess.Board()
        return

    _tactic_load_into_state(fp, payload)



def on_exit_tactic_mode() -> None:
    exit_tactic_mode_hard()


def on_next_tactic() -> None:
    fp, payload = load_random_tactic(
        selected_tags=st.session_state.get("tactic_filter_tags", []),
        match_all=bool(st.session_state.get("tactic_filter_match_all", False)),
    )

    if not payload:
        st.session_state.tactic_feedback = (
            "No tactics match your current filter." if st.session_state.get("tactic_filter_tags") else "No tactics saved yet."
        )
        return

    _tactic_load_into_state(fp, payload)

def on_submit_tactic_move() -> None:
    payload = st.session_state.get("tactic_payload") or {}
    moves = payload.get("moves_san") or []
    off = int(st.session_state.get("tactic_offset", 0))

    if st.session_state.get("tactic_complete"):
        return
    if not moves or off >= len(moves):
        st.session_state.tactic_complete = True
        return

    guess_raw = (st.session_state.get("tactic_input") or "").strip()
    st.session_state.tactic_input = ""
    if not guess_raw:
        return

    expected_raw = moves[off]

    b = st.session_state.tactic_board

    expected_mv, _ = parse_san_lenient(b, expected_raw)
    if expected_mv is None:
        st.session_state.tactic_feedback = "Tactic data error (expected move is invalid for this position)."
        st.session_state.tactic_complete = True
        return

    guess_mv, options = parse_san_lenient(b, guess_raw)
    if guess_mv is None:
        st.session_state.tactic_feedback = "Illegal or unrecognized move — try again."
        if options:
            st.session_state.tactic_feedback += " (Did you mean: " + ", ".join(options[:6]) + "?)"
        return

    if guess_mv != expected_mv:
        if guess_mv.from_square == expected_mv.from_square:
            st.session_state.tactic_feedback = "Right piece, wrong move — try again."
        else:
            st.session_state.tactic_feedback = "Incorrect — try again."
        return

    # Correct move
    b.push(expected_mv)
    st.session_state.tactic_offset = off + 1
    st.session_state.tactic_feedback = "Correct."
    st.session_state.tactic_reveal_next = False

    _tactic_autoplay_black_if_needed()

    if int(st.session_state.get("tactic_offset", 0)) >= len(moves):
        st.session_state.tactic_complete = True
        st.session_state.tactic_feedback = "Tactic complete ✅"

def on_go_to_tactic_game() -> None:
    """
    Exit tactic mode and open the source game for the currently loaded tactic,
    jumping to the tactic start position (White to move).
    """
    payload = st.session_state.get("tactic_payload") or {}
    src = _resolve_path_str(payload.get("source_game_file"))
    if not src:
        st.session_state.tactic_feedback = "This tactic isn’t linked to a source game."
        return

    p = Path(src)
    if not p.exists():
        st.session_state.tactic_feedback = f"Source game file not found: {p}"
        return

    start_ply = _safe_int(payload.get("start_ply", 0), 0)

    # Exit tactic mode first (so UI swaps back to game UI)
    st.session_state.tactic_mode = False
    st.session_state.tactic_feedback = ""
    st.session_state.tactic_input = ""

    # Load the game, then jump to the tactic's start position
    load_game_from_path(p)

    # Clamp and reposition to the tactic start (position after moves[:start_ply])
    start_ply = max(0, min(start_ply, len(st.session_state.get("moves_san", []))))
    _reposition_board(start_ply)

    st.session_state.flash = "Opened source game at tactic start."

def on_clear_tactic_filter() -> None:
    st.session_state.tactic_filter_tags = []
    st.session_state.tactic_filter_match_all = False

def on_reveal_next_tactic_move() -> None:
    st.session_state.tactic_reveal_next = True

def on_hide_next_tactic_move() -> None:
    st.session_state.tactic_reveal_next = False



# ===================== Callbacks: Tactic Notes =====================
def on_save_tactic_note() -> None:
    fp = st.session_state.get("tactic_current_file")
    if not fp:
        st.session_state.tactic_note_flash = "No tactic loaded."
        return

    p = Path(fp)
    payload = _read_json(p)
    if not payload:
        st.session_state.tactic_note_flash = "Couldn't read tactic file."
        return

    payload["note"] = str(st.session_state.get("tactic_note_draft") or "").rstrip()
    if _write_json(p, payload):
        st.session_state.tactic_payload = payload
        st.session_state.tactic_note_flash = "Saved tactic notes."
    else:
        st.session_state.tactic_note_flash = "Couldn't write tactic file."


def on_revert_tactic_note() -> None:
    payload = st.session_state.get("tactic_payload") or {}
    st.session_state.tactic_note_draft = str(payload.get("note", "") or "")
    st.session_state.tactic_note_flash = "Reverted."


# ===================== Callbacks: Tactic Creation (from Saved Line) =====================
def on_set_tactic_start_here() -> None:
    moves = st.session_state.get("moves_san", [])
    if not moves:
        return

    attach = _current_attach_ply()

    # Start trigger: Black move index (odd), with room for at least one White move after it.
    if attach % 2 == 0:
        attach = max(1, attach - 1)

    if attach < 1 or attach >= len(moves) - 2:
        st.session_state.flash = "Pick a Black move that has at least one White move after it."
        return

    st.session_state.tactic_start_black_ply = attach
    st.session_state.tactic_start_ply = attach + 1  # first White move in the tactic
    st.session_state.tactic_end_ply = st.session_state.tactic_start_ply
    st.session_state.tactic_start_selected = True
    st.session_state.tactic_end_selected = False

    _reposition_board(st.session_state.tactic_start_ply + 1)


def on_set_tactic_end_here() -> None:
    moves = st.session_state.get("moves_san", [])
    if not moves or not st.session_state.get("tactic_start_selected"):
        return

    start_raw = int(st.session_state.get("tactic_start_ply", 0))
    end_raw = _current_attach_ply()

    start_ply, end_ply = normalize_tactic_window(len(moves), start_raw, end_raw)
    st.session_state.tactic_end_ply = end_ply
    st.session_state.tactic_end_selected = True


def on_reset_tactic_selection() -> None:
    st.session_state.tactic_start_selected = False
    st.session_state.tactic_end_selected = False
    st.session_state.tactic_start_black_ply = None
    st.session_state.tactic_start_ply = 0
    st.session_state.tactic_end_ply = 0
    _reposition_board(0)


def on_save_tactic_from_tab() -> None:
    if not st.session_state.get("tactic_start_selected", False):
        st.session_state.flash = "Select a tactic start first."
        return
    if not st.session_state.get("tactic_end_selected", False):
        st.session_state.flash = "Select a tactic end first."
        return

    cur_file = st.session_state.get("current_file")
    moves = st.session_state.get("moves_san", [])
    if not cur_file or not moves:
        st.session_state.flash = "Open a saved game with moves to create a tactic."
        return

    start_raw = int(st.session_state.get("tactic_start_ply", 0))
    end_raw = int(st.session_state.get("tactic_end_ply", len(moves) - 1))
    start_ply, end_ply = normalize_tactic_window(len(moves), start_raw, end_raw)

    if start_ply % 2 != 0 or end_ply % 2 != 0:
        st.session_state.flash = "Internal error: tactic window must start/end on a White move."
        return

    try:
        _apply_moves_to_board(moves[:start_ply])
        _apply_moves_to_board(moves[: end_ply + 1])
        game_payload = _read_json(Path(cur_file)) or {}
        label = (game_payload.get("source") or "").strip() or Path(cur_file).name
        url = game_payload.get("url") or ""
    except Exception as e:
        st.session_state.flash = f"Could not validate tactic slice: {e}"
        return

    outp = save_tactic_json(
        source_game_file=cur_file,
        source_label=label,
        url=url,
        start_ply=start_ply,
        end_ply=end_ply,
        game_moves=moves,
        note=st.session_state.get("tactic_new_note", ""),
        tags=compute_tactic_piece_tags(
            start_fen=_apply_moves_to_board(moves[:start_ply]).fen(),
            tactic_moves_san=moves[start_ply : end_ply + 1],
        ),
    )

    st.session_state.flash = f"Saved tactic: {outp.name}"
    st.session_state.tactic_new_note = ""


# ===================== Sidebar =====================
def on_clear_search() -> None:
    st.session_state.search_move = ""
    st.session_state.search_side = "White"


files = sorted(DATA_DIR.glob("game_*.json"), reverse=True)


def _run_pending_actions() -> None:
    ss = st.session_state
    if ss.get("pgn_pending_import", False):
        ss.pgn_pending_import = False
        # Runs at the top of a rerun, before widgets instantiate
        on_import_selected_pgn(do_rerun=False)
        st.rerun()

_run_pending_actions()

with st.sidebar:
    st.header("Actions")

    st.button("➕ New game / Reset", use_container_width=True, on_click=on_clear)
    st.button("🔄 Update all games", use_container_width=True, on_click=on_update_all_games)

    # Search
    q_move = st.text_input(
        "Move (SAN)",
        placeholder="Search games by move",
        key="search_move",
        label_visibility="collapsed",
    )

    q_side = st.radio(
        "Side",
        options=["White", "Black"],
        index=0,
        key="search_side",
        horizontal=True,
        label_visibility="collapsed",
    )

    # Show "Clear search" only when there's something to clear
    search_active = bool((q_move or "").strip())
    if search_active:
        st.button(
            "Clear search",
            key="clear_search",
            use_container_width=True,
            on_click=on_clear_search,
    )

    # Tactic mode toggle (show only one button)
    if st.session_state.get("tactic_mode"):
        st.button(
            "Exit tactic mode",
            key="btn_exit_tactic_mode",
            use_container_width=True,
            on_click=on_exit_tactic_mode,
        )
    else:
        st.button(
            "🎯 Tactic mode",
            key="btn_enter_tactic_mode",
            use_container_width=True,
            on_click=on_enter_tactic_mode,
        )


    side_map = {"White": "white", "Black": "black"}

    st.divider()
    st.subheader("Saved games")

    if (q_move or "").strip():
        hits = find_games_with_move(q_move.strip(), side=side_map[q_side])

        if not hits:
            st.caption(f'_No games contain "{q_move.strip()}" ({q_side})._')
        else:
            for fp in hits:
                payload = _read_json(fp) or {}
                source = (payload.get("source") or "").strip() or "(untitled)"
                st.button(
                    source,
                    key=f"open_search_{fp.name}",
                    use_container_width=True,
                    on_click=(lambda p=fp: load_game_from_path(p)),
                )
    else:
        if not files:
            st.caption("_No saved games yet._")
        else:
            seen_labels = set()
            for fp in files:
                payload = _read_json(fp)
                if not payload:
                    st.button(f"{fp.name} (unreadable)", key=f"bad_{fp.name}", use_container_width=True, disabled=True)
                    continue

                ts = payload.get("timestamp", fp.stem.replace("game_", ""))
                source = (payload.get("source") or "").strip() or "(untitled)"
                label = source if source not in seen_labels else f"{source} — {ts}"
                seen_labels.add(source)

                st.button(
                    label,
                    key=f"open_{fp.name}",
                    use_container_width=True,
                    on_click=(lambda p=fp: load_game_from_path(p)),
                )

def on_show_tactic_notes() -> None:
    st.session_state.tactic_show_notes = True

def on_hide_tactic_notes() -> None:
    st.session_state.tactic_show_notes = False


# ===================== Main Layout =====================

def _render_tag_pills(tags: List[str]) -> None:
    tags = [str(t).strip() for t in (tags or []) if str(t).strip()]
    if not tags:
        st.caption("_No tags on this tactic._")
        return
    html = "<div class='tag-pills'>" + "".join(f"<span>{t}</span>" for t in tags) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_tactic_left_solver_ui() -> None:
    """
    LEFT column in tactic mode: keep the *guess* UX here.
    """
    st.subheader("Your move")

    payload = st.session_state.get("tactic_payload")
    if not payload:
        msg = st.session_state.get("tactic_feedback") or "No tactic loaded."
        st.markdown(f"<div class='flash'>{msg}</div>", unsafe_allow_html=True)
        st.caption("Use the right panel to load a tactic (and set filters).")
        return

    moves = payload.get("moves_san") or []
    off = int(st.session_state.get("tactic_offset", 0))
    complete = bool(st.session_state.get("tactic_complete", False))

    trigger = (st.session_state.get("tactic_trigger_text") or "").strip()
    if trigger:
        st.caption(f"Starting position is after **{trigger}**")


    if complete:
        st.markdown("✅ **Tactic complete**")
        st.button("Next tactic", type="primary", on_click=on_next_tactic, use_container_width=True)
    else:
        st.text_input(
            "Tactic move",
            key="tactic_input",
            placeholder="e.g., Bxc7+, Ne5, Qxd5...",
            label_visibility="collapsed",
            on_change=on_submit_tactic_move,
        )
        st.button("Submit", on_click=on_submit_tactic_move, use_container_width=True)
        # --- Reveal next move (kept on left because it’s part of solving) ---
        if off < len(moves):
            r1, r2 = st.columns([0.35, 0.65])
            with r1:
                if not st.session_state.get("tactic_reveal_next", False):
                    st.button("Reveal", on_click=on_reveal_next_tactic_move, use_container_width=True)
                else:
                    st.button("Hide", on_click=on_hide_next_tactic_move, use_container_width=True)

            with r2:
                if st.session_state.get("tactic_reveal_next", False):
                    side = "White" if (off % 2 == 0) else "Black"
                    st.markdown(
                        f"<div class='flash'><b>Next ({side}):</b> {moves[off]}</div>",
                        unsafe_allow_html=True,
                    )

    if st.session_state.get("tactic_feedback"):
        st.markdown(f"<div class='flash'>{st.session_state.tactic_feedback}</div>", unsafe_allow_html=True)


def render_tactic_right_panel() -> None:
    """
    RIGHT column in tactic mode: info, navigation, filter, notes, spoiler solution.
    Notes are now embedded in Info (below Solution), not a separate tab.
    """
    st.subheader("Tactic panel")

    # Tabs: Notes removed (now lives in Info)
    tab_info, tab_filter, tab_manage = st.tabs(["Info", "Filter", "All tactics"])


    payload = st.session_state.get("tactic_payload") or None

    with tab_info:
        if not payload:
            st.markdown(
                "<div class='soft-panel'><b>No tactic loaded.</b><br>"
                "<span class='small-muted'>Use Filter → Next tactic, or just Next tactic.</span>"
                "</div>",
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.button("Next tactic", on_click=on_next_tactic, use_container_width=True)
            with c2:
                st.button("Clear filter", on_click=on_clear_tactic_filter, use_container_width=True)
            return

        moves = payload.get("moves_san") or []

        # Controls
        b1, b2 = st.columns(2)
        with b1:
            st.button("Next tactic", on_click=on_next_tactic, use_container_width=True)
        with b2:
            st.button("Go to game", on_click=on_go_to_tactic_game, use_container_width=True)

        # --- Spoilers (single-level) ---
        st.markdown("**Spoilers**")

        st.checkbox("Show tags", key="tactic_show_tags")
        if st.session_state.get("tactic_show_tags", False):
            _render_tag_pills(payload.get("tags") or [])

        st.checkbox("Show solution line", key="tactic_show_solution")
        if st.session_state.get("tactic_show_solution", False):
            st.code(format_tactic_solution(payload))

        st.checkbox("Show notes", key="tactic_show_notes")
        if st.session_state.get("tactic_show_notes", False):
            st.text_area(
                "Notes",
                key="tactic_note_draft",
                height=160,
                placeholder="Themes, key squares, why it works, common mistakes, etc.",
                label_visibility="collapsed",
            )

            n1, n2 = st.columns(2)
            with n1:
                st.button("Save notes", on_click=on_save_tactic_note, use_container_width=True)
            with n2:
                st.button("Revert", on_click=on_revert_tactic_note, use_container_width=True)

            if st.session_state.get("tactic_note_flash"):
                st.markdown(
                    f"<div class='flash'>{st.session_state.tactic_note_flash}</div>",
                    unsafe_allow_html=True,
                )

        with tab_manage:
            # Build catalog (already cached by mtimes)
            file_mtimes = _tactic_file_mtimes()
            catalog = build_tactics_catalog_cached(file_mtimes)

            total = len(catalog)
            if total == 0:
                st.caption("_No tactics saved yet._")
                return

            # Controls
            st.text_input(
                "Search tactics",
                key="tactic_manage_query",
                placeholder="Search by source, trigger, tag…",
                label_visibility="collapsed",
            ) 

            # Reuse existing tag filter (same as training Filter tab)
            selected = st.session_state.get("tactic_filter_tags", [])
            match_all = bool(st.session_state.get("tactic_filter_match_all", False))

            q = (st.session_state.get("tactic_manage_query") or "").strip().lower()
            lim = int(st.session_state.get("tactic_manage_limit", 50) or 50)

            def _matches(it: dict) -> bool:
                payload = it.get("payload") or {}
                tags = it.get("tags") or []
                if selected and not _tactic_matches_selected_tags(tags, selected, match_all):
                    return False

                if not q:
                    return True

                label = str(payload.get("source_label") or "")
                trig = _tactic_trigger_text(payload) or ""
                name = Path(it.get("path") or "").name
                hay = " ".join([label, trig, name, " ".join(tags)]).lower()
                return q in hay

            filtered = [it for it in catalog if _matches(it)]

            st.caption(f"Showing: **{min(len(filtered), lim)}** / {len(filtered)} matched (total tactics: {total})")

            pending = st.session_state.get("tactic_delete_pending")

            for it in filtered[:lim]:
                path_str = str(it.get("path") or "")
                payload = it.get("payload") or {}
                tags = it.get("tags") or []

                src_label = (payload.get("source_label") or "").strip() or "(untitled)"
                start_ply = _safe_int(payload.get("start_ply", 0), 0)
                start_move_num = (start_ply // 2) + 1
                trigger = _tactic_trigger_text(payload)

                # Compact row
                st.markdown("<div class='comment-card'>", unsafe_allow_html=True)
                title = f"**{src_label}**  <span class='small-muted'>· starts move {start_move_num}</span>"
                st.markdown(title, unsafe_allow_html=True)

                meta_bits = []
                if trigger:
                    meta_bits.append(f"after {trigger}")
                if tags:
                    meta_bits.append(f"{len(tags)} tag(s)")
                if meta_bits:
                    st.caption(" · ".join(meta_bits))

                c1, c2, c3 = st.columns([0.34, 0.33, 0.33])
                with c1:
                    st.button(
                        "Load / Train",
                        key=f"tman_load_{Path(path_str).name}",
                        use_container_width=True,
                        on_click=make_load_tactic_from_manage(path_str, payload),
                    )
                with c2:
                    st.button(
                        "Go to game",
                        key=f"tman_goto_{Path(path_str).name}",
                        use_container_width=True,
                        on_click=on_go_to_tactic_game,
                        disabled=not _paths_equal(st.session_state.get("tactic_current_file"), path_str),
                        help="Enabled only for the currently loaded tactic.",
                    )
                with c3:
                    if pending == path_str:
                        d1, d2 = st.columns(2)
                        with d1:
                            st.button(
                                "Confirm",
                                key=f"tman_del_yes_{Path(path_str).name}",
                                use_container_width=True,
                                on_click=make_delete_tactic_confirm(path_str),
                            )
                        with d2:
                            st.button(
                                "Cancel",
                                key=f"tman_del_no_{Path(path_str).name}",
                                use_container_width=True,
                                on_click=on_cancel_delete_tactic,
                            )
                    else:
                        st.button(
                            "Delete",
                            key=f"tman_del_{Path(path_str).name}",
                            use_container_width=True,
                            on_click=lambda p=path_str: on_mark_delete_tactic(p),
                        )

                st.markdown("</div>", unsafe_allow_html=True)


    with tab_filter:
        # --- Filter UI (piece tags) ---
        file_mtimes = _tactic_file_mtimes()
        catalog = build_tactics_catalog_cached(file_mtimes)
        all_tags = _all_tactic_tags_from_catalog(catalog)

        st.markdown(
            "<div class='soft-panel'>"
            "<b>Filter tactics</b><br>"
            "<span class='small-muted'>Filter by pieces that moved or were captured in the tactic window.</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.multiselect(
            "Pieces",
            options=all_tags,
            key="tactic_filter_tags",
            help="Filters tactics by tagged pieces (moving or captured).",
        )
        st.checkbox(
            "Require all selected pieces",
            key="tactic_filter_match_all",
            help="If off: match ANY selected tag. If on: match ALL selected tags.",
        )

        selected = st.session_state.get("tactic_filter_tags", [])
        match_all = bool(st.session_state.get("tactic_filter_match_all", False))

        total = len(catalog)
        matched = sum(
            1 for it in catalog
            if _tactic_matches_selected_tags(it.get("tags") or [], selected, match_all)
        )
        st.caption(f"Matching tactics: **{matched}** / {total}")

        c1, c2 = st.columns(2)
        with c1:
            st.button(
                "Next tactic",
                key="btn_next_tactic_filter",
                on_click=on_next_tactic,
                use_container_width=True,
            )
        with c2:
            st.button("Clear filter", on_click=on_clear_tactic_filter, use_container_width=True)



top_left, top_right = st.columns([0.5, 0.5])

# ---------- Board + Controls ----------
with top_left:

    st.markdown(
        f"<div class='board-context-title'>{board_context_title()}</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.get("tactic_mode") and st.session_state.get("tactic_payload"):
        board_to_show = st.session_state.tactic_board
        last = board_to_show.move_stack[-1] if board_to_show.move_stack else None
        svg = chess_svg.board(
            board=board_to_show,
            lastmove=last,
            coordinates=True,
            size=520,
            orientation=chess.WHITE,  # tactic training always as White
        )
    else:
        last = st.session_state.board.move_stack[-1] if st.session_state.board.move_stack else None
        orientation = chess.WHITE if not st.session_state.flipped else chess.BLACK
        svg = chess_svg.board(
            board=st.session_state.board,
            lastmove=last,
            coordinates=True,
            size=520,
            orientation=orientation,
        )

    st.markdown('<div class="board-wrap">', unsafe_allow_html=True)
    components.html(svg, height=540, scrolling=False)
    st.markdown("</div>", unsafe_allow_html=True)

    # Latest comment under board (Explore or Walkthrough)
    read_only = is_read_only()
    walkthrough = bool(st.session_state.get("walkthrough_mode", False))

    if read_only or walkthrough:
        moves = st.session_state.moves_san
        comments = st.session_state.comments
        cur_ply = int(st.session_state.current_ply_idx)

        if moves and comments:
            visible = [(idx, c) for idx, c in enumerate(comments) if is_comment_visible_at_ply(c, cur_ply)]
            if visible:
                latest_idx, latest = sorted(visible, key=lambda item: _safe_int(item[1].get("start", 0), 0))[-1]
                start_ply = _safe_int(latest.get("start", 0), 0)
                header = f"**{format_span(start_ply, moves)}**"
                revealed = set(st.session_state.get("walkthrough_revealed", []))

                st.markdown('<div class="comment-card" style="margin-top:.5rem;">', unsafe_allow_html=True)

                if walkthrough and latest_idx not in revealed:
                    st.markdown(
                        f"{header}<br><span class='small-muted'>[Comment hidden — click Reveal]</span>",
                        unsafe_allow_html=True,
                    )
                    st.button(
                        "Reveal comment",
                        key=f"reveal_underboard_{latest_idx}",
                        on_click=make_reveal_comment(latest_idx),
                        use_container_width=True,
                    )
                else:
                    st.markdown(f"{header}<br>{latest.get('text', '')}", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Tactic Mode UI ----------
    if st.session_state.get("tactic_mode"):
        render_tactic_left_solver_ui()


    # ---------- Normal Move Entry UI ----------
    else:
        # Show move entry only for:
        # - new/templated games (current_file is None), OR
        # - saved games explicitly switched to Edit mode
        show_move_entry = (st.session_state.get("current_file") is None) or (st.session_state.get("view_mode") == "edit")

        if not show_move_entry:
            st.caption("_Explore mode: move entry hidden. Click **Edit** to modify moves._")
        else:
            st.subheader("Enter next move (SAN)")

            index = build_next_move_index_cached(_game_file_mtimes())

            prefix = tuple(st.session_state.moves_san[: st.session_state.current_ply_idx])
            suggestions = get_suggestions(prefix, index, limit=12)

            is_at_end = (st.session_state.current_ply_idx == len(st.session_state.moves_san))
            read_only = is_read_only()

            if suggestions:
                rowsize = 6
                for start in range(0, len(suggestions), rowsize):
                    cols = st.columns(rowsize)
                    for col, (mv, cnt) in zip(cols, suggestions[start : start + rowsize]):
                        col.button(
                            f"{mv}  ·  {cnt}",
                            key=f"sugg_{prefix}_{mv}_{cnt}_{start}",
                            on_click=make_on_add_suggested(mv) if not read_only else None,
                            disabled=read_only or not is_at_end,
                        )
            else:
                st.caption("_No suggestions for this position yet (or no saved games)._")

            st.text_input(
                "Next move",
                placeholder="e.g., d4, d5, Bf4, Nbd2, Bxc7+",
                key="move_input",
                on_change=on_add_from_text if (not read_only and is_at_end) else None,
                disabled=read_only or not is_at_end,
                label_visibility="collapsed",
            )

            col_add, col_undo, col_flip = st.columns([1, 1, 1])
            with col_add:
                st.button(
                    "Add",
                    type="primary",
                    on_click=on_add_from_text,
                    use_container_width=True,
                    disabled=read_only or not is_at_end,
                )
            with col_undo:
                st.button(
                    "Undo",
                    on_click=on_undo,
                    use_container_width=True,
                    disabled=read_only or not is_at_end or not st.session_state.moves_san,
                )
            with col_flip:
                st.button("Flip", on_click=on_flip, use_container_width=True)

            if st.session_state.flash:
                st.markdown(f'<div class="flash">{st.session_state.flash}</div>', unsafe_allow_html=True)



# ---------- Tabs ----------
with top_right:
    if st.session_state.get("tactic_mode", False):
        # (Optional) show nothing at all:
        render_tactic_right_panel()
    else:
        # Pick label based on whether we’re in New Game mode
        SIMILAR_TAB_LABEL = "Matching Games" if (st.session_state.get("current_file") is None) else "Similar Games / Delete"

        tab_labels = ["Moves & Comments", "Game Info & Save", SIMILAR_TAB_LABEL, "Coverage Map"]

        # Only show Import/Export when you're in a New Game (unsaved) state
        if st.session_state.get("current_file") is None:
            tab_labels.append("Import / Export")

        show_tactics_tab = bool(st.session_state.get("current_file")) and (not moves_dirty_vs_loaded())
        if show_tactics_tab:
            tab_labels.append("Tactics")

        tabs = st.tabs(tab_labels)

        # Map labels to tabs so optional sections don't shift indices
        tab_map = {name: tab for name, tab in zip(tab_labels, tabs)}

        tab_moves = tab_map["Moves & Comments"]
        tab_meta = tab_map["Game Info & Save"]
        tab_similar = tab_map[SIMILAR_TAB_LABEL]   # <-- use the variable here
        tab_coverage = tab_map["Coverage Map"]

        tab_import_export = tab_map.get("Import / Export")
        tab_tactics = tab_map.get("Tactics")




        # ===== Tab 1: Moves & Comments =====
        with tab_moves:
            if st.session_state.get("current_file"):
                mode = st.session_state.get("view_mode", "edit")
                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    st.button(
                        "Explore",
                        key="btn_explore_mode",
                        on_click=on_set_explore_mode,
                        use_container_width=True,
                        type="secondary",
                        disabled=(mode == "explore"),
                    )
                with mcol2:
                    st.button(
                        "Edit",
                        key="btn_edit_mode",
                        on_click=on_set_edit_mode,
                        use_container_width=True,
                        type="secondary",
                        disabled=(mode == "edit"),
                    )
                with mcol3:
                    st.button(
                        "Use as Template",
                        key="btn_use_as_template",
                        on_click=on_use_as_template,
                        use_container_width=True,
                        type="secondary",
                    )


            st.subheader("Move list")
            render_moves_bubbles(st.session_state.moves_san)

            # Navigation controls
            if st.session_state.moves_san:
                max_ply_list = len(st.session_state.moves_san)
                nav_cols = st.columns(4)
                with nav_cols[0]:
                    st.button("|◀", key="movelist_start", on_click=on_go_to_start, use_container_width=True,
                            disabled=st.session_state.current_ply_idx == 0)
                with nav_cols[1]:
                    st.button("◀", key="movelist_back", on_click=on_step_backward, use_container_width=True,
                            disabled=st.session_state.current_ply_idx == 0)
                with nav_cols[2]:
                    st.button("▶", key="movelist_forward", on_click=on_step_forward, use_container_width=True,
                            disabled=st.session_state.current_ply_idx == max_ply_list)
                with nav_cols[3]:
                    st.button("▶|", key="movelist_end", on_click=on_go_to_end, use_container_width=True,
                            disabled=st.session_state.current_ply_idx == max_ply_list)

            st.markdown("---")
            st.subheader("Comments")

            can_walkthrough = (st.session_state.get("current_file") is None) or is_read_only()
            if can_walkthrough:
                st.checkbox(
                    "Walkthrough mode",
                    key="walkthrough_mode",
                    help=(
                        "Show where comments occur as you step through the line, "
                        "but hide their text until you explicitly reveal them."
                    ),
                )
            else:
                st.session_state.walkthrough_mode = False
                st.session_state.walkthrough_revealed = []

            moves = st.session_state.moves_san
            has_moves = bool(moves)
            current_ply = int(st.session_state.current_ply_idx)
            max_ply = len(moves) - 1 if has_moves else 0

            raw_comments = st.session_state.comments
            walkthrough = bool(st.session_state.get("walkthrough_mode", False))
            revealed_set = set(st.session_state.get("walkthrough_revealed", []))

            if walkthrough:
                st.markdown(
                    "<div class='soft-panel'>"
                    "<b>Walkthrough mode is ON</b><br>"
                    "<span class='small-muted'>Comments are hidden in this panel. "
                    "Step through the line and reveal comments under the board.</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            else:
                # Visible comments list
                if raw_comments and has_moves:
                    visible = [(idx, c) for idx, c in enumerate(raw_comments) if is_comment_visible_at_ply(c, current_ply)]

                    def sort_key(item):
                        idx, c = item
                        is_original_here = (
                            c.get("origin") == "original"
                            and str(c.get("source_file", "")).strip() == str(st.session_state.get("current_file", "")).strip()
                        )
                        return (1 if is_original_here else 0, _safe_int(c.get("start", 0), 0), c.get("text", ""))

                    if visible:
                        for idx, c in sorted(visible, key=sort_key, reverse=True):
                            with st.container():
                                st.markdown('<div class="comment-card">', unsafe_allow_html=True)

                                ccols = st.columns([0.7, 0.15, 0.15])

                                move_span = format_span(_safe_int(c.get("start", 0), 0), moves)
                                header_text = f"**@ {move_span}**"

                                if _is_comment_editable_here(c):
                                    header_text += " | Original"
                                elif c.get("origin") == "imported":
                                    source_label = c.get("source_label") or (Path(c.get("source_file", "")).name if c.get("source_file") else "Unknown Source")
                                    header_text += f" | Imported from: {source_label}"
                                else:
                                    source_label = c.get("source_label") or (Path(c.get("source_file", "")).name if c.get("source_file") else "Unsaved Template")
                                    header_text += f" | From: {source_label}"

                                is_revealed = (not walkthrough) or (idx in revealed_set)

                                with ccols[0]:
                                    body = c.get("text", "")
                                    if walkthrough and not is_revealed:
                                        body = "<span class='small-muted'>[Comment hidden — click Reveal]</span>"
                                    st.markdown(f"{header_text}<br>{body}", unsafe_allow_html=True)

                                with ccols[1]:
                                    editable = _is_comment_editable_here(c)
                                    if walkthrough and not is_revealed:
                                        st.button(
                                            "Reveal",
                                            key=f"reveal_c_{idx}_{c.get('id', 'no_id')}",
                                            on_click=make_reveal_comment(idx),
                                            use_container_width=True,
                                        )
                                    else:
                                        st.button(
                                            "Edit",
                                            key=f"edit_c_{idx}_{c.get('id', 'no_id')}",
                                            on_click=make_start_edit_comment(idx) if editable else None,
                                            disabled=not editable,
                                            use_container_width=True,
                                        )

                                with ccols[2]:
                                    deletable = _is_comment_deletable_here(c)
                                    st.button(
                                        "Delete",
                                        key=f"del_c_{idx}_{c.get('id', 'no_id')}",
                                        on_click=make_delete_comment(idx) if deletable else None,
                                        disabled=not deletable,
                                        use_container_width=True,
                                    )

                                st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.caption("_No comments yet at or before this position._")

                elif raw_comments and not has_moves:
                    st.caption("_There are comments saved for this game, but they are attached to moves. Add moves or load the full line to see them._")
                else:
                    st.caption("_No comments yet for this game._")

                # Add / Edit comment panel
                edit_idx = st.session_state.get("edit_comment_idx")
                if edit_idx is not None and not (0 <= edit_idx < len(st.session_state.comments)):
                    st.session_state.edit_comment_idx = None
                    edit_idx = None

                st.markdown('<div class="soft-panel">', unsafe_allow_html=True)

                if has_moves:
                    if edit_idx is not None:
                        st.text_area(
                            "Edit comment text",
                            key="new_comment_text",
                            height=110,
                            placeholder="e.g., Good vs ...c5. Watch out for ...Qa5.",
                        )

                        locked_ply = _safe_int(st.session_state.comments[edit_idx].get("start", 0), 0)
                        st.caption(ply_label(locked_ply, moves))

                        c1, c2 = st.columns(2)
                        with c1:
                            st.button("Confirm edit", on_click=on_confirm_edit_comment, use_container_width=True)
                        with c2:
                            st.button("Cancel", on_click=on_cancel_edit_comment, use_container_width=True)

                    else:
                        st.markdown("**Add comment**")
                        st.text_area(
                            "Comment text",
                            key="new_comment_text",
                            height=100,
                            placeholder="e.g., Great vs …c5. Watch out for …Qa5.",
                            label_visibility="collapsed",
                        )

                        attach_ply = 0
                        if current_ply > 0:
                            attach_ply = min(current_ply - 1, max_ply)

                        st.caption(f"This comment will be attached to: {ply_label(attach_ply, moves)}")
                        st.button("Add comment", on_click=on_add_comment, use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)

        # ===== Tab 2: Game Info & Save =====
        with tab_meta:
            st.subheader("Game Info & Save")

            col_a, col_b = st.columns([0.6, 0.4])
            with col_a:
                st.text_area(
                    "Source (where this game/line came from)",
                    key="source_text",
                    placeholder="e.g., book name + page, video title, author...",
                    height=80,
                )
            with col_b:
                st.text_input(
                    "URL (optional)",
                    key="url_text",
                    placeholder="e.g., https://www.youtube.com/...    or https://www.chess.com/... ",
                )
                if (st.session_state.url_text or "").strip():
                    st.markdown(f"🔗 **Reference:** [{st.session_state.url_text}]({st.session_state.url_text})")

            if not st.session_state.get("current_file"):
                st.button(
                    "Save as new game to local database",
                    type="primary",
                    on_click=on_save,
                    disabled=is_read_only(),
                )

            if st.session_state.get("current_file"):
                st.button(
                    "Save changes (overwrite game)",
                    type="primary",
                    on_click=on_save_overwrite_current,
                    disabled=is_read_only(),
                )
            else:
                st.caption("_Open a saved game to enable overwriting an existing game._")

            # Mode summary
            cur_file = st.session_state.get("current_file")
            moves = st.session_state.get("moves_san", [])
            cur_ply = int(st.session_state.get("current_ply_idx", 0))

            if cur_file:
                line1 = "Old game"
            elif st.session_state.get("templated_from"):
                line1 = "Templated game"
            else:
                line1 = "New game"

            if line1 in ("New game", "Templated game"):
                line2 = "Unsaved"
            else:
                baseline = st.session_state.get("loaded_snapshot")
                dirty = (baseline is not None) and (_snapshot_current_game_state() != baseline)
                line2 = "Edited/Unsaved" if dirty else "Unchanged"

            line3 = "Explore Mode" if is_read_only() else "Edit mode"

            can_walkthrough = (cur_file is None) or is_read_only()
            wt_requested = bool(st.session_state.get("walkthrough_mode", False))
            wt_on = wt_requested and can_walkthrough
            line4 = "Walkthrough mode: On" if wt_on else "Walkthrough mode: Off"

            if not moves or cur_ply <= 0:
                line5 = "Position: start"
            elif cur_ply >= len(moves):
                try:
                    line5 = f"Position: end of game (last move {ply_label(len(moves) - 1, moves)})"
                except Exception:
                    line5 = f"Position: end of game (last move {len(moves)})"
            else:
                try:
                    line5 = f"Position: {ply_label(cur_ply - 1, moves)}"
                except Exception:
                    line5 = f"Position: ply {cur_ply}"

            st.markdown(
                f"""
                <div class="soft-panel" style="margin-bottom:.75rem;">
                <b>Current mode</b><br>
                <span class="small-muted">{line1}</span><br>
                <span class="small-muted">{line2}</span><br>
                <span class="small-muted">{line3}</span><br>
                <span class="small-muted">{line4}</span><br>
                <span class="small-muted">{line5}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ===== Tab 3: Similar Games / Delete =====
        with tab_similar:
            MIN_MOVES_FOR_NEWGAME_MATCH = 4  # counts SAN plies in moves_san

            if not st.session_state.get("current_file"):
                st.subheader("Matching saved games (current position)")

                total_played = len(st.session_state.get("moves_san", []))
                if total_played < MIN_MOVES_FOR_NEWGAME_MATCH:
                    st.caption(f"_Play at least {MIN_MOVES_FOR_NEWGAME_MATCH} moves to enable this feature._")
                else:
                    prefix = st.session_state.moves_san[: int(st.session_state.current_ply_idx)]
                    plen = len(prefix)

                    if plen == 0:
                        st.caption("_At the start position._")
                    else:
                        try:
                            st.caption(f"Matching by prefix through **{ply_label(plen - 1, prefix)}**")
                        except Exception:
                            st.caption(f"Matching by prefix length: {plen} ply(ies)")

                    games_index = build_games_index_cached(_game_file_mtimes())
                    matches = find_games_matching_prefix(prefix, games_index)

                    if not matches:
                        st.caption("_No saved games match this position yet._")
                    else:
                        st.caption(f"Matches: **{len(matches)}**")
                        for m in matches[:25]:
                            open_col, info_col = st.columns([0.22, 0.78], vertical_alignment="center")

                            with open_col:
                                st.button(
                                    "Open",
                                    key=f"open_match_{m['path'].name}_{plen}",
                                    on_click=(lambda p=m["path"]: load_game_from_path(p)),
                                    use_container_width=True,
                                )

                            with info_col:
                                nxt = m.get("next_san")
                                if nxt:
                                    # show next move with move number context if possible
                                    try:
                                        nxt_label = ply_label(plen, m["moves"])
                                    except Exception:
                                        nxt_label = nxt
                                    next_text = f"Next: **{nxt_label}**"
                                else:
                                    next_text = "_(Line ends here)_"

                                rem_plies = int(m.get("remaining_plies", 0) or 0)
                                rem_moves = rem_plies // 2  # round down

                                st.markdown(
                                    f"**{m['label']}**  \n"
                                    f"<span class='small-muted'>{next_text} · remaining moves: {rem_moves}</span>",
                                    unsafe_allow_html=True,
                                )

            else:
                st.subheader("Similar Games")

                if st.session_state.get("current_file"):
                    cur_path = Path(st.session_state["current_file"])
                    similar_games = _find_similar_games(cur_path, limit=5)

                    if not similar_games:
                        st.caption("_No similar games found yet._")
                    else:
                        for t in similar_games:
                            open_col, meta_col = st.columns([0.22, 0.78], vertical_alignment="center")
                            with open_col:
                                st.button(
                                    "Open",
                                    key=f"open_similar_{t['path'].name}",
                                    on_click=(lambda p=t["path"]: load_game_from_path(p)),
                                    use_container_width=True,
                                )
                            with meta_col:
                                st.markdown(
                                    f"**{t['label']}**  \n"
                                    f'<span class="small-muted">shares {t["shared_moves"]} move(s)'
                                    f' · diverges after {t.get("diverge_label", "n/a")}</span>',
                                    unsafe_allow_html=True,
                                )
                else:
                    st.caption("_Open a saved game to see similar games._")

                st.markdown("---")
                st.subheader("Delete Game")

                if st.session_state.get("current_file"):
                    cur_path = Path(st.session_state["current_file"])
                    payload = _read_json(cur_path) or {}
                    source = payload.get("source", "") or cur_path.name

                    st.markdown(
                        '<div class="soft-panel" style="border-color: rgba(239,68,68,.35); '
                        'background: rgba(239,68,68,.06);">'
                        '<span class="small-muted">This removes the game file and all original comments from the database. This cannot be undone.</span>',
                        unsafe_allow_html=True,
                    )

                    if st.session_state.delete_pending == str(cur_path):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.button(
                                "Confirm delete",
                                key="confirm_delete_current",
                                use_container_width=True,
                                on_click=make_delete_confirm(cur_path),
                            )
                        with c2:
                            st.button(
                                "Cancel",
                                key="cancel_delete_current",
                                use_container_width=True,
                                on_click=on_cancel_delete_current,
                            )
                    else:
                        st.button(
                            f"Delete {source}",
                            key="mark_delete_current",
                            use_container_width=True,
                            on_click=on_mark_delete_current,
                        )

                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.caption("_Open a saved game to see delete options._")

        # ===== Tab 4: Coverage Map =====
        with tab_coverage:
            st.subheader("Coverage map (all saved lines)")

            if not files:
                st.caption("_No saved games yet — save at least one line to see coverage._")
            else:
                coverage_tree = build_coverage_tree_cached(_game_file_mtimes())

                moves = st.session_state.moves_san
                has_moves = bool(moves)

                attach_ply = _current_attach_ply() if has_moves else 0
                default_full_moves = float(max(1, min(20, (attach_ply // 2) + 1)))

                if "coverage_slider" not in st.session_state:
                    st.session_state.coverage_slider = default_full_moves
                elif (
                    st.session_state.get("coverage_slider_synced", True)
                    and attach_ply != st.session_state.get("coverage_last_attach_ply", 0)
                ):
                    st.session_state.coverage_slider = default_full_moves

                max_moves = st.slider(
                    "Max moves to show",
                    min_value=1.0,
                    max_value=20.0,
                    step=0.5,
                    format="%.1f",
                    help="Measured in full moves; 0.5 = one half-move (one ply).",
                    key="coverage_slider",
                )

                if st.session_state.get("coverage_slider_synced", True):
                    if abs(max_moves - default_full_moves) > 1e-9:
                        st.session_state.coverage_slider_synced = False

                st.session_state.coverage_last_attach_ply = attach_ply

                max_depth_plies = int(max_moves * 2)
                coverage_html = _render_coverage_html(coverage_tree, max_depth=max_depth_plies)
                st.markdown(coverage_html, unsafe_allow_html=True)

        # ===== Tab 5: Tactics (creation) =====
        if tab_tactics is not None:
            with tab_tactics:
                st.subheader("Create a tactic from this line")

                moves = st.session_state.get("moves_san", [])
                cur_file = st.session_state.get("current_file")

                if not cur_file or not moves:
                    st.caption("_Open a saved game with moves to create a tactic._")
                else:
                    render_moves_bubbles(moves)
                    moves_len = len(moves)

                    # Step 1: pick start (after Black move)
                    st.markdown("### 1) Starting position")
                    st.caption("Use the arrows to step through **Black moves only** (positions where White is to play).")

                    allowed_start = _allowed_ply_idxs_after_black(moves_len)
                    snav = st.columns(4)
                    with snav[0]:
                        st.button("|◀", key="tstart_first",
                                on_click=(lambda: _reposition_board(allowed_start[0] if allowed_start else 0)),
                                use_container_width=True, disabled=not allowed_start)
                    with snav[1]:
                        st.button("◀", key="tstart_prev",
                                on_click=(lambda: _jump_to_allowed_ply(allowed_start, -1)),
                                use_container_width=True, disabled=not allowed_start)
                    with snav[2]:
                        st.button("▶", key="tstart_next",
                                on_click=(lambda: _jump_to_allowed_ply(allowed_start, +1)),
                                use_container_width=True, disabled=not allowed_start)
                    with snav[3]:
                        st.button("▶|", key="tstart_last",
                                on_click=(lambda: _reposition_board(allowed_start[-1] if allowed_start else moves_len)),
                                use_container_width=True, disabled=not allowed_start)

                    st.button("Set start", use_container_width=True,
                            on_click=on_set_tactic_start_here, disabled=not allowed_start)

                    # Step 2: pick end (after White move)
                    st.markdown("### 2) Final position")
                    end_disabled = not st.session_state.get("tactic_start_selected", False)

                    if end_disabled:
                        st.caption("Select a start first.")
                    else:
                        st.caption("Now step through **White moves only**, starting after the first White move in the window.")

                    start_ply = int(st.session_state.get("tactic_start_ply", 0))
                    allowed_end = _allowed_ply_idxs_after_white(moves_len, start_ply) if not end_disabled else []

                    enav = st.columns(4)
                    with enav[0]:
                        st.button("|◀", key="tend_first",
                                on_click=(lambda: _reposition_board(allowed_end[0] if allowed_end else start_ply + 1)),
                                use_container_width=True, disabled=end_disabled or not allowed_end)
                    with enav[1]:
                        st.button("◀", key="tend_prev",
                                on_click=(lambda: _jump_to_allowed_ply(allowed_end, -1)),
                                use_container_width=True, disabled=end_disabled or not allowed_end)
                    with enav[2]:
                        st.button("▶", key="tend_next",
                                on_click=(lambda: _jump_to_allowed_ply(allowed_end, +1)),
                                use_container_width=True, disabled=end_disabled or not allowed_end)
                    with enav[3]:
                        st.button("▶|", key="tend_last",
                                on_click=(lambda: _reposition_board(allowed_end[-1] if allowed_end else moves_len)),
                                use_container_width=True, disabled=end_disabled or not allowed_end)

                    st.button("Set end", use_container_width=True, on_click=on_set_tactic_end_here,
                            disabled=end_disabled or not allowed_end)

                    st.button("Reset selection", use_container_width=True, on_click=on_reset_tactic_selection)

                    # Show normalized selection
                    start_raw = int(st.session_state.get("tactic_start_ply", 0))
                    end_raw = int(st.session_state.get("tactic_end_ply", len(moves) - 1))
                    start_ply_n, end_ply_n = normalize_tactic_window(len(moves), start_raw, end_raw)

                    st.markdown(
                        f"<div class='soft-panel'>"
                        f"<b>Selected window</b><br>"
                        f"<span class='small-muted'>Start: {format_span(start_ply_n, moves)} &nbsp;•&nbsp; "
                        f"End: {format_span(end_ply_n, moves)}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    st.text_area(
                        "Tactic notes (optional)",
                        key="tactic_new_note",
                        height=110,
                        placeholder="What’s the point? What should you remember?",
                    )

                start_selected = bool(st.session_state.get("tactic_start_selected", False))
                end_selected = bool(st.session_state.get("tactic_end_selected", False))

                st.button(
                    "Save tactic",
                    type="primary",
                    use_container_width=True,
                    on_click=on_save_tactic_from_tab,
                    disabled=(not start_selected) or (not end_selected),
                )

                st.markdown("---")
                st.subheader("Other tactics from this game")

                cur_file = st.session_state.get("current_file")
                titems = list_tactics_for_game(cur_file) if cur_file else []

                if not titems:
                    st.caption("_No tactics saved yet for this game._")
                else:
                    for fp, payload in titems[:20]:
                        label = payload.get("source_label") or fp.name
                        tags = payload.get("tags") or []
                        if tags:
                            st.caption("Tags: " + ", ".join(tags))

                        span = ""
                        try:
                            sp = int(payload.get("start_ply", 0))
                            span = f" (Starts at move {(sp // 2) + 1})"
                        except Exception:
                            span = ""


                        c1, c2 = st.columns([0.25, 0.75], vertical_alignment="center")
                        with c1:
                            st.button(
                                "Train",
                                key=f"train_{fp.name}",
                                use_container_width=True,
                                on_click=(lambda p=fp, pl=payload: (
                                    st.session_state.__setitem__("tactic_mode", True),
                                    _tactic_load_into_state(p, pl),
                                )),
                            )
                        with c2:
                            st.markdown(f"**{label}**{span}")

        if tab_import_export is not None:
            with tab_import_export:
                st.subheader("Import / Export")

                # ---------- 1) PGN Import ----------
                st.markdown("### 1) Import PGN (chess.com / lichess)")
                st.caption("Imports the mainline from a single PGN game into your current (unsaved) workspace.")

                pgn_file = st.file_uploader(
                    "PGN file",
                    type=["pgn"],
                    accept_multiple_files=False,
                    key=f"pgn_upload_main_{st.session_state.pgn_uploader_nonce}",
                    label_visibility="collapsed",
                )

                if pgn_file is not None:
                    b = pgn_file.getvalue()
                    fp = _pgn_fingerprint(pgn_file.name, b)

                    if st.session_state.get("pgn_fingerprint") != fp:
                        games, errs = parse_pgn_upload(b)
                        st.session_state.pgn_fingerprint = fp
                        st.session_state.pgn_games = games
                        st.session_state.pgn_parse_errors = errs

                        if games and not errs:
                            st.session_state.pgn_selected_idx = 0
                            if st.session_state.get("pgn_auto_import_single", True) and len(games) == 1:
                                st.session_state.pgn_pending_import = True
                                st.rerun()

                    errs = st.session_state.get("pgn_parse_errors") or []
                    if errs:
                        for e in errs[:6]:
                            st.caption(f"⚠️ {e}")
                    else:
                        games = st.session_state.get("pgn_games") or []
                        if games:
                            g = games[0]
                            st.caption(f"✅ Detected: {g['label']} · {len(g['moves_san'])} plies")
                

                st.markdown("### Or paste PGN / movetext")
                st.caption("Paste a full PGN (with headers) or just a move line like: `1. d4 d5 2. Bf4 ...`")

                st.text_area(
                    "Paste PGN",
                    key="pgn_paste_text",
                    height=160,
                    placeholder="Paste PGN here…",
                    label_visibility="collapsed",
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.button("Parse pasted PGN", on_click=on_parse_pgn_paste, use_container_width=True)
                with c2:
                    st.button("Clear", on_click=reset_pgn_import_ui, use_container_width=True)

                # If we have parsed games (from file OR paste), let user choose + import
                games = st.session_state.get("pgn_games") or []
                errs = st.session_state.get("pgn_parse_errors") or []

                if errs:
                    for e in errs[:6]:
                        st.caption(f"⚠️ {e}")

                if games:
                    if len(games) > 1:
                        st.selectbox(
                            "Choose game",
                            options=list(range(len(games))),
                            format_func=lambda i: games[i]["label"],
                            key="pgn_selected_idx",
                        )
                        st.caption(f"Parsed **{len(games)}** game(s). Select one to import.")
                    else:
                        st.caption(f"✅ Ready: {games[0]['label']} · {len(games[0]['moves_san'])} plies")

                    st.button(
                        "Import selected game",
                        type="primary",
                        on_click=on_mark_pgn_pending_import,
                        use_container_width=True,
                        disabled=bool(errs),
                    )


                st.divider()

                # ---------- 2) Import MyChessNotebook package ----------
                st.markdown("### 2) Import MyChessNotebook package")
                st.caption("Upload a single exported game JSON, or a ZIP package containing multiple games (and optional tactics).")

                mcn_file = st.file_uploader(
                    "MyChessNotebook export (.json or .zip)",
                    type=["json", "zip"],
                    accept_multiple_files=False,
                    key="mcn_import_uploader",
                    label_visibility="collapsed",
                )

                if mcn_file is not None:
                    report = import_mcn_upload(mcn_file.name, mcn_file.getvalue())

                    if report["errors"]:
                        for e in report["errors"][:8]:
                            st.markdown(f"<div class='flash'>❌ {e}</div>", unsafe_allow_html=True)

                    if report["warnings"]:
                        with st.expander(f"Warnings ({len(report['warnings'])})"):
                            for w in report["warnings"][:50]:
                                st.caption(f"⚠️ {w}")

                    if report["imported_games"]:
                        st.markdown(
                            f"<div class='soft-panel'><b>Imported games:</b> {len(report['imported_games'])}<br>"
                            f"<span class='small-muted'>Saved into your local database.</span></div>",
                            unsafe_allow_html=True,
                        )
                        if report["imported_tactics"]:
                            st.caption(f"Imported tactics: **{report['imported_tactics']}**")

                st.divider()

                # ---------- 3) Export ----------
                st.markdown("### 3) Export")
                st.caption("Select games to export. The ZIP will include those games (and optionally tactics).")

                games_index = build_games_index_cached(_game_file_mtimes())
                # label -> path
                label_to_path = {g["label"]: g["path"] for g in games_index}

                selected_labels = st.multiselect(
                    "Games to export",
                    options=list(label_to_path.keys()),
                )

                include_tactics = st.checkbox("Include tactics for selected games", value=True)
                include_sources = st.checkbox(
                    "Include source games needed for imported comments",
                    value=True,
                    help="If selected games have imported comments pointing to other games, include those source games too.",
                )

                if selected_labels:
                    selected_paths = [label_to_path[lbl] for lbl in selected_labels if lbl in label_to_path]
                    zip_bytes, meta = export_mcn_zip(
                        selected_game_paths=selected_paths,
                        include_tactics=include_tactics,
                        include_source_games_for_imported_comments=include_sources,
                    )

                    st.caption(f"Will export: **{meta['games']}** game(s), **{meta['tactics']}** tactic(s)")
                    if meta["warnings"]:
                        with st.expander(f"Export warnings ({len(meta['warnings'])})"):
                            for w in meta["warnings"][:50]:
                                st.caption(f"⚠️ {w}")

                    st.download_button(
                        "Download export ZIP",
                        data=zip_bytes,
                        file_name=f"MyChessNotebook_export_{time.strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )
                else:
                    st.caption("_Select at least one game to enable export._")
