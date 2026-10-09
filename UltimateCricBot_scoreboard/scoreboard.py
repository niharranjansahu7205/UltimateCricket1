"""
Live Scoreboard — /scoreboard (Team Cricket)

Posts one pinned, auto-updating scoreboard message for a running Team Cricket
match, with a 🔄 Refresh button.

Design
------
- Every number on the scoreboard is *derived from SQLite* (team_match_events +
  team_live_state + team_match_players), never from the in-memory TeamMatch.
  So Refresh is a pure read-and-render: it can't score a ball, rotate strike or
  touch match state, and a scoreboard still renders correctly after a restart.
- Layers: database.py (reads/writes) -> derive_*/build_* (pure functions, no
  Telegram, no DB) -> handlers + update worker (Telegram only).
- Auto-update is event-driven: teamcricket.py calls request_update() after each
  committed delivery. One short-lived worker task per match coalesces bursts
  and respects a minimum edit interval — no polling loop, nothing per user.
- Refresh button: callback_data = "sb:<match_id>:<r|l|c>" (<= ~20 bytes).
      r = refresh current view   l = live view   c = full scorecard
      p = popup (compact alert shown from the match's number-pad button)
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import re
import time
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest, Forbidden, RetryAfter, TelegramError
from telegram.ext import ContextTypes

import database as db
from database import is_banned

logger = logging.getLogger(__name__)

SEP = "━━━━━━━━━━━━━━━━━━"
MAX_TEXT = 4000                 # Telegram hard limit is 4096
MIN_EDIT_INTERVAL = 1.2         # seconds between auto edits of one scoreboard
USER_COOLDOWN = 1.5             # seconds between refresh taps per user
CB_RE = re.compile(r"sb:(\d+):([rlcp])")

BALL_EMOJI = {"0": "0️⃣", "1": "1️⃣", "2": "2️⃣", "3": "3️⃣", "4": "4️⃣", "5": "5️⃣", "6": "6️⃣", "W": "❌"}


def _limits() -> tuple[int, int]:
    # Imported lazily: teamcricket imports this module lazily too.
    import teamcricket as tc
    return tc.TEAM_OVERS, tc.BALLS_PER_OVER


def _esc(x) -> str:
    return html.escape(str(x) if x is not None else "", quote=False)


# ---------------------------------------------------------------------------
# Pure derivation (events -> numbers). No I/O.
# ---------------------------------------------------------------------------

def derive_innings(events: list[dict], innings: int, balls_per_over: int) -> dict:
    """Aggregates one innings from its ball events.

    This engine has no wides/no-balls/byes (every delivery is legal and all
    runs come off the bat), so `extras` is always 0 — but team total is still
    computed from the events themselves, not by summing batter rows.
    """
    evs = [e for e in events if e["innings"] == innings]
    bats: dict[int, dict] = {}
    bowls: dict[int, dict] = {}
    order: list[int] = []
    total = wickets = legal = extras = 0
    for e in evs:
        o = str(e["outcome"])
        sid, bid = e["striker_id"], e["bowler_id"]
        b = bats.setdefault(sid, {"runs": 0, "balls": 0, "fours": 0, "sixes": 0, "out": False, "dismissal": None})
        w = bowls.setdefault(bid, {"balls": 0, "runs": 0, "wickets": 0})
        if sid not in order:
            order.append(sid)
        legal += 1
        b["balls"] += 1
        w["balls"] += 1
        if o == "W":
            wickets += 1
            b["out"] = True
            b["dismissal"] = e.get("dismissal")
            w["wickets"] += 1
        elif o.isdigit():
            r = int(o)
            total += r
            b["runs"] += r
            w["runs"] += r
            if r == 4:
                b["fours"] += 1
            elif r == 6:
                b["sixes"] += 1
    return {
        "events": evs, "bats": bats, "bowls": bowls, "bat_order": order,
        "total": total + extras, "wickets": wickets, "legal": legal, "extras": extras,
    }


def _sr(b: dict) -> str:
    return f"{(b['runs'] / b['balls']) * 100:.1f}" if b["balls"] else "0.0"


def _overs(legal: int, bpo: int) -> str:
    return f"{legal // bpo}.{legal % bpo}"


def _rate(runs: int, legal: int, bpo: int) -> str:
    return f"{runs / (legal / bpo):.2f}" if legal else "0.00"


def ball_label(e: dict) -> str:
    return BALL_EMOJI.get(str(e["outcome"]), str(e["outcome"]))


# ---------------------------------------------------------------------------
# Rendering (pure: snapshot dict -> HTML text)
# ---------------------------------------------------------------------------

def _team_label(m: dict, side: Optional[str]) -> str:
    return (m.get("team_a_label") if side == "a" else m.get("team_b_label")) or ("Team A" if side == "a" else "Team B")


def _status_line(snap: dict, running: bool, innings_done: bool) -> str:
    st = snap["match"].get("status")
    if st == "completed":
        return "🏁 MATCH COMPLETED"
    if st == "cancelled":
        return "❌ MATCH CANCELLED"
    if not running:
        return "⏸ MATCH NOT RUNNING (bot restarted)"
    live = snap["live"]
    if live is None or live.get("stage") in ("toss_call", "bat_bowl"):
        return "🟡 TOSS / SETUP"
    if live.get("stage") == "order_pick":
        return "🟡 BATTING ORDER BEING SET"
    if innings_done:
        return "🟠 INNINGS BREAK"
    return "🔴 LIVE MATCH"


def build_scoreboard(snap: dict, running: bool, view: str = "l") -> str:
    default_overs, bpo = _limits()
    overs_limit = (snap.get("live") or {}).get("overs") or default_overs
    text = _build_card(snap, running, overs_limit, bpo) if view == "c" else _build_live(snap, running, overs_limit, bpo)
    if len(text) > MAX_TEXT:
        text = text[: MAX_TEXT - 20].rstrip() + "\n…(truncated)"
    return text


def _build_live(snap: dict, running: bool, overs_limit: int, bpo: int) -> str:
    m, live = snap["match"], snap["live"]
    names = {p["user_id"]: p["name"] for p in snap["players"]}
    nm = lambda uid: _esc(names.get(uid, "?"))  # noqa: E731
    finished = m.get("status") in ("completed", "cancelled")
    head = [f"🏏 <b>CRICKET {'' if finished else 'LIVE '}SCOREBOARD</b>", SEP, ""]

    if live is None:
        head += [
            f"🟦 <b>{_esc(_team_label(m, 'a'))}</b>  vs  🟥 <b>{_esc(_team_label(m, 'b'))}</b>",
            _status_line(snap, running, False), "", "<i>Match is being set up — innings not started yet.</i>",
        ]
        return "\n".join(head)

    inn = live["innings"]
    bat_side, bowl_side = live["batting_side"], live["bowling_side"]
    d = derive_innings(snap["events"], inn, bpo)
    team_size = sum(1 for p in snap["players"] if p["side"] == bat_side)
    max_balls = overs_limit * bpo
    target = live.get("target") if inn == 2 else None
    all_out = team_size > 0 and d["wickets"] >= team_size - 1
    innings_done = (
        m.get("status") == "completed"
        or all_out
        or d["legal"] >= max_balls
        or (target is not None and d["total"] >= target)
    )

    out = head + [
        f"🏆 <b>{_esc(_team_label(m, bat_side))}</b>  <i>(Innings {inn})</i>  🔑 <code>{m['match_id']}</code>",
        _status_line(snap, running, innings_done),
        "",
        f"📊 SCORE: <b>{d['total']}/{d['wickets']}</b>",
        f"⏱ Overs: {_overs(d['legal'], bpo)} / {overs_limit}",
        f"⚡ Run Rate: {_rate(d['total'], d['legal'], bpo)}",
    ]
    if target is not None:
        need = target - d["total"]
        left = max_balls - d["legal"]
        if need > 0 and not innings_done:
            out.append(f"🎯 Target: <b>{target}</b> — need <b>{need}</b> from {left} ball(s)")
        else:
            out.append(f"🎯 Target: <b>{target}</b>")
    if inn == 2:
        d1 = derive_innings(snap["events"], 1, bpo)
        out.append(f"📌 Inn 1: {_esc(_team_label(m, bowl_side))} {d1['total']}/{d1['wickets']} ({_overs(d1['legal'], bpo)} ov)")

    # ---- current over ------------------------------------------------------
    out += ["", SEP, "", "🎯 <b>BALL-BY-BALL — CURRENT OVER</b>", ""]
    if d["events"]:
        if innings_done:
            over_no = d["events"][-1]["over_no"]
        else:
            over_no = live.get("overs_completed", 0) + 1
        cur = [e for e in d["events"] if e["over_no"] == over_no]
        if cur:
            suffix = "  ✅ <i>innings completed</i>" if innings_done else ""
            out.append(f"Over {over_no}:  " + " │ ".join(ball_label(e) for e in cur) + suffix)
        else:
            prev = [e for e in d["events"] if e["over_no"] == over_no - 1]
            out.append(f"Over {over_no}: <i>waiting for first ball</i>")
            if prev:
                out.append("Last over:  " + " │ ".join(ball_label(e) for e in prev))
    else:
        out.append("<i>No balls bowled yet.</i>")

    # ---- batsmen at the crease --------------------------------------------
    lineup = live.get("lineup") or []
    if lineup and not innings_done:
        active = []
        for uid, mark, label in ((live.get("striker_id"), "🟢", " *"), (live.get("non_striker_id"), "⚪", "")):
            if uid is None:
                continue
            b = d["bats"].get(uid)
            if b is not None and b["out"]:
                continue  # a dismissed batter is never shown as active
            b = b or {"runs": 0, "balls": 0, "fours": 0, "sixes": 0}
            active += [
                f"{mark} <b>{nm(uid)}</b>{label}",
                f"Runs: {b['runs']} ({b['balls']} balls)",
                f"4️⃣ Fours: {b['fours']} | 6️⃣ Sixes: {b['sixes']}",
                f"Strike Rate: {_sr(b)}",
                "",
            ]
        if active:
            out += ["", SEP, "", "🏏 <b>BATSMAN STATISTICS</b>", ""] + active[:-1]

    # ---- batting contribution ---------------------------------------------
    if lineup:
        batted = list(lineup[: live.get("next_in_idx", 0)])
        for uid in d["bat_order"]:            # safety net: anyone who faced a ball
            if uid not in batted:
                batted.append(uid)
        rows = []
        for i, uid in enumerate(batted, 1):
            b = d["bats"].get(uid, {"runs": 0, "balls": 0, "out": False, "dismissal": None})
            if b["out"]:
                tail = f"  ❌ {_esc(b['dismissal'] or 'out')}"
            elif not innings_done and uid == live.get("striker_id"):
                tail = " *"
            else:
                tail = ""
            rows.append(f"{i}. {nm(uid)} — {b['runs']} ({b['balls']}){tail}")
        if rows:
            out += ["", SEP, "", "📈 <b>BATTING CONTRIBUTION</b>", ""] + rows
            out += ["", f"➕ Extras: {d['extras']}", f"💥 Total: <b>{d['total']}</b>"]

    # ---- bowling -----------------------------------------------------------
    bid = live.get("bowler_id")
    if bid is not None:
        w = d["bowls"].get(bid, {"balls": 0, "runs": 0, "wickets": 0})
        eco = f"{w['runs'] / (w['balls'] / bpo):.2f}" if w["balls"] else "0.00"
        out += [
            "", SEP, "", "⚾ <b>BOWLING</b>", "",
            f"🎯 Bowler: <b>{nm(bid)}</b>",
            f"Overs: {_overs(w['balls'], bpo)}",
            f"Runs Conceded: {w['runs']}",
            f"Wickets: {w['wickets']}",
            f"Economy: {eco}",
        ]

    if m.get("status") == "completed" and m.get("result"):
        out += ["", SEP, "", _esc(m["result"])]
    return "\n".join(out)


def _build_card(snap: dict, running: bool, overs_limit: int, bpo: int) -> str:
    m = snap["match"]
    names = {p["user_id"]: p["name"] for p in snap["players"]}
    nm = lambda uid: _esc(names.get(uid, "?"))  # noqa: E731
    out = ["📊 <b>FULL SCORECARD</b>", SEP]
    live = snap["live"]
    innings_seen = sorted({e["innings"] for e in snap["events"]} | ({live["innings"]} if live else set()))
    if not innings_seen:
        out += ["", "<i>No balls bowled yet.</i>"]
    for inn in innings_seen:
        d = derive_innings(snap["events"], inn, bpo)
        # batting side of that innings: from live state if current, else the other side's players
        side = None
        if live and inn == live["innings"]:
            side = live["batting_side"]
        elif d["bat_order"]:
            side = next((p["side"] for p in snap["players"] if p["user_id"] == d["bat_order"][0]), None)
        out += ["", f"🏏 <b>Innings {inn} — {_esc(_team_label(m, side))}</b>: {d['total']}/{d['wickets']} ({_overs(d['legal'], bpo)} ov)"]
        for uid in d["bat_order"]:
            b = d["bats"][uid]
            status = _esc(b["dismissal"] or "out") if b["out"] else "not out"
            out.append(f"• {nm(uid)} — {b['runs']} ({b['balls']}) 4s:{b['fours']} 6s:{b['sixes']} SR {_sr(b)} [{status}]")
        for uid, w in d["bowls"].items():
            eco = f"{w['runs'] / (w['balls'] / bpo):.2f}" if w["balls"] else "0.00"
            out.append(f"🎯 {nm(uid)} — {_overs(w['balls'], bpo)} ov, {w['runs']} R, {w['wickets']} W (Econ {eco})")
    out += ["", SEP, _status_line(snap, running, False)]
    if m.get("status") == "completed" and m.get("result"):
        out.append(_esc(m["result"]))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Popup (callback-answer alert) — Telegram allows max 200 chars, plain text
# ---------------------------------------------------------------------------

POPUP_MAX = 200


def _u16(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _short(name, n: int = 9) -> str:
    name = str(name or "?")
    return name if len(name) <= n else name[: n - 1] + "…"


def build_popup(snap: dict, running: bool) -> str:
    """Ultra-compact scoreboard for the 'Scoreboard' button under the number
    pad. Plain text, <= 200 UTF-16 units (Telegram's alert limit)."""
    overs_default, bpo = _limits()
    m, live = snap["match"], snap["live"]
    names = {p["user_id"]: p["name"] for p in snap["players"]}
    if live is None:
        return "🏏 Match is being set up — innings not started yet."

    overs_limit = live.get("overs") or overs_default
    inn = live["innings"]
    d = derive_innings(snap["events"], inn, bpo)
    team_size = sum(1 for p in snap["players"] if p["side"] == live["batting_side"])
    target = live.get("target") if inn == 2 else None
    done = (
        m.get("status") == "completed"
        or (team_size > 0 and d["wickets"] >= team_size - 1)
        or d["legal"] >= overs_limit * bpo
        or (target is not None and d["total"] >= target)
    )

    lines = [f"🏏 {_short(_team_label(m, live['batting_side']), 10)} {d['total']}/{d['wickets']} "
             f"({_overs(d['legal'], bpo)}/{overs_limit})"]
    rr = f"RR {_rate(d['total'], d['legal'], bpo)}"
    if target is not None and not done:
        rr += f" | 🎯{target} need {max(target - d['total'], 0)}"
    lines.append(rr)

    if m.get("status") == "completed":
        lines.append("🏁 Match completed")
    elif m.get("status") == "cancelled":
        lines.append("❌ Match cancelled")
    elif not running:
        lines.append("⏸ Not running")
    elif done:
        lines.append("✅ Innings completed")

    if d["events"]:
        over_no = d["events"][-1]["over_no"] if done else live.get("overs_completed", 0) + 1
        cur = [str(e["outcome"]).replace("W", "❌") for e in d["events"] if e["over_no"] == over_no]
        if cur:
            lines.append(f"Over {over_no}: " + " │ ".join(cur))
        else:  # new over not started yet -> show the one just finished
            prev = [str(e["outcome"]).replace("W", "❌") for e in d["events"] if e["over_no"] == over_no - 1]
            lines.append(f"Last over: " + " │ ".join(prev) if prev else f"Over {over_no}: –")

    if not done and m.get("status") == "active":
        bits = []
        for uid, mark, star in ((live.get("striker_id"), "🟢", "*"), (live.get("non_striker_id"), "⚪", "")):
            b = d["bats"].get(uid) if uid is not None else None
            if uid is None or (b and b["out"]):
                continue
            b = b or {"runs": 0, "balls": 0}
            bits.append(f"{mark}{_short(names.get(uid))}{star} {b['runs']}({b['balls']})")
        if bits:
            lines.append("  ".join(bits))
        bid = live.get("bowler_id")
        if bid is not None:
            w = d["bowls"].get(bid, {"balls": 0, "runs": 0, "wickets": 0})
            lines.append(f"🎯{_short(names.get(bid))} {_overs(w['balls'], bpo)}-{w['runs']}-{w['wickets']}")

    # Drop least important lines first until it fits.
    text = "\n".join(lines)
    while _u16(text) > POPUP_MAX and len(lines) > 2:
        lines.pop()
        text = "\n".join(lines)
    while _u16(text) > POPUP_MAX:
        text = text[:-2]
    return text


def build_keyboard(match_id: int, view: str) -> InlineKeyboardMarkup:
    second = (
        InlineKeyboardButton("🏏 Live Score", callback_data=f"sb:{match_id}:l")
        if view == "c"
        else InlineKeyboardButton("📊 Full Scorecard", callback_data=f"sb:{match_id}:c")
    )
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Refresh Scoreboard", callback_data=f"sb:{match_id}:r")],
        [second],
    ])


# ---------------------------------------------------------------------------
# Render + edit (shared by the Refresh button and the auto-update worker)
# ---------------------------------------------------------------------------

_locks: dict[int, asyncio.Lock] = {}
_dirty: set[int] = set()
_workers: dict[int, asyncio.Task] = {}
_last_auto: dict[int, float] = {}
_user_last: dict[tuple[int, int], float] = {}


def _lock(match_id: int) -> asyncio.Lock:
    return _locks.setdefault(match_id, asyncio.Lock())


def _is_running(match_id: int) -> bool:
    import teamcricket as tc
    return any(
        g.db_match_id == match_id and g.stage not in ("done", "cancelled")
        for g in tc.TEAM_MATCHES.values()
    )


async def render_and_edit(bot, match_id: int) -> str:
    """Reads the latest persisted state and edits the scoreboard message.
    Returns: 'updated' | 'same' | 'none' | 'gone' | 'busy' | 'error'.
    Never raises, never mutates match state."""
    try:
        snap = await asyncio.to_thread(db.get_scoreboard_snapshot, match_id)
    except Exception:
        logger.exception("Scoreboard DB read failed for match %s", match_id)
        return "error"
    if snap is None or snap["scoreboard"] is None:
        return "none"
    sb = snap["scoreboard"]
    view = sb.get("view") or "l"
    finished = snap["match"].get("status") in ("completed", "cancelled")
    try:
        text = build_scoreboard(snap, _is_running(match_id), view)
    except Exception:
        logger.exception("Scoreboard render failed for match %s", match_id)
        return "error"

    digest = hashlib.sha1(f"{view}\n{text}".encode()).hexdigest()
    if digest == sb.get("last_hash"):
        if finished and sb.get("pinned"):
            await _unpin(bot, sb)
        return "same"

    try:
        await bot.edit_message_text(
            chat_id=sb["chat_id"], message_id=sb["message_id"], text=text,
            parse_mode=ParseMode.HTML, reply_markup=build_keyboard(match_id, view),
        )
    except BadRequest as e:
        msg = str(e).lower()
        if "not modified" in msg:
            await asyncio.to_thread(db.update_team_scoreboard, match_id, digest)
            return "same"
        if "not found" in msg or "can't be edited" in msg or "message_id_invalid" in msg:
            logger.warning("Scoreboard message gone for match %s: %s", match_id, e)
            return "gone"
        logger.error("Scoreboard edit rejected for match %s: %s", match_id, e)
        return "error"
    except RetryAfter as e:
        logger.warning("Scoreboard rate-limited for match %s (retry in %ss)", match_id, e.retry_after)
        return "busy"
    except TelegramError:
        logger.exception("Scoreboard edit failed for match %s", match_id)
        return "error"

    await asyncio.to_thread(db.update_team_scoreboard, match_id, digest)
    if finished and sb.get("pinned"):
        await _unpin(bot, sb)
    return "updated"


async def _unpin(bot, sb: dict):
    try:
        await bot.unpin_chat_message(chat_id=sb["chat_id"], message_id=sb["message_id"])
    except TelegramError as e:
        logger.info("Could not unpin scoreboard %s: %s", sb["message_id"], e)
    await asyncio.to_thread(db.update_team_scoreboard, sb["match_id"], None, None, False)


# ---------------------------------------------------------------------------
# Auto-update (event-driven, coalesced, one short-lived task per match)
# ---------------------------------------------------------------------------

def request_update(bot, match_id: Optional[int]):
    """Called by the game engine AFTER a state change is committed to SQLite.
    Fire-and-forget: never blocks or breaks the ball flow."""
    if not match_id:
        return
    _dirty.add(match_id)
    t = _workers.get(match_id)
    if t is not None and not t.done():
        return  # the running worker will pick the dirty flag up
    _workers[match_id] = asyncio.create_task(_worker(bot, match_id))


async def _worker(bot, match_id: int):
    try:
        while match_id in _dirty:
            wait = MIN_EDIT_INTERVAL - (time.monotonic() - _last_auto.get(match_id, 0.0))
            if wait > 0:
                await asyncio.sleep(wait)
            _dirty.discard(match_id)
            async with _lock(match_id):
                res = await render_and_edit(bot, match_id)
            _last_auto[match_id] = time.monotonic()
            if res == "busy":
                _dirty.add(match_id)
                await asyncio.sleep(3)
            elif res in ("gone", "none"):
                _dirty.discard(match_id)
                if res == "gone":
                    logger.warning("Auto-update stopped for match %s: scoreboard message is gone", match_id)
    except Exception:
        logger.exception("Scoreboard worker crashed for match %s", match_id)
    finally:
        _workers.pop(match_id, None)


# ---------------------------------------------------------------------------
# /scoreboard
# ---------------------------------------------------------------------------

def _find_running_match(chat_id: int, reply_msg_id: Optional[int]):
    import teamcricket as tc
    live = [
        g for g in tc.TEAM_MATCHES.values()
        if g.chat_id == chat_id and g.db_match_id and g.stage not in ("lobby", "done", "cancelled")
    ]
    if reply_msg_id is not None:
        for g in live:
            if g.message_id == reply_msg_id:
                return g, live
    if len(live) == 1:
        return live[0], live
    return None, live


async def scoreboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat, msg, user = update.effective_chat, update.message, update.effective_user
    if chat.type not in ("group", "supergroup"):
        await msg.reply_text("🏏 /scoreboard works inside groups, during a /teamcricket match.")
        return
    if is_banned(user.id):
        await msg.reply_text("🚫 You are banned from playing UltimateCric.")
        return

    reply_id = msg.reply_to_message.message_id if msg.reply_to_message else None
    game, live_list = _find_running_match(chat.id, reply_id)
    if game is None:
        if len(live_list) > 1:
            await msg.reply_text("Multiple Team Cricket matches are running here — reply to the match message with /scoreboard.")
        else:
            await msg.reply_text("No live Team Cricket match in this group right now. Start one with /teamcricket!")
        return

    mid = game.db_match_id
    async with _lock(mid):
        # One scoreboard per match: retire the previous one if it exists.
        old = await asyncio.to_thread(db.get_team_scoreboard, mid)
        if old:
            try:
                await context.bot.unpin_chat_message(chat_id=old["chat_id"], message_id=old["message_id"])
            except TelegramError:
                pass
            try:
                await context.bot.delete_message(chat_id=old["chat_id"], message_id=old["message_id"])
            except TelegramError:
                pass

        snap = await asyncio.to_thread(db.get_scoreboard_snapshot, mid)
        text = build_scoreboard(snap, True, "l")
        sent = await msg.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=build_keyboard(mid, "l"))

        pinned = False
        try:
            await context.bot.pin_chat_message(chat_id=chat.id, message_id=sent.message_id, disable_notification=True)
            pinned = True
        except (Forbidden, BadRequest) as e:
            logger.info("Could not pin scoreboard in chat %s: %s", chat.id, e)
        except TelegramError:
            logger.exception("Pin failed in chat %s", chat.id)

        await asyncio.to_thread(db.save_team_scoreboard, mid, chat.id, sent.message_id, pinned)
        digest = hashlib.sha1(f"l\n{text}".encode()).hexdigest()
        await asyncio.to_thread(db.update_team_scoreboard, mid, digest)

    if not pinned:
        await msg.reply_text(
            "ℹ️ I couldn't pin the scoreboard. Make me an admin with the <b>Pin Messages</b> permission "
            "and run /scoreboard again. It will still auto-update.",
            parse_mode=ParseMode.HTML,
        )


# ---------------------------------------------------------------------------
# Callback: sb:<match_id>:<r|l|c>
# ---------------------------------------------------------------------------

async def scoreboard_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        m = CB_RE.fullmatch(q.data or "")
        if not m or q.message is None:
            await q.answer("Invalid action.", show_alert=True)
            return
        mid, action = int(m.group(1)), m.group(2)

        if action == "p":
            await _popup(q, mid)
            return

        row = await asyncio.to_thread(db.get_team_scoreboard, mid)
        if row is None:
            await q.answer("This scoreboard no longer exists.", show_alert=True)
            return
        if q.message.chat.id != row["chat_id"]:
            await q.answer("This scoreboard isn't available here.", show_alert=True)
            return
        if q.message.message_id != row["message_id"]:
            await q.answer("Outdated scoreboard — use /scoreboard for the current one.", show_alert=True)
            return
        if is_banned(q.from_user.id):
            await q.answer("🚫 You are banned from playing UltimateCric.", show_alert=True)
            return

        now = time.monotonic()
        key = (mid, q.from_user.id)
        if now - _user_last.get(key, 0.0) < USER_COOLDOWN:
            await q.answer("Please wait a moment…")
            return
        _user_last[key] = now
        if len(_user_last) > 2000:
            _user_last.clear()

        async with _lock(mid):
            if action in ("l", "c"):
                await asyncio.to_thread(db.update_team_scoreboard, mid, None, action)
            res = await render_and_edit(context.bot, mid)

        answers = {
            "updated": "Scoreboard updated!",
            "same": "Scoreboard is already up to date.",
            "busy": "Telegram is busy — try again in a few seconds.",
            "gone": "This scoreboard message can't be edited anymore. Use /scoreboard.",
        }
        await q.answer(answers.get(res, "Couldn't refresh right now. Please try again."))
    except Exception:
        logger.exception("Scoreboard callback failed")
        try:
            await q.answer("Couldn't refresh right now. Please try again.")
        except Exception:
            pass


async def _popup(q, match_id: int):
    """'📊 Scoreboard' button under the number pad: read-only alert popup."""
    snap = await asyncio.to_thread(db.get_scoreboard_snapshot, match_id)
    if snap is None:
        await q.answer("This match no longer exists.", show_alert=True)
        return
    if q.message.chat.id != snap["match"]["chat_id"]:
        await q.answer("This match isn't available here.", show_alert=True)
        return
    if is_banned(q.from_user.id):
        await q.answer("🚫 You are banned from playing UltimateCric.", show_alert=True)
        return
    await q.answer(build_popup(snap, _is_running(match_id)), show_alert=True)
