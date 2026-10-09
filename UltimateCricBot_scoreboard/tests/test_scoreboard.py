"""Tests for the live scoreboard. Drives the REAL Team Cricket engine
(teamcricket._resolve_ball etc.) against a temporary SQLite file, with a
fake Telegram bot that records every API call.

Run:  python -m pytest -q tests/
"""
import asyncio
import os
import sys
import types

import pytest
from telegram.error import BadRequest, RetryAfter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database as db            # noqa: E402
import scoreboard as sb          # noqa: E402
import teamcricket as tc         # noqa: E402

CHAT = -1001
A = [(1, "Alice"), (2, "Bob"), (3, "Cara")]
B = [(11, "Dan"), (12, "Eve"), (13, "Fay")]


class FakeBot:
    def __init__(self):
        self.edits, self.sent, self.pins, self.unpins, self.deleted = [], [], [], [], []
        self.fail_with = None

    async def edit_message_text(self, chat_id, message_id, text, parse_mode=None, reply_markup=None):
        if self.fail_with:
            raise self.fail_with
        self.edits.append((chat_id, message_id, text, reply_markup))

    async def send_message(self, chat_id, text, parse_mode=None):
        self.sent.append(text)

    async def pin_chat_message(self, chat_id, message_id, disable_notification=True):
        self.pins.append(message_id)

    async def unpin_chat_message(self, chat_id, message_id=None):
        self.unpins.append(message_id)

    async def delete_message(self, chat_id, message_id):
        self.deleted.append(message_id)

    def sb_edits(self):
        return [e for e in self.edits if e[1] == 999]


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    db.init_db()
    tc.TEAM_MATCHES.clear()
    for d in (sb._locks, sb._dirty, sb._workers, sb._last_auto, sb._user_last):
        d.clear() if hasattr(d, "clear") else None
    monkeypatch.setattr(sb, "MIN_EDIT_INTERVAL", 0.0)
    monkeypatch.setattr(sb, "USER_COOLDOWN", 0.0)
    yield


@pytest.fixture
def no_auto(monkeypatch):
    """Turn off event-driven auto-update so manual Refresh can be tested alone."""
    monkeypatch.setattr(sb, "request_update", lambda *a, **k: None)


def make_game(a=A, b=B):
    game = tc.TeamMatch(match_id=500, chat_id=CHAT, message_id=500, creator_id=1, creator_name="Alice")
    game.team_a.players, game.team_a.captain_id = list(a), a[0][0]
    game.team_b.players, game.team_b.captain_id = list(b), b[0][0]
    game.db_match_id = db.create_team_match(CHAT, 500, "Team A", "Team B")
    for side, team in (("a", game.team_a), ("b", game.team_b)):
        for uid, n in team.players:
            db.record_team_match_player(game.db_match_id, uid, n, side, uid == team.captain_id)
            game.batting_stats[uid] = tc.BattingStats()
            game.bowling_stats[uid] = tc.BowlingStats()
    game.stage = "toss_call"
    TEAM = tc.TEAM_MATCHES
    TEAM[500] = game
    return game


async def start_innings(ctx, game, side="a", bowler=None):
    await tc._begin_innings(ctx, game, side)
    team = tc._side_obj(game, side)
    game.lineup = [u for u, _ in team.players]
    game.order_pick_pool = []
    await tc._confirm_batting_order(ctx, game)
    other = tc._side_obj(game, tc._other_side(side))
    await tc._set_bowler_and_start_ball(ctx, game, bowler or other.players[0][0])
    tc._cancel_timer(game)


async def ball(ctx, game, bat, bowl):
    game.picks = {game.striker_id: bat, game.current_bowler_id: bowl}
    game.turn = "bowler"
    await tc._resolve_ball(ctx, game)
    tc._cancel_timer(game)


def ctx_of(bot):
    return types.SimpleNamespace(bot=bot)


def snap(game):
    return db.get_scoreboard_snapshot(game.db_match_id)


def render(game, view="l", running=True):
    return sb.build_scoreboard(snap(game), running, view)


def run(coro):
    return asyncio.run(coro)


# 1 ------------------------------------------------------------------------
def test_new_innings_zero_zero():
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot); g = make_game()
        await start_innings(ctx, g)
        t = render(g)
        assert "SCORE: <b>0/0</b>" in t and "Run Rate: 0.00" in t
        assert "No balls bowled yet" in t
        assert "🟢 <b>Alice</b> *" in t and "⚪ <b>Bob</b>" in t
        assert "Strike Rate: 0.0" in t
        assert "🎯 Bowler: <b>Dan</b>" in t
        assert "LIVE MATCH" in t
    run(go())


# 2,3,7,8 ------------------------------------------------------------------
def test_dot_single_boundary_rotation_and_stats():
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot); g = make_game()
        await start_innings(ctx, g)
        await ball(ctx, g, 0, 1)      # dot, Alice
        await ball(ctx, g, 1, 2)      # single -> strike to Bob
        assert g.striker_id == 2
        await ball(ctx, g, 4, 0)      # Bob four
        s = snap(g)
        d = sb.derive_innings(s["events"], 1, 6)
        assert d["total"] == 5 and d["wickets"] == 0 and d["legal"] == 3 and d["extras"] == 0
        assert d["bats"][1] == {"runs": 1, "balls": 2, "fours": 0, "sixes": 0, "out": False, "dismissal": None}
        assert d["bats"][2]["runs"] == 4 and d["bats"][2]["fours"] == 1
        t = render(g)
        assert "0️⃣ │ 1️⃣ │ 4️⃣" in t
        assert "SCORE: <b>5/0</b>" in t
        assert "🟢 <b>Bob</b> *" in t and "Runs: 4 (1 balls)" in t and "Strike Rate: 400.0" in t
        assert "➕ Extras: 0" in t and "💥 Total: <b>5</b>" in t
        assert "1. Alice — 1 (2)" in t and "2. Bob — 4 (1) *" in t
        assert "Overs: 0.3 / 3" in t
        # DB tables agree with the in-memory engine
        row = [p for p in s["players"] if p["user_id"] == 2][0]
        assert row["name"] == "Bob"
    run(go())


def test_over_completion_shows_only_current_over():
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot); g = make_game()
        await start_innings(ctx, g)
        for _ in range(6):
            await ball(ctx, g, 2, 1)           # six doubles, no strike change
        assert g.overs_completed == 1 and g.stage == "bowler_pick"
        assert g.striker_id == 2               # end-of-over swap
        t = render(g)
        assert "Over 2: <i>waiting for first ball</i>" in t and "Last over:" in t
        await tc._set_bowler_and_start_ball(ctx, g, 12)
        tc._cancel_timer(g)
        await ball(ctx, g, 6, 0)
        t = render(g)
        assert "Over 2:  6️⃣" in t and "2️⃣ │ 2️⃣" not in t.split("CURRENT OVER")[1].split("BATSMAN")[0]
        assert "Overs: 1.1 / 3" in t and "SCORE: <b>18/0</b>" in t
        assert "🎯 Bowler: <b>Eve</b>" in t
    run(go())


# 5 ------------------------------------------------------------------------
def test_wicket_dismissal_and_replacement():
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot); g = make_game()
        await start_innings(ctx, g)
        await ball(ctx, g, 3, 1)
        await ball(ctx, g, 5, 5)               # matching numbers -> wicket (Bob on strike)
        t = render(g)
        assert "SCORE: <b>3/1</b>" in t
        assert "❌ │" in t or "│ ❌" in t
        assert g.striker_id == 3               # Cara in
        assert "🟢 <b>Cara</b> *" in t
        active = t.split("BATSMAN STATISTICS")[1].split("BATTING CONTRIBUTION")[0]
        assert "Bob" not in active             # dismissed never shown as active
        assert "3. Cara — 0 (0) *" in t
        assert "2. Bob — 0 (1)  ❌" in t
    run(go())


# 13 -----------------------------------------------------------------------
def test_all_out_innings_completed_and_final_match():
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot)
        g = make_game(A[:2], B[:2])            # 2v2 -> all out after 1 wicket
        await start_innings(ctx, g)
        await ball(ctx, g, 6, 1)
        await ball(ctx, g, 6, 2)
        await ball(ctx, g, 3, 3)               # wicket, all out, innings 1 = 12
        assert g.innings == 2 and g.target == 13 and g.stage == "order_pick"
        t = render(g)
        assert "(Innings 2)" in t and "BATTING ORDER BEING SET" in t
        assert "Inn 1: Team A 12/1 (0.3 ov)" in t
        tc._cancel_timer(g)
        g.lineup = [11, 12]; g.order_pick_pool = []
        await tc._confirm_batting_order(ctx, g)
        await tc._set_bowler_and_start_ball(ctx, g, 1)
        tc._cancel_timer(g)
        await ball(ctx, g, 6, 0)
        await ball(ctx, g, 6, 0)               # 12 -> not yet
        t = render(g)
        assert "Target: <b>13</b> — need <b>1</b> from 16 ball(s)" in t
        await ball(ctx, g, 1, 0)               # target reached -> match over
        s = snap(g)
        assert s["match"]["status"] == "completed"
        final = render(g, running=False)
        assert "MATCH COMPLETED" in final and "LIVE" not in final
        assert "won by" in final
        assert "innings completed" in final
        card = render(g, view="c", running=False)
        assert "Innings 1" in card and "Innings 2" in card and "Alice" in card and "Dan" in card
        # extras always 0 here and total == sum of events
        d2 = sb.derive_innings(s["events"], 2, 6)
        assert d2["total"] == 13
    run(go())


# 9,10 ---------------------------------------------------------------------
def test_refresh_is_read_only_and_dedupes(no_auto):
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot); g = make_game()
        await start_innings(ctx, g)
        await ball(ctx, g, 2, 1)
        db.save_team_scoreboard(g.db_match_id, CHAT, 999, True)
        before = snap(g)
        assert await sb.render_and_edit(bot, g.db_match_id) == "updated"
        assert await sb.render_and_edit(bot, g.db_match_id) == "same"
        assert await sb.render_and_edit(bot, g.db_match_id) == "same"
        assert len(bot.sb_edits()) == 1
        after = snap(g)
        assert before["events"] == after["events"] and before["live"] == after["live"]
        await ball(ctx, g, 1, 0)
        assert await sb.render_and_edit(bot, g.db_match_id) == "updated"
        assert len(bot.sb_edits()) == 2
        assert "SCORE: <b>3/0</b>" in bot.sb_edits()[-1][2]
        kb = bot.sb_edits()[-1][3].inline_keyboard
        assert kb[0][0].callback_data == f"sb:{g.db_match_id}:r" and "Refresh" in kb[0][0].text
        assert all(len(b.callback_data.encode()) <= 64 for row in kb for b in row)
    run(go())


def fake_query(data, chat_id=CHAT, message_id=999, user_id=7):
    answers = []
    async def answer(text=None, show_alert=False):
        answers.append(text)
    q = types.SimpleNamespace(
        data=data, answer=answer, from_user=types.SimpleNamespace(id=user_id),
        message=types.SimpleNamespace(chat=types.SimpleNamespace(id=chat_id), message_id=message_id),
    )
    return types.SimpleNamespace(callback_query=q), answers


# 11 + callback validation --------------------------------------------------
def test_callback_validation_and_simultaneous_refresh(no_auto):
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot); g = make_game()
        await start_innings(ctx, g)
        await ball(ctx, g, 2, 1)
        db.save_team_scoreboard(g.db_match_id, CHAT, 999, True)
        mid = g.db_match_id

        for bad, kw in (("sb:abc:r", {}), ("sb:99999:r", {}), (f"sb:{mid}:r", {"chat_id": -5}), (f"sb:{mid}:r", {"message_id": 1})):
            u, ans = fake_query(bad, **kw)
            await sb.scoreboard_callback(u, ctx)
            assert ans and ("updated" not in ans[0])
        assert bot.sb_edits() == []

        results = []
        async def click(uid):
            u, ans = fake_query(f"sb:{mid}:r", user_id=uid)
            await sb.scoreboard_callback(u, ctx)
            results.append(ans[0])
        await asyncio.gather(*[click(i) for i in range(10, 18)])
        assert len(bot.sb_edits()) == 1
        assert sorted(results).count("Scoreboard updated!") == 1
        assert results.count("Scoreboard is already up to date.") == 7
        # switch view
        u, ans = fake_query(f"sb:{mid}:c")
        await sb.scoreboard_callback(u, ctx)
        assert "FULL SCORECARD" in bot.sb_edits()[-1][2]
    run(go())


# 12 -----------------------------------------------------------------------
def test_duplicate_and_concurrent_delivery_commit_once():
    async def go():
        g = make_game()
        mid = g.db_match_id
        live = {"innings": 1, "batting_side": "a", "bowling_side": "b", "stage": "ball_wait", "striker_id": 1,
                "non_striker_id": 2, "bowler_id": 11, "lineup": [1, 2, 3], "next_in_idx": 2}
        bat = {"runs": 4, "balls": 1, "fours": 1, "sixes": 0, "is_out": False, "dismissal": None}
        bowl = {"overs": 0.2, "runs_conceded": 4, "wickets": 0}
        args = (mid, 1, 1, 1, 1, 11, "4", None, bat, bowl, live)
        res = await asyncio.gather(*[asyncio.to_thread(db.commit_team_ball, *args) for _ in range(6)])
        assert res.count(True) == 1 and res.count(False) == 5
        assert len(snap(g)["events"]) == 1
    run(go())


# 15 -----------------------------------------------------------------------
def test_restart_recovery_from_database():
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot); g = make_game()
        await start_innings(ctx, g)
        await ball(ctx, g, 6, 1); await ball(ctx, g, 0, 2)
        db.save_team_scoreboard(g.db_match_id, CHAT, 999, True)
        live_text = render(g)
        tc.TEAM_MATCHES.clear()                       # "restart": in-memory state gone
        assert sb._is_running(g.db_match_id) is False
        assert await sb.render_and_edit(bot, g.db_match_id) == "updated"
        t = bot.sb_edits()[-1][2]
        assert "SCORE: <b>6/0</b>" in t and "NOT RUNNING" in t
        assert "6️⃣ │ 0️⃣" in t and "🟢 <b>Alice</b> *" in t
        assert "LIVE MATCH" in live_text
    run(go())


# 7 (telegram errors) --------------------------------------------------------
def test_telegram_edit_errors_do_not_crash():
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot); g = make_game()
        await start_innings(ctx, g)
        db.save_team_scoreboard(g.db_match_id, CHAT, 999, False)
        mid = g.db_match_id
        bot.fail_with = BadRequest("Message to edit not found")
        assert await sb.render_and_edit(bot, mid) == "gone"
        bot.fail_with = BadRequest("Message is not modified: ...")
        assert await sb.render_and_edit(bot, mid) == "same"
        bot.fail_with = RetryAfter(5)
        db.update_team_scoreboard(mid, "stale-hash")
        assert await sb.render_and_edit(bot, mid) == "busy"
        bot.fail_with = RuntimeError("boom")
        db.update_team_scoreboard(mid, "stale-hash")
        u, ans = fake_query(f"sb:{mid}:r"); u.callback_query.message.message_id = 999
        await sb.scoreboard_callback(u, ctx)       # must swallow
        assert ans and "boom" not in ans[0]
    run(go())


# 14 / auto-update coalescing + finish unpin -------------------------------
def test_auto_update_after_each_ball_is_coalesced():
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot); g = make_game()
        await start_innings(ctx, g)
        db.save_team_scoreboard(g.db_match_id, CHAT, 999, True)
        for _ in range(4):
            await ball(ctx, g, 1, 2)
        await asyncio.sleep(0.2)
        edits = bot.sb_edits()
        assert 1 <= len(edits) <= 5
        assert "SCORE: <b>4/0</b>" in edits[-1][2]      # last render has the latest ball
        assert not sb._workers                            # worker finished, no polling loop
    run(go())


def test_finished_match_unpins_scoreboard():
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot)
        g = make_game(A[:2], B[:2])
        await start_innings(ctx, g)
        db.save_team_scoreboard(g.db_match_id, CHAT, 999, True)
        await ball(ctx, g, 6, 1); await ball(ctx, g, 3, 3)           # all out, 6
        tc._cancel_timer(g)
        g.lineup = [11, 12]; g.order_pick_pool = []
        await tc._confirm_batting_order(ctx, g)
        await tc._set_bowler_and_start_ball(ctx, g, 1); tc._cancel_timer(g)
        await ball(ctx, g, 2, 2)                                      # all out chasing -> Team A wins
        await asyncio.sleep(0.3)
        assert snap(g)["match"]["status"] == "completed"
        assert "MATCH COMPLETED" in bot.sb_edits()[-1][2]
        assert 999 in bot.unpins
        assert db.get_team_scoreboard(g.db_match_id)["pinned"] == 0
    run(go())


def test_html_escaping_of_names():
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot)
        g = make_game([(1, "<b>Al&ice"), (2, "Bob"), (3, "Cara")], B)
        await start_innings(ctx, g)
        t = render(g)
        assert "<b>Al&ice" not in t and "&lt;b&gt;Al&amp;ice" in t
        assert len(t) < 4096
    run(go())


def test_twelve_player_team_fits_telegram_limit():
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot)
        a = [(i, f"Player{i:02d}LongName") for i in range(1, 13)]
        b = [(100 + i, f"Opp{i:02d}LongName") for i in range(1, 13)]
        g = make_game(a, b)
        await start_innings(ctx, g)
        for _ in range(8):
            await ball(ctx, g, 4, 4)           # a wicket every ball
            if g.stage == "bowler_pick":
                await tc._set_bowler_and_start_ball(ctx, g, 101 + g.overs_completed); tc._cancel_timer(g)
        t = render(g)
        assert len(t) < 4096 and "Player09" in t
        assert len(render(g, "c")) < 4096
    run(go())


def test_scoreboard_command_posts_pins_and_persists(no_auto):
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot); g = make_game()
        await start_innings(ctx, g)
        await ball(ctx, g, 4, 1)
        replies = []

        async def reply_text(text, parse_mode=None, reply_markup=None):
            replies.append((text, reply_markup))
            return types.SimpleNamespace(message_id=999)

        msg = types.SimpleNamespace(reply_text=reply_text, reply_to_message=None)
        upd = types.SimpleNamespace(
            effective_chat=types.SimpleNamespace(id=CHAT, type="supergroup"),
            effective_user=types.SimpleNamespace(id=1), message=msg,
        )
        await sb.scoreboard_command(upd, ctx)
        assert "SCORE: <b>4/0</b>" in replies[0][0] and replies[0][1] is not None
        assert bot.pins == [999]
        row = db.get_team_scoreboard(g.db_match_id)
        assert row["message_id"] == 999 and row["pinned"] == 1
        # second call replaces the old scoreboard
        await sb.scoreboard_command(upd, ctx)
        assert bot.deleted == [999]

        # no running match -> friendly message, nothing created
        tc.TEAM_MATCHES.clear(); replies.clear()
        await sb.scoreboard_command(upd, ctx)
        assert "No live Team Cricket match" in replies[0][0]
    run(go())


def test_custom_overs_end_innings_and_show_on_scoreboard(no_auto):
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot)
        g = make_game(); g.overs = 1
        await start_innings(ctx, g)
        for _ in range(6):
            await ball(ctx, g, 1, 2)       # 1 over = 6 balls, all singles
        assert g.innings == 2 and g.target == 7      # innings ended after ONE over
        assert snap(g)["live"]["overs"] == 1
        assert "Overs: 1.0 / 1" in render(g) or "Inn 1:" in render(g)
    run(go())


def test_overs_button_rules_and_command_arg():
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot)
        g = make_game(); g.stage = "lobby"; g.message_id = 500
        async def press(uid, n):
            ans = []
            async def answer(t=None, show_alert=False): ans.append(t)
            q = types.SimpleNamespace(data=f"tc:500:overs:{n}", answer=answer, from_user=types.SimpleNamespace(id=uid),
                                      message=types.SimpleNamespace(message_id=500))
            await tc.team_overs_callback(types.SimpleNamespace(callback_query=q), ctx)
            return ans[0]
        assert "Only the creator" in await press(99, 5) and g.overs == 3      # outsider rejected
        assert "Overs set to 5" in await press(1, 5) and g.overs == 5          # captain ok
        assert "Invalid" in await press(1, 99) and g.overs == 5                # out of range
        assert "Overs per innings: <b>5</b>" in bot.edits[-1][2]
        g.stage = "toss_call"
        assert "before the match starts" in await press(1, 2) and g.overs == 5  # locked after start

        # /teamcricket 7
        sent = types.SimpleNamespace(message_id=700)
        async def reply_text(text, **kw): return sent
        upd = types.SimpleNamespace(
            effective_chat=types.SimpleNamespace(id=CHAT, type="supergroup"),
            effective_user=types.SimpleNamespace(id=5, first_name="Zed", username=None),
            message=types.SimpleNamespace(reply_text=reply_text))
        await tc.teamcricket_start(upd, types.SimpleNamespace(bot=bot, args=["7"]))
        assert tc.TEAM_MATCHES[700].overs == 7
        await tc.teamcricket_start(upd, types.SimpleNamespace(bot=bot, args=["500"]))
        assert tc.TEAM_MATCHES[700].overs == 3   # invalid value -> default, with a warning reply
    run(go())


def test_overs_column_migrates_on_existing_db(tmp_path, monkeypatch):
    import sqlite3
    path = str(tmp_path / "old.db")
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE team_live_state (match_id INTEGER PRIMARY KEY, innings INTEGER NOT NULL, batting_side TEXT,"
              " bowling_side TEXT, stage TEXT, striker_id INTEGER, non_striker_id INTEGER, bowler_id INTEGER, lineup TEXT,"
              " next_in_idx INTEGER DEFAULT 0, target INTEGER, overs_completed INTEGER DEFAULT 0,"
              " balls_this_over INTEGER DEFAULT 0, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    c.commit(); c.close()
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db(); db.init_db()
    cols = [r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(team_live_state)")]
    assert "overs" in cols


def test_popup_button_under_number_pad_and_alert(no_auto):
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot)
        a = [(i, f"VeryLongPlayerName{i:02d}") for i in range(1, 13)]
        b = [(100 + i, f"AnotherLongOpponent{i:02d}") for i in range(1, 13)]
        g = make_game(a, b); g.overs = 20
        await start_innings(ctx, g)
        for _ in range(5):
            await ball(ctx, g, 6, 1)
        await ball(ctx, g, 3, 3)                      # wicket
        kb = tc._pick_keyboard(g).inline_keyboard
        assert kb[-1][0].text == "📊 Scoreboard" and kb[-1][0].callback_data == f"sb:{g.db_match_id}:p"
        assert len(kb[0]) + len(kb[1]) == 7           # number buttons intact

        alerts = []
        async def answer(text=None, show_alert=False): alerts.append((text, show_alert))
        def q(chat=CHAT, uid=7):
            return types.SimpleNamespace(data=f"sb:{g.db_match_id}:p", answer=answer, from_user=types.SimpleNamespace(id=uid),
                                         message=types.SimpleNamespace(chat=types.SimpleNamespace(id=chat), message_id=500))
        before = snap(g)
        await sb.scoreboard_callback(types.SimpleNamespace(callback_query=q()), ctx)
        text, alert = alerts[-1]
        assert alert is True and sb._u16(text) <= 200
        assert "30/1" in text and "❌" in text and "Last over:" in text
        assert snap(g)["events"] == before["events"] and snap(g)["live"] == before["live"]   # read-only
        assert bot.edits == [e for e in bot.edits if e[1] == 500]                              # no scoreboard edits needed

        await sb.scoreboard_callback(types.SimpleNamespace(callback_query=q(chat=-5)), ctx)
        assert "isn't available" in alerts[-1][0]
        # always within the limit, even mid-chase with target
        s = snap(g)
        s["live"]["target"] = 99; s["live"]["innings"] = 1
        assert sb._u16(sb.build_popup(s, True)) <= 200
    run(go())


import history as hist  # noqa: E402


def _history_update(args, chat=CHAT, uid=7):
    replies = []
    async def reply_text(text, **kw): replies.append(text)
    upd = types.SimpleNamespace(effective_chat=types.SimpleNamespace(id=chat, type="supergroup"),
                                effective_user=types.SimpleNamespace(id=uid), message=types.SimpleNamespace(reply_text=reply_text))
    return upd, types.SimpleNamespace(args=args, bot=None), replies


def test_history_by_key_shows_ball_by_ball_and_scorecards(no_auto):
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot)
        g = make_game(A[:2], B[:2])
        await start_innings(ctx, g)
        await ball(ctx, g, 6, 1); await ball(ctx, g, 4, 2); await ball(ctx, g, 3, 3)   # Alice 10, then out (Alice)
        tc._cancel_timer(g)
        g.lineup = [11, 12]; g.order_pick_pool = []
        await tc._confirm_batting_order(ctx, g)
        await tc._set_bowler_and_start_ball(ctx, g, 1); tc._cancel_timer(g)
        await ball(ctx, g, 6, 0); await ball(ctx, g, 6, 0)                              # Dan 12 -> wins
        key = g.db_match_id
        assert any(f"/history {key}" in t for t in bot.sent)       # key printed on result card

        upd, c, replies = _history_update([str(key)])
        await hist.history_command(upd, c)
        t = "\n".join(replies)
        assert f"<code>{key}</code>" in t and "MATCH HISTORY" in t
        assert "INNINGS 1" in t and "INNINGS 2" in t and "BALL-BY-BALL" in t
        assert "Alice — <b>10</b> (3)" in t and "❌ Bowled" in t or "❌" in t
        assert "0.1  Alice — 6️⃣ 6 run(s) 💥 SIX!" in t
        assert "0.3  Alice — ❌ <b>OUT</b>" in t and "b Dan" in t or "b Dan" in t
        assert "Over 1</b> • 🎯 Dan • 10 run(s), 1 wkt • total 10" in t
        assert "won by" in t

        # '#key' style, wrong group, unknown key, junk
        upd, c, r = _history_update([f"#{key}"]); await hist.history_command(upd, c); assert "MATCH HISTORY" in r[0]
        upd, c, r = _history_update([str(key)], chat=-777); await hist.history_command(upd, c); assert "No match with that key" in r[0]
        upd, c, r = _history_update(["999999"]); await hist.history_command(upd, c); assert "No match with that key" in r[0]
        upd, c, r = _history_update(["abc"]); await hist.history_command(upd, c); assert "Invalid key" in r[0]
        upd, c, r = _history_update([]); await hist.history_command(upd, c); assert "Usage" in r[0] and str(key) in r[0]
    run(go())


def test_history_long_match_is_paginated_under_limit(no_auto):
    async def go():
        bot = FakeBot(); ctx = ctx_of(bot)
        a = [(i, f"Player{i:02d}") for i in range(1, 13)]
        b = [(100 + i, f"Opp{i:02d}") for i in range(1, 13)]
        g = make_game(a, b); g.overs = 20
        await start_innings(ctx, g)
        for _ in range(120):
            await ball(ctx, g, 2, 1)
            if g.stage == "bowler_pick":
                await tc._set_bowler_and_start_ball(ctx, g, 101 + (g.overs_completed % 11)); tc._cancel_timer(g)
        pages = hist.build_history(snap(g))
        assert len(pages) >= 2 and all(len(p) <= 4096 for p in pages)
        assert "Over 19" in "\n".join(pages)
    run(go())
