"""Unit tests for agent 2.9.2 (round 11): concurrent vote candidates (AGENT_VOTE_PARALLEL), second-check adjudication
with a fresh fix conversation (AGENT_VERIFY_FIX2), whitespace-insensitive edit blocks, the exec allowance halving after
a timeout, NA reasons in check scripts and thread-safe House accounting."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agent"))
sys.path.insert(0, str(HERE))

import agent  # noqa: E402
from test_v22_features import FakeHouse, _solver  # noqa: E402

F = "```"


def _py(code: str):
    return (F + "python\n" + code + "\n" + F, "stop")


def _script(v):
    return _py("import json, os\njson.dump({'v': " + str(v) + "}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))")


def _value(s):
    return json.loads((s.out_dir / "f0.json").read_text())["v"]


def _primary(s, v):
    val, _, _ = s.run_attempt(agent.extract_code(_script(v)[0]), "a1")
    assert val.ok


class LabelHouse(FakeHouse):
    """Replies keyed by request label (thread-safe), so concurrent candidates get deterministic answers."""

    def __init__(self, by_label, cfg, delay=0.0):
        super().__init__([], cfg)
        self.by_label, self.delay, self.lock = dict(by_label), delay, threading.Lock()
        self.rate_limited = False
        self.threads = set()

    def chat(self, messages, *, thinking=None, max_tokens=None, label="call", temperature=None):
        with self.lock:
            self.calls.append((label, [m["role"] for m in messages], messages[-1]["content"], thinking))
            self.requests_made += 1
            self.threads.add(threading.get_ident())
            if label not in self.by_label:
                raise agent.ModelError(f"no scripted reply for {label}")
            reply = self.by_label[label]
        if self.delay:
            time.sleep(self.delay)
        return reply


def _vote_solver(tmp_path, by_label, delay=0.0, **flags):
    s = _solver(tmp_path, n_deliv=1, vote=True, **flags)
    s.house = LabelHouse(by_label, s.cfg, delay)
    s.task_context = "task"
    return s


# ---- concurrent vote ------------------------------------------------------------------------------------------------
def test_vote_parallel_requests_two_candidates_at_once(tmp_path, capsys):
    s = _vote_solver(tmp_path, {"vote1": _script(1), "vote2": _script(3)}, delay=0.6, vote_parallel=2, vote_max=2)
    _primary(s, 1)
    t0 = time.time()
    s.vote(True)
    elapsed = time.time() - t0
    labels = sorted(c[0] for c in s.house.calls)
    assert labels == ["vote1", "vote2"] and len(s.house.threads) == 2, (labels, s.house.threads)
    assert elapsed < 1.1, elapsed                       # two 0.6 s calls overlapped (serial would take >= 1.2 s)
    assert _value(s) == 1 and s.vote_agreed and s.house.requests_made == 2
    assert "requested concurrently" in capsys.readouterr().out


def test_vote_parallel_falls_back_to_one_at_a_time_when_rate_limited(tmp_path):
    s = _vote_solver(tmp_path, {"vote1": _script(1), "vote2": _script(3), "vote3": _script(4)}, vote_parallel=2, vote_max=3)
    s.house.rate_limited = True
    _primary(s, 1)
    s.vote(True)
    assert [c[0] for c in s.house.calls] == ["vote1"] and _value(s) == 1 and s.vote_agreed


def test_vote_parallel_replaces_an_outvoted_answer_and_flags_agreement(tmp_path):
    s = _vote_solver(tmp_path, {"vote1": _script(2), "vote2": _script(2)}, vote_parallel=2, vote_max=2)
    _primary(s, 1)
    s.vote(True)
    assert _value(s) == 2 and s.vote_agreed and s.best_ok and sorted(c[0] for c in s.house.calls) == ["vote1", "vote2"]


def test_vote_without_agreement_leaves_the_flag_down(tmp_path):
    s = _vote_solver(tmp_path, {"vote1": _script(2), "vote2": _script(3)}, vote_max=2)
    _primary(s, 1)
    s.vote(True)
    assert _value(s) == 1 and not s.vote_agreed and [c[0] for c in s.house.calls] == ["vote1", "vote2"]


def test_vote_parallel_batches_shrink_when_requests_run_low(tmp_path):
    s = _vote_solver(tmp_path, {"vote1": _script(2), "vote2": _script(3), "vote3": _script(4)}, vote_parallel=2, vote_max=3)
    s.house.requests_made = 16   # 4 left: a pair needs 5, so candidates come one at a time
    _primary(s, 1)
    s.vote(True)
    assert len(s.house.threads) == 1 and [c[0] for c in s.house.calls][:1] == ["vote1"]


def test_vote_candidate_thread_failure_is_contained(tmp_path):
    s = _vote_solver(tmp_path, {"vote1": _script(1)}, vote_parallel=2, vote_max=2)  # vote2 has no reply -> ModelError
    _primary(s, 1)
    s.vote(True)
    assert _value(s) == 1 and s.vote_agreed


# ---- House accounting under threads ----------------------------------------------------------------------------------
def test_house_admission_is_atomic_under_threads(monkeypatch):
    import requests

    class Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}

    def fake_post(url, json=None, headers=None, timeout=None):
        time.sleep(0.05)
        return Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    cfg = agent.Config(endpoint="http://127.0.0.1:1", max_requests=5)
    house = agent.House(cfg, agent.Budget(600, 700))
    results = []
    lock = threading.Lock()

    def call():
        try:
            house.chat([{"role": "user", "content": "x"}], label="t")
            r = "ok"
        except agent.BudgetExhausted:
            r = "exhausted"
        with lock:
            results.append(r)

    threads = [threading.Thread(target=call) for _ in range(9)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert results.count("ok") == 5 and results.count("exhausted") == 4 and house.requests_made == 5
    assert len(house.call_log) == 5 and house.requests_left() == 0


def test_config_parses_the_new_flags(monkeypatch):
    monkeypatch.setenv("AGENT_VOTE_PARALLEL", "7")
    monkeypatch.setenv("AGENT_VERIFY_FIX2", "1")
    monkeypatch.setenv("AGENT_VOTE_MAX", "5")
    cfg = agent.Config.from_env()
    assert cfg.vote_parallel == 3 and cfg.verify_fix2 and cfg.vote_max == 5
    monkeypatch.delenv("AGENT_VOTE_PARALLEL")
    monkeypatch.delenv("AGENT_VERIFY_FIX2")
    cfg = agent.Config.from_env()
    assert cfg.vote_parallel == 1 and not cfg.verify_fix2


# ---- edit blocks -------------------------------------------------------------------------------------------------------
def test_apply_edit_blocks_matches_despite_whitespace_differences():
    code = "x = 1\nif x:\n    y  =  f( x , 2 )\n    z = 3\nprint(y)"
    reply = "<<<<<<< SEARCH\nif x:\n  y = f(x, 2)\n=======\nif x:\n    y = g(x)\n>>>>>>> REPLACE"
    out = agent.apply_edit_blocks(code, reply)
    assert out == "x = 1\nif x:\n    y = g(x)\n    z = 3\nprint(y)"
    # exact matches still win, an ambiguous fuzzy match is refused, an absent one too
    assert agent.apply_edit_blocks("a = 1\nb = 2", "<<<<<<< SEARCH\nb = 2\n=======\nb = 3\n>>>>>>> REPLACE") == "a = 1\nb = 3"
    assert agent.apply_edit_blocks("q=1\nw=2\nq=1\n", "<<<<<<< SEARCH\nq = 1\n=======\nq = 2\n>>>>>>> REPLACE") is None
    assert agent.apply_edit_blocks("q=1\n", "<<<<<<< SEARCH\nr = 1\n=======\nr = 2\n>>>>>>> REPLACE") is None


def test_fuzzy_span_ignores_blank_lines_in_the_search():
    lines = ["def f():", "    a = 1", "", "    return a"]
    assert agent._fuzzy_span(lines, "a = 1\n\n\nreturn a\n") == (1, 3)
    assert agent._fuzzy_span(lines, "") is None


# ---- exec allowance after a timeout -------------------------------------------------------------------------------------
def test_exec_allowance_halves_after_each_timeout(tmp_path, monkeypatch):
    s = _solver(tmp_path, n_deliv=1)
    s.house = FakeHouse([], s.cfg)
    base = s._exec_timeout()
    assert base == 270.0                      # max(150, 0.45 x 600) under the test budget
    s.exec_timeouts = 1
    assert s._exec_timeout() == 135.0
    s.exec_timeouts = 3
    assert s._exec_timeout() == 60.0          # floor
    s.exec_timeouts = 0
    monkeypatch.setattr(agent, "run_script", lambda *a, **k: agent.RunResult(-9, "", "", True, 1.0))
    val, res, fb = s.run_attempt("print(1)", "a1")
    assert s.exec_timeouts == 1 and not val.ok and "next run is limited to 135 seconds" in fb and "coarser grids" in fb


# ---- verify: NA reasons, failure logging --------------------------------------------------------------------------------
def _verify_solver(tmp_path, replies, v=2.5, **flags):
    s = _solver(tmp_path, n_deliv=1, verify=True, **flags)
    s.house = FakeHouse(list(replies), s.cfg)
    s.task_context = "task"
    s.plan_json = {"key_quantities": [{"file": "f0.json", "path": "v", "agg": "value", "meaning": "the value v"}]}
    _primary(s, v)
    return s


def _check(v, extra=""):
    return _py("print('CHECK q1 " + str(v) + extra + "')\nprint('CHECKS DONE')")


def test_verify_na_reasons_trigger_one_repair(tmp_path, capsys):
    s = _verify_solver(tmp_path, [_check("NA KeyError: close_px"), _check(2.5)])
    s.verify()
    assert [c[0] for c in s.house.calls] == ["verify", "verify-fix"]
    assert "could not compute" in s.house.calls[1][2] and "KeyError: close_px" in s.house.calls[1][2]
    out = capsys.readouterr().out
    assert "printed NA for most quantities" in out and "1 agree, 0 disagree" in out and _value(s) == 2.5
    assert "NA <short reason" in s.house.calls[0][2]


def test_verify_keeps_the_first_values_when_the_repair_is_worse(tmp_path):
    s = _verify_solver(tmp_path, [_check("NA x"), _py("raise RuntimeError('boom')")])
    s.verify()
    assert len(s.house.calls) == 2 and _value(s) == 2.5


def test_verify_logs_the_reason_of_an_unusable_check(tmp_path, capsys):
    s = _verify_solver(tmp_path, [_py("raise RuntimeError('nope')"), _py("raise ValueError('again')")])
    s.verify()
    out = capsys.readouterr().out
    assert "check script failed; one repair [RuntimeError: nope]" in out and "unusable; skipped [ValueError: again]" in out


# ---- verify FIX2 ----------------------------------------------------------------------------------------------------------
EDIT_5 = ("<<<<<<< SEARCH\njson.dump({'v': 2.5}\n=======\njson.dump({'v': 5.0}\n>>>>>>> REPLACE", "stop")
EDIT_26 = ("<<<<<<< SEARCH\njson.dump({'v': 2.5}\n=======\njson.dump({'v': 2.6}\n>>>>>>> REPLACE", "stop")


def test_verify_fix2_confirmed_disagreement_is_fixed_in_a_fresh_conversation(tmp_path, capsys):
    s = _verify_solver(tmp_path, [_check(5.0), _check(5.0), EDIT_5], verify_fix2=True)
    s.verify()
    labels = [c[0] for c in s.house.calls]
    assert labels == ["verify", "verify2", "verify-edit1"], labels
    assert "second, separate re-derivation" in s.house.calls[1][2]
    fix = s.house.calls[2]
    assert fix[1] == ["system", "user"] and "script=2.5 independent #1=5 independent #2=5" in fix[2] and "json.dump" in fix[2]
    assert _value(s) == 5.0 and "5.0" in s.last_code
    assert "1 disagreement(s) confirmed" in capsys.readouterr().out


def test_verify_fix2_second_check_disagreeing_keeps_outputs(tmp_path, capsys):
    s = _verify_solver(tmp_path, [_check(5.0), _check(7.0)], verify_fix2=True)
    s.verify()
    assert [c[0] for c in s.house.calls] == ["verify", "verify2"] and _value(s) == 2.5
    assert "0 disagreement(s) confirmed, 0 resolved in our favour, 1 unresolved" in capsys.readouterr().out


def test_verify_fix2_second_check_confirming_ours_keeps_outputs(tmp_path, capsys):
    s = _verify_solver(tmp_path, [_check(5.0), _check(2.5)], verify_fix2=True)
    s.verify()
    assert len(s.house.calls) == 2 and _value(s) == 2.5
    assert "1 resolved in our favour" in capsys.readouterr().out


def test_verify_fix2_restores_outputs_when_the_fix_misses(tmp_path):
    s = _verify_solver(tmp_path, [_check(5.0), _check(5.0), EDIT_26], verify_fix2=True)
    prev = s.last_code
    s.verify()
    assert len(s.house.calls) == 3 and _value(s) == 2.5 and s.last_code == prev


def test_verify_fix2_unusable_second_check_keeps_outputs(tmp_path):
    s = _verify_solver(tmp_path, [_check(5.0), _py("raise RuntimeError('x')")], verify_fix2=True)
    s.verify()
    assert len(s.house.calls) == 2 and _value(s) == 2.5


def test_verify_skipped_after_a_vote_agreement_in_fix2_mode(tmp_path, capsys):
    s = _verify_solver(tmp_path, [_check(5.0)], verify_fix2=True)
    s.vote_agreed = True
    s.verify()
    assert not s.house.calls and "skipped (an independent solution agreed" in capsys.readouterr().out
    s.cfg = agent.Config(endpoint="http://127.0.0.1:1", memory=False, selfcheck=False, do_plan=False, verify=True)
    s.house.replies.append(("VERDICT: KEEP", "stop"))
    s.verify()   # without FIX2 the flag is ignored: the usual check + judge run
    assert [c[0] for c in s.house.calls] == ["verify", "verify-judge1"]


def test_verify_fix2_agreeing_check_costs_one_request(tmp_path):
    s = _verify_solver(tmp_path, [_check(2.5)], verify_fix2=True)
    s.verify()
    assert [c[0] for c in s.house.calls] == ["verify"] and _value(s) == 2.5
