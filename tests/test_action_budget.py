"""Action budgets exclude observations, but idle observation loops stay bounded."""
from types import SimpleNamespace

import pytest

from gameagent.agent import ActionValidator
from gameagent.models import Action, ActionType, Decision, ExecutionResult, Observation
from gameagent.runtime.runner import AgentRunner, RunnerOptions


def make_runner(monkeypatch, tmp_path, decisions, **options):
    rows, executed, frames, events = [], [], [], []
    clock = [0.0]
    iterator = iter(decisions)
    def capture(frame_id, previous):
        frames.append(frame_id)
        return Observation(frame_id, 0., 100, 100)
    def decide(obs):
        kind = next(iterator)  # Exhaustion means the runner exceeded its budget.
        clock[0] += 1
        return Decision(Action(ActionType(kind), x=30, y=30, duration_ms=0))
    def execute(action, obs):
        executed.append(action.type.value)
        return ExecutionResult(ok=True)
    logger = SimpleNamespace(path=tmp_path, rule_action_trace_path=tmp_path/'trace',
        start_video_recording=lambda: None, log_step=lambda **kw: rows.append(kw),
        finalize=lambda: events.append('finalized'))
    runner = AgentRunner(SimpleNamespace(capture=capture),
        SimpleNamespace(model_name='test', decide=decide), SimpleNamespace(execute=execute),
        ActionValidator(), logger, RunnerOptions(**{
            'step_budget_mode': 'actions', 'max_episode_steps': 50,
            'tick_interval_ms': 0, 'settle_after_action_ms': 0, **options}))
    monkeypatch.setattr(runner, '_install_signal_handlers', lambda: None)
    monkeypatch.setattr('gameagent.runtime.runner.time.monotonic', lambda: clock[0])
    return runner, rows, executed, frames, events


def test_fifty_actions_allow_one_hundred_additional_waits(monkeypatch, tmp_path):
    runner, rows, executed, frames, events = make_runner(monkeypatch, tmp_path,
                                                        ['wait', 'wait', 'tap'] * 50)
    assert runner.run() == 0
    assert frames == list(range(1, 151))
    assert executed.count('tap') == 50 and executed.count('wait') == 100
    assert rows[-1]['extra']['action_step'] == 50
    assert rows[-1]['extra']['observation_step'] == 150
    assert sum(r['extra']['counted_action'] for r in rows) == 50
    assert events == ['finalized']


def test_wait_and_noop_do_not_consume_action_budget(monkeypatch, tmp_path):
    runner, rows, executed, _, _ = make_runner(monkeypatch, tmp_path,
        ['wait', 'noop', 'tap'], max_episode_steps=1)
    assert runner.run() == 0
    assert [r['extra']['action_step'] for r in rows] == [0, 0, 1]
    assert executed == ['wait', 'noop', 'tap']


def test_observation_mode_preserves_legacy_budget(monkeypatch, tmp_path):
    runner, rows, _, _, _ = make_runner(monkeypatch, tmp_path, ['wait', 'wait', 'tap'],
        step_budget_mode='observations', max_episode_steps=2)
    assert runner.run() == 0
    assert len(rows) == 2


def test_idle_timeout_prevents_infinite_wait_and_late_action(monkeypatch, tmp_path):
    runner, rows, executed, _, events = make_runner(monkeypatch, tmp_path,
        ['wait', 'wait', 'tap'], max_idle_seconds=3)
    assert runner.run() == 2
    assert runner.logger.finish_reason == 'idle_timeout'
    assert runner.logger.action_steps == 0
    assert executed == ['wait', 'wait', 'noop']  # Late model response cannot click.
    assert len(rows) == 3
    assert events == ['finalized']


def test_dry_run_is_bounded_by_planned_actions(monkeypatch, tmp_path):
    runner, rows, executed, _, _ = make_runner(monkeypatch, tmp_path,
        ['wait', 'tap'], max_episode_steps=1, dry_run=True)
    assert runner.run() == 0
    assert executed == ['noop', 'noop']
    assert rows[-1]['extra']['action_step'] == 1


@pytest.mark.parametrize('passive', ['wait', 'noop'])
def test_passive_observation_skips_extra_settle_but_tap_keeps_it(monkeypatch, tmp_path, passive):
    runner, _, _, _, _ = make_runner(monkeypatch, tmp_path,
        [passive, 'tap', 'tap'], max_episode_steps=2, settle_after_action_ms=300)
    sleeps = []
    monkeypatch.setattr('gameagent.runtime.runner.time.sleep', sleeps.append)
    assert runner.run() == 0
    # No settle after WAIT/NOOP or after the final action; 300ms after the first tap.
    assert sleeps == [.3]


def test_dry_run_does_not_settle_for_unexecuted_inputs(monkeypatch, tmp_path):
    runner, _, _, _, _ = make_runner(monkeypatch, tmp_path,
        ['tap', 'tap'], max_episode_steps=2, settle_after_action_ms=300, dry_run=True)
    sleeps = []
    monkeypatch.setattr('gameagent.runtime.runner.time.sleep', sleeps.append)
    assert runner.run() == 0
    assert sleeps == []


def test_explicit_tick_interval_still_applies_without_settle(monkeypatch, tmp_path):
    runner, _, _, _, _ = make_runner(monkeypatch, tmp_path, [],
        tick_interval_ms=500, settle_after_action_ms=300)
    sleeps = []
    monkeypatch.setattr('gameagent.runtime.runner.time.sleep', sleeps.append)
    monkeypatch.setattr('gameagent.runtime.runner.time.perf_counter', lambda: 10.2)
    runner._wait_before_next_observation(10.0, after_input=False)
    assert sleeps == pytest.approx([.3])


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
def test_idle_timeout_must_be_finite_positive(monkeypatch, tmp_path, timeout):
    with pytest.raises(ValueError, match='max_idle_seconds'):
        make_runner(monkeypatch, tmp_path, [], max_idle_seconds=timeout)
