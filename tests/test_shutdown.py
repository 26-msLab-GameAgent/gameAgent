"""Exercise real SIGINT, HTTP-wait cancellation and playable video finalization."""
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from PIL import Image

from gameagent.models import Action, ActionType, Decision, ExecutionResult, Observation
from gameagent.storage.episode import EpisodeLogger


CHILD = r'''
import io, os, signal, sys, time
from pathlib import Path
from PIL import Image
from gameagent.agent import ActionValidator
from gameagent.models import Action, ActionType, Decision, ExecutionResult, Observation
from gameagent.runtime.runner import AgentRunner, RunnerOptions
from gameagent.storage.episode import EpisodeLogger

root=Path(sys.argv[1]); mode=sys.argv[2]
buffer=io.BytesIO(); Image.new('RGB',(64,64),'blue').save(buffer,format='PNG')
class Capture:
    def capture(self, frame_id, previous):
        if mode=='capture' and frame_id==2:
            print('READY',flush=True); time.sleep(120)
        return Observation(frame_id,time.time(),64,64,image_bytes=buffer.getvalue())
class Model:
    model_name='shutdown-test'
    def decide(self, obs):
        if mode=='idle-timeout':
            time.sleep(.02)
            return Decision(Action(ActionType.WAIT,duration_ms=0))
        if mode=='action-budget' and obs.frame_id % 3:
            return Decision(Action(ActionType.WAIT,duration_ms=0))
        if mode.startswith('automatic') and obs.frame_id==2:
            if mode=='automatic-recording':
                deadline=time.monotonic()+5
                while not (root/'recorder-ready').exists():
                    assert time.monotonic()<deadline
                    time.sleep(.01)
            return Decision.from_dict({'action': {'type': 'noop'},
                'pipeline': {'screen_routing': {'stop_requested': True,
                    'stop_reason': 'no verified progress within step budget'}}})
        if mode in ('decision','recording') and obs.frame_id==2:
            print('READY',flush=True)
            try:
                time.sleep(120)
            except Exception:
                raise AssertionError('stop was swallowed by a normal retry handler')
        return Decision(Action(ActionType.TAP,x=20,y=20))
class Control:
    def execute(self, action, obs):
        with (root/'control_calls.txt').open('a') as f: f.write(action.type.value+'\n')
        if mode=='action':
            print('READY',flush=True); time.sleep(120)
        return ExecutionResult(ok=True)
logger=EpisodeLogger(root/'run',timestamped_run_dir=False,
    video_recording=mode in ('recording','automatic-recording'),adb_path=str(root/'fake-adb'))
if mode=='logging':
    original=logger.log_step
    def log(*args,**kwargs):
        original(*args,**kwargs)
        os.kill(os.getpid(),signal.SIGINT)
    logger.log_step=log
if mode=='decision':
    original_encode=logger._encode_video_with_pyav
    def encode():
        # A second Ctrl+C while finalizing must not destroy the video.
        os.kill(os.getpid(),signal.SIGINT)
        original_encode()
    logger._encode_video_with_pyav=encode
runner=AgentRunner(Capture(),Model(),Control(),ActionValidator(),logger,
    RunnerOptions(tick_interval_ms=0,settle_after_action_ms=0,
        max_episode_steps=2 if mode=='action-budget' else 10,
        step_budget_mode='actions', max_idle_seconds=.01 if mode=='idle-timeout' else 120))
if mode=='settling':
    def settle(*args):
        print('READY',flush=True); time.sleep(120)
    runner._wait_before_next_observation=settle
raise SystemExit(runner.run())
'''


def run_child(tmp_path, mode):
    script = tmp_path/'runner.py'
    script.write_text(CHILD)
    output = tmp_path/'stdout.txt'
    with output.open('w') as log:
        proc = subprocess.Popen([sys.executable, '-u', str(script), str(tmp_path), mode],
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if mode not in ('logging', 'action-budget', 'idle-timeout') and not mode.startswith('automatic'):
                deadline = time.monotonic() + 15
                while 'READY' not in output.read_text():
                    assert proc.poll() is None, output.read_text()
                    assert time.monotonic() < deadline, output.read_text()
                    time.sleep(.02)
                if mode == 'recording':
                    while not (tmp_path/'recorder-ready').exists():
                        assert time.monotonic() < deadline
                        time.sleep(.02)
                # Like terminal Ctrl+C: signal the entire foreground process group.
                os.killpg(proc.pid, signal.SIGINT)
            expected_exit = (0 if mode=='action-budget' else 2
                             if mode.startswith('automatic') or mode=='idle-timeout' else 130)
            assert proc.wait(timeout=20) == expected_exit, output.read_text()
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait()
    return output.read_text()


def load_steps(tmp_path):
    return [json.loads(line) for line in (tmp_path/'run/steps.jsonl').read_text().splitlines()]


def assert_video_and_summary(tmp_path, finish_reason='interrupted'):
    import av
    with av.open(str(tmp_path/'run/demonstration.mp4')) as video:
        assert sum(1 for _ in video.decode(video=0)) > 0
    summary = json.loads((tmp_path/'run/run_summary.json').read_text())
    assert summary['finish_reason'] == finish_reason
    assert summary['video_created']
    document = json.loads((tmp_path/'run/demonstration.json').read_text())
    assert document['session']['video']['status'] == 'finished'
    return document


def test_ctrl_c_interrupts_model_wait_and_preserves_completed_step(tmp_path):
    run_child(tmp_path, 'decision')
    steps = load_steps(tmp_path)
    assert len(steps) == 2
    assert steps[0]['execution']['ok']
    assert steps[1]['extra']['interrupted_phase'] == 'decision'
    assert steps[1]['decision']['action']['type'] == 'noop'
    assert steps[1]['execution']['metadata']['execution_status'] == 'not_executed'
    assert (tmp_path/'control_calls.txt').read_text() == 'tap\n'
    trace = (tmp_path/'run/rule_action_trace.jsonl').read_text().splitlines()
    assert len(trace) == 2 and all(json.loads(line) for line in trace)
    document = assert_video_and_summary(tmp_path)
    assert len(document['actions']) == 1
    assert len(list((tmp_path/'run/frames').glob('*.png'))) == 2


def test_ctrl_c_during_action_records_uncertain_execution(tmp_path):
    run_child(tmp_path, 'action')
    steps = load_steps(tmp_path)
    assert len(steps) == 1
    assert steps[0]['extra']['interrupted_phase'] == 'action'
    assert steps[0]['extra']['execution_uncertain']
    assert not steps[0]['execution']['ok']
    assert assert_video_and_summary(tmp_path)['actions'] == []


def test_ctrl_c_during_logging_finishes_record_without_duplicates(tmp_path):
    run_child(tmp_path, 'logging')
    assert len(load_steps(tmp_path)) == 1
    assert len(assert_video_and_summary(tmp_path)['actions']) == 1


@pytest.mark.parametrize('phase', ['capture', 'settling'])
def test_ctrl_c_preserves_last_completed_step_between_decisions(tmp_path, phase):
    run_child(tmp_path, phase)
    assert len(load_steps(tmp_path)) == 1
    assert len(assert_video_and_summary(tmp_path)['actions']) == 1


@pytest.mark.parametrize('mode', ['recording', 'automatic-recording'])
def test_continuous_recorder_stops_and_creates_playable_mp4(tmp_path, mode):
    import av
    raw = tmp_path/'fixture.h264'
    with av.open(str(raw), 'w', format='h264') as output:
        stream = output.add_stream('libx264', rate=30)
        stream.width = stream.height = 64
        stream.pix_fmt = 'yuv420p'
        for _ in range(12):
            frame = av.VideoFrame.from_image(Image.new('RGB', (64, 64), 'green'))
            for packet in stream.encode(frame): output.mux(packet)
        for packet in stream.encode(): output.mux(packet)
    fake_adb = tmp_path/'fake-adb'
    fake_adb.write_text(f'''#!{sys.executable}
import signal,sys,time
from pathlib import Path
root=Path(__file__).parent
def stop(*args):
    (root/'recorder-stopped').touch()
    raise SystemExit(0)
signal.signal(signal.SIGINT,stop)
sys.stdout.buffer.write((root/'fixture.h264').read_bytes()); sys.stdout.buffer.flush()
(root/'recorder-ready').touch()
while True: time.sleep(1)
''')
    fake_adb.chmod(0o755)
    output = run_child(tmp_path, mode)
    reason = 'unknown_recovery_exhausted' if mode.startswith('automatic') else 'interrupted'
    document = assert_video_and_summary(tmp_path, reason)
    assert 'returning to terminal' in output
    assert document['session']['video']['capture_method'] == 'adb_screenrecord'
    assert (tmp_path/'recorder-stopped').exists()
    assert not (tmp_path/'run/.screenrecord.h264').exists()


def test_automatic_stop_saves_playable_fallback_video_and_exits(tmp_path):
    output = run_child(tmp_path, 'automatic')
    assert 'saving video and exiting' in output
    assert 'returning to terminal' in output
    document = assert_video_and_summary(tmp_path, 'unknown_recovery_exhausted')
    assert len(document['actions']) == 1
    assert len(load_steps(tmp_path)) == 2
    assert (tmp_path/'control_calls.txt').read_text() == 'tap\nnoop\n'
    summary = json.loads((tmp_path/'run/run_summary.json').read_text())
    assert summary['stop_reason'] == 'no verified progress within step budget'


@pytest.mark.parametrize('mode,reason,actions,observations', [
    ('action-budget', 'completed', 2, 6), ('idle-timeout', 'idle_timeout', 0, 1)])
def test_action_budget_and_idle_timeout_save_video_and_return(tmp_path, mode, reason, actions, observations):
    output = run_child(tmp_path, mode)
    document = assert_video_and_summary(tmp_path, reason)
    assert 'returning to terminal' in output
    assert len(document['actions']) == actions
    summary = json.loads((tmp_path/'run/run_summary.json').read_text())
    assert summary['step_budget_mode'] == 'actions'
    assert summary['action_steps'] == actions
    assert summary['observation_steps'] == observations


def test_ffmpeg_timeout_preserves_frames_and_finishes(tmp_path, monkeypatch):
    logger = EpisodeLogger(tmp_path, timestamped_run_dir=False)
    buffer = io.BytesIO()
    Image.new('RGB', (64, 64)).save(buffer, format='PNG')
    logger.log_step(Observation(1, time.time(), 64, 64, image_bytes=buffer.getvalue()),
                    Decision(Action(ActionType.NOOP)), ExecutionResult(ok=True))
    def fail_pyav():
        raise RuntimeError('codec unavailable')
    def timeout_ffmpeg(command, **kwargs):
        assert kwargs['timeout'] == 120
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(logger, '_encode_video_with_pyav', fail_pyav)
    monkeypatch.setattr('gameagent.storage.episode.shutil.which', lambda name: '/fake/ffmpeg')
    monkeypatch.setattr('gameagent.storage.episode.subprocess.run', timeout_ffmpeg)
    logger.finalize()
    summary = json.loads((tmp_path/'run_summary.json').read_text())
    assert not summary['video_created']
    assert summary['video_status'] == 'video_error'
    assert 'TimeoutExpired' in summary['error']
    assert len(list((tmp_path/'frames').glob('*.png'))) == 1
    assert logger._finalized


def test_failed_conversion_keeps_raw_stream(tmp_path, monkeypatch):
    logger = EpisodeLogger(tmp_path, timestamped_run_dir=False)
    logger._recording_raw_path.write_bytes(b'recoverable stream')
    def fail(): raise RuntimeError('codec unavailable')
    monkeypatch.setattr(logger, '_encode_raw_h264', fail)
    assert not logger._convert_raw_recording()
    assert logger._recording_raw_path.read_bytes() == b'recoverable stream'
    logger.finalize()
    assert json.loads((tmp_path/'run_summary.json').read_text())['raw_video_retained']


def test_capture_error_still_finalizes_and_restores_signals(tmp_path):
    from gameagent.runtime.runner import AgentRunner, RunnerOptions
    from gameagent.agent import ActionValidator
    from types import SimpleNamespace
    previous = signal.getsignal(signal.SIGINT)
    logger = EpisodeLogger(tmp_path, timestamped_run_dir=False)
    def fail(*args): raise RuntimeError('ADB disconnected')
    runner = AgentRunner(SimpleNamespace(capture=fail), SimpleNamespace(model_name='test'),
                         None, ActionValidator(), logger, RunnerOptions())
    with pytest.raises(RuntimeError, match='ADB disconnected'):
        runner.run()
    assert signal.getsignal(signal.SIGINT) == previous
    summary = json.loads((tmp_path/'run_summary.json').read_text())
    assert summary['finish_reason'] == 'error'
    assert (tmp_path/'steps.jsonl').is_file()
    assert logger._finalized


@pytest.mark.parametrize('extra,ok', [({'dry_run': True}, True),
                                    ({'stale_observation': True}, True), ({}, False)])
def test_skipped_or_failed_actions_not_claimed_in_demonstration(tmp_path, extra, ok):
    logger = EpisodeLogger(tmp_path, timestamped_run_dir=False)
    buffer = io.BytesIO(); Image.new('RGB', (64, 64)).save(buffer, format='PNG')
    logger.log_step(Observation(1, time.time(), 64, 64, image_bytes=buffer.getvalue()),
                    Decision(Action(ActionType.TAP, x=20, y=20)), ExecutionResult(ok=ok), extra)
    logger.finalize()
    assert json.loads((tmp_path/'demonstration.json').read_text())['actions'] == []
