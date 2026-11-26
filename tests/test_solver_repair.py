import json

import pytest

from gameagent.solvers.registry import write_json
from gameagent.solvers.repair import RepairSession, diagnose


def candidate(folder, score=0., kind='perception', fallback=0):
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'manifest.json'
    write_json(path,{'status':'candidate','perception_validation':{'status':'passed' if kind=='action' else 'failed'},
                    'validation':{'changed':2,'completions':1,'vlm_fallback_executed':fallback}})
    write_json(folder/'perception_report.json',{'board_examples':0 if kind=='reference' else 1,
        'results':[{'status':'failed','errors':['label mismatch'],'bbox_iou':score,
                    'max_cell_offset':1-score,'label_agreement':score}]})
    return path


def test_diagnosis_selects_component_and_no_vlm_success_credit(tmp_path):
    assert diagnose(candidate(tmp_path/'ref',kind='reference'))['kind']=='reference'
    assert 'observe' in diagnose(candidate(tmp_path/'obs'))['focus']
    good=diagnose(candidate(tmp_path/'policy',kind='action'))
    fallback=diagnose(candidate(tmp_path/'fallback',kind='action',fallback=1))
    assert 'choose_action' in good['focus']
    assert good['score']>fallback['score']


def test_regression_keeps_best_and_stagnation_has_report(tmp_path):
    first=candidate(tmp_path/'first',.8)
    session=RepairSession(tmp_path/'session.json',10,1000,2,first)
    worse=candidate(tmp_path/'worse',.2)
    assert session.record(worse)==first
    equal=candidate(tmp_path/'equal',.8)
    with pytest.raises(RuntimeError,match='No improvement'):
        session.record(equal)
    session.finish('stagnation')
    report=json.loads(session.path.read_text())
    assert report['best']['manifest']==str(first)
    assert report['unresolved']['kind']=='perception'
    assert report['status']=='stopped'


def test_improvement_resets_stagnation_and_budget_counts_requests(tmp_path):
    first=candidate(tmp_path/'first',.3)
    session=RepairSession(tmp_path/'session.json',1,1000,2,first)
    session.record(candidate(tmp_path/'worse',.2))
    better=candidate(tmp_path/'better',.8)
    assert session.record(better)==better and session.stalled==0
    session.request()
    with pytest.raises(RuntimeError,match='call budget'):
        session.request()
    assert session.data['generation_reference_calls']==1
    session.started-=1001
    with pytest.raises(RuntimeError,match='time budget'):
        session.check()


@pytest.mark.parametrize('finish,expected', [('idle_timeout', 'candidate'),
    ('unknown_recovery_exhausted', 'candidate'), ('error', 'candidate'),
    ('interrupted', 'candidate'), ('completed', 'active')])
def test_final_outcome_overrides_midrun_promotion(tmp_path, finish, expected):
    from gameagent.solvers.cli import reconcile_trial
    manifest = tmp_path/'manifest.json'
    write_json(manifest, {'status': 'active', 'last_trial_result': 'passed_local_board',
                         'validation': {'executed': 6, 'changed': 5, 'completions': 1}})
    steps = tmp_path/'steps.jsonl'
    steps.write_text('')
    write_json(tmp_path/'run_summary.json', {'finish_reason': finish,
               'stop_reason': 'no actionable input for 120 seconds' if finish == 'idle_timeout' else ''})
    reconcile_trial(manifest, [steps])
    data = json.loads(manifest.read_text())
    assert data['status'] == expected
    assert data['validation']['completions'] == 1
    if finish != 'completed':
        assert data['last_trial_result'] == 'failed_' + finish
        assert data['last_trial_result'] in diagnose(manifest)['reasons']


def test_nonzero_runner_without_summary_cannot_remain_active(tmp_path):
    from gameagent.solvers.cli import reconcile_trial
    manifest = tmp_path/'manifest.json'
    write_json(manifest, {'status': 'active'})
    reconcile_trial(manifest, [], returncode=2)
    assert json.loads(manifest.read_text())['status'] == 'candidate'


@pytest.mark.parametrize('game,finish,fallback,changes,active',[
    ('game_2048','completed',False,2,True),
    ('game_2048','idle_timeout',False,2,False),
    ('game_2048','completed',True,2,False),
    ('game_2048','completed',False,0,False),
    ('candy_crush_soda','completed',False,2,False)])
def test_2048_final_progress_does_not_require_completion(tmp_path,game,finish,fallback,changes,active):
    import hashlib
    from gameagent.solvers.cli import reconcile_trial
    source='test source';(tmp_path/'solver.py').write_text(source)
    sha=hashlib.sha256(source.encode()).hexdigest()
    manifest=tmp_path/'manifest.json'
    write_json(manifest,{'game':game,'status':'candidate','sha256':sha,
                        'perception_validation':{'status':'passed','sha256':sha}})
    rows=[]
    for i in range(3):
        rows.append({'decision':{'action':{'type':'swipe'},'raw_response':{'pipeline':{'screen_routing':{
            'route':'board_single_call' if fallback and i==2 else 'generated_solver',
            'generated_solver':{'executed':2,'changed':changes,'completions':0,'runtime_model_calls':1}}}}},
            'execution':{'ok':True}})
    steps=tmp_path/'steps.jsonl';steps.write_text('\n'.join(json.dumps(r) for r in rows))
    write_json(tmp_path/'run_summary.json',{'finish_reason':finish,'action_steps':3})
    reconcile_trial(manifest,[steps])
    d=json.loads(manifest.read_text())
    assert (d['status']=='active') == active
    if active:
        assert d['last_trial_result']=='passed_local_progress'
        assert d['validation']['completions']==0 and d['validation']['executed']==3
        assert d['runtime_vlm_free'] is False
