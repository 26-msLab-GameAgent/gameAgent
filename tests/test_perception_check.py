import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image

from gameagent.solvers import build, cli
from gameagent.solvers.perception_check import (
    compare_observation, reference_cases, validate_perception,
)
from gameagent.solvers.registry import LocalSolver, write_json
from gameagent.server.vlm_server import PreviousAction

SOURCE = '''
def observe(image, context):
    w, h = image.size
    if image.getpixel((0, 0)) == (255, 255, 255):
        return None
    return {"signature":"rgby", "confidence":0.95,
        "board_bbox":[int(w*.1),int(h*.1),int(w*.9),int(h*.9)],
        "grid":{"rows":2,"cols":2,
        "centers":[[[int(w*.3),int(h*.3)],[int(w*.7),int(h*.3)]],
                   [[int(w*.3),int(h*.7)],[int(w*.7),int(h*.7)]]],
        "labels":[["red","green"],["blue","yellow"]]}}
def choose_action(state, context):
    return None
'''


def artifact(folder, source=SOURCE, status='candidate', gate='pending'):
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'solver.py').write_text(source)
    digest = hashlib.sha256(source.encode()).hexdigest()
    path = folder/'manifest.json'
    write_json(path, {'game':'test', 'version':folder.name, 'status':status, 'sha256':digest,
                     'perception_validation':{'status':gate,'sha256':digest}})
    return path


def reference():
    return {'id':'image_0','kind':'grid','confidence':.99,'width':100,'height':100,
            'board_bbox':[10,10,90,90], 'rows':2,'cols':2,
            'centers':[[[30,30],[70,30]],[[30,70],[70,70]]],
            'labels':[['red','green'],['blue','yellow']]}


def observation():
    r = reference()
    return {'signature':'rgby','confidence':.95,'board_bbox':r['board_bbox'],
            'grid':{k:r[k] for k in ('rows','cols','centers','labels')}}


@pytest.mark.parametrize('problem', ['shape','bbox','centers','labels','unknown','missing'])
def test_high_confidence_cannot_override_perception_error(problem):
    obs=observation()
    obs['confidence']=.99
    if problem=='shape':
        obs['grid']={'rows':4,'cols':5,'labels':[['blue']*5 for _ in range(4)],
                     'centers':[[[10+c*15,10+r*20] for c in range(5)] for r in range(4)]}
    elif problem=='bbox': obs['board_bbox']=[0,0,99,99]
    elif problem=='centers': obs['grid']['centers'][0][0]=[15,15]
    elif problem=='labels': obs['grid']['labels'][0][0]='blue'
    elif problem=='unknown': obs['grid']['labels']=[['unknown']*2 for _ in range(2)]
    else: obs.pop('grid')
    assert compare_observation(obs,reference())['status']=='failed'


def test_false_positive_on_menu_fails():
    ref={'kind':'nonplay','confidence':.99}
    assert compare_observation(observation(),ref)['status']=='failed'
    assert compare_observation(None,ref)['status']=='passed'


def test_complete_check_uses_native_scaled_and_negative_frames(tmp_path):
    manifest=artifact(tmp_path)
    board=tmp_path/'board.png'; Image.new('RGB',(100,100),'black').save(board)
    menu=tmp_path/'menu.png'; Image.new('RGB',(100,100),'white').save(menu)
    refs=[{**reference(),'image_path':str(board)},
          {'id':'menu','kind':'nonplay','confidence':.99,'width':100,'height':100,'image_path':str(menu)}]
    report=validate_perception(manifest,refs)
    assert report['status']=='passed'
    assert len(report['results'])==3
    assert cli.perception_ready(manifest)


def test_no_board_does_not_pass(tmp_path):
    manifest=artifact(tmp_path)
    assert validate_perception(manifest,[])['status']=='failed'
    assert not cli.perception_ready(manifest)


def test_bad_reference_is_not_silently_trusted(tmp_path):
    image=tmp_path/'frame.png'; Image.new('RGB',(100,100)).save(image)
    cases=[{'id':'image_0','image_path':str(image)}]
    with pytest.raises(ValueError,match='every supplied'):
        reference_cases(cases,{'frames':[]})
    bad={**reference(),'board_bbox':[0,0,999,999]}
    with pytest.raises(ValueError,match='normalized'):
        reference_cases(cases,{'frames':[bad]})


def test_initial_generation_is_independently_checked(monkeypatch,tmp_path):
    image=tmp_path/'frame.png'; Image.new('RGB',(100,100),'black').save(image)
    (tmp_path/'configs').symlink_to(build.ROOT/'configs',target_is_directory=True)
    monkeypatch.setattr(build,'ROOT',tmp_path)
    calls=[]
    def api(parts,*args,**kwargs):
        calls.append(kwargs.get('system_prompt'))
        if kwargs.get('system_prompt'):
            assert SOURCE not in json.dumps(parts)
            return {'frames':[{'id':'image_0','kind':'grid','confidence':.99,
                               'board_bbox':[.1,.1,.9,.9],'rows':2,'cols':2,
                               'centers':[[[.3,.3],[.7,.3]],[[.3,.7],[.7,.7]]],
                               'labels':[['red','green'],['blue','yellow']]}]}
        return {'source':SOURCE,'objective':'test'}
    monkeypatch.setattr(build,'api_generate',api)
    manifest=build.generate('test','rules',[image],model='test',base_url='https://unused')
    assert len(calls)==2 and calls[0] is None and calls[1]
    assert cli.perception_ready(manifest)
    assert manifest.with_name('perception_report.json').exists()
    repaired = build.generate('test','rules',[image],model='test',base_url='https://unused',previous=manifest)
    assert len(calls) == 3 and calls[-1] is None  # Only code repair, no changed reference judge.
    assert cli.perception_ready(repaired)
    assert json.loads(repaired.with_name('reference.json').read_text())['reused_from'] == str(manifest)


def test_pending_solver_cannot_run_or_promote(tmp_path):
    solver=LocalSolver(artifact(tmp_path))
    assert solver.infer('unused.png',{}) is None
    assert not cli.perception_ready(solver.path)


def test_fresh_solver_bypasses_reuse_and_old_source(monkeypatch,tmp_path):
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(cli,'prepare_reference',lambda *a, **kw: tmp_path/'capture.png')
    monkeypatch.setattr(cli,'reusable_solver',lambda *a: pytest.fail('fresh must not look up existing solver'))
    seen=[]
    def generate(*a,**kw):
        seen.append(kw['previous'])
        return artifact(tmp_path/'new',gate='passed')
    monkeypatch.setattr(cli,'generate',generate)
    monkeypatch.setattr(cli,'trial',lambda *a:'active')
    cli.main(['bootstrap','--game','test','--video','input.mp4','--model','test',
              '--runtime-config','unused','--fresh-solver'])
    assert seen==[None]


def test_failed_perception_is_repaired_before_any_trial(monkeypatch,tmp_path):
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(cli,'prepare_reference',lambda *a, **kw: tmp_path/'capture.png')
    seen=[]; trials=[]
    def generate(*a,**kw):
        seen.append(kw['previous'])
        return artifact(tmp_path/str(len(seen)),gate='failed' if len(seen)==1 else 'passed')
    monkeypatch.setattr(cli,'generate',generate)
    monkeypatch.setattr(cli,'trial',lambda manifest,*a: trials.append(manifest) or 'active')
    cli.main(['bootstrap','--game','test','--video','input.mp4','--model','test','--runtime-config','unused'])
    assert seen==[None,tmp_path/'1/manifest.json']
    assert trials==[tmp_path/'2/manifest.json']


def test_fresh_and_previous_are_incompatible():
    with pytest.raises(SystemExit):
        cli.main(['generate','--game','test','--model','test','--fresh-solver','--previous','old.json'])


def test_executed_vlm_board_fallback_fails_local_trial(tmp_path):
    solver=LocalSolver(artifact(tmp_path,gate='passed'))
    req=SimpleNamespace(frame_id=1,previous_action=None,metadata={})
    trace={'state':'B','route':'board_single_call','classification_source':'semantic',
           'classification':{'confidence':1},'errors':[],'model_call_count':1}
    result={'action':{'type':'tap','x':30,'y':30},'pipeline':{'screen_routing':trace}}
    solver.record(req,result,'unused.png')
    req.frame_id=2
    req.previous_action=PreviousAction(type='tap',x=30,y=30)
    req.metadata={'previous_execution_ok':True}
    trace['route']='board_reobserve'
    result['action']={'type':'wait'}
    solver.record(req,result,'unused.png')
    assert solver.stats['vlm_fallback_executed']==1
    assert solver.manifest['last_trial_result']=='failed_vlm_fallback'
    # Even later completion and sufficient local actions must not erase that failure.
    solver.stats.update(executed=10,changed=10,completions=1)
    solver.record(req,result,'unused.png')
    assert solver.manifest['status']=='candidate'


@pytest.mark.parametrize('extra,expected', [({},'candidate'), ({'dry_run':True},'active'), ({'stale_observation':True},'active')])
def test_last_fallback_action_reconciled_without_next_frame(tmp_path,extra,expected):
    manifest=artifact(tmp_path,gate='passed')
    data=json.loads(manifest.read_text()); data['status']='active'
    manifest.write_text(json.dumps(data))
    steps=tmp_path/'steps.jsonl'
    steps.write_text(json.dumps({'decision':{'action':{'type':'swipe'},
        'raw_response':{'pipeline':{'screen_routing':{'route':'board_single_call'}}}},
        'execution':{'ok':True},'extra':extra})+'\n')
    cli.reconcile_trial(manifest,[steps])
    assert json.loads(manifest.read_text())['status']==expected


def test_nongrid_objects_validate_without_grid(tmp_path):
    image=tmp_path/'screen.png'; Image.new('RGB',(100,100)).save(image)
    frames=reference_cases([{'id':'screen','image_path':str(image)}], {'frames':[{
        'id':'screen','kind':'nongrid','confidence':.99,'board_bbox':[0,0,1,1],
        'objects':[{'label':'target','bbox':[.2,.2,.4,.4]}]}]})
    ref=frames[0]
    obs={'confidence':.99,'board_bbox':ref['board_bbox'],'objects':deepcopy(ref['objects'])}
    assert compare_observation(obs,ref)['status']=='passed'
    obs['objects'][0]['label']='player'
    assert compare_observation(obs,ref)['status']=='failed'
    obs['objects']=[]
    assert compare_observation(obs,ref)['status']=='failed'


def test_nongrid_native_and_scaled_gate(tmp_path):
    source='''
def observe(image, context):
    w,h=image.size
    return {"signature":"target", "confidence":.99,"board_bbox":[0,0,w-1,h-1],
            "objects":[{"label":"target","bbox":[int(w*.2),int(h*.2),int(w*.4),int(h*.4)]}]}
def choose_action(state,context):
    return None
'''
    manifest=artifact(tmp_path,source=source)
    image=tmp_path/'screen.png'; Image.new('RGB',(100,100)).save(image)
    ref={'id':'screen','image_path':str(image),'kind':'nongrid','confidence':.99,
         'width':100,'height':100,'board_bbox':[0,0,99,99],
         'objects':[{'label':'target','bbox':[20,20,40,40]}]}
    report=validate_perception(manifest,[ref])
    assert report['status']=='passed'
    assert len(report['results'])==2


@pytest.mark.parametrize('metadata,blocked', [({'previous_execution_ok':True},True),
    ({'previous_execution_ok':False},False),
    ({'previous_execution_ok':True,'dry_run':True},False),
    ({'previous_execution_ok':True,'skipped_action_frame_id':1},False)])
def test_repeat_guard_requires_executed_action_and_two_observations(tmp_path,metadata,blocked):
    solver=LocalSolver(artifact(tmp_path,gate='passed'))
    action={'type':'swipe','x':30,'y':30,'x2':70,'y2':30}
    solver.last={'observation':observation()}
    solver.action_watch={'signature':'rgby','action':action,'frame_id':1,'executed':False,'unchanged':0}
    req=SimpleNamespace(frame_id=2,previous_action=PreviousAction(**action),metadata=metadata)
    solver.check_action_progress(req)
    assert not solver.failed_actions
    req.frame_id=3
    solver.check_action_progress(req)
    assert bool(solver.guard_repeated_action(action))==blocked
    assert solver.guard_repeated_action({**action,'x':30,'y':70,'x2':70,'y2':70}) is None
    solver.last['observation']['signature']='new-state'
    assert solver.guard_repeated_action(action) is None


def test_missing_reference_stops_before_paid_regeneration(monkeypatch,tmp_path):
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(cli,'prepare_reference',lambda *a, **kw:tmp_path/'capture.png')
    calls=[]
    def generate(*a,**kw):
        calls.append(True)
        path=artifact(tmp_path/'new',gate='failed')
        data=json.loads(path.read_text())
        data['perception_validation']['board_examples']=0
        write_json(path,data)
        return path
    monkeypatch.setattr(cli,'generate',generate)
    monkeypatch.setattr(cli,'trial',lambda *a:pytest.fail('no live input allowed'))
    with pytest.raises(SystemExit,match='insufficient input evidence'):
        cli.main(['bootstrap','--game','test','--video','test.mp4','--model','test',
                  '--runtime-config','unused','--attempts','3'])
    assert len(calls)==1


@pytest.mark.parametrize('failure',['unordered','low_confidence'])
def test_reference_repair_keeps_solver_and_retries_only_reference(monkeypatch,tmp_path,failure):
    from gameagent.solvers.perception_check import request_references
    image=tmp_path/'board.png';Image.new('RGB',(100,100)).save(image)
    ref={'id':'board','kind':'grid','confidence':.99,'board_bbox':[.1,.1,.9,.9],
         'rows':2,'cols':2,'centers':[[[.3,.3],[.7,.3]],[[.3,.7],[.7,.7]]],
         'labels':[['red','green'],['blue','yellow']]}
    bad=deepcopy(ref)
    if failure=='unordered': bad['centers'][1]=deepcopy(bad['centers'][0])
    else: bad['confidence']=.72
    calls=[]
    def generate(parts,*args,**kwargs):
        calls.append(deepcopy(parts))
        return {'frames':[bad if len(calls)==1 else ref]}
    monkeypatch.setattr(build,'api_generate',generate)
    source=tmp_path/'solver.py';source.write_text('preserved')
    result=request_references(tmp_path,[{'id':'board','image_path':str(image)}],
                              'test','https://unused',1000,None)
    assert len(calls)==2 and result[0]['confidence']==.99
    assert 'failed validation' in calls[1][-1]['text']
    assert source.read_text()=='preserved'
    assert (tmp_path/'reference_error_1.json').exists()


@pytest.mark.parametrize('game,geometry_bad,expected', [
    ('tasty_travel', False, 'provisional'), ('candy_crush_soda', False, 'failed'),
    ('tasty_travel', True, 'failed')])
def test_tasty_label_disagreement_allows_unvalidated_trial_only(tmp_path,game,geometry_bad,expected):
    source = SOURCE.replace('"red","green"','"yellow","purple"')
    source = source.replace('"rows":2,"cols":2,', '"rows":2,"cols":2,"access":[["available","available"],["available","available"]],')
    if geometry_bad:
        source = source.replace('int(w*.1),int(h*.1),int(w*.9),int(h*.9)', '0,0,int(w*.5),int(h*.5)')
    manifest = artifact(tmp_path, source=source)
    data = json.loads(manifest.read_text()); data['game'] = game; write_json(manifest,data)
    image = tmp_path/'board.png'; Image.new('RGB',(100,100),'black').save(image)
    report = validate_perception(manifest,[{**reference(),'image_path':str(image),
        'access':[['available','available'],['available','available']]}])
    assert report['status'] == expected
    assert cli.perception_ready(manifest) == (expected == 'provisional')
    data = json.loads(manifest.read_text())
    assert data['status'] != 'active'
    if expected == 'provisional':
        solver=LocalSolver(manifest)
        assert solver.infer(image,{'width':100,'height':100}) is not None
        data['perception_validation']['sha256'] = 'stale'
        write_json(manifest,data)
        assert not cli.perception_ready(manifest)
