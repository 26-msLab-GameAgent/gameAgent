from copy import deepcopy
import json

import pytest
from PIL import Image

from gameagent.solvers.registry import grid_action_error, LocalSolver, write_json
from gameagent.solvers.worker import symbolic_signature
from gameagent.solvers.perception_check import validate_perception
from test_perception_check import SOURCE, artifact, reference
from test_generated_solver import frame_result
from gameagent.server.vlm_server import PreviousAction
from types import SimpleNamespace


def grid():
    return {'rows':2,'cols':2,'centers':[[[30,30],[70,30]],[[30,70],[70,70]]],
            'labels':[['tea_2','tea_2'],['empty','unknown']],
            'access':[['available','locked'],['available','unknown']],
            'unlockable':[[False,True],[False,False]]}


@pytest.mark.parametrize('case', ['locked_source','number_lock','wrong_item','color_only','missing_access','unknown_source'])
def test_unsafe_drag_blocked(case):
    g=grid(); a={'type':'swipe','x':30,'y':30,'x2':70,'y2':30}
    if case=='locked_source': g['access'][0][0]='locked'
    if case=='number_lock': g['unlockable'][0][1]=False
    if case=='wrong_item': g['labels'][0][1]='tea_3'
    if case=='color_only': g['labels'][0]=['yellow','yellow']
    if case=='missing_access': g.pop('access')
    if case=='unknown_source': g['access'][0][0]='unknown'
    assert grid_action_error(a,g,required=True)


def test_matching_movable_item_can_unlock_cloth_destination():
    assert grid_action_error({'type':'swipe','x':30,'y':30,'x2':70,'y2':30},grid(),required=True) is None


def test_feature_jitter_cannot_count_as_progress_or_reset_failed_watch(tmp_path):
    path=artifact(tmp_path, gate='passed'); solver=LocalSolver(path)
    obs={'signature':'pixels1','confidence':.95,'board_bbox':[10,10,90,90],'grid':grid()}
    obs['signature']=symbolic_signature(obs)
    action={'type':'tap','x':30,'y':30}
    solver.last={'observation':obs,'action':action}
    result=frame_result(); result['action']=action
    req=SimpleNamespace(frame_id=1,previous_action=None,metadata={})
    solver.record(req,result,tmp_path/'unused.png')
    for i in (2,3):
        changed=deepcopy(obs); changed['signature']='pixel jitter '+str(i)
        changed['grid']['centers'][0][0][0]+=i
        changed['state']={'histogram':[.1*i,.2]}
        changed['signature']=symbolic_signature(changed)
        assert changed['signature']==obs['signature']
        solver.last={'observation':changed,'action':action}
        req.frame_id=i; req.metadata={'previous_execution_ok':True}
        req.previous_action=PreviousAction(**action) if i==2 else PreviousAction(type='wait')
        failed=solver.check_action_progress(req)
        wait=frame_result('B','board_reobserve'); wait['action']={'type':'wait'}
        solver.record(req,wait,tmp_path/'unused.png')
    assert failed is True
    assert solver.stats['executed']==1 and solver.stats['changed']==0
    assert solver.guard_repeated_action(action)
    changed['grid']['access'][0][1]='available'
    assert symbolic_signature(changed)!=obs['signature']


@pytest.mark.parametrize('case',['missing','false_available','locked_action'])
def test_tasty_safety_failure_cannot_become_provisional(tmp_path,case):
    access=[['available','locked'],['available','available']]
    actual=deepcopy(access)
    if case=='false_available':actual[0][1]='available'
    source=SOURCE
    if case!='missing':
        source=source.replace('"rows":2,"cols":2,',f'"rows":2,"cols":2,"access":{actual!r},')
    if case in {'false_available','locked_action'}:
        source=source.replace('def choose_action(state, context):\n    return None',
            'def choose_action(state, context):\n    x,y=state["grid"]["centers"][0][1]\n    return {"type":"tap","x":x,"y":y}')
    manifest=artifact(tmp_path,source=source)
    data=json.loads(manifest.read_text()); data['game']='tasty_travel'; write_json(manifest,data)
    image=tmp_path/'board.png'; Image.new('RGB',(100,100),'black').save(image)
    report=validate_perception(manifest,[{**reference(),'image_path':str(image),'access':access}])
    assert report['status']=='failed'
    assert all(r['status']=='failed' for r in report['results'])
    assert all(r['failure_category']=='interaction_safety' for r in report['results'])
