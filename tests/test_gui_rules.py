import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gameagent.server.gui_rules import GuiRules


def setup_rule(tmp_path, response):
    path=tmp_path/'any_game.yaml'
    path.write_text('rules:\n  - id: collect\n    states: [B, P]\n    when: reward available\n    target: claim button\n    success: claim button disappears\n')
    rules=GuiRules(path)
    calls=[]
    def ground(*args):
        calls.append(args)
        return {'type':'tap','x':30,'y':15,'duration_ms':80}, {}
    server=SimpleNamespace(gui_actor=SimpleNamespace(ground=ground),_grounding_calls=[],
                           _generate_with_prompt=lambda *a:json.dumps(response))
    req=SimpleNamespace(frame_id=1,previous_action=None,metadata={},screen=SimpleNamespace(width=100,height=100))
    return rules,server,req,calls


def test_no_visible_target_never_calls_actor(tmp_path):
    rules,server,req,calls=setup_rule(tmp_path,{'target_id':None})
    assert rules.choose(server,req,Path('frame.png'),{'state':'B'}) is None
    assert calls==[]


def test_generic_rule_grounding_and_bounded_retries(tmp_path):
    response={'target_id':'collect','bbox':[20,10,40,25],'evidence':'visible claim button','previous_succeeded':False}
    rules,server,req,calls=setup_rule(tmp_path,response)
    for i in (1,2):
        req.frame_id=i
        action=rules.choose(server,req,Path('frame.png'),{'state':'B'})
        assert action['x']==30
        req.previous_action=SimpleNamespace(**action)
        req.metadata={'previous_execution_ok':True}
        rules.acknowledge(req)
    req.frame_id=3
    assert rules.choose(server,req,Path('frame.png'),{'state':'B'}) is None
    assert len(calls)==2
    assert any(e['event']=='unverified_outcome' for e in rules.events)


def test_success_is_only_recorded_after_executed_input(tmp_path):
    response={'target_id':'collect','bbox':[20,10,40,25],'evidence':'button'}
    rules,server,req,calls=setup_rule(tmp_path,response)
    action=rules.choose(server,req,Path('frame.png'),{'state':'B'})
    req.previous_action=SimpleNamespace(**action)
    req.metadata={'previous_execution_ok':True,'dry_run':True}
    rules.acknowledge(req)
    assert rules.pending is None
    assert rules.events[-1]['event']=='not_executed'
    req.metadata={}
    action=rules.choose(server,req,Path('frame.png'),{'state':'B'})
    req.previous_action=SimpleNamespace(**action);req.metadata={'previous_execution_ok':True}
    rules.acknowledge(req)
    response.update(target_id=None,previous_succeeded=True,outcome_evidence='claim button gone')
    rules.choose(server,req,Path('frame.png'),{'state':'P'})
    assert rules.attempts['collect']==0
    assert any(e['event']=='verified_success' for e in rules.events)


def test_actor_error_or_outside_coordinate_returns_to_normal_route(tmp_path):
    rules,server,req,calls=setup_rule(tmp_path,{'target_id':'collect','bbox':[20,10,40,25],'evidence':'button'})
    server.gui_actor.ground=lambda *a:({'type':'tap','x':99,'y':99},{})
    assert rules.choose(server,req,Path('frame.png'),{'state':'B'}) is None
    assert server._grounding_calls[-1]['status']=='error'


def test_rules_survive_new_server_and_state_gating(tmp_path):
    rules,server,req,calls=setup_rule(tmp_path,{'target_id':'collect','bbox':[20,10,40,25],'evidence':'button'})
    assert rules.choose(server,req,Path('frame.png'),{'state':'S'}) is None
    assert calls==[]
    assert GuiRules(rules.path).rules==rules.rules


def test_router_prioritizes_generic_gui_rule_over_local_solver(monkeypatch,tmp_path):
    import hashlib
    from PIL import Image
    from gameagent.solvers.registry import write_json
    from gameagent.server.adaptive_router import AdaptiveRoutingServer
    from gameagent.server.vlm_server import DecisionServer,DecideRequest,Screen
    rules,_,_,_=setup_rule(tmp_path,{})
    source='''
def observe(image, context):
    return {"signature":"board","confidence":1,"board_bbox":[10,40,90,95]}
def choose_action(state, context):
    return {"type":"tap","x":50,"y":60,"duration_ms":80}
'''
    (tmp_path/'solver.py').write_text(source)
    manifest=tmp_path/'manifest.json'
    write_json(manifest,{'game':'another_game','version':'test','status':'candidate','sha256':hashlib.sha256(source.encode()).hexdigest()})
    profile=tmp_path/'profile.json'
    write_json(profile,{'name':'another_game','solver_first':True,'generated_solver':str(manifest),'gui_rules':str(rules.path)})
    def generate(self,req,path,prompt,*args):
        if 'Evaluate the developer GUI rules' in prompt:
            return json.dumps({'target_id':'collect','bbox':[20,10,40,25],'evidence':'claim button visible'})
        return json.dumps({'state':'B','confidence':1,'board_confidence':1,'board_bbox':[10,40,90,95],
                           'evidence':'board','controls':[]})
    monkeypatch.setattr(DecisionServer,'_generate_with_prompt',generate)
    server=AdaptiveRoutingServer(model_id='openai:test',profile_path=str(profile),screen_routing={
        'classifier_model':'openai:test','adaptive':{'templates':str(tmp_path/'missing.json'),'orb_references':None}})
    server.gui_actor=SimpleNamespace(ground=lambda *a:({'type':'tap','x':30,'y':15,'duration_ms':80},{}))
    image=tmp_path/'frame.png';Image.new('RGB',(100,100),(200,0,0)).save(image)
    result=server.decide(DecideRequest(frame_id=1,image_path=str(image),screen=Screen(width=100,height=100)))
    assert result['pipeline']['screen_routing']['route']=='gui_rule'
    assert result['action']['y']==15
    assert server.generated_solver.pending is None
