import json
from pathlib import Path

import pytest
from PIL import Image

from gameagent.solvers import build, tutorial
from gameagent.solvers.perception_check import compare_observation
from gameagent.solvers.registry import LocalSolver


def action_file(path):
    data={'session':{'screen':{'width':160,'height':120},'video':{'started_offset_ms':200}},
          'actions':[{'type':'tap','start_ms':1200,'end_ms':1300,'position':{'x':40,'y':30}}]}
    path.write_text(json.dumps(data));return path


def video_file(path):
    import cv2
    import numpy as np
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,120))
    assert writer.isOpened()
    for i in range(40): writer.write(np.full((120,160,3),i,dtype=np.uint8))
    writer.release();return path


def test_video_time_offset_and_explicit_video_time(tmp_path):
    p=action_file(tmp_path/'actions.json')
    assert tutorial.load_actions(p)['actions'][0]['start_ms']==1000
    data=json.loads(p.read_text());data['actions'][0].update(video_start_ms=500,video_end_ms=600)
    p.write_text(json.dumps(data))
    assert tutorial.load_actions(p)['actions'][0]['start_ms']==500
    data['actions'][0]['position']['x']=999;p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='coordinates'):
        tutorial.load_actions(p)


def test_action_alignment_and_out_of_video_rejection(tmp_path):
    timeline=tutorial.load_actions(action_file(tmp_path/'actions.json'))
    video=video_file(tmp_path/'tutorial.mp4')
    parts,record=tutorial.aligned_evidence(video,timeline,tmp_path/'evidence')
    frames=record['events'][0]['frames']
    assert frames[0]['timestamp_ms']<1000
    assert frames[1]['timestamp_ms']>1100
    assert len(parts)==5
    timeline['actions'][0].update(start_ms=9000,end_ms=9100)
    with pytest.raises(ValueError,match='overlap'):
        tutorial.aligned_evidence(video,timeline,tmp_path/'bad')


def test_locked_identity_ignored_but_access_mismatch_fails():
    ref={'kind':'grid','confidence':1,'width':100,'height':100,'board_bbox':[10,10,90,90],
         'rows':2,'cols':2,'centers':[[[30,30],[70,30]],[[30,70],[70,70]]],
         'labels':[['red','locked'],['locked','locked']],
         'access':[['available','locked'],['locked','locked']]}
    obs={'confidence':1,'board_bbox':ref['board_bbox'],'grid':{
        k:ref[k] for k in ('rows','cols','centers','access')}}
    obs['grid']['labels']=[['red','unknown'],['unknown','unknown']]
    assert compare_observation(obs,ref)['status']=='passed'
    obs['grid']['access']=[['available','available'],['locked','locked']]
    assert 'cell interactability disagrees with reference' in compare_observation(obs,ref)['errors']


def test_video_generation_skips_action_specification(monkeypatch,tmp_path):
    video=video_file(tmp_path/'tutorial.mp4');actions=action_file(tmp_path/'tutorial.json')
    (tmp_path/'configs').symlink_to(build.ROOT/'configs',target_is_directory=True)
    monkeypatch.setattr(build,'ROOT',tmp_path);monkeypatch.setattr(tutorial,'ROOT',tmp_path)
    spec={'representation':'grid','objective':'match red','cell_labels':['unknown','empty','locked','red'],
          'state_fields':[],'rules':[{'description':'tap observed','action_ids':[0],'confidence':.5}],
          'interaction_rules':[],'success_signals':[],'uncertainties':['tap outcome unconfirmed']}
    source='''
def observe(image, context):
    w,h=image.size
    return {"signature":"r", "confidence":1,"board_bbox":[int(w*.1),int(h*.1),int(w*.9),int(h*.9)],
      "grid":{"rows":2,"cols":2,"centers":[[[int(w*.3),int(h*.3)],[int(w*.7),int(h*.3)]],
      [[int(w*.3),int(h*.7)],[int(w*.7),int(h*.7)]]],"labels":[["red","unknown"],["unknown","unknown"]],
      "access":[["available","locked"],["locked","locked"]]}}
def choose_action(state,context):
    return None
'''
    calls=[]
    def api(parts,*args,**kw):
        prompt=kw.get('system_prompt');calls.append(prompt)
        if prompt==tutorial.SPEC_PROMPT:
            assert 'before' in json.dumps(parts) and 'start_norm' in json.dumps(parts)
            return spec
        if prompt:
            assert 'Shared, fallible game specification' not in json.dumps(parts)
            ids=[p['text'].split(': ',1)[1] for p in parts if p.get('text','').startswith('Frame id:')]
            return {'frames':[{'id':i,'kind':'grid','confidence':1,'board_bbox':[.1,.1,.9,.9],
                'rows':2,'cols':2,'centers':[[[.3,.3],[.7,.3]],[[.3,.7],[.7,.7]]],
                'labels':[['red','unknown'],['unknown','unknown']],
                'access':[['available','locked'],['locked','locked']]} for i in ids]}
        assert 'Shared game specification' not in json.dumps(parts)
        return {'source':source,'objective':'match red'}
    monkeypatch.setattr(build,'api_generate',api)
    manifest=build.generate('test','',video=video,actions_json=actions,model='test',base_url='https://unused')
    assert len(calls)==2
    assert tutorial.SPEC_PROMPT not in calls
    assert not manifest.with_name('game_spec.json').exists()
    assert json.loads(manifest.read_text())['perception_validation']['status']=='passed'
    # Repair reuses the video-only reference, without action evidence or specification.
    build.generate('test','',video=video,actions_json=actions,model='test',base_url='https://unused',previous=manifest)
    assert len(calls)==3


@pytest.mark.parametrize('explicit', [False, True])
def test_cli_ignores_paired_action_json(monkeypatch,tmp_path,explicit):
    from gameagent.solvers import cli
    from gameagent.solvers.registry import write_json
    video=tmp_path/'demo.mp4';video.touch();actions=action_file(video.with_suffix('.json'))
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(cli,'prepare_reference',lambda *a, **kw:tmp_path/'board.png')
    seen=[]
    def generate(*args,**kwargs):
        seen.append(kwargs.get('actions_json'))
        manifest=tmp_path/'manifest.json';write_json(manifest,{'status':'active'})
        return manifest
    monkeypatch.setattr(cli,'generate',generate);monkeypatch.setattr(cli,'trial',lambda *a:'active')
    cli.main(['bootstrap','--game','test','--video',str(video),'--model','test','--runtime-config','unused']
             + (['--actions-json',str(actions)] if explicit else []))
    assert seen==[None]


def test_reference_sampling_uses_action_frames_and_is_idempotent(tmp_path):
    from gameagent.solvers.perception_check import action_reference_cases
    from gameagent.solvers.registry import write_json
    source=tmp_path/'source.png';Image.new('RGB',(20,20)).save(source)
    timeline=tmp_path/'timeline.json'
    write_json(timeline,{'events':[
      {'id':1,'type':'swipe','frames':[{'phase':'before','image_path':str(source)},
         {'phase':'after','image_path':str(source)}],'after_may_be_unsettled':True},
      {'id':2,'type':'tap','confounded_by_next_action':True,
       'frames':[{'phase':'before','image_path':str(source)}]}]})
    write_json(tmp_path/'game_spec.json',{'evidence':str(timeline)})
    cases=action_reference_cases(tmp_path,[])
    assert [c['id'] for c in cases]==['action_1_before']
    assert action_reference_cases(tmp_path,cases)==cases
