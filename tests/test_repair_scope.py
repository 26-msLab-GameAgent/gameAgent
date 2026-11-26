import pytest
from gameagent.solvers.repair_scope import repair_scope,check_repair_scope,regressed_cases

SOURCE='''
LIMIT = 60
def detect(image):
    return [1, 2]
def observe(image, context):
    box = detect(image)
    if box is None:
        return None
    return {"box": box}
def choose_action(state, context):
    return None
'''


def test_only_failed_branch_can_change():
    scope=repair_scope(SOURCE,[{'diagnostics':{'returns':[{'function':'observe','line':8,'returned':None}]}}],'perception')
    assert scope['sites'][0]['source']=='if box is None:\n        return None'
    check_repair_scope(SOURCE,SOURCE.replace('if box is None:', 'if box is None or len(box) < 2:'),scope)
    for altered in [SOURCE.replace('LIMIT = 60','LIMIT = 80'), SOURCE.replace('return [1, 2]','return [5, 6]'),
                    SOURCE+'\ndef new_helper():\n    return 1\n', SOURCE.replace('box = detect(image)','box = [1,2]')]:
        with pytest.raises(ValueError,match='outside'):check_repair_scope(SOURCE,altered,scope)


def test_missing_trace_limits_to_entry_function():
    scope=repair_scope(SOURCE,[],'action')
    assert 'def choose_action' in scope['sites'][0]['source']
    check_repair_scope(SOURCE,SOURCE.replace('def choose_action(state, context):\n    return None','def choose_action(state, context):\n    return {"type":"wait"}'),scope)
    with pytest.raises(ValueError):check_repair_scope(SOURCE,SOURCE.replace('LIMIT = 60','LIMIT = 80'),scope)


def test_interaction_contract_failure_allows_observation_and_policy_only():
    scope=repair_scope(SOURCE,[{'failure_category':'interaction_safety'}],'perception')
    assert len(scope['sites'])==2
    check_repair_scope(SOURCE,SOURCE.replace('return {"box": box}', 'return {"box": box, "access": []}'),scope)
    check_repair_scope(SOURCE,SOURCE.replace('def choose_action(state, context):\n    return None',
                      'def choose_action(state, context):\n    return {"type":"wait"}'),scope)
    with pytest.raises(ValueError):
        check_repair_scope(SOURCE,SOURCE.replace('LIMIT = 60','LIMIT = 80'),scope)


def test_regressions_require_same_image_and_include_missing_cases():
    old={'results':[{'id':'a','scale':1,'status':'passed'},{'id':'a','scale':.75,'status':'failed'}]}
    new={'results':[{'id':'a','scale':1,'status':'failed'},{'id':'a','scale':.75,'status':'passed'}]}
    refs=[{'id':'a','image_sha256':'same'}]
    assert regressed_cases(old,new,refs,refs)==[{'id':'a','scale':1}]
    assert regressed_cases(old,new,refs,[{'id':'a','image_sha256':'different'}])==[]
