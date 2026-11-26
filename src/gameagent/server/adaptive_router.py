"""0916 router: cheap verified paths, one semantic call, bounded progress recovery."""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from .fast_screens import VerifiedScreens, thumbnail, stationary, board_stationary, same_board_chrome
from .screen_router import ScreenRoutingServer, normalize_screen, guard_board, CLASSIFY_GUI
from .vlm_server import _parse_model_json, _plan_candy_crush_swipe, _candy_board_signature


PROMPT = CLASSIFY_GUI.replace(
    'U: uncertain, unsupported, failure/retry, permission/account/payment screen.',
    'R: explicit level failure with a visible free Retry button. Retry may consume a life; '
    'never confuse it with purchasing extra moves. Return role retry for the free button.\n'
    'U: genuinely unrecognized or unsupported screen. Account/permission screens have no controls.'
).replace('confirm,skip.', 'confirm,skip,retry.').replace('S|B|P|C|T|L|U', 'S|B|P|C|T|L|U|R') + '''
Return concise JSON only. Keep evidence to one sentence. Confidence is telemetry, not a
reason to replace a recognizable L/R/UI screen with U. Identify L for ongoing animation.
For B return one optional fallback_action (adjacent swipe only) if a valid match is visible;
otherwise null. Never select a booster. Coordinates are original pixels.
For UI targets identify visible free controls with grounding_description; GUI Actor supplies
their coordinates. UI bbox is optional; board_bbox is mandatory for B. Do not guess missing
controls or stage numbers.
For B also return moves_remaining as {"value":15,"confidence":0.95} when the
remaining MOVES counter is clearly readable, otherwise null. Read only the moves
counter, never the goal count, level number, score, booster count or lives. Do not
infer it from previous actions. For every non-B screen use moves_remaining:null.
'''

DEFAULT_ADAPTIVE = {
    'templates': 'assets/ui_templates.json',
    'orb_references': 'assets/router_references/candy_crush_soda',
    'orb_min_good_matches': 12,
    'orb_min_inliers': 8,
    'orb_min_margin': 1.25,
    'short_wait_ms': 200,
    'reobserve_steps': 2,
    'max_recovery_calls': 3,
    'max_no_progress_steps': 14,
    'max_retries_per_stage': 2,
    'max_same_ui_attempts': 2,
    'board_refresh_steps': 8,
    'local_board': True,
}


def changed_known_grid(before, after):
    """Do not mistake changing unknown-cell coverage or grid size for progress."""
    if not before or not after:
        return False
    old, new = before.split('|'), after.split('|')
    return len(old) == len(new) and any(
        a != b and a != 'unknown' and b != 'unknown' for a, b in zip(old, new))


def read_moves_remaining(raw, threshold):
    """Only accept an explicit, confident counter, never loose visible_text digits."""
    value = raw.get('moves_remaining')
    if not isinstance(value, dict):
        return None
    count, confidence = value.get('value'), value.get('confidence')
    if (type(count) is int and 0 <= count <= 999
            and type(confidence) in (int, float) and math.isfinite(confidence)
            and threshold <= confidence <= 1):
        return count
    return None


class AdaptiveRoutingServer(ScreenRoutingServer):
    def __init__(self, *args, screen_routing=None, **kwargs):
        config = deepcopy(screen_routing or {})
        adaptive = config.pop('adaptive', {})
        if not isinstance(adaptive, dict) or set(adaptive)-set(DEFAULT_ADAPTIVE):
            raise ValueError('invalid adaptive configuration')
        self.adaptive = {**DEFAULT_ADAPTIVE, **adaptive}
        for key in ('short_wait_ms', 'reobserve_steps', 'max_recovery_calls',
                    'max_no_progress_steps', 'max_retries_per_stage',
                    'max_same_ui_attempts', 'board_refresh_steps'):
            if type(self.adaptive[key]) is not int or self.adaptive[key] < 1:
                raise ValueError(f'adaptive.{key} must be a positive integer')
        if type(self.adaptive['local_board']) is not bool:
            raise ValueError('adaptive.local_board must be boolean')
        for key in ('orb_min_good_matches', 'orb_min_inliers'):
            if type(self.adaptive[key]) is not int or self.adaptive[key] < 4:
                raise ValueError(f'adaptive.{key} must be an integer >= 4')
        if (type(self.adaptive['orb_min_margin']) not in (int, float)
                or self.adaptive['orb_min_margin'] < 1.0):
            raise ValueError('adaptive.orb_min_margin must be >= 1.0')
        super().__init__(*args, screen_routing=config, **kwargs)
        path = Path(self.adaptive['templates'])
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        orb_path = self.adaptive['orb_references']
        if orb_path:
            orb_path = Path(orb_path)
            if not orb_path.is_absolute():
                orb_path = Path(__file__).resolve().parents[3] / orb_path
        self.fast_screens = VerifiedScreens(
            path,
            orb_references=orb_path,
            orb_min_good_matches=self.adaptive['orb_min_good_matches'],
            orb_min_inliers=self.adaptive['orb_min_inliers'],
            orb_min_margin=self.adaptive['orb_min_margin'],
        )
        from gameagent.solvers.registry import LocalSolver
        generated = (self.profile or {}).get('generated_solver')
        self.generated_solver = LocalSolver(generated) if generated and self.routing['enable_board_solver'] else None
        from .gui_rules import GuiRules
        game = self.generated_solver.manifest.get('game') if self.generated_solver else (self.profile or {}).get('name')
        rules_path = (self.profile or {}).get('gui_rules')
        if not rules_path and isinstance(game,str) and game.replace('_','').isalnum():
            rules_path = Path(__file__).resolve().parents[3]/'configs'/'gui_rules'/f'{game}.yaml'
        self.gui_rules = GuiRules(rules_path) if rules_path else None
        self._reset_adaptive()

    def _reset_adaptive(self):
        if getattr(self,'gui_rules',None):
            self.gui_rules.reset()
        self._pixels = None
        self._board_pixels = None
        self._board_state = None
        self._last_signature = None
        self._last_moves_remaining = None
        self._swipes_since_moves = 0
        self._board_stage = None
        self._observed_signature = None
        self._last_state = None
        self._last_action = None
        self._last_ui = None
        self._failed_ui = defaultdict(int)
        self._retry_counts = defaultdict(int)
        self._pending_retry = None
        self._pending_effect = None
        self._blocked_templates = set()
        self._recovery_calls = 0
        self._idle_steps = 0
        self._deferred = 0
        self._board_age = 0
        self._board_recovery_pending = False
        self._forced_vlm_reason = None
        self._stop_reason = None

    def decide(self, req):
        with self._decision_lock:
            if req.frame_id <= 1:
                self._reset_adaptive()
                if self.generated_solver:
                    self.generated_solver.reset_episode()
            return super().decide(req)

    def _short_wait(self, reason):
        return {'type': 'wait', 'duration_ms': self.adaptive['short_wait_ms'], 'reason': reason}

    def _semantic(self, req, path, state_hint=None):
        # One call, including malformed output handling: never fan out to the legacy pipeline.
        instruction = 'Choose a different visible free target after failure; do not repeat it.'
        context = {
            'screen': [req.screen.width, req.screen.height],
            'completed': self._stage_progress['completed'],
            'recent_actions': self._history[-2:],
            'failed_targets': [list(k) for k, v in self._failed_ui.items() if v],
            'instruction': instruction,
        }
        if self._forced_vlm_reason:
            context['board_recovery_reason'] = self._forced_vlm_reason
            context['instruction'] += ' Local board actions failed. Propose a different valid board action from the current image.'
            if self.generated_solver:
                context['failed_board_actions'] = [a for actions in self.generated_solver.failed_actions.values()
                                                   for a in actions][-8:]
        if state_hint:
            context['orb_state_hint'] = state_hint
            context['instruction'] = (
                f'ORB already classified the foreground screen as {state_hint}. '
                'Keep that state unless the image clearly contradicts it. '
                'Use the existing model/GUI Actor path to identify the allowed free '
                'control or board details needed for the next action. '
                + instruction
            )
        base_prompt = PROMPT
        if self.generated_solver or (self.profile or {}).get('prepare_board_only'):
            base_prompt = base_prompt.replace('merge/match mobile game', 'mobile game')
            base_prompt = base_prompt.replace(
                'For B return one optional fallback_action (adjacent swipe only) if a valid match is visible;\notherwise null.',
                'For B return an optional fallback_action following the supplied game rules, or null. '
                'Allowed actions: tap, double_tap, swipe, long_press, wait, noop. '
                'All coordinates must be inside board_bbox; never touch purchases or boosters.')
            context['game_profile'] = self.profile
        prompt = base_prompt + '\nContext: ' + json.dumps({
            **context,
        }, ensure_ascii=False)
        output = self._generate_with_prompt(req, path, prompt, 'screen_classifier', True)
        return _parse_model_json(output)

    @staticmethod
    def _ui_key(state, control):
        stage = control.get('stage') or state.get('stage') or {}
        return (state['state'], stage.get('ordinal', 0), control.get('role'),
                control.get('label', ''), tuple(control.get('bbox') or []))

    def _board_signature(self, image_path):
        if self.generated_solver:
            obs = (self.generated_solver.last or {}).get('observation')
            return obs['signature'] if obs and obs['confidence'] >= .8 else None
        return _candy_board_signature(self._solver_profile, image_path)

    def _board_action(self, image_path, image_fingerprint, signature):
        if self.generated_solver:
            return (self.generated_solver.last or {}).get('action') if signature else None
        return _plan_candy_crush_swipe(
            self._solver_profile, image_path, self._history, image_fingerprint,
            signature, self._failed_swipes_by_board, self._failed_cells_by_board,
        ) if self._solver_profile and signature else None

    def _guard_board_action(self, action, state, allowed):
        from gameagent.solvers.registry import is_order_sale
        if self.generated_solver:
            local = self.generated_solver.last or {}
            # The exception is only for the current validated local proposal,
            # never a stale control or an unrelated VLM fallback coordinate.
            if (action == local.get('action') and 'tap' in allowed
                    and is_order_sale(action, local.get('observation'), self.generated_solver.manifest.get('game'))):
                return None
        return guard_board(action, state, allowed)

    def _decide_pipeline(self, req, image_path, image_fingerprint, board_signature, started):
        if self.gui_rules:
            self.gui_rules.events = []
            self.gui_rules.acknowledge(req)
        if self.generated_solver:
            self.generated_solver.infer(image_path, {
                'width': req.screen.width, 'height': req.screen.height,
                'frame_id': req.frame_id, 'history': self._history[-8:],
                'profile': self.profile,
            })
            action_progress = self.generated_solver.check_action_progress(req)
            if action_progress:
                self._forced_vlm_reason = (
                    'Local perception unavailable after input; require VLM recovery'
                    if action_progress == 'unobserved' else
                    'Confirmed ineffective board action; require VLM recovery')
                self._board_recovery_pending = True
            proposal = (self.generated_solver.last or {}).get('action')
            if (not self._forced_vlm_reason and self._board_recovery_pending and proposal and self._board_state
                    and proposal.get('type') in {'tap','double_tap','swipe','long_press'}
                    and not self.generated_solver.guard_repeated_action(proposal)
                    and not self._guard_board_action(proposal,self._board_state,self.routing['states']['B']['allowed_actions'])):
                # Newly delivered failure feedback may unlock a different local move.
                self._board_recovery_pending = False
        result = self._decide_adaptive(req, image_path, image_fingerprint, board_signature, started)
        if self.generated_solver:
            self.generated_solver.record(req, result, image_path)
        return result

    def _decide_adaptive(self, req, image_path, image_fingerprint, board_signature, started):
        self._screen_calls, self._grounding_calls = [], []
        fallback_reason = None
        self._active_screen = None
        errors = []
        previous_outcome = None
        raw = None
        source = 'deferred'
        action = None
        route = 'reobserve'
        stop = self._stop_reason is not None
        local_started = time.perf_counter()
        fast_screen_started = time.perf_counter()
        pixels = thumbnail(image_path)
        stable = stationary(self._pixels, pixels)
        skipped = req.metadata.get('skipped_action_frame_id')
        actual_previous = getattr(req.previous_action, 'type', None)
        pending = self._pending_effect
        if pending and not pending['executed']:
            if (skipped == pending['frame_id'] or req.metadata.get('dry_run', False)
                    or req.metadata.get('previous_execution_ok') is False
                    or actual_previous != pending['action']['type']):
                self._pending_effect = pending = None
            else:
                pending['executed'] = True
                # Wait observations do not consume the no-progress action budget.
                self._idle_steps += 1
                if pending['mode'] == 'B' and pending['action']['type'] == 'swipe':
                    self._swipes_since_moves += 1
                if pending['mode'] == 'R' and pending.get('retry_stage'):
                    self._retry_counts[pending['retry_stage']] += 1

        signature = None
        observed_signature = None
        grid_stable = False
        local_board = False
        orb_state_hint = None
        raw = self.fast_screens.classify(pixels, req.screen, image_path=image_path)
        if self._board_recovery_pending:
            raw = None
        fast_screen_ms = round((time.perf_counter() - fast_screen_started) * 1000, 2)
        if raw and raw.get('evidence') in self._blocked_templates:
            raw = None
        if raw:
            source = raw.pop('_classification_source', 'local_template')
            # ORB owns the coarse screen state. UI controls still come from the
            # existing semantic/GUI Actor path, so a map reference without a
            # fixed level tile never degenerates into repeated waits.
            if source == 'orb' and raw.get('state') in {'S', 'P', 'C', 'T'}:
                orb_state_hint = raw['state']
                orb_fallback = deepcopy(raw)
                try:
                    semantic = self._semantic(req, image_path, state_hint=orb_state_hint)
                except Exception as exc:
                    errors.append(f'{type(exc).__name__}: {str(exc)[:200]}')
                    semantic = None
                if isinstance(semantic, dict):
                    semantic['state'] = orb_state_hint
                    raw = semantic
                    source = 'orb_ui_semantic'
                else:
                    raw = orb_fallback
        elif (not self._board_recovery_pending and self.adaptive['local_board'] and self._board_state
              and self._board_age < self.adaptive['board_refresh_steps']
              and same_board_chrome(self._board_pixels, pixels,
                                   self._board_state['board_bbox'], req.screen)):
            observed_signature = self._board_signature(image_path)
            grid_stable = bool(
                observed_signature and observed_signature == self._observed_signature
                and board_stationary(self._pixels, pixels,
                                     self._board_state['board_bbox'], req.screen))
            if grid_stable:
                signature = observed_signature
                raw = deepcopy(self._board_state)
                raw['evidence'] = 'Stable local grid and confirmed board chrome'
                raw['board_confidence'] = 1.0
                local_board = True
                source = 'local_board'

        # After input / while an animation resolves, give the next capture a chance first.
        if raw is None and not stop:
            if self._pixels is not None and self._deferred < self.adaptive['reobserve_steps']:
                self._deferred += 1
                raw = {'state': 'L' if not stable else 'U', 'confidence': 0,
                       'evidence': 'Short reobservation before semantic recovery', 'controls': []}
            elif self._recovery_calls >= self.adaptive['max_recovery_calls']:
                stop = True
                self._stop_reason = 'semantic recovery budget exhausted without verified progress'
            else:
                self._deferred = 0
                self._board_recovery_pending = False
                source = 'semantic'
                try:
                    raw = self._semantic(req, image_path)
                except Exception as exc:
                    errors.append(f'{type(exc).__name__}: {str(exc)[:200]}')
        raw = raw or {'state': 'U', 'confidence': 0, 'evidence': 'No recognized screen', 'controls': []}
        state = normalize_screen(raw, req.screen, self.routing['min_confidence'],
                                 semantic_ui=self.gui_actor is not None, preserve_state=True)
        self._active_screen = state
        mode = state['state']
        moves_remaining = (read_moves_remaining(raw, self.routing['min_confidence'])
                           if mode == 'B' and source == 'semantic' else None)
        # A cached board classification does not contain a fresh counter reading.
        state['moves_remaining'] = moves_remaining
        # Fresh counter evidence overrides noisy local grid signatures.
        if (self.generated_solver and mode == 'B' and moves_remaining is not None
                and moves_remaining == self._last_moves_remaining
                and self._swipes_since_moves >= 2):
            self._forced_vlm_reason = 'Moves counter unchanged after multiple swipes; require VLM recovery'
        current_stage = state.get('stage')
        stage_changed = bool(mode == 'B' and current_stage and self._board_stage
                             and current_stage != self._board_stage)
        if mode not in {'B', 'L', 'U'} or stage_changed:
            self._last_signature = self._last_moves_remaining = None
            self._swipes_since_moves = 0
            self._board_stage = None
            if pending and pending['mode'] == 'B':
                self._pending_effect = pending = None
        if mode == 'B' and current_stage:
            self._board_stage = deepcopy(current_stage)

        # Only trusted visual transitions reset the episode budget, not L/U/B label flicker.
        progress = False
        progress_evidence = 'Verified UI transition or new completion'
        executed = bool(pending and pending['executed'])
        if executed and not stationary(pending['pixels'], pixels):
            if mode == 'B' and pending['mode'] in {'S', 'T', 'R'}:
                progress = True
            elif pending['mode'] == 'C' and mode == 'S':
                progress = True
            elif pending['mode'] == 'S' and mode == 'T':
                progress = True
            elif pending['mode'] == 'S' and mode == 'S' and pending['role'] == 'stage' and state.get('stage'):
                progress = True
            elif pending['mode'] == 'P' and mode in {'R', 'S', 'B'}:
                progress = True
        if mode == 'C' and state.get('stage'):
            stage = state['stage']
            if stage not in self._stage_progress['completed']:
                self._stage_progress['completed'].append(dict(stage))
                self._stage_progress['last_completed'] = dict(stage)
                progress = True
        if mode == 'B':
            signature = signature or self._board_signature(image_path)
            observed_signature = signature
            # A current semantic/ORB B confirmation or a repeated local grid is
            # the stability gate. The candy grid signature is the progress
            # signal; screen classification alone never counts as progress.
            if (executed and pending['action']['type'] == 'swipe'
                    and changed_known_grid(pending['signature'], signature)
                    and (source in {'semantic', 'orb'} or grid_stable)):
                progress = True
                progress_evidence = 'Verified changed grid after executed swipe'
            if (executed and pending['mode'] == 'B' and pending['action']['type'] == 'swipe'
                    and moves_remaining is not None and self._last_moves_remaining is not None
                    and 0 < self._last_moves_remaining - moves_remaining <= self._swipes_since_moves):
                progress = True
                progress_evidence = (f'Verified remaining moves decreased: '
                                     f'{self._last_moves_remaining} -> {moves_remaining}')
            if moves_remaining is not None:
                self._last_moves_remaining = moves_remaining
                self._swipes_since_moves = 0
            if source in {'semantic', 'orb'}:
                self._board_pixels, self._board_state = pixels.copy(), deepcopy(state)
                self._board_age = 0
            else:
                self._board_age += 1
        elif mode not in {'L', 'U'}:
            self._board_state = None
            self._board_pixels = None
        if progress:
            self._recovery_calls = self._idle_steps = 0
            self._pending_effect = pending = None
            previous_outcome = {'status': 'success', 'evidence': progress_evidence}
        elif executed and mode not in {'L', 'U'} and stable and pending.get('ui_key'):
            self._failed_ui[pending['ui_key']] += 1
            previous_outcome = {'status': 'failure', 'evidence': 'Expected UI transition did not occur'}
            if pending.get('template'):
                self._blocked_templates.add(pending['template'])
            self._pending_effect = None
        self._last_ui = None
        # Disabled for the 50-action board experiment. Board-level progress is
        # still recorded in previous_outcome/board_progress, but unchanged
        # boards no longer terminate the run before the configured action cap.
        # if self._idle_steps > self.adaptive['max_no_progress_steps']:
        #     stop = True
        #     self._stop_reason = 'no verified progress within action budget'
        if self.generated_solver and self._idle_steps > self.adaptive['max_no_progress_steps']:
            stop = True
            self._stop_reason = 'generated solver: no verified progress within action budget'

        if (not stop and mode == 'B' and (self.profile or {}).get('prepare_board_only')
                and source in {'semantic', 'orb', 'local_template'}
                and state['confidence'] >= self.routing['min_confidence']):
            stop = True
            self._stop_reason = 'UI preparation complete: gameplay screen reached'
        elif (not stop and mode == 'B' and (self.profile or {}).get('prepare_board_only')
                and source == 'semantic' and state.get('board_bbox')
                and type(raw.get('board_confidence')) in (int,float)
                and self.routing['min_confidence'] <= raw['board_confidence'] <= 1):
            # A visible board with a tutorial hint may have ambiguous S/B/T state.
            # Capture evidence only; independent perception validation still gates play.
            stop = True
            self._stop_reason = 'UI preparation: provisional board reference captured; validation required'
        rule_action = None
        if (not stop and self.gui_rules and source != 'deferred' and mode in {'B','S','P','C','T','R'}
                and (mode != 'B' or not (self.profile or {}).get('prepare_board_only'))):
            rule_action = self.gui_rules.choose(self, req, image_path, state)
        if rule_action:
            action, route = rule_action, 'gui_rule'
        elif not stop and mode == 'B' and (self.profile or {}).get('prepare_board_only'):
            # Unconfirmed/cached guesses keep waiting; preparation never dispatches board moves.
            action = self._short_wait('Waiting for a confident gameplay screen')
            route = 'board_reobserve'
            self._board_age = self.adaptive['board_refresh_steps']
        elif not stop and mode == 'B':
            self._record_previous_action_outcome(req, signature)
            action = self._board_action(image_path, image_fingerprint, signature)
            fallback_reason = self._forced_vlm_reason
            if fallback_reason:
                action = None
            route = ('generated_solver' if self.generated_solver and action else
                     'local_board_solver' if local_board else 'board_solver')
            # An unsafe solver proposal is unavailable, just like no proposal.
            # Reject it before choosing the VLM fallback from this same frame.
            if action:
                rejected = (self.generated_solver.guard_repeated_action(action) if self.generated_solver else None)
                rejected = rejected or self._guard_board_action(action, state, self.routing['states']['B']['allowed_actions'])
                if rejected:
                    errors.append(rejected)
                    action = None
            if action is None and source == 'semantic':
                candidate = raw.get('fallback_action')
                if self.generated_solver and isinstance(candidate, dict):
                    from gameagent.solvers.registry import validate_result
                    try:
                        validate_result({'observation': {
                            'signature': 'fallback', 'confidence': 1,
                            'board_bbox': state['board_bbox'],
                        }, 'action': candidate}, req.screen.width, req.screen.height)
                        action = candidate
                        route = 'board_single_call'
                    except ValueError as exc:
                        errors.append(str(exc))
                elif isinstance(candidate, dict) and candidate.get('type') == 'swipe':
                    points = [candidate.get(k) for k in ('x', 'y', 'x2', 'y2')]
                    if all(type(p) is int for p in points):
                        x, y, x2, y2 = points
                        l, t, r, b = state['board_bbox']
                        if ((x == x2) != (y == y2)) and 15 <= abs(x-x2)+abs(y-y2) <= min(r-l,b-t)/3:
                            action = {**candidate, 'duration_ms': 220}
                            route = 'board_single_call'
            if action:
                rejected = (self.generated_solver.guard_repeated_action(
                    action, semantic_fallback=(route == 'board_single_call')) if self.generated_solver else None)
                rejected = rejected or self._guard_board_action(action, state, self.routing['states']['B']['allowed_actions'])
                if rejected:
                    errors.append(rejected)
                    action = None
            if action is None:
                route = 'board_reobserve'
                action = self._short_wait('No valid board move; reobserve then single-call recovery')
                self._board_age = self.adaptive['board_refresh_steps']
                # ORB recognizing B does not mean the policy can act. Do not let a
                # perpetual ORB hit starve bounded semantic recovery after local failure.
                if not self.generated_solver or not self.generated_solver.action_watch:
                    self._board_recovery_pending = True
            elif route == 'board_single_call':
                self._forced_vlm_reason = None
                self._board_recovery_pending = False
        elif not stop and mode in {'S', 'C', 'T', 'P', 'R'}:
            state['controls'] = [c for c in state['controls']
                                 if self._failed_ui[self._ui_key(state, c)]
                                 < self.adaptive['max_same_ui_attempts']]
            retry_stage = str((state.get('stage') or {}).get('ordinal', 'unknown'))
            if mode == 'R' and self._retry_counts[retry_stage] >= self.adaptive['max_retries_per_stage']:
                stop = True
                self._stop_reason = 'retry budget exhausted for level ' + retry_stage
            else:
                # Semantic controls have already passed role/free/enabled/confidence checks.
                # GUI Actor supplies the coordinates; requiring the classifier's bbox here
                # would discard the very controls that need grounding. An optional bbox
                # remains a hard constraint in GuiActorClient.
                action, route = self._ui_action(req, image_path, state)
                if action.get('type') == 'tap':
                    target_id = (self._grounding_calls[-1].get('target_id')
                                 if self._grounding_calls else None)
                    selected = next((c for c in state['controls'] if c['id'] == target_id), None)
                    if selected is None and not self.gui_actor:
                        selected = next((c for c in state['controls'] if c.get('bbox')
                                         and c['bbox'][0] <= action['x'] <= c['bbox'][2]
                                         and c['bbox'][1] <= action['y'] <= c['bbox'][3]), None)
                    if selected:
                        self._last_ui = self._ui_key(state, selected)
                elif source in {'local_template', 'orb'}:
                    # A known scene with no usable target must eventually reach semantic recovery.
                    self._blocked_templates.add(raw['evidence'])
                    self._deferred = self.adaptive['reobserve_steps']
        # Count unresolved semantic observations, not every API call. A usable B
        # or UI action is normal routing; repeated ineffective actions are still
        # bounded by the independent no-progress/UI/retry limits.
        if source == 'semantic' and not progress and not stop:
            usable_action = action and action.get('type') in {'tap', 'swipe'}
            if not usable_action:
                self._recovery_calls += 1
        if stop:
            route = 'recovery_exhausted'
            action = {'type': 'noop', 'reason': self._stop_reason or 'recovery exhausted'}
        if action is None and source == 'deferred':
            action = self._short_wait(
                f'Local reobservation {self._deferred}/{self.adaptive["reobserve_steps"]}; '
                'semantic classifier not called')
        action = action or self._short_wait(
            'Classifier: animation; reobserve' if mode == 'L'
            else 'Classifier: unrecognized screen; reobserve')
        local_ms = int((time.perf_counter()-local_started)*1000)
        self._pixels = pixels
        self._observed_signature = observed_signature
        self._last_state = mode
        self._last_action = dict(action)
        if action['type'] in {'tap', 'swipe'}:
            # A new input starts a new animation window, including local moves.
            self._deferred = 0
            self._pending_effect = {
                'action': dict(action), 'mode': mode,
                # Keep the last confirmed board through intermittent detection
                # failures; a fallback swipe must not erase its comparison base.
                'signature': (signature or self._last_signature) if mode == 'B' else None,
                'pixels': pixels.copy(), 'executed': False, 'frame_id': req.frame_id,
                'ui_key': self._last_ui,
                'role': self._last_ui[2] if self._last_ui else None,
                'retry_stage': (str((state.get('stage') or {}).get('ordinal', 'unknown'))
                                if self._last_ui and self._last_ui[2] == 'retry' else None),
                'template': raw.get('evidence') if source in {'local_template', 'orb'} else None,
            }
        if signature:
            self._last_signature = signature
        self._pending_interactions.clear()  # No multi-call legacy outcome/rule pipeline.
        self._active_screen = None
        elapsed = int((time.perf_counter()-started)*1000)
        annotation = {
            'schema_version': 2, 'taxonomy_version': 'SBPCTLRU-v3',
            'state': mode, 'classification': state, 'raw_classification': raw,
            'classification_source': source, 'route': route,
            'action_origin': ('vlm_fallback' if route == 'board_single_call' else
                              'local_solver' if route in {'generated_solver', 'local_board_solver', 'board_solver'} else route),
            'fallback_reason': fallback_reason,
            'stage_progress': deepcopy(self._stage_progress),
            'calls': self._screen_calls, 'model_call_count': len(self._screen_calls),
            'hosted_call_count': sum(c['model'].startswith(('openai:', 'anthropic:', 'gemini:'))
                                     for c in self._screen_calls),
            'grounding_calls': self._grounding_calls, 'grounding_call_count': len(self._grounding_calls),
            'unknown_recovery': {'calls_without_progress': self._recovery_calls,
                                 'unresolved_semantic_calls': self._recovery_calls,
                                 'no_progress_steps': self._idle_steps,
                                 'no_progress_actions': self._idle_steps,
                                 'max_calls': self.adaptive['max_recovery_calls']},
            'board_progress': {'moves_remaining': moves_remaining,
                               'last_known_moves_remaining': self._last_moves_remaining,
                               'swipes_since_counter_read': self._swipes_since_moves},
            'stability': {'whole_screen': stable, 'local_grid': grid_stable},
            'fast_screen_ms': fast_screen_ms,
            'orb_state_hint': orb_state_hint,
            'retry_counts': dict(self._retry_counts), 'stop_requested': stop,
            'stop_reason': self._stop_reason, 'errors': errors,
            'gui_rule_events': list(self.gui_rules.events) if self.gui_rules else [],
            'total_ms': elapsed, 'routing_wall_ms': local_ms,
        }
        return {'action': action, 'intent': action.get('reason', route),
                'confidence': state['confidence'], 'observation_summary': state['evidence'],
                'image_fingerprint': image_fingerprint, 'board_signature': signature,
                'model_name': 'adaptive_screen_router_0916', 'latency_ms': elapsed,
                'pipeline': {'screen_routing': annotation, 'previous_outcome': previous_outcome,
                             'perception': {'summary': state['evidence'],
                                            'visible_text': state.get('visible_text', '')},
                             'plan': {'screen_mode': mode, 'route': route}}}
