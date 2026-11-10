"""Regression tests for conversational orchestration without audio hardware."""
import subprocess
import sys
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from adapters.dispatcher import AdapterDispatcher
from adapters.grok_adapter import GrokAdapter
from adapters.guaardvark_adapter import GuaardvarkAdapter
from core.audio_recorder import AudioRecorder
from core.conversation import Conversation
from core.council import run_council
from core.diagnostics import validate_config
from core.state_manager import StateManager
from core.turn_context import TurnContext
from core.tts_engine import TTSEngine
from terminal.tmux_bridge import TmuxBridge


class StateTests(unittest.TestCase):
    def test_followup_starts_after_long_turn(self):
        with patch('core.state_manager.time.monotonic', return_value=100) as clock:
            state = StateManager()
            state.activate_assistant('grok')
            state.begin_turn()
            clock.return_value = 200
            self.assertTrue(state.is_in_active_window())
            state.finish_turn()
            clock.return_value = 229
            self.assertTrue(state.is_in_active_window())
            clock.return_value = 231
            self.assertFalse(state.is_in_active_window())

    def test_history_is_bounded(self):
        state = StateManager()
        for n in range(110):
            state.record_turn('grok', str(n), 'reply')
        self.assertEqual(len(state.conversation_history), 100)
        self.assertEqual(state.conversation_history[0]['prompt'], '10')


class AudioTests(unittest.TestCase):
    def test_barge_in_preserves_onset_and_next_frame(self):
        tts = Mock(is_playing=True)
        recorder = AudioRecorder(tts_engine=tts)
        audio = np.full((800, 1), 0.1, dtype=np.float32)
        with patch('core.audio_recorder.time.monotonic', side_effect=[1, 1.05, 1.1, 1.15]):
            for _ in range(3):
                recorder._audio_callback(audio, 800, None, None)
            self.assertTrue(recorder._interrupt.is_set())
            tts.stop.assert_not_called()  # callback only signals the control worker
            tts.is_playing = False
            recorder._audio_callback(audio, 800, None, None)
        self.assertTrue(recorder._is_speaking)
        self.assertEqual(sum(map(len, recorder._audio_buffer)), 3200)
        self.assertEqual(recorder._ignore_until, 0)

    def test_natural_playback_tail_is_ignored(self):
        recorder = AudioRecorder(tts_engine=Mock(is_playing=False))
        recorder._tts_was_playing = True
        with patch('core.audio_recorder.time.monotonic', return_value=1):
            recorder._audio_callback(np.ones((800, 1)), 800, None, None)
        self.assertFalse(recorder._is_speaking)
        self.assertEqual(recorder._ignore_until, 1.4)

    def test_queue_drops_oldest(self):
        recorder = AudioRecorder()
        for n in range(5):
            recorder._audio_buffer = [np.array([n])]
            recorder._emit_utterance()
        self.assertEqual(recorder._utterance_queue.qsize(), 3)
        self.assertEqual(recorder._utterance_queue.get()[1][0], 2)

    def test_rejects_non_whisper_sample_rate(self):
        with self.assertRaisesRegex(ValueError, '16000'):
            AudioRecorder(sample_rate=48000)


class PlaybackTests(unittest.TestCase):
    def test_cancellation_during_synthesis_prevents_playback(self):
        engine = TTSEngine()
        context = TurnContext()
        voice = Mock()
        def synthesize(*args, **kwargs):
            context.cancel()
            yield Mock(audio_float_array=np.zeros(100), sample_rate=16000)
        voice.synthesize.side_effect = synthesize
        with patch.object(engine, 'get_or_load_voice', return_value=voice), \
             patch('core.tts_engine.sd.OutputStream') as output:
            engine.speak('Hello there.', context=context)
            output.assert_not_called()
        self.assertFalse(engine.is_playing)

    def test_cancelled_turn_does_not_record_history(self):
        state = StateManager()
        conversation = Conversation(state, Mock(), Mock())
        context = TurnContext()
        context.cancel()
        try:
            self.assertFalse(conversation.record_turn('grok', 'hello', 'late', context))
            self.assertEqual(state.conversation_history, [])
        finally:
            conversation.close()


class CouncilTests(unittest.TestCase):
    def test_parallel_context_snapshotted_and_completion_order(self):
        release = threading.Event()
        snapshots = []
        replies = []
        def transcript(member):
            snapshots.append(member)
            return 'before'
        def dispatch(member, prompt):
            self.assertEqual(snapshots, ['slow', 'fast'])
            self.assertIn('before', prompt)
            if member == 'slow':
                self.assertTrue(release.wait(2))
            return member, 'voice', 0
        def reply(member, *args):
            replies.append(member)
            if member == 'fast':
                release.set()
        run_council('hello', ['slow', 'fast'], {}, dispatch, transcript, on_reply=reply)
        self.assertEqual(replies, ['fast', 'slow'])

    def test_member_order_and_sequential_context(self):
        seen = []
        def dispatch(member, prompt):
            seen.append(prompt)
            return member, 'voice', 0
        run_council('hello', ['a', 'b'], {}, dispatch, lambda _: str(len(seen)),
                    parallel=False)
        self.assertIn('\n0\n', seen[0])
        self.assertIn('\n1\n', seen[1])
        results = run_council('hello', ['a', 'b'], {}, dispatch, lambda _: '', reply_order='member')
        self.assertEqual([r[0] for r in results], ['a', 'b'])

    def test_cancel_stops_remaining_replies(self):
        context = TurnContext()
        heard = []
        def reply(member, *args):
            heard.append(member)
            context.cancel()
        run_council('hello', ['a', 'b'], {}, lambda *a, **kw: ('reply', 'voice', 0),
                    lambda _: '', parallel=False, context=context, on_reply=reply)
        self.assertEqual(heard, ['a'])


class AdapterTests(unittest.TestCase):
    def test_guaardvark_readonly_does_not_contact_backend(self):
        adapter = GuaardvarkAdapter({})
        with patch('adapters.guaardvark_adapter.requests.post') as post:
            self.assertIn('read-only', adapter.execute('do something'))
            post.assert_not_called()

    def test_cancelled_http_response_discarded(self):
        adapter = GuaardvarkAdapter({'permissions': 'full'})
        context = TurnContext()
        def post(*args, **kwargs):
            context.cancel()
            return Mock(status_code=200, json=lambda: {'response': 'late'})
        with patch('adapters.guaardvark_adapter.requests.post', side_effect=post):
            self.assertEqual(adapter.execute('hello', context=context), '')

    def test_cancelled_dispatch_does_not_execute(self):
        dispatcher = AdapterDispatcher({'grok': {}})
        context = TurnContext()
        context.cancel()
        with patch.object(dispatcher.adapters['grok'], 'execute') as execute:
            self.assertEqual(dispatcher.dispatch('grok', 'hello', context=context)[0], '')
            execute.assert_not_called()

    def test_process_group_terminated_on_cancellation(self):
        adapter = GrokAdapter({})
        context = TurnContext()
        proc = Mock(pid=123)
        def communicate(**kwargs):
            if not context.cancelled.is_set():
                context.cancel()
                raise subprocess.TimeoutExpired('test', 0.1)
            return '', ''
        proc.communicate.side_effect = communicate
        with patch('adapters.base_adapter.subprocess.Popen', return_value=proc), \
             patch('adapters.base_adapter.os.killpg') as kill:
            self.assertEqual(adapter.run_subprocess(['fake'], context=context), '')
            kill.assert_called_once()

    def test_real_subprocess_output_and_timeout(self):
        adapter = GrokAdapter({})
        self.assertEqual(adapter.run_subprocess([sys.executable, '-c', 'print("hello")']), 'hello')
        self.assertIn('too long', adapter.run_subprocess(
            [sys.executable, '-c', 'import time; time.sleep(10)'], timeout=0.05))


class TmuxTests(unittest.TestCase):
    def test_unverified_session_rejected(self):
        tmux = TmuxBridge()
        tmux.tmux_bin = 'tmux'
        with patch.object(tmux, 'session_exists', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'permissions'):
                tmux.create_cockpit_session({})

    def test_stable_panes_workspace_permissions_and_literal_input(self):
        tmux = TmuxBridge()
        tmux.tmux_bin = 'tmux'
        adapter = GrokAdapter({'workspace': '/tmp'})
        with patch.object(tmux, 'session_exists', return_value=False), \
             patch.object(tmux, '_run', return_value=Mock(stdout='%42\n')) as run:
            tmux.create_cockpit_session({'grok': adapter, 'alias': adapter})
            launch = run.call_args_list[0].args
            self.assertIn('/tmp', launch)
            self.assertIn('--deny Bash', launch[-1])
            self.assertTrue(tmux.send_prompt('alias', 'C-c'))
            self.assertIn(unittest.mock.call('send-keys', '-t', '%42', '-l', '--', 'C-c'), run.call_args_list)
            self.assertFalse(tmux.send_prompt('team', 'hello'))

    def test_failed_send_not_acknowledged(self):
        tmux = TmuxBridge()
        tmux.panes['grok'] = '%1'
        with patch.object(tmux, '_run', side_effect=subprocess.CalledProcessError(1, 'tmux')):
            self.assertFalse(tmux.send_prompt('grok', 'hello'))


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.state = StateManager(default_assistant='grok')
        self.state.activate_assistant('grok')
        self.dispatcher = AdapterDispatcher({'grok': {}})
        self.tts = Mock()
        self.conversation = Conversation(self.state, self.tts, self.dispatcher)

    def tearDown(self):
        self.conversation.close()
        self.conversation.executor.shutdown(wait=True)

    def test_stop_cancels_active_backend_and_sleep_clears_routing(self):
        entered, exited = threading.Event(), threading.Event()
        context = TurnContext()
        def work(ctx):
            entered.set()
            ctx.cancelled.wait(2)
            exited.set()
        self.conversation.submit(context, work)
        self.assertTrue(entered.wait(2))
        self.assertTrue(self.conversation.control('stop', 'grok'))
        self.assertTrue(exited.wait(2))
        self.assertTrue(context.cancelled.is_set())
        self.assertTrue(self.conversation.control('go to sleep', 'grok'))
        self.assertIsNone(self.state.active_assistant)

    def test_new_conversation_resets_session_and_history(self):
        adapter = self.dispatcher.adapters['grok']
        adapter.session_id = 'old'
        self.state.record_turn('grok', 'hello', 'reply')
        self.conversation.control('new conversation', 'grok')
        self.conversation.executor.submit(lambda: None).result(2)
        self.assertIsNone(adapter.session_id)
        self.assertEqual(self.state.conversation_history, [])

    def test_repeat_uses_last_spoken_reply(self):
        self.conversation.last_reply = ('hello', 'voice', 2)
        self.conversation.control('repeat that', 'grok')
        self.conversation.future.result(2)
        self.tts.speak.assert_called_once()
        self.assertEqual(self.tts.speak.call_args.args, ('hello',))

    def test_invalid_config(self):
        with self.assertRaisesRegex(ValueError, '16000'):
            validate_config({'audio': {'sample_rate': 48000}})
