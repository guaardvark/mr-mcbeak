"""The team file, spoken team requests, control phrases and saved conversations."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mcbeak
from adapters.dispatcher import AdapterDispatcher
from adapters.grok_adapter import GrokAdapter
from core import team
from core.conversation import Conversation
from core.sessions import SessionStore

CONFIG = {
    "system": {"name": "Mr. McBeak", "default_assistant": "claude"},
    "coordinator_wake_words": ["mr mcbeak"],
    "council": {"wake_words": ["team", "hey team"], "members": ["claude", "grok"]},
    "assistants": {
        "claude": {"name": "Claude Code", "target_id": "claude", "cli_cmd": "claude", "permissions": "full",
                   "wake_words": ["claude", "hey claude"], "phonetic_variants": ["cloud"],
                   "voice": "kokoro:am_michael", "mcp": "none"},
        "grok": {"name": "Grok", "target_id": "grok", "cli_cmd": "grok",
                 "wake_words": ["grok"], "voice": "en_US-libritts-high.onnx", "voice_speaker": 3},
        "guaardvark": {"name": "Guaardvark", "target_id": "guaardvark", "wake_words": ["guaardvark"],
                       "voice": "en_US-libritts-high.onnx", "voice_speaker": 42},
    },
}


def config():
    import copy
    return copy.deepcopy(CONFIG)


class ResolveTests(unittest.TestCase):
    def test_own_name_keeps_the_kinds_identity(self):
        out = team.resolve([{"name": "Claude", "kind": "claude"}], config())
        self.assertEqual(list(out), ["claude"])
        self.assertEqual(out["claude"]["name"], "Claude Code")
        self.assertEqual(out["claude"]["voice"], "kokoro:am_michael")
        self.assertEqual(out["claude"]["wake_words"], ["claude", "hey claude"])
        self.assertEqual(out["claude"]["mcp"], "none")

    def test_new_member_is_readonly_with_its_own_words_and_voice(self):
        out = team.resolve([{"name": "Claude", "kind": "claude"}, {"name": "Ada", "kind": "claude"}], config())
        ada = out["ada"]
        self.assertEqual(ada["permissions"], "readonly")
        self.assertIn("hey ada", ada["wake_words"])
        self.assertNotEqual((ada["voice"], ada["voice_speaker"]),
                            (out["claude"]["voice"], out["claude"]["voice_speaker"]))
        self.assertNotIn("phonetic_variants", ada)

    def test_guaardvark_member_is_not_refused_as_readonly(self):
        out = team.resolve([{"name": "Guaardvark"}], config())
        self.assertEqual(out["guaardvark"]["permissions"], "full")

    def test_names_that_sound_alike_are_refused(self):
        for name in ("Grock", "Cloud", "Stop", "Team", "X"):
            with self.assertRaises(team.TeamError, msg=name):
                team.resolve([{"name": "Claude"}, {"name": "Grok"}, {"name": name, "kind": "claude"}], config())

    def test_unknown_kind_and_setting_are_refused(self):
        with self.assertRaises(team.TeamError):
            team.resolve([{"name": "Ada", "kind": "siri"}], config())
        with self.assertRaises(team.TeamError):
            team.resolve([{"name": "Ada", "kind": "claude", "permision": "full"}], config())


class TeamFileTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "team.yaml"
        self.which = patch("core.team.shutil.which", side_effect=lambda n: None if n == "grok" else f"/bin/{n}")
        self.which.start()
        self.addCleanup(self.which.stop)

    def test_without_a_file_the_team_is_what_is_installed(self):
        cfg = team.apply(config(), path=self.path)
        self.assertEqual(list(cfg["assistants"]), ["claude", "guaardvark"])
        self.assertEqual(cfg["council"]["members"], ["claude"])
        self.assertIn("grok command is not installed", cfg["team_notes"][0])

    def test_add_then_remove(self):
        entry = team.add_member(config(), "Ada", "claude", path=self.path, model="claude-haiku-5-5")
        self.assertEqual(entry["name"], "Ada")
        names = [m["name"] for m in team.read_team(self.path)]
        self.assertEqual(names, ["Claude Code", "Guaardvark", "Ada"])
        cfg = team.apply(config(), path=self.path)
        self.assertEqual(cfg["council"]["members"], ["claude", "guaardvark", "ada"])
        team.remove_member(config(), "claude", path=self.path)
        cfg = team.apply(config(), path=self.path)
        self.assertEqual(list(cfg["assistants"]), ["guaardvark", "ada"])
        self.assertEqual(cfg["system"]["default_assistant"], "guaardvark")

    def test_member_whose_program_is_missing_cannot_be_added(self):
        with self.assertRaisesRegex(team.TeamError, "not installed"):
            team.add_member(config(), "Grok", path=self.path)
        self.assertFalse(self.path.exists())

    def test_last_member_cannot_be_removed(self):
        team.write_team([{"name": "Claude", "kind": "claude"}], self.path)
        with self.assertRaises(team.TeamError):
            team.remove_member(config(), "Claude", path=self.path)


class DispatcherTests(unittest.TestCase):
    def test_two_members_of_one_kind_and_update_keeps_unchanged_adapters(self):
        members = {"grok": {"target_id": "grok"}, "rex": {"kind": "grok", "name": "Rex"}}
        dispatcher = AdapterDispatcher(members)
        first, rex = dispatcher.adapters["grok"], dispatcher.adapters["rex"]
        self.assertIsNot(first, rex)
        self.assertEqual(rex.name, "Rex")
        first.session_id = "abc"
        added, removed = dispatcher.update({"grok": {"target_id": "grok"}, "max": {"kind": "grok", "name": "Max"}})
        self.assertEqual((added, removed), (["max"], ["rex"]))
        self.assertIs(dispatcher.adapters["grok"], first)
        self.assertEqual(first.session_id, "abc")

    def test_unknown_member_is_not_sent_to_someone_else(self):
        dispatcher = AdapterDispatcher({"grok": {}})
        with patch.object(dispatcher.adapters["grok"], "execute") as execute:
            self.assertIn("no team member", dispatcher.dispatch("nobody", "hello")[0])
            execute.assert_not_called()


class SessionTests(unittest.TestCase):
    def test_saved_conversation_is_restored_for_same_kind_and_folder(self):
        path = Path(tempfile.mkdtemp()) / "sessions.json"
        adapter = GrokAdapter({})
        adapter.session_id = "abc-123"
        SessionStore(path).save("grok", adapter)
        again = GrokAdapter({})
        self.assertTrue(SessionStore(path).restore("grok", again))
        self.assertEqual(again.session_id, "abc-123")
        elsewhere = GrokAdapter({"folder": tempfile.mkdtemp()})
        self.assertFalse(SessionStore(path).restore("grok", elsewhere))
        self.assertFalse(SessionStore(path, max_age_hours=0).restore("grok", GrokAdapter({})))
        store = SessionStore(path)
        store.forget("grok")
        self.assertFalse(SessionStore(path).restore("grok", GrokAdapter({})))

    def test_stale_saved_conversation_starts_fresh(self):
        path = Path(tempfile.mkdtemp()) / "sessions.json"
        saved = GrokAdapter({})
        saved.session_id = "gone"
        SessionStore(path).save("grok", saved)
        dispatcher = AdapterDispatcher({"grok": {}}, sessions=SessionStore(path))
        adapter = dispatcher.adapters["grok"]
        calls = []

        def execute(prompt, cwd=None, context=None):
            calls.append(adapter.session_id)
            adapter.last_error = "No conversation found" if adapter.session_id else None
            return "error" if adapter.session_id else "hello"
        with patch.object(adapter, "execute", side_effect=execute):
            self.assertEqual(dispatcher.dispatch("grok", "hi")[0], "hello")
        self.assertEqual(calls, ["gone", None])


class SpokenRequestTests(unittest.TestCase):
    def test_team_requests(self):
        self.assertEqual(mcbeak.team_request("Who's on the team?"), ("who", ""))
        self.assertEqual(mcbeak.team_request("asks Codex to join."), ("join", "codex"))
        self.assertEqual(mcbeak.team_request("Please add Grok to the team"), ("join", "grok"))
        self.assertEqual(mcbeak.team_request("Ask Grok to leave."), ("leave", "grok"))
        self.assertEqual(mcbeak.team_request("remove Cursor from the team"), ("leave", "cursor"))
        self.assertIsNone(mcbeak.team_request("ask Claude to review the code"))
        self.assertIsNone(mcbeak.team_request("add a test to the team module"))

    def test_control_phrases(self):
        command = Conversation.control_command
        self.assertEqual(command("Stop."), "stop")
        self.assertEqual(command("stop please"), "stop")
        self.assertEqual(command("Okay, please stop talking now."), "stop")
        self.assertEqual(command("say that again"), "repeat that")
        self.assertEqual(command("start over"), "new conversation")
        self.assertIsNone(command("stop the server"))
        self.assertIsNone(command("please"))
        self.assertIsNone(command("repeat the last command in the terminal"))


if __name__ == "__main__":
    unittest.main()
