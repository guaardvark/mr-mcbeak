"""The transcript exporter: reading the log, building the script, writing each format."""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import transcript
from core.speech_stream import SentenceFeed

LOG = """\
2026-01-31 20:59:58 [INFO] Speaking (claude): 'Left over from a session that rotated away.'
2026-01-31 21:00:00 [INFO] Status endpoint on http://127.0.0.1:8214/health
2026-01-31 21:00:00 [INFO] Starting Mr. McBeak...
2026-01-31 21:00:00 [INFO] Registered assistant adapter: Claude Code (claude)
2026-01-31 21:00:00 [INFO] Registered assistant adapter: Grok (grok)
2026-01-31 21:00:00 [INFO] Registered assistant adapter: Cursor Agent (cursor)
2026-01-31 21:00:10 [INFO] STT clip: samples=29600 dur=1.85s rms=0.0868 peak=0.8000
2026-01-31 21:00:10 [INFO] Tone: happy 0.76, angry 0.18, neutral 0.03
2026-01-31 21:00:10 [INFO] turn=aaa stage=transcription elapsed=0.303s
2026-01-31 21:00:10 [INFO] 🎙️  Heard (0.30s): "Claude, what is "two" plus two?"
2026-01-31 21:00:10 [INFO] 🎯 Matched Target: [CLAUDE] (Similarity: 1.00)
2026-01-31 21:00:10 [INFO] Dispatching prompt to [Claude Code]: 'what is "two" plus two?...'
2026-01-31 21:00:12 [INFO] Speaking (claude) as the reply arrives
2026-01-31 21:00:13 [INFO] turn=aaa stage=first_audio elapsed=3.270s
2026-01-31 21:00:15 [INFO] Spoke (claude) as the reply arrived: "It's four. Ask me another."
2026-01-31 21:00:20 [INFO] STT clip: samples=1 dur=1.00s rms=0.1 peak=0.8
2026-01-31 21:00:20 [INFO] 🎙️  Heard (0.10s): "Did you see the game last night?"
2026-01-31 21:00:20 [INFO] Tone: neutral 0.90, sad 0.05, happy 0.01
2026-01-31 21:00:30 [INFO] STT clip: samples=1 dur=1.00s rms=0.1 peak=0.8
2026-01-31 21:00:30 [INFO] Tone: angry 0.98, neutral 0.01, happy 0.00
2026-01-31 21:00:30 [INFO] 🎙️  Heard (0.10s): "Hey team."
2026-01-31 21:00:30 [INFO] 🎯 Matched Target: [TEAM] (Similarity: 1.00)
2026-01-31 21:00:31 [INFO] team listening (30s follow-up window, no spoken ack)
2026-01-31 21:00:34 [INFO] STT clip: samples=1 dur=1.00s rms=0.1 peak=0.8
2026-01-31 21:00:34 [INFO] 🎙️  Heard (0.09s): "Everybody ready?"
2026-01-31 21:00:34 [INFO] Tone: surprised 0.91, neutral 0.05, happy 0.00
2026-01-31 21:00:34 [INFO] 🎯 Matched Target: [TEAM] (Similarity: 1.00)
2026-01-31 21:00:35 [INFO] Dispatching prompt to [Claude Code]: 'You are Claude Code, one voice in a spoken group conversatio...'
2026-01-31 21:00:35 [INFO] Dispatching prompt to [Grok]: 'You are Grok, one voice in a spoken group conversation with ...'
2026-01-31 21:00:35 [WARNING] [Cursor Agent] Exit code 1: Error: Authentication required.
Traceback line that carries no timestamp
2026-01-31 21:00:35 [INFO] Speaking (cursor): 'Cursor Agent could not answer and will sit out team questions until you speak to it directly.'
2026-01-31 21:00:40 [INFO] Speaking (grok): "Grok says: Ready. Mic's working."
2026-01-31 21:00:44 [INFO] Speaking (claude): 'Claude Code says: Ready.'
2026-01-31 21:00:50 [INFO] Audio recorder stopped.
2026-01-31 21:05:00 [INFO] Status endpoint on http://127.0.0.1:8214/health
2026-01-31 21:05:00 [INFO] Starting Mr. McBeak...
2026-01-31 21:05:00 [INFO] Registered assistant adapter: Claude Code (claude)
2026-01-31 21:05:05 [INFO] STT clip: samples=1 dur=1.00s rms=0.1 peak=0.8
2026-01-31 21:05:05 [INFO] 🎙️  Heard (0.10s): "Claude, are you there?"
2026-01-31 21:05:05 [INFO] 🎯 Matched Target: [CLAUDE] (Similarity: 1.00)
2026-01-31 21:05:06 [INFO] Speaking (claude) as the reply arrives
2026-01-31 21:05:09 [INFO] Speaking (Mr. McBeak): 'On the team: Claude Code.'
"""


def records():
    return [r for r in map(transcript.parse_line, LOG.splitlines()) if r]


def sessions():
    return transcript.split_sessions(records())


def first():
    return transcript.select(transcript.build_entries(sessions()[0].records))


class Reading(unittest.TestCase):
    def test_a_log_line_gives_time_level_and_message(self):
        when, level, message = transcript.parse_line("2026-01-31 21:00:00 [WARNING] something: odd\n")
        self.assertEqual((when, level, message), (datetime(2026, 1, 31, 21, 0, 0), "WARNING", "something: odd"))

    def test_a_continuation_line_is_not_a_record(self):
        self.assertIsNone(transcript.parse_line("Traceback line that carries no timestamp"))

    def test_a_session_runs_from_one_start_to_the_next(self):
        found = sessions()
        self.assertEqual([s.number for s in found], [1, 2])
        self.assertEqual(found[0].name, "Mr. McBeak")
        self.assertEqual(found[0].start, datetime(2026, 1, 31, 21, 0, 0))
        # The status line logged just before the second start belongs to the second session
        self.assertEqual(found[0].end, datetime(2026, 1, 31, 21, 0, 50))
        self.assertTrue(found[1].records[0][2].startswith("Status endpoint"))

    def test_lines_before_the_first_start_are_dropped(self):
        self.assertNotIn("Left over", " ".join(e.text for e in first()))

    def test_rotated_logs_are_read_oldest_first(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "mcbeak.log"
            for name in ("mcbeak.log.2", "mcbeak.log.1", "mcbeak.log", "mcbeak.log.bak"):
                (Path(folder) / name).write_text("x\n")
            self.assertEqual([p.name for p in transcript.log_files(log)],
                             ["mcbeak.log.2", "mcbeak.log.1", "mcbeak.log"])


class Building(unittest.TestCase):
    def test_what_was_said_keeps_its_quotes_and_names_who_it_was_to(self):
        you = first()[0]
        self.assertEqual((you.kind, you.speaker, you.to), ("you", "You", "Claude Code"))
        self.assertEqual(you.text, 'Claude, what is "two" plus two?')
        self.assertEqual((you.tone, you.tone_score), ("happy", 0.76))

    def test_a_streamed_reply_takes_its_text_from_the_completion_line_and_its_time_from_the_start(self):
        reply = first()[1]
        self.assertEqual((reply.kind, reply.speaker, reply.text), ("reply", "Claude Code", "It's four. Ask me another."))
        self.assertEqual(reply.time, datetime(2026, 1, 31, 21, 0, 12))
        self.assertFalse(reply.interrupted)

    def test_an_interrupted_streamed_reply_is_marked(self):
        lines = [(datetime(2026, 1, 31, 21, 0, 0), "INFO", "Speaking (claude) as the reply arrives"),
                 (datetime(2026, 1, 31, 21, 0, 2), "INFO", "Spoke (claude) as the reply arrived, interrupted: 'Well,'")]
        entry, = transcript.build_entries(lines)
        self.assertEqual((entry.text, entry.interrupted, entry.speaker), ("Well,", True, "Claude"))

    def test_a_streamed_reply_from_an_older_log_becomes_a_note(self):
        entries = transcript.build_entries(sessions()[1].records)
        self.assertEqual((entries[1].kind, entries[1].speaker), ("note", "Claude Code"))
        self.assertIn("not logged", entries[1].text)

    def test_the_apps_own_speech_is_named_after_the_app(self):
        self.assertEqual(transcript.build_entries(sessions()[1].records)[2].speaker, "Mr. McBeak")

    def test_speech_to_nobody_is_left_out_unless_asked_for(self):
        self.assertNotIn("Did you see the game last night?", [e.text for e in first()])
        everything = transcript.select(transcript.build_entries(sessions()[0].records), unaddressed=True)
        game = next(e for e in everything if e.text.startswith("Did you see"))
        self.assertEqual((game.to, game.tone), ("", "neutral"))

    def test_a_name_on_its_own_joins_the_question_that_follows(self):
        question = first()[2]
        self.assertEqual(question.text, "Hey team. Everybody ready?")
        self.assertEqual(question.time, datetime(2026, 1, 31, 21, 0, 30))
        self.assertEqual((question.to, question.tone), ("Team", "surprised"))

    def test_a_team_round_groups_the_answers_under_the_question(self):
        question, failed, grok, claude = first()[2:]
        self.assertEqual({e.round for e in (question, failed, grok, claude)}, {1})
        self.assertIsNone(first()[0].round)
        self.assertEqual((failed.kind, failed.speaker, failed.text), ("note", "Cursor Agent", "could not answer"))
        self.assertEqual((grok.speaker, grok.text), ("Grok", "Ready. Mic's working."))
        self.assertEqual((claude.speaker, claude.text), ("Claude Code", "Ready."))

    def test_a_reply_cut_by_the_old_log_format_ends_in_an_ellipsis(self):
        line = (datetime(2026, 1, 31, 21, 0, 0), "INFO", f"Speaking (grok): {'a' * 120!r}")
        self.assertTrue(transcript.build_entries([line])[0].text.endswith("a…"))


class Times(unittest.TestCase):
    DAY = datetime(2026, 1, 31, 21, 0, 0)

    def test_a_clock_time_lands_on_the_sessions_day(self):
        self.assertEqual(transcript.parse_when("21:54", self.DAY), datetime(2026, 1, 31, 21, 54))
        self.assertEqual(transcript.parse_when("21:54:30", self.DAY), datetime(2026, 1, 31, 21, 54, 30))

    def test_an_iso_time_is_taken_as_written(self):
        self.assertEqual(transcript.parse_when("2026-02-01T08:05", self.DAY), datetime(2026, 2, 1, 8, 5))

    def test_anything_else_is_refused(self):
        with self.assertRaises(ValueError):
            transcript.parse_when("9pm", self.DAY)

    def test_since_and_until_bound_the_entries(self):
        kept = transcript.select(first(), since=datetime(2026, 1, 31, 21, 0, 30), until=datetime(2026, 1, 31, 21, 0, 40))
        self.assertEqual([e.speaker for e in kept], ["You", "Cursor Agent", "Grok"])


class Writing(unittest.TestCase):
    def test_markdown(self):
        text = transcript.render_markdown(first(), "Mr. McBeak", "Session 1, 2026-01-31 21:00:00 to 21:00:50")
        self.assertEqual(text.splitlines(), [
            "# Mr. McBeak transcript",
            "",
            "Session 1, 2026-01-31 21:00:00 to 21:00:50",
            "",
            '**21:00:10 You** (to Claude Code; tone: happy): Claude, what is "two" plus two?',
            "",
            "**21:00:12 Claude Code:** It's four. Ask me another.",
            "",
            "**21:00:30 You** (to Team; tone: surprised): Hey team. Everybody ready?",
            "",
            "- **21:00:35 Cursor Agent:** *could not answer*",
            "- **21:00:40 Grok:** Ready. Mic's working.",
            "- **21:00:44 Claude Code:** Ready.",
        ])

    def test_no_tone_leaves_the_tone_out(self):
        text = transcript.render_markdown(first(), "Mr. McBeak", "Session 1", tone=False)
        self.assertNotIn("tone", text)
        self.assertIn("**21:00:10 You** (to Claude Code):", text)

    def test_plain_text_indents_a_team_round(self):
        lines = transcript.render_text(first(), "Mr. McBeak", "Session 1").splitlines()
        self.assertEqual(lines[3], '[21:00:10] You (to Claude Code; tone: happy): Claude, what is "two" plus two?')
        self.assertEqual(lines[6], "    [21:00:35] Cursor Agent: (could not answer)")
        self.assertEqual(lines[7], "    [21:00:40] Grok: Ready. Mic's working.")

    def test_srt_counts_from_the_origin_and_leaves_notes_out(self):
        blocks = transcript.render_srt(first(), datetime(2026, 1, 31, 21, 0, 0)).split("\n\n")
        self.assertEqual(blocks[0], '1\n00:00:10,000 --> 00:00:12,000\nYou: Claude, what is "two" plus two?')
        self.assertEqual(blocks[1], "2\n00:00:12,000 --> 00:00:14,000\nClaude Code: It's four. Ask me another.")
        self.assertEqual(len(blocks), 5)
        self.assertNotIn("could not answer", "".join(blocks))

    def test_srt_clock_carries_hours_and_milliseconds(self):
        self.assertEqual(transcript._srt_clock(3725.5), "01:02:05,500")

    def test_json_round_trips(self):
        data = json.loads(transcript.render_json(first(), "Mr. McBeak", "Session 1"))
        self.assertEqual(data["name"], "Mr. McBeak")
        self.assertEqual(data["entries"][1], {
            "time": "2026-01-31 21:00:12", "kind": "reply", "speaker": "Claude Code",
            "text": "It's four. Ask me another.", "to": "", "tone": "", "tone_score": 0.0,
            "round": None, "interrupted": False})
        bare = json.loads(transcript.render_json(first(), "Mr. McBeak", "Session 1", tone=False))
        self.assertNotIn("tone", bare["entries"][0])


class Command(unittest.TestCase):
    def run_it(self, *argv):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "mcbeak.log"
            log.write_text(LOG, encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = transcript.main([a.replace("{dir}", folder) for a in argv], log)
            written = Path(folder) / "out.srt"
            return code, out.getvalue(), err.getvalue(), written.read_text() if written.exists() else None

    def test_the_last_session_is_the_default(self):
        code, out, _, _ = self.run_it()
        self.assertEqual(code, 0)
        self.assertIn("Session 2, 2026-01-31 21:05:00 to 21:05:09", out)
        self.assertIn("Claude, are you there?", out)

    def test_a_session_by_number_and_a_time_range(self):
        code, out, _, _ = self.run_it("--session", "1", "--since", "21:00:30", "--format", "txt", "--no-tone")
        self.assertEqual(code, 0)
        self.assertTrue(out.splitlines()[3].startswith("[21:00:30] You (to Team): Hey team."))

    def test_until_a_minute_takes_in_the_whole_minute(self):
        _, out, _, _ = self.run_it("--session", "1", "--until", "21:00", "--format", "txt")
        self.assertIn("[21:00:44] Claude Code: Ready.", out)

    def test_list_shows_every_session(self):
        code, out, _, _ = self.run_it("--session", "list")
        self.assertEqual((code, len(out.splitlines())), (0, 2))
        self.assertIn('"Claude, what is "two" plus two?"', out.splitlines()[0])

    def test_out_writes_a_file_and_srt_counts_from_since(self):
        code, out, err, written = self.run_it("--session", "1", "--format", "srt", "--since", "21:00:30",
                                              "--out", "{dir}/out.srt")
        self.assertEqual((code, out), (0, ""))
        self.assertIn("Wrote", err)
        self.assertTrue(written.startswith("1\n00:00:00,000 --> "))

    def test_an_unknown_session_or_time_exits_with_a_message(self):
        self.assertEqual(self.run_it("--session", "9")[0], 1)
        self.assertIn("not a time", self.run_it("--since", "9pm")[2])


class StreamedText(unittest.TestCase):
    def test_delivered_is_what_the_speaker_took(self):
        feed = SentenceFeed(max_sentences=3)
        feed.add("One. Two. Three is still being written")
        taken = feed.sentences()
        self.assertEqual(feed.delivered, "")
        self.assertEqual(next(taken), "One.")
        self.assertEqual(feed.delivered, "One.")
        feed.mute()
        self.assertEqual(list(taken), [])
        self.assertEqual((feed.delivered, feed.spoken), ("One.", "One. Two."))

    def test_delivered_matches_spoken_when_the_reply_plays_out(self):
        feed = SentenceFeed(max_sentences=3)
        feed.add("One. Two.\n")
        feed.close()
        self.assertEqual(list(feed.sentences()), ["One.", "Two."])
        self.assertEqual(feed.delivered, feed.spoken)


if __name__ == "__main__":
    unittest.main()
