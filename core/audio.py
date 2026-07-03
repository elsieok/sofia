import time
import fluidsynth

NOTES = {"do": 60, "re": 62, "mi": 64, "fa": 65, "so": 67, "la": 69, "ti": 71}

class NotePlayer:
    def __init__(self):
        self.fs = fluidsynth.Synth()
        self.fs.start()

        self.sfid = self.fs.sfload("assets/soundfonts/198_Juno106_LeadSynth.sf2")

        # Use TWO separate MIDI channels
        self.left_channel = 0
        self.right_channel = 1

        self.fs.program_select(self.left_channel, self.sfid, 0, 0)
        self.fs.program_select(self.right_channel, self.sfid, 0, 0)

        self.active_notes = {
            "left": None,
            "right": None
        }

    def play_note(self, hand, note):
        midi = NOTES.get(note)
        if midi is None:
            return

        channel = self.left_channel if hand == "left" else self.right_channel

        # stop only that hand's note
        if self.active_notes[hand] == midi:
            return

        if self.active_notes[hand] is not None:
            self.fs.noteoff(channel, self.active_notes[hand])

        self.fs.noteon(channel, midi, 100)
        self.active_notes[hand] = midi

    def stop_note(self, hand):
        channel = self.left_channel if hand == "left" else self.right_channel

        if self.active_notes[hand] is not None:
            self.fs.noteoff(channel, self.active_notes[hand])
            self.active_notes[hand] = None

    def stop_synth(self):
        self.stop_note("left")
        self.stop_note("right")
        self.fs.delete()