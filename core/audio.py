import time
import fluidsynth

NOTES = {"do": 60, "re": 62, "mi": 64, "fa": 65, "so": 67, "la": 69, "ti": 71}

class NotePlayer:
    def __init__(self):
        self.fs = fluidsynth.Synth()
        self.fs.start()

        self.sfid = self.fs.sfload("assets/soundfonts/198_Juno106_LeadSynth.sf2")
        self.fs.program_select(0, self.sfid, 0, 0)

        self.current_note = None

    def play_note(self, note):
        midi = NOTES.get(note)
        if midi is None:
            return

        # Already playing this note
        if self.current_note == midi:
            return

        # Stop previous note
        if self.current_note is not None:
            self.fs.noteoff(0, self.current_note)

        # Start new note
        self.fs.noteon(0, midi, 100)
        self.current_note = midi

    def stop_note(self):
        if self.current_note is not None:
            self.fs.noteoff(0, self.current_note)
            self.current_note = None

    def stop_synth(self):
        self.stop_note()
        self.fs.delete()
