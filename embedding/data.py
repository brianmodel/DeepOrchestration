# from Orchestration.midi.read_midi import Read_midi
# from embedding import path
# from embedding.utils import transpose, stream_to_pr

import glob
from pathlib import Path
from embedding.utils import tokenize


class DataIterator:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for song in Path(self.path).glob("**/*.mid"):
            yield tokenize(song)


# data = DataIterator(
#     "/Users/brianmodel/Downloads/130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]/Classical_Piano_piano-midi.de_MIDIRip"
# )

