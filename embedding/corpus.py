import sys

sys.path.append("/Users/brianmodel/Desktop/gatech/VIP/DeepOrchestration")
import glob
from pathlib import Path
from embedding import base_path
from gensim import corpora
from embedding.utils import add_to_corpus, transpose


class Corpus:
    def __iter__(self):
        for path in Path(base_path).glob("**/*.mid"):
            pass

    @staticmethod
    def create_corpus():
        corpus = set()
        # for path in Path(
        #     base_path
        #     + "/Classical Archives - The Greats (MIDI)/Bach/Bwv001- 400 Chorales/Bwv768 Chorale and Variations"
        # ).glob("**/*.mid"):
        path = "/Users/brianmodel/Downloads/130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]/Classical Archives - The Greats (MIDI)/Bach/Bwv001- 400 Chorales/Bwv768 Chorale and Variations/bsgjg_a.mid"
        song = transpose(path)
        song = song.chordify()
        add_to_corpus(song, corpus)
        corpus = corpora.Dictionary([list(corpus)])
        print(corpus.token2id)
        corpus.save("tmp.dict")
        return corpus

    @staticmethod
    def read_corpus(path):
        pass


Corpus.create_corpus()
