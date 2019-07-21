from embedding.utils import transpose, stream_to_pr, tokenize, embedded, in_vocab
import glob
from pathlib import Path

from embedding.data import DataIterator
from gensim.models.word2vec import Word2Vec
import multiprocessing

# data = DataIterator(
#     "/Users/brianmodel/Downloads/130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]/Classical_Piano_piano-midi.de_MIDIRip"
# )

data = DataIterator(
    "/home/brian/data/130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]/Classical_Piano_piano-midi.de_MIDIRip/"
)


print("BEFORE")
model = Word2Vec(
    sentences=data, size=300, window=5, negative=10, workers=multiprocessing.cpu_count()
)
print("AFTER")
model.save("word2vec_nooctive_enharmonic.model")

# model = Word2Vec.load("word2vec_nooctive.model")
