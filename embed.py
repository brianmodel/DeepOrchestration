from embedding.utils import transpose, stream_to_pr, tokenize
import glob
from pathlib import Path

from embedding.data import DataIterator
from gensim.models.word2vec import Word2Vec


data = DataIterator(
    "/Users/brianmodel/Downloads/130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]/Classical_Piano_piano-midi.de_MIDIRip/sinding"
)

print("BEFORE")
model = Word2Vec(sentences=data, size=500, window=5)
print("AFTER")
model.save("word2vec_nooctive.model")

# model = Word2Vec.load("word2vec.model")
# print(model["G3"])
# print(model.most_similar("G3"))

