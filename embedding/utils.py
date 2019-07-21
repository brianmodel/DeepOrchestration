import music21
from music21.interval import Interval
from music21.pitch import Pitch
import numpy as np

from Orchestration.midi.write_midi import write_midi
from embedding import note_mapping


def tokenize(file):
    transposed = transpose(file)
    transposed = transposed.chordify()
    tokens = stream_to_tokens(transposed)
    return tokens


def transpose(file):
    score = music21.converter.parse(file)
    key = score.analyze("key")

    if key.mode == "minor":
        i = Interval(key.tonic, Pitch("A"))
    else:
        i = Interval(key.tonic, Pitch("C"))

    transposed = score.transpose(i)
    return transposed


def stream_to_pr(stream):
    pr = np.empty((0, 128))
    for element in stream.notes:
        quant = np.zeros((1, 128))
        if isinstance(element, music21.chord.Chord):
            for note in element:
                quant[0][note_to_index(str(note.pitch))] = 1
        else:
            quant[0][note_to_index(str(element.pitch))] = 1
        if len(pr) == 0 or np.count_nonzero(quant) != 0 and (pr[-1] != quant).any():
            pr = np.append(pr, quant, axis=0)
    return pr


def add_to_corpus(stream, corpus):
    """
    Adding the string representation of the chord: i.e. C3G3
    """
    for element in stream.notes:
        quant = ""
        if isinstance(element, music21.chord.Chord):
            for note in element:
                quant += str(note.pitch)
        else:
            quant += str(element.pitch)
        if quant != "":
            corpus.add(quant)


def stream_to_tokens(stream):
    tokens = []
    for element in stream.notes:
        quant = ""
        if isinstance(element, music21.chord.Chord):
            for note in element:
                mapped = str(note_mapping.get(str(note.pitch)[:-1], -1))
                if mapped not in quant:
                    quant += mapped + " "
                # if str(note.pitch)[:-1] not in quant:
                #     quant += str(note.pitch)[:-1]
        else:
            quant += str(note_mapping.get(str(note.pitch)[:-1], -1)) + " "
            # quant += str(element.pitch)
        if quant != "":
            tokens.append(quant.strip())
            # tokens.append(quant)
    return tokens


def note_to_index(note):
    mapping = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    base = mapping[note[0]]
    scalar = int(note[-1])
    index = (scalar - 1) * 12 + base + 24
    if len(note) == 3:
        if note[1] == "-":
            index -= 1
        elif note[1] == "#":
            index += 1
    return index


def index_to_note(index):
    pass


def embedded(model, chord):
    if in_vocab(model, chord):
        return model.wv[chord]
    else:
        return model.wv.vectors.mean(0)


def in_vocab(model, word):
    if word in model:
        return True

    return False
