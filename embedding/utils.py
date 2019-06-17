import music21
from music21.interval import Interval
from music21.pitch import Pitch


def transpose(file):
    score = music21.converter.parse(file)
    key = score.analyze("key")

    if key.mode == "minor":
        i = Interval(key.tonic, Pitch("A"))
    else:
        i = Interval(key.tonic, Pitch("C"))

    transposed = score.transpose(i)
    for part in transposed.parts:
        print(part)
    # transposed = transposed.chordify()
    transposed.write("midi", "test.mid")
    # print(transposed.__dict__)
    # for element in transposed.notesAndRests:
    #     print(element)
