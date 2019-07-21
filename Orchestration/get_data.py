import os
import pickle
import numpy as np

import torch

from Orchestration.midi.read_midi import Read_midi
from Orchestration.midi.write_midi import write_midi
from Orchestration import data_path, base_path, inst_mapping

from embedding.utils import transpose
from pathlib import Path
import music21
from music21.interval import Interval
from music21.pitch import Pitch
from scipy.ndimage.interpolation import shift
from gensim.models import Word2Vec
from embedding.utils import embedded

def get_train_data(source="bouliane_aligned", fix=True):
    """Method for getting training data for machine learning
    
    Keyword Arguments:
        source {str} -- which folder you want the data from (default: {"bouliane_aligned"})
    
    Returns:
        dict -- the training data
    """

    cashe = os.path.join(base_path, "Orchestration/cashe/" + source)
    X = []
    y = []
    for point in os.listdir(cashe):
        point_path = os.path.join(cashe, point)
        if point_path[-9:] == ".DS_Store":
            continue
        for score in os.listdir(point_path):
            with open(os.path.join(point_path, score), "rb") as f:
                part = pickle.load(f)
                # Have to correct for some piano scores having more than 1 piano part
                if "solo" in os.path.join(point_path, score):
                    total = None
                    for key in part:
                        if total is None:
                            total = part[key]
                        else:
                            total = np.add(total, part[key])
                    part = {"Kboard": total}
                    X.append(total)
                else:
                    corrected_part = {}
                    for inst in part:
                        if inst in inst_mapping:
                            correct_inst = inst_mapping[inst]
                            if correct_inst in corrected_part:
                                corrected_part[correct_inst] = np.add(part[inst], corrected_part[correct_inst])
                                indices = corrected_part[correct_inst] > 127
                                corrected_part[correct_inst][indices] = 127
                            else:
                                corrected_part[correct_inst] = part[inst]
                    y.append(corrected_part)
    # Fixing the imperfections in the data
    for i in range(len(X)):
        inst = y[i][list(y[i].keys())[0]]
        if len(X[i]) != len(inst):
            diff = abs(len(inst) - len(X[i]))
            if len(X[i]) < len(inst):
                for j in range(diff):
                    X[i] = np.append(X[i], [X[i][-1]], axis=0)
            else:
                for j in range(diff):
                    X[i] = X[i][:-1, :]
    if fix:
        add_instruments(y)
    return X, y
 

def get_str_data(source='bouliane_aligned'):
    X = []
    y = []
    path = os.path.join(base_path, "data/" + source)
    embedding = Word2Vec.load("/Users/brianmodel/Desktop/gatech/VIP/DeepOrchestration/word2vec_nooctive_enharmonic.model")

    for point in os.listdir(path):
        point_path = os.path.join(source, point)
        if point_path[-9:] == ".DS_Store":
            continue
        songs = []
        for song in Path(path+"/"+point).glob("*.mid"):
            songs.append(str(song))
        if "solo" in songs[0]:
            solo = songs[0]
            orch = songs[1]
        else:
            solo = songs[1]
            orch = songs[0]

        solo_score = music21.converter.parse(solo)
        key = solo_score.analyze("key")

        if key.mode == "minor":
            i = Interval(key.tonic, Pitch("A"))
        else:
            i = Interval(key.tonic, Pitch("C"))
        semitones = i.chromatic.semitones
        solo_data = Read_midi(
            solo, 8
        ).read_file()

        # Correct the data
        total = None
        for key in solo_data:
            if total is None:
                total = solo_data[key]
            else:
                total = np.add(total, solo_data[key])
        
        if semitones != 0:
            total = transpose_solo_pr(total, semitones)
        tokens = pr_to_tokens(total)

        orch_data = Read_midi(
            orch, 8
        ).read_file()

        orch_data = fix_orch_pr(orch_data)
        if semitones != 0:
            orch_data = tranpose_orch_pr(orch_data, semitones)
        # orch_pr_to_tokens(orch_data)
        y.append(orch_data)

        # Make sure the lengths are the same
        for i in range(len(tokens)):
            inst = orch_data[list(orch_data.keys())[0]]
            if len(tokens) != len(inst):
                diff = abs(len(inst) - len(tokens))
                if len(tokens) < len(inst):
                    for j in range(diff):
                        tokens.append(tokens[-1])
                else:
                    tokens = tokens[:len(tokens)-diff]

        # Convert to embedded vectors
        embedded_tokens = np.empty((0, 300))
        for i in range(len(tokens)):
            embedded_tokens = np.append(embedded_tokens, embedded(embedding, tokens[i]).reshape((1, 300)), axis=0)
        embedded_tokens = embedded_tokens.reshape(embedded_tokens.shape[0], 1, 300)
        X.append(embedded_tokens)
    return X, y

def transpose_solo_pr(pr, semitones):
    for i in range(len(pr)):
        pr[i] = shift(pr[i], semitones)
    return pr

def fix_orch_pr(pr):
    corrected_part = {}
    for inst in pr:
        if inst in inst_mapping:
            correct_inst = inst_mapping[inst]
            if correct_inst in corrected_part:
                corrected_part[correct_inst] = np.add(pr[inst], corrected_part[correct_inst])
                indices = corrected_part[correct_inst] > 127
                corrected_part[correct_inst][indices] = 127
            else:
                corrected_part[correct_inst] = pr[inst]
    return corrected_part

def tranpose_orch_pr(pr, semitones):
    for inst in pr:
        pr[inst] = transpose_solo_pr(pr[inst], semitones)
    return pr

def pr_to_tokens(pr):
    tokens = []
    for quant in pr:
        chord = ""
        notes = np.nonzero(quant)
        for note in notes[0]:
            val = str(int(note)%12)
            if val not in chord:
                chord += str(int(note)%12) + ' '
        tokens.append(chord.strip())
    return tokens

def orch_pr_to_tokens(pr):
    for inst in pr:
        pr[inst] = pr_to_tokens(pr[inst])

def add_instruments(y):
    instruments = set()
    for orch in y:
        for inst in orch.keys():
            instruments.add(inst)
    instruments = list(instruments)
    for i in range(len(y)):
        for inst in instruments:
            if inst not in y[i]:
                shape = y[i][list(y[i].keys())[0]].shape
                y[i][inst] = np.zeros(shape)

def filter(data, n_lines=None):
    data = data.astype(int)
    if n_lines == None:
        for i in range(len(data)):
            for j in range(len(data[i])):
                if data[i][j] > 127:
                    data[i][j] = 127
                elif data[i][j] < 0:
                    data[i][j] = 0
    else:
        for i in range(len(data)):
            argmax = data[i].argsort()[-1*n_lines:][::-1]
            vals = {}
            for j in argmax:
                vals[j] = data[i][j]
            data[i] = np.zeros(128)
            for j in vals.keys():
                data[i][j] = vals[j]
                if data[i][j] > 127:
                    data[i][j] = 127
                elif data[i][j] < 10:
                    data[i][j] = 0
    return data

def inst_to_midi(data, inst):
    data = filter(data, 2)
    # with open('temp.txt', 'a') as f:
    #     f.write("BEGIN")
    #     for line in data:
    #         f.write(str(line))
    #     f.write("END")
    output_path=os.path.join(base_path, base_path + "/Orchestration/out/{}.mid".format(inst))
    write_midi({inst: data}, 8, output_path)

def orch_to_midi(data):
    output_path=os.path.join(base_path, base_path + "/Orchestration/out/orch.mid")
    write_midi(data, 8, output_path)

def piano_to_midi(data, name='piano'):
    data = filter(data)
    output_path=os.path.join(base_path, base_path + "/Orchestration/out/{}.mid".format(name))
    write_midi({"Kboard": data}, 8, output_path)


def cashe_data(path):
    """Method that cashes all the parsed midi files from a certain
    directory in the data set, and stores it in the similarly structured
    directory called cashe
    
    Arguments:
        path {str} -- path to directory that contains a single data set
    """

    quantization = 8
    set_name = path.split("/")[-1]
    cashe = os.path.join(base_path, "Orchestration/cashe")
    if not os.path.exists(cashe):
        os.mkdir(cashe)

    cashed_set_dir = os.path.join(cashe, set_name)
    if not os.path.exists(cashed_set_dir):
        os.mkdir(cashed_set_dir)

    for sample in os.listdir(path):
        if sample == ".DS_Store":
            continue
        sample_path = os.path.join(path, sample)
        for file in os.listdir(sample_path):
            if file[-4:] == ".mid":
                data = Read_midi(
                    os.path.join(sample_path, file), quantization
                ).read_file()
                if not os.path.exists(os.path.join(cashed_set_dir, sample)):
                    os.mkdir(os.path.join(cashed_set_dir, sample))
                with open(
                    os.path.join(cashed_set_dir, sample + "/" + file[:-4]), "wb"
                ) as handle:
                    pickle.dump(data, handle)


def cashe_order(order):
    """Method to cashe the order of instruments in orchestration
    
    Arguments:
        order {list} -- order of instruments in orchestration
    """

    with open(os.path.join(base_path, "Orchestration/cashe/order"), "wb") as handle:
        pickle.dump(order, handle)


def read_order():
    """Helper method to get the order of instruments
    
    Returns:
        list -- order of the instruments
    """

    with open(os.path.join(base_path, "Orchestration/cashe/order"), "rb") as handle:
        order = pickle.load(handle)
    return order
