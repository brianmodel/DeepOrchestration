import os
import pickle
import numpy as np

import torch

from Orchestration.midi.read_midi import Read_midi
from Orchestration.midi.write_midi import write_midi
from Orchestration import data_path, base_path


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
                    y.append(part)
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
