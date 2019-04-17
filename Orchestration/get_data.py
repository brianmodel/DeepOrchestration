import os
import pickle
import numpy as np

import torch

from Orchestration.midi.read_midi import Read_midi
from Orchestration.midi.write_midi import write_midi
from Orchestration import data_path, base_path


def get_train_data(source="bouliane_aligned"):
    """Method for getting training data for machine learning

    Keyword Arguments:
        source {str} -- which folder you want the data from (default: {"bouliane_aligned"})

    Returns:
        dict -- the training data
    """

    if os.path.isfile(os.path.join(base_path, "Orchestration/cashe/data")):
        with open(os.path.join(base_path, "Orchestration/cashe/data"), "rb") as handle:
            data = pickle.load(handle)
        return data

    else:
        cashe = os.path.join(base_path, "Orchestration/cashe/" + source)
        X = []
        y = []
        for point in os.listdir(cashe):
            point_path = os.path.join(cashe, point)
            if not "." in point[0]:
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
                                    np.add(total, part[key])
                            part = {"Kboard": total}
                            X.append(total)
                        else:
                            y.append(part)
        y = matrix_orch(y)
        y = vectorize_orch(y)
        with open(os.path.join(base_path, "Orchestration/cashe/data"), "wb") as handle:
            data = [X, y]
            pickle.dump(data, handle)
        return X, y


def piano_roll_to_tensor(pr):
    return torch.as_tensor(pr)


def orchestra_to_tensor(orch):
    pass


def vectorize_orch(data):
    """Method to convert orchestra to a tensor

    Arguments:
        data {dict} -- orchestra dictionary

    Returns:
        np.array -- tensor representation of the orchestra
    """

    order = read_order()
    vect = []
    for instrument in order:
        vect.append(data[instrument])
    return vect


def vectorize_orch(data):
    vect = []
    for orch in data:
        deep_y = [[] for i in range(len(orch[0]))]
        for inst in orch:
            for j in range(len(inst)):
                deep_y[j].extend(inst[j])
        vect.append(np.array(deep_y))
        print("HERE")
    return vect


def matrix_orch(data):
    """Method that puts ALL the orchestra data into a matrix from the dictionary

    Arguments:
        data {arr} -- orchestration array containing dictionaries
    """
    vect = []
    if os.path.isfile(os.path.join(base_path, "Orchestration/cashe/order")):
        order = read_order()
    else:
        order = set()
        for orch in data:
            for instrument in orch:
                order.add(instrument)
        order = list(order)
        order = sorted(order)
        cashe_order(order)

    for orch in data:
        vect.append([])
        nan = np.zeros(orch[list(orch.keys())[0]].shape)
        for instrument in order:
            if instrument in orch:
                vect[-1].append(orch[instrument])
            else:
                vect[-1].append(nan)
    return vect


def devectorize_orch(data):
    """converting orchestra tensor into a dictionary

    Arguments:
        data {np.array} -- orchestra tensor

    Returns:
        dict -- dictionary representation of orchestra
    """
    mat = [[] for i in range(74)]
    for row in data:
        for i in range(len(mat)):
            mat[i].append(row[i * 128 : (i + 1) * 128])
            for j in range(len(mat[i][-1])):
                if mat[i][-1][j] < 10:
                    mat[i][-1][j] = 0
                mat[i][-1][j] *= 20
                if mat[i][-1][j] > 127:
                    mat[i][-1][j] = 127
    mat = np.array(mat)
    mat = mat.astype(int)
    orch = {}
    order = read_order()
    for i in range(len(order)):
        orch[order[i]] = mat[i]
    return orch


def orch_to_midi(data, output_path=os.path.join(base_path, "Orchestration/temp.mid")):
    """Method to convert orchestra tensor to midi

    Arguments:
        data {np.array} -- orchestration matrix

    Keyword Arguments:
        output_path {str} -- path where to write midi (default: {os.path.join(base_path, "Orchestration/temp.mid")})
    """

    pr = devectorize_orch(data)
    write_midi(pr, 8, output_path)


def piano_to_midi(data, output_path=os.path.join(base_path, "Orchestration/temp.mid")):
    """Helper to convert piano matrix to midi

    Arguments:
        data {np.array} -- piano roll matrix

    Keyword Arguments:
        output_path {str} -- path where to write midi (default: {os.path.join(base_path, "Orchestration/temp.mid")})
    """

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
