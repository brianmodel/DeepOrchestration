import os
import pickle

from Orchestration.midi.read_midi import Read_midi
from Orchestration import data_path, base_path


def get_train_data(source="bouliane_aligned"):
    cashe = os.path.join(base_path, "Orchestration/cashe/" + source)
    data = []
    for point in os.listdir(cashe):
        point_path = os.path.join(cashe, point)
        scores = []
        for score in os.listdir(point_path):
            with open(os.path.join(point_path, score), "rb") as f:
                scores.append([pickle.load(f)])
        data.append(scores)
    return data


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
