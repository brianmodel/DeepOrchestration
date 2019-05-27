from app import ALLOWED_EXTENSIONS, DOWNLOAD_FOLDER
from Orchestration.model import MultipleRNN
from Orchestration.midi.read_midi import Read_midi
from Orchestration.midi.write_midi import write_midi
from Orchestration.get_data import filter
import numpy as np


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def process_file(path, filename, instrument):
    generate_orchestration(path, filename, instrument)


def generate_orchestration(path, filename, instrument):
    # Generate the orchestration, and save file to downloads directory
    rnn = MultipleRNN()
    quantization = 8
    data = Read_midi(path, quantization).read_file()
    total = None
    for inst in data:
        if total is None:
            total = data[inst]
        else:
            total = np.add(total, data[inst])
    print("SHAPE TOT ", total.shape)
    pred = rnn.predict(total, instrument, save=False)
    # pred = filter(pred, 2)
    pred = filter(pred, 2)
    print("SHAPE ", pred.shape)
    write_midi({instrument: pred}, 8, DOWNLOAD_FOLDER + filename)
