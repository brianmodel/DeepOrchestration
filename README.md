# DeepOrchestration
### Created as part of the DeepScore research project — Georgia Tech Robotic Musicianship lab

This repo contains several models which map piano to orchestral scores as a way of simulating the work of a human orchestrator. 


## Data
This project is inspired by the research done McGill University:
https://hal.archives-ouvertes.fr/hal-01578292/document
This study accumulated several datasets of piano and corresponding orchestral scores. Our models were trained on this data.

## Models
Our repo contains two types of sequence-to-sequence machine learning models. On the **naveen2** branch, there is a model which uses a single RNN to read in the piano score data directly and generate an orchestral mapping, while the **multipleRNN** branch contains a model which is being trained on an instrument specific basis, with individual RNNs mapping each part in the orchestra.
