# Orchestration_checked

This is a database for **projective orchestration learning**. It contains scores of orchestral pieces and their piano reduction (or equivalently piano pieces and their orchestration). Orchestrations and reductions have been performed by famous composers or composition teachers.

Scores are in the **midi format**.

The database is splitted in several folders according to the origin af the files :
- each folder (*liszt_classical_archives/*), contains subfolders indexed by number.
- a csv file with the same name as the folder contains metadata about the files contained in this folder (*liszt_classical_archives.csv*)
- in each indexed subfolders, there are two midifiles and corresponding csv files (*liszt_classical_archives/32*)
- midi files are an orchestration and its piano version (*liszt_classical_archives/32/beet9m2.mid* and *liszt_classical_archives/32/symphony_9_2_orch.mid*)
- csv contains the mapping between the name of the midi tracks and a normalized nomenclatura for the instrument names
(*liszt_classical_archives/32/beet9m2.csv* and *liszt_classical_archives/32/symphony_9_2_orch.csv*)
