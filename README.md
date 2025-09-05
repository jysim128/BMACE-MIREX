# BMACE: Mamba-Based Automatic Chord Estimation

This package contains the inference code for BMACE, submitted to **MIREX 2025 – Audio Chord Estimation**.

## Installation
```bash
pip install -r requirements.txt
```

## # Make BMACE.sh executable (first time only)
```bash
chmod +x BMACE.sh
```

## Running BMACE
```bash
./BMACE.sh prepare
./BMACE.sh do_chord_identification "/path/to/input.wav" "/path/to/output.txt"
```
##Quick Test Example
We provide a toy audio file under examples/.
```bash
./BMACE.sh do_chord_identification examples/01-I_Saw_Her_Standing_There.wav results/01-I_Saw_Her_Standing_There.wav.txt
```

This writes a chord lab file in MIREX format:
```bash
0.000000 0.092880 C:maj
0.092880 0.185760 G:min
...
```
