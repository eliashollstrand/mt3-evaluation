# Audio-to-MIDI Transcription
Bachelor's thesis at KTH about Automatic Music Transcription (AMT) in the form of audio-to-MIDI conversion and the effects on generalization when training a ML-model on multiple instruments (piano and guitar).

## Initialization
Clone the repo
```
git clone <repo>
```

Install virtualenv
```
pip install virtualenv
```

Navigate to the repo folder and create environment
```
virtualenv env
```

Activate the environment
```
//mac
source env/bin/activate

//CMD
env\Scripts\activate.bat

//Powershell
env\Scripts\activate.ps1
```

Install the necessary packages
```
pip install -r requirements.txt
```

## Download datasets
The datasets used in this project are the MAESTRO and GuitarSet which can be downloaded from the respective websites:

- [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro) (V3.0.0)
- [GuitarSet Dataset](https://github.com/marl/guitarset) (V1.1.0)

## Set up paths
Create a `.env` file in the repo folder. In this file you should add the paths where you dowloaded the datasets by writing the following:
```
MAESTRO_PATH="<path>"
GUITARSET_PATH="<path>"
```