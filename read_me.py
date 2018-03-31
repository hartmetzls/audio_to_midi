1. download music from saarland music dataset:
http://resources.mpi-inf.mpg.de/SMD/SMD_MIDI-Audio-Piano-Music.html

mkdir audio

wget "http://resources.mpi-inf.mpg.de/SMD/SMD_MIDI-Audio-Piano-Music.html" -e robots=off -r -l1 -nd --no-parent -A.mp3

mkdir midi

wget "http://resources.mpi-inf.mpg.de/SMD/SMD_MIDI-Audio-Piano-Music.html" -e robots=off -r -l1 -nd --no-parent -A.mid

install necessary packages to tensorflow enviro:
source activate tensorflow
pip install mido
pip install librosa

On windows:
Used https://online-audio-converter.com/ for http://resources.mpi-inf.mpg.de/SMD/SMD_MIDI-Audio-Piano-Music.html files because of
This error:
raise NoBackendError()
audioread.NoBackendError
When trying to load

It worked.


On linux you can instead do:
sudo apt-get install libav-tools
