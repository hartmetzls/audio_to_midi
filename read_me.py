# Obtain dataset:

# BUILD FROM SCRATCH
# Download music from Saarland Music dataset:
# mkdir audio
# wget "http://resources.mpi-inf.mpg.de/SMD/SMD_MIDI-Audio-Piano-Music.html" -e robots=off -r -l1 -nd --no-parent -A.mp3
# mkdir midi
# wget "http://resources.mpi-inf.mpg.de/SMD/SMD_MIDI-Audio-Piano-Music.html" -e robots=off -r -l1 -nd --no-parent -A.mid
# Run create_dataset.
#
# OR use your own method to download from
# http: // resources.mpi-inf.mpg.de / SMD / SMD_MIDI-Audio-Piano-Music.html
#
# OR use
# cqt_segments_midi_segments.pkl

# Standard necessary libraries:
# tensorflow
# keras (incl dependency for keras: conda install h5py)

# Install necessary packages to env/virtualenv containing tensorflow:
# source activate tensorflow
# pip install mido
# pip install librosa

# Note: On Windows, when trying to load the audio with librosa, I got the following error:
# raise NoBackendError()
# audioread.NoBackendError
# If you get that error, install FFmpeg: https://www.wikihow.com/Install-FFmpeg-on-Windows
# The equivalent Linux fix is:
# sudo apt-get install libav-tools