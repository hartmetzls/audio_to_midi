# audio_to_midi
A CNN which converts piano audio to a simplified MIDI format

Copyright (c) 2018 Lillian Neff

OBTAIN DATASET
To get started, download the data from the Saarland Music dataset, putting the audio in one directory named "audio" and
the MIDI in another directory called "midi":

  * mkdir audio  
  * wget "http://resources.mpi-inf.mpg.de/SMD/SMD_MIDI-Audio-Piano-Music.html" -e robots=off -r -l1 -nd --no-parent -A.mp3  
  * mkdir midi  
  * wget "http://resources.mpi-inf.mpg.de/SMD/SMD_MIDI-Audio-Piano-Music.html" -e robots=off -r -l1 -nd --no-parent -A.mid  

In create_dataset.py, in the main function, follow the instructions to set directory_str to the filepath where you
downloaded the dataset. Run create_dataset. The file models.py contains the code for the final model.

Libraries used:

collections  
keras (incl dependency for keras: conda install h5py)  
librosa  
math  
matplotlib  
mido  
ntpath  
numpy  
os  
pickle  
random  
scikit-learn  
shutil  
tensorflow  
time  

Install necessary packages to env/virtualenv containing tensorflow:  
source activate tensorflow  
pip install mido  
pip install librosa  

Note:   
On Windows, when trying to load the audio with librosa, I got the following error:  
raise NoBackendError()  
audioread.NoBackendError  
If you get that error, install FFmpeg: https://www.wikihow.com/Install-FFmpeg-on-Windows  
The equivalent Linux fix is:  
sudo apt-get install libav-tools