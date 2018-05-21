# audio_to_midi
A Convolutional Neural Network which converts piano audio to a simplified MIDI format. 
The final model takes as input an audio file (mono- or polyphonic) 
and outputs a simplified MIDI output with the corresponding notes and 
note duration. This output can then be reconstructed into a standard MIDI file format.

#### Main objective
The automated conversion executed by the CNN is a step toward the larger 
goal of Automatic Music Transcription (AMT). AMT and Music Information Retrieval have many applications in industry, including Digital 
Audio Workstation software development and music recommendation systems. 

#### Setup

To get started, download the data from the Saarland Music dataset, putting the audio in one directory named "audio" and
the MIDI in another directory called "midi":

``` 
mkdir audio 
wget "http://resources.mpi-inf.mpg.de/SMD/SMD_MIDI-Audio-Piano-Music.html" -e robots=off -r -l1 -nd --no-parent -A.mp3  
mkdir midi  
wget "http://resources.mpi-inf.mpg.de/SMD/SMD_MIDI-Audio-Piano-Music.html" -e robots=off -r -l1 -nd --no-parent -A.mid
```  
  
In create_dataset.py, in the main function, follow the instructions to set directory_str to the filepath where you
downloaded the dataset. Run create_dataset. The file models.py contains the code for the final model.

#### Libraries

TensorFlow install instructions:  
https://www.tensorflow.org/install/

Keras install instructions:  
https://keras.io/#installation

collections  
keras (see instructions above for dependencies)   
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
tensorflow (see instructions above for dependencies)   
time  


#### Issues 
On Windows, when loading in the audio files with librosa, the following error may arise:  
> raise NoBackendError()  
> audioread.NoBackendError  

If the above error is raised, try installing FFmpeg:   
https://www.wikihow.com/Install-FFmpeg-on-Windows      


On Linux, the equivalent fix is:   
```sudo apt-get install libav-tools```

#### Copyright
Copyright (c) 2018 Lillian Neff