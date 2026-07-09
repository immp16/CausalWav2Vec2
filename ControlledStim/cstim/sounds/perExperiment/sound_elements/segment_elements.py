import voxpopuli
import numpy as np
from dataclasses import dataclass,field
from cstim.sounds.perExperiment.sound_elements import Sound
from scipy.io.wavfile import read, write
from io import BytesIO
import torch
from julius import  resample_frac
from typing import Tuple,Union
from pathlib import Path

@dataclass
class SoundSegment(Sound):
    filename : Union[Path,str] = ""
    start : float = 0 # in seconds
    stop : float = 0 # in seconds 
    
    def __post_init__(self):
        # TODO memmap the soundfile?
        sr,sd = read(self.filename)
        self.samplerate = sr
        self.sound = sd[int(np.floor(self.start*self.samplerate)):int(np.ceil(self.stop*self.samplerate))]
        self.duration = (1.0*self.sound.shape[0])/self.samplerate
    
    def __eq__(self, other):
        if not isinstance(other, SoundSegment):
            # don't attempt to compare against unrelated types
            return NotImplemented
        if self.filename==other.filename and self.start==other.start and self.stop==other.stop:
            return True
        return False