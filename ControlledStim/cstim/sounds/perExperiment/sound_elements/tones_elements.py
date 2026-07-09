from cstim.sounds.perExperiment.sound_elements.sound_class import Sound
import numpy as np
import librosa
from  dataclasses import dataclass,field
from typing import Union

## Remark from Pierre:
# I am not sure how useful is Gaussian_RN.
#
def gaussian_N(samplerate,duration):
    n_samples = int(samplerate * duration)
    return np.random.normal(0, 1, n_samples)
@dataclass
class Gaussian_N(Sound):
    name : str = "Gaussian_N"
    def __post_init__(self) -> None:
        self.sound = gaussian_N(self.samplerate,self.duration)

def gaussian_RN(samplerate,duration):
    # Generate a gaussian noise being repeated one time.
    # Pierre: I don't understand this. also you used:
    # n_samples = int(samplerate * duration / 2000)
    # I am not sure why
    noise = gaussian_N(samplerate,duration)
    r_noise = np.concatenate((noise, noise))
    return r_noise

@dataclass
class Gaussian_RN(Sound):
    name : str = "Gaussian_RN"
    def __post_init__(self) -> None:
        self.sound = gaussian_RN(self.samplerate,self.duration)

def bip_randomPitch(samplerate: int,duration: float,fs: Union[list[float],np.ndarray]):
    dis = np.random.uniform(0, 1, len(fs))
    # weights for random linear combination of pure tones...
    return np.transpose(np.transpose(
        np.stack([librosa.tone(f, sr=samplerate, duration=duration) for f in fs],
                 axis=0)) @ np.transpose(dis))
@dataclass
class Bip_randPitch(Sound):
    name : str = "Bip_randomPitch"
    fs : Union[list[float],np.ndarray] = field(default=list) # frequencies of the pure tones.
    def __post_init__(self) -> None:
        self.sound = bip_randomPitch(self.samplerate,self.duration,self.fs)

def bip(samplerate: int,duration: float,fs: Union[list[float],np.ndarray]):
    return np.mean(np.stack([librosa.tone(f, sr=samplerate, duration=duration) for f in fs],
                 axis=0),axis=0)
@dataclass
class Bip(Sound):
    name : str = "Bip"
    fs : Union[list[float],np.ndarray] = field(default=list) # frequencies of the pure tones.
    first_freq : int = field(init=False)
    def __post_init__(self) -> None:
        self.sound = bip(self.samplerate,self.duration,self.fs)
        self.first_freq = self.fs[0]
    
    def __eq__(self, other):
        if not isinstance(other, Bip):
            # don't attempt to compare against unrelated types
            return NotImplemented
        if np.all(self.fs==other.fs) and self.duration==other.duration and self.first_freq==other.first_freq and self.samplerate==other.samplerate:
            return True
        return False

@dataclass
class Silence(Sound):
    name : str = "Silence"
    def __post_init__(self) -> None:
        self.sound = np.zeros(int(self.samplerate * self.duration))