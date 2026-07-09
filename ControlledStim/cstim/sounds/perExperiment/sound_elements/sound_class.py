import dataclasses
import copy
import numpy as np
from dataclasses import dataclass,field

@dataclass(frozen=False)
class Sound:
    name: str = field(default_factory=str)
    samplerate: int = 16000
    duration: float = 0.05 # in seconds
    sound: np.ndarray = field(init=False)
    def __post_init__(self) -> None:
        self.sound = np.zeros(int(self.duration*self.samplerate))


def ramp_sound(s:Sound,cosine_rmp_length:float = 0.005) -> Sound:
    # Warning: modify in place the sounds.
    # creating ramps
    hanning_window = np.hanning(int(cosine_rmp_length * s.samplerate))
    hanning_window = hanning_window[:int(np.floor(hanning_window.shape[0] / 2))]
    newS = copy.deepcopy(s) #dataclasses.replace(s)
    # filtering tones with ramps:
    newS.sound[:hanning_window.shape[0]] = newS.sound[:hanning_window.shape[0]] * hanning_window
    newS.sound[-hanning_window.shape[0]:] = newS.sound[-hanning_window.shape[0]:] * hanning_window[::-1]
    return newS

def normalize_sound(s:Sound) -> Sound:
    if np.sum(s.sound**2)!=0:
        newS = copy.deepcopy(s)
        newS.sound = (newS.sound - np.mean(newS.sound,axis=-1,keepdims=True))/ np.std(newS.sound, axis=-1, keepdims=True)
        return newS
    else:
        return s

from typing import Optional
def pitch_shift(s:Sound,pitch_orig: Optional[float]=None,
                pitch_target:Optional[float]=None,
                ratio_pitch:Optional[float]=None) -> Sound:
    # Performs pitch shiffting using the pyrubberband package
    ## This requires to have installed rubberband
    # as well as the rubberband-CLI
    # Note:
    # On linux the steps are: install meson,
    # then install rubberband code (download and compile)
    # then sudo apt-get install rubberband-cli

    import pyrubberband.pyrb as pyrb
    if ratio_pitch is None:
        assert  pitch_target is not None
        assert pitch_orig is not None
        ratio_pitch = pitch_target/pitch_orig
    nstep = np.log(ratio_pitch) / np.log(1.05946)
    newS = copy.deepcopy(s)
    newS.sound = pyrb.pitch_shift(newS.sound, sr=s.samplerate
                                  , n_steps=nstep)
    return newS


@dataclass(frozen=False)
class Sound_pool(list[Sound]):
    # The sound_pool is simply a list of Sound elements, but we define
    # specific methods to it.
    def __post_init__(self):
        self.picked = []

    @classmethod
    def from_list(cls,ls : list[Sound]):
        s = Sound_pool()
        for l in ls:
            s.append(l)
        return s
    def pick_norepeat(self) -> Sound:
        ## memory cach the sound that were picked so that we can sample without repeat:
        pick = np.random.choice(np.setdiff1d(range(self.__len__()),self.picked),1)[0]
        self.picked.append(pick)
        return self[pick]
    def pick_norepeat_n(self,n:int) -> list[Sound]:
        ## memory cach the n sounds that were picked so that we can sample without repeat:
        picks = np.random.choice(np.setdiff1d(range(self.__len__()),self.picked),n,replace=False)
        for p in picks:
            self.picked.append(p)
        return [self[p] for p in picks]
    def clear_picked(self):
        self.picked = []

    def boundpick_norepeat_n(self,n:int,min_d:float,max_d:float,on_feature:str) -> list[Sound]:
        # Pick sounds in the list but at maximal and minimal distance from each other,
        #  acording from on_feature so that we make sure they have distinct properties.
        ### TODO: instead of using __getattribute__ force the sounds to be sortable!
        picks = []
        for _ in range(n):
            ## Find all possible sound:
            if len(self.picked) > 0:
                possible_s = []
                for idl,l in [(e,self[e]) for e in np.setdiff1d(range(self.__len__()),self.picked)]:
                    l_atrr = l.__getattribute__(on_feature)
                    if np.all([abs(l_atrr-self[e].__getattribute__(on_feature))>min_d for e in self.picked]) and \
                            np.all([abs(l_atrr-self[e].__getattribute__(on_feature))<max_d for e in self.picked]):
                        possible_s+=[idl]
                assert len(possible_s)>0
                self.picked.append(possible_s[np.random.choice(range(len(possible_s)), 1, replace=False)[-1]])
            else:
                self.picked.append(np.random.choice(range(self.__len__()), 1, replace=False)[0])
            picks += [self.picked[-1]]
        return [self[p] for p in picks]

