import numpy as np
import pandas as pd

from cstim.sounds.perExperiment.sequences import ToneList,Local_Deviant,Local_Standard,Sequence,RandomPattern
from cstim.sounds.perExperiment.sound_elements import Bip,Silence,EnglishSyllable
from cstim.sounds.perExperiment.sound_elements import Sound_pool,Sound
from cstim.sounds.perExperiment.protocols.ProtocolGeneration import Protocol_independentTrial
from cstim.sounds.perExperiment.sound_elements import ramp_sound,normalize_sound
from dataclasses import dataclass,field

from typing import Union,Tuple
@dataclass
class RandRegDev_LocalGlobal(Protocol_independentTrial):
    name : str = "RandRegDev_LocalGlobal"
    sequence_isi : float = 0.75
    isi : float = 0.1
    duration_tone : float = 0.05
    samplerate : int = 16000
    motif_repeat : int = 3
    global_standard : str = "localstandard"
    is_deviant : bool = True
    tones_fs : Union[list[float],np.ndarray] = field(default_factory=list)
    rand_size : int = 15
    s_rand: list[Sound] = field(default=None)
    s_reg: list[Sound] = field(default=None)

    def __post_init__(self):
        self.name = self.name+"_"+self.global_standard
        sounds = [Bip(name="bip-" + str(idf), samplerate=self.samplerate, duration=self.duration_tone, fs=[f]) for
                  idf, f in enumerate(self.tones_fs)]
        # Note: naming the bip is useful to one who is where.
        self.sound_pool = Sound_pool.from_list(sounds)
        self.randSeq = ToneList(isi=self.isi, cycle=self.rand_size)
        if self.global_standard =="localstandard":
            self.regSeq = Local_Standard(isi=self.isi)
            self.devSeq = Local_Deviant(isi=self.isi)
        else:
            self.regSeq = Local_Deviant(isi=self.isi)
            self.devSeq = Local_Standard(isi = self.isi)

    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool],list[Sequence]]:
        ## Instantiate the vocabularies:
        if not self.s_rand is None:
            all_pool = [self.s_rand]
        else:
            s_rand = Sound_pool.from_list(self.sound_pool.pick_norepeat_n(self.rand_size))
            all_pool = [s_rand]

        if not self.s_reg is None:
            all_pool += [self.s_reg for _ in range(self.motif_repeat)] + [self.s_reg]
        else:
            if self.s_rand is None:
                s_reg = Sound_pool.from_list(s_rand.pick_norepeat_n(2))
            else:
                s_reg = Sound_pool.from_list(self.s_rand.pick_norepeat_n(2))
                self.s_rand.clear_picked()
            all_pool += [s_reg for _ in range(self.motif_repeat)] + [s_reg]
        all_seq = [self.randSeq] + [self.regSeq for _ in range(self.motif_repeat)] + [self.devSeq]
        return all_pool,all_seq
    def _trial(self) -> tuple[list[Sound],int,pd.DataFrame]:
        ''' Trial implements the logic of the protocol for one trial.'''
        all_pool, all_seq = self._getPoolAndSeq()
        all_sound = []
        nb_element = 0
        for p,seq in zip(all_pool, all_seq):
            s_p = seq(p) # combine sequence and pool
            ## Apply sound modifications:
            s_p = [normalize_sound(ramp_sound(s,cosine_rmp_length=0.005)) for s in s_p]
            all_sound += s_p
            nb_element += np.sum([type(s)!= Silence for s in s_p])
            if self.sequence_isi > 0:
                all_sound += [Silence(samplerate=self.samplerate, duration=self.sequence_isi)]
        # should be a list of Sound
        self.sound_pool.clear_picked()

        ### Further returns additional trial information to be stored:
        df_info = pd.DataFrame.from_dict({"isi":[self.isi],"sequence_isi":[self.sequence_isi],
                                          "global_standard":[self.global_standard],"deviant":self.is_deviant})

        return (all_sound,nb_element,df_info)

    def samplePool(self) -> Tuple[list[Sound], list[Sound]]:
        assert self.s_rand is None
        self.s_rand = Sound_pool.from_list(self.sound_pool.pick_norepeat_n(self.rand_size))
        self.s_reg = Sound_pool.from_list(self.s_rand.pick_norepeat_n(2))
        return self.s_rand, self.s_reg

    def fixPoolSampled(self, s_rand: list[Sound], s_reg: list[Sound]) -> None:
        self.s_rand = s_rand
        self.s_reg = s_reg
    def sampleBoundPool(self,min_freqDist:float,max_freqDist:float) -> Tuple[list[Sound],list[Sound]]:
        assert self.s_rand is None
        self.s_rand  = Sound_pool.from_list(self.sound_pool.pick_norepeat_n(16))
        self.sound_pool.clear_picked()
        self.s_reg = Sound_pool.from_list(self.sound_pool.boundpick_norepeat_n(2,min_freqDist,max_freqDist,"first_freq"))
        return self.s_rand,self.s_reg

@dataclass
class RandRegDev_LocalGlobal_orig(RandRegDev_LocalGlobal):
    is_deviant : bool = False
    def __post_init__(self):
        self.name = self.name+"_"+self.global_standard
        sounds = [Bip(name="bip-" + str(idf), samplerate=self.samplerate, duration=self.duration_tone, fs=[f]) for
                  idf, f in enumerate(self.tones_fs)]
        # Note: naming the bip is useful to one who is where.
        self.sound_pool = Sound_pool.from_list(sounds)
        self.randSeq = ToneList(isi=self.isi, cycle=self.rand_size)
        if self.global_standard =="localstandard":
            self.regSeq = Local_Standard(isi=self.isi)
            self.devSeq = Local_Standard(isi=self.isi)
        else:
            self.regSeq = Local_Deviant(isi=self.isi)
            self.devSeq = Local_Deviant(isi = self.isi)



from cstim.sounds.perExperiment.sound_elements.segment_elements import SoundSegment
from typing import List
from pathlib import Path

@dataclass
class RandRegDev_LocalGlobal_otherStim(RandRegDev_LocalGlobal):
    sound_paths : List[Union[str,Path]] = ""
    start: list[float] = 0
    stop: list[float] = 0.05
    
    def __post_init__(self):
        super().__post_init__()
        sounds = [SoundSegment(name="bip-" + str(idf), 
                               filename = self.sound_paths[idf],
                               start = self.start[idf],
                               stop = self.stop[idf]) for idf in range(len(self.sound_paths))]
        # Note: naming the bip is useful to know who is where.
        self.sound_pool = Sound_pool.from_list(sounds)


@dataclass
class RandRegDev_LocalGlobal_syllable(RandRegDev_LocalGlobal):    
    def __post_init__(self):
        syllables =   np.array([["t","u"],["p","i"],["r","o"],["b","i"],["d","a"],["k","u"],
                     ["g","o"],["l","a"],["b","u"],["p","a"],["d","o"],["t","i"]])
        self.syllables = ["".join(e) for e in syllables]
        self.rand_size = len(self.syllables)
        super().__post_init__()
        sounds = [EnglishSyllable(name="syllable-" + str(ids), 
                                    syllable=s,samplerate=self.samplerate,duration=self.duration_tone)
                                    for ids,s in enumerate(self.syllables)]
        self.sound_pool = Sound_pool.from_list(sounds)