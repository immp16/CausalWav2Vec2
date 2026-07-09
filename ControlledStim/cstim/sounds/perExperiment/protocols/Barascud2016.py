import numpy as np
import pandas as pd

from cstim.sounds.perExperiment.sequences import ToneList, RandomPattern
from cstim.sounds.perExperiment.sound_elements import Bip,Silence,EnglishSyllable,Sound_pool
from cstim.sounds.perExperiment.sound_elements import Sound_pool,Sound
from cstim.sounds.perExperiment.protocols.ProtocolGeneration import Protocol_independentTrial
from cstim.sounds.perExperiment.sound_elements import ramp_sound,normalize_sound
from dataclasses import dataclass,field

from typing import Union, Tuple

@dataclass
class RandRegRand(Protocol_independentTrial):
    name:          str           = "RandRegRand_Barascud"
    cycle_len:     int           = 4                                                                                                                   
    n_cycles:      int           = 20
    rand_size:     int           = 16                                                                                                                  
    tones_fs:      Union[list, np.ndarray] = field(default_factory=list)
    isi:           float         = 0.06                                                                                                                
    sequence_isi:  float         = 0.0                                                                                                                 
    duration_tone: float         = 0.10
    samplerate:    int           = 16000                                                                                                               
    s_rand:        object        = field(default=None)
    s_reg:         object        = field(default=None)                                                                                                 
  
    def __post_init__(self):                                                                                                                           
        self.name = f"{self.name}_k{self.cycle_len}"
        sounds = [                                                                                                                                     
            Bip(name=f"bip-{i}", samplerate=self.samplerate,                                                                                           
                duration=self.duration_tone, fs=[f])                                                                                                   
            for i, f in enumerate(self.tones_fs)                                                                                                       
        ]                                                                                                                                              
        self.sound_pool = Sound_pool.from_list(sounds)
        self.randSeq    = ToneList(isi=self.isi, cycle=self.rand_size)                                                                                 
        self.regSeq     = ToneList(isi=self.isi, cycle=self.cycle_len)                                                                                 
        self.randSeqEnd = RandomPattern(
            isi=self.isi, nb_unique_elements=self.rand_size, len=self.rand_size                                                                        
        )       
                                                                                                                                                        
    def samplePool(self) -> Tuple:                                                                                                                     
        assert self.s_rand is None
        self.s_rand = Sound_pool.from_list(                                                                                                            
            self.sound_pool.pick_norepeat_n(self.rand_size)
        )                                                                                                                                              
        self.s_reg = Sound_pool.from_list(
            self.s_rand.pick_norepeat_n(self.cycle_len)                                                                                                
        )       
        return self.s_rand, self.s_reg                                                                                                                 
                
    def fixPoolSampled(self, s_rand, s_reg) -> None:                                                                                                   
        self.s_rand = s_rand
        self.s_reg  = s_reg                                                                                                                            
                
    def _getPoolAndSeq(self) -> Tuple:
        if self.s_rand is not None:
            all_pool = [self.s_rand] + [self.s_reg] * self.n_cycles + [self.s_rand]                                                                    
        else:                                                                                                                                          
            s_rand = Sound_pool.from_list(                                                                                                             
                self.sound_pool.pick_norepeat_n(self.rand_size)                                                                                        
            )   
            s_reg = Sound_pool.from_list(s_rand.pick_norepeat_n(self.cycle_len))
            all_pool = [s_rand] + [s_reg] * self.n_cycles + [s_rand]                                                                                   
        all_seq = [self.randSeq] + [self.regSeq] * self.n_cycles + [self.randSeqEnd]
        return all_pool, all_seq                                                                                                                       
                
    def _trial(self) -> tuple:                                                                                                                         
        all_pool, all_seq = self._getPoolAndSeq()
        all_sound  = []                                                                                                                                
        nb_element = 0                                                                                                                                 
        for p, seq in zip(all_pool, all_seq):
            s_p = seq(p)                                                                                                                               
            s_p = [normalize_sound(ramp_sound(s, cosine_rmp_length=0.005)) for s in s_p]                                                               
            all_sound  += s_p                                                                                                                          
            nb_element += sum(type(s) != Silence for s in s_p)                                                                                         
            if self.sequence_isi > 0:                                                                                                                  
                all_sound += [Silence(samplerate=self.samplerate, duration=self.sequence_isi)]
        self.sound_pool.clear_picked()                                                                                                                 
        df_info = pd.DataFrame({                                                                                                                       
            "isi":         [self.isi],                                                                                                                 
            "sequence_isi":[self.sequence_isi],                                                                                                        
            "cycle_len":   [self.cycle_len],
            "n_cycles":    [self.n_cycles],                                                                                                            
        })      
        return all_sound, nb_element, df_info                                                                                                          
  
                                                                                                                                                        
@dataclass      
class RandRegRand_MatchedRandom(RandRegRand):
    """                                                                                                                                                
    Same yoked pools, but the middle segment is a fresh random permutation
    of the k cycle tones each cycle — removes repetition while exactly                                                                                 
    matching tone identity and per-cycle frequency.                                                                                                    
    """                                                                                                                                                
    def _getPoolAndSeq(self) -> Tuple:                                                                                                                 
        if self.s_rand is not None:                                                                                                                    
            all_pool = [self.s_rand] + [self.s_reg] * self.n_cycles + [self.s_rand]
        else:                                                                                                                                          
            s_rand = Sound_pool.from_list(
                self.sound_pool.pick_norepeat_n(self.rand_size)                                                                                        
            )   
            s_reg = Sound_pool.from_list(s_rand.pick_norepeat_n(self.cycle_len))
            all_pool = [s_rand] + [s_reg] * self.n_cycles + [s_rand]

        middle_seqs = []                                                                                                                               
        for _ in range(self.n_cycles):
            rp = RandomPattern(                                                                                                                        
                isi=self.isi, nb_unique_elements=self.cycle_len, len=self.cycle_len                                                                    
            )
            rp.pattern = np.random.permutation(self.cycle_len).tolist()                                                                                
            middle_seqs.append(rp)                                                                                                                     
  
        all_seq = [self.randSeq] + middle_seqs + [self.randSeqEnd]                                                                                     
        return all_pool, all_seq


from cstim.sounds.perExperiment.sound_elements.segment_elements import SoundSegment
from typing import List
from pathlib import Path

@dataclass
class RandRegRand_otherStim(RandRegRand):
    """
        RandReg paradigm with different stimulis.
    """
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
class RandRegRand_syllable(RandRegRand):
    """
        RandReg paradigm with syllable stimulis.
    """
    
    def __post_init__(self):
        syllables =   np.array([["t","u"],["p","i"],["r","o"],["b","i"],["d","a"],["k","u"],
                     ["g","o"],["l","a"],["b","u"],["p","a"],["d","o"],["t","i"]])
        self.syllables = ["".join(e) for e in syllables]
        self.rand_voc = len(self.syllables)
        super().__post_init__()


        sounds = [EnglishSyllable(name="syllable-" + str(ids), 
                                    syllable=s,samplerate=self.samplerate,duration=self.duration_tone)
                                    for ids,s in enumerate(self.syllables)]
        self.sound_pool = Sound_pool.from_list(sounds)