import pandas as pd
from cstim.sounds.perExperiment.sequences import lot_patterns,ToneList,Sequence,RandomPattern
from cstim.sounds.perExperiment.sound_elements import Bip,Silence,EnglishSyllable
from cstim.sounds.perExperiment.sound_elements import Sound_pool,Sound
from cstim.sounds.perExperiment.protocols.ProtocolGeneration import Protocol_independentTrial
from cstim.sounds.perExperiment.sound_elements import ramp_sound,normalize_sound
from dataclasses import dataclass,field
import numpy as np
from typing import Union,Tuple

@dataclass
class LOT(Protocol_independentTrial):
    name : str = "Classical_LOT"
    sequence_isi : float = 0.5
    isi : float = 0.2
    duration_tone : float = 0.05
    samplerate : int = 16000
    motif_repeat : int = 10
    lot_seq : str = "pairs"
    tones_fs : Union[list[float],np.ndarray,list[np.ndarray]] = field(default_factory=list)

    s_reg : list[Sound] = field(default=None)
    def __post_init__(self):
        self.name = self.name+"_"+self.lot_seq
        sounds = []
        for idf, f in enumerate(self.tones_fs):
            if type(f)==np.ndarray or type(f)==list:
                sounds+=[Bip(name="bip-" + str(idf), samplerate=self.samplerate, duration=self.duration_tone, fs=f)]
            else:
                sounds+=[Bip(name="bip-" + str(idf), samplerate=self.samplerate, duration=self.duration_tone, fs=[f])]
        self.sound_pool = Sound_pool.from_list(sounds)
        # Note: naming the bip is useful to know who is where.
        self.regSeq = lot_patterns[self.lot_seq](isi=self.isi)

    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool],list[Sequence]]:
        ## Instantiate the vocabularies:
        if self.s_reg is None:
            s_reg = Sound_pool.from_list(self.sound_pool.pick_norepeat_n(2))
        else:
            s_reg = self.s_reg
        all_pool = [s_reg for _ in range(self.motif_repeat)]
        all_seq = [self.regSeq for _ in range(self.motif_repeat)]
        return all_pool,all_seq

    def _trial(self) -> tuple[list[Sound],int,pd.DataFrame]:
        ''' Trial implements the logic of the protocol for one trial.'''
        all_pool, all_seq = self._getPoolAndSeq()
        all_sound = []
        nb_element = 0
        for p,seq in zip(all_pool, all_seq):
            s_p = seq(p) # combine sequence and pool
            ## Apply sound modifications:
            s_p = [ramp_sound(s,cosine_rmp_length=0.005) for s in s_p]
            all_sound += s_p
            nb_element += np.sum([type(s)!= Silence for s in s_p])
            if self.sequence_isi > 0:
                all_sound += [Silence(samplerate=self.samplerate, duration=self.sequence_isi)]
        # should be a list of Sound
        self.sound_pool.clear_picked()
        ### Further returns additional trial information to be stored:
        df_info = pd.DataFrame.from_dict({"isi":[self.isi],"sequence_isi":[self.sequence_isi],"lot_seq":[self.lot_seq]})
        return (all_sound,nb_element,df_info)

@dataclass
class LOT_generalize(LOT):
    ### Same as LOT but at each repetition of the motif we vary the set of tones used:
    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool],list[Sequence]]:
        ## Instantiate the vocabularies:
        all_pool = []
        for i in range(self.motif_repeat):
            s_reg = Sound_pool.from_list(self.sound_pool.pick_norepeat_n(2))
            all_pool += [s_reg]
            if (i+1)*2>= len(self.sound_pool):
                # we have used all available tones and reinitialize all the possible pairs to choose from
                self.sound_pool.clear_picked()
        all_seq = [self.regSeq for _ in range(self.motif_repeat)]
        return all_pool,all_seq

@dataclass
class LOT_deviant(LOT):
    deviant :int =-1 # to choose the deviant index
    is_complementary : bool = False # to invert the sequence' tones
    s_reg : list[Sound] = field(default=None)
    id_lot_seq : int = 0
    base_name : str = field(init=False)
    def __post_init__(self):
        super(LOT_deviant,self).__post_init__()
        self.id_lot_seq = 0
        self.lot_seq = list(lot_patterns.keys())[self.id_lot_seq]
        self.regSeq = lot_patterns[self.lot_seq](isi=self.isi)
        self.devSeq = lot_patterns[self.lot_seq](isi=self.isi)
        if self.deviant >= 0:
            self.devSeq.as_deviant_pattern(self.deviant)
        self.base_name = self.name
        self.name = self.base_name + self.lot_seq
    def _trial(self) -> tuple[list[Sound],int,pd.DataFrame]:
        ## A trial function implementing some memory
        # such that for every two tones, we generate every sequence with every possible deviant combinations
        assert self.regSeq.__class__ == self.devSeq.__class__
        if self.s_reg is None:
            all_pool,all_seq = self._getPoolAndSeq()
            self.s_reg = all_pool[0]

        all_sound, nb_element, df_info = super(LOT_deviant,self)._trial()
        df_info["deviantId"] = self.deviant
        if self.deviant!=-1:
            df_info["deviantpos"] = self.devSeq.deviant_pos[self.deviant]
        df_info["is_complementary"] = self.is_complementary

        # update the deviant:
        self.deviant += 1
        if self.deviant >= 0 and self.deviant < 4:
            self.devSeq = lot_patterns[self.lot_seq](isi=self.isi)
            self.devSeq.as_deviant_pattern(self.deviant)
        elif self.deviant >= 4 and self.is_complementary:
            self.deviant = -1

            self.is_complementary = False
            if self.id_lot_seq == len(list(lot_patterns.keys())) - 1: # reinitialize the protocol
                self.s_reg = None
                self.id_lot_seq = 0
                self.lot_seq = list(lot_patterns.keys())[self.id_lot_seq]
                self.regSeq = lot_patterns[self.lot_seq](isi=self.isi)
                self.devSeq = lot_patterns[self.lot_seq](isi=self.isi)

                self.name = self.base_name + self.lot_seq
            else:
                self.id_lot_seq += 1
                self.lot_seq = list(lot_patterns.keys())[self.id_lot_seq]
                self.regSeq = lot_patterns[self.lot_seq](isi=self.isi)
                self.devSeq = lot_patterns[self.lot_seq](isi=self.isi)
                self.name =  self.base_name + self.lot_seq
        elif self.deviant >= 4 and not self.is_complementary:
            self.is_complementary = True
            self.deviant = -1
            self.devSeq = lot_patterns[self.lot_seq](isi=self.isi)
            self.s_reg = self.s_reg[::-1]  ## complementary sequence
        return (all_sound, nb_element, df_info)
    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool],list[Sequence]]:
        ## Instantiate the vocabularies:
        if self.s_reg is None:
            s_reg = Sound_pool.from_list(self.sound_pool.pick_norepeat_n(2))
        else:
            s_reg = self.s_reg
        all_pool = [s_reg for _ in range(self.motif_repeat)]
        all_seq = [self.regSeq for _ in range(self.motif_repeat-1)] +[self.devSeq]
        return all_pool,all_seq


#### Similar protocol but adapted to fit a Random-Regular-Random organization:
@dataclass
class RandRegRand_LOT(Protocol_independentTrial):
    name : str = "RandRegRand_LOT"
    sequence_isi : float = 0.3
    isi : float = 0.2
    duration_tone : float = 0.05
    samplerate : int = 16000
    motif_repeat : int = 3
    lot_seq : str = "pairs"
    tones_fs : Union[list[float],np.ndarray] = field(default_factory=list)
    rand_size : int = 16
    s_rand: list[Sound] = field(default=None)
    s_reg: list[Sound] = field(default=None)

    def __post_init__(self):
        self.name = self.name+"_"+self.lot_seq
        sounds = [Bip(name="bip-" + str(idf), samplerate=self.samplerate, duration=self.duration_tone, fs=[f]) for
                  idf, f in enumerate(self.tones_fs)]
        # Note: naming the bip is useful to know who is where.
        self.sound_pool = Sound_pool.from_list(sounds)
        self.randSeq = ToneList(isi=self.isi, cycle=self.rand_size)
        self.regSeq = lot_patterns[self.lot_seq](isi=self.isi)
        self.randSeqEnd = RandomPattern(isi=self.isi, nb_unique_elements=self.rand_size, len=self.rand_size)

        ## Make sure the first random tone breaks the sequence:
        if self.randSeqEnd.pattern[0] == self.regSeq.pattern[0]:
            self.randSeqEnd.pattern[0] = 1-self.randSeqEnd.pattern[0]

    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool],list[Sequence]]:
        ## Instantiate the vocabularies:
        if not self.s_rand is None:
                all_pool = [self.s_rand] + [self.s_reg for _ in range(self.motif_repeat)] + [self.s_rand]
        else:
            s_rand = Sound_pool.from_list(self.sound_pool.pick_norepeat_n(self.rand_size))
            s_reg = Sound_pool.from_list(s_rand.pick_norepeat_n(2))
            all_pool = [s_rand] + [s_reg for _ in range(self.motif_repeat)] + [s_rand]
        all_seq = [self.randSeq] + [self.regSeq for _ in range(self.motif_repeat)] + [self.randSeqEnd]
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
        df_info = pd.DataFrame.from_dict({"isi":[self.isi],"sequence_isi":[self.sequence_isi],"lot_seq":[self.lot_seq]})

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
        self.s_rand  = Sound_pool.from_list(self.sound_pool.pick_norepeat_n(self.rand_size))
        self.sound_pool.clear_picked()
        self.s_reg = Sound_pool.from_list(self.sound_pool.boundpick_norepeat_n(2,min_freqDist,max_freqDist,"first_freq"))
        return self.s_rand,self.s_reg


@dataclass
class RandRegRand_LOT_deviant(RandRegRand_LOT):
    deviant : int = 0
    def __post_init__(self):
        sounds = [Bip(name="bip-"+str(idf),samplerate=self.samplerate, duration=self.duration_tone, fs=[f]) for idf,f in enumerate(self.tones_fs)]
        # Note: naming the bip is useful to one who is where.
        self.sound_pool = Sound_pool.from_list(sounds)
        self.randSeq = ToneList(isi=self.isi, cycle=self.rand_size)
        self.regSeq = lot_patterns[self.lot_seq](isi=self.isi)

        self.devSeq = lot_patterns[self.lot_seq](isi=self.isi)
        self.devSeq.as_deviant_pattern(self.deviant)

    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool], list[Sequence]]:
        ## Instantiate the vocabularies:
        if not self.s_rand is None:
            all_pool = [self.s_rand] + [self.s_reg for _ in range(self.motif_repeat)] + [self.s_reg]
        else:
            s_rand = Sound_pool.from_list(self.sound_pool.pick_norepeat_n(self.rand_size))
            s_reg = Sound_pool.from_list(s_rand.pick_norepeat_n(2))
            all_pool = [s_rand] + [s_reg for _ in range(self.motif_repeat)] + [s_reg]
        all_seq = [self.randSeq] + [self.regSeq for _ in range(self.motif_repeat)] + [self.devSeq]
        return all_pool, all_seq

    def _trial(self) -> tuple[list[Sound], int, pd.DataFrame]:
        all_sound,nb_element,df_info = super(RandRegRand_LOT_deviant,self)._trial()
        df_info = df_info.join(pd.DataFrame({"deviantpos":[self.devSeq.deviant_pos[self.deviant]],"deviant":[self.deviant]}).set_axis(df_info.index))
        return (all_sound,nb_element,df_info)


@dataclass
class RandRegRand_LOT_orig(RandRegRand_LOT):
    def __post_init__(self):
        sounds = [Bip(name="bip-"+str(idf),samplerate=self.samplerate, duration=self.duration_tone, fs=[f]) for idf,f in enumerate(self.tones_fs)]
        # Note: naming the bip is useful to one who is where.
        self.sound_pool = Sound_pool.from_list(sounds)
        self.randSeq = ToneList(isi=self.isi, cycle=self.rand_size)
        self.regSeq = lot_patterns[self.lot_seq](isi=self.isi)
    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool], list[Sequence]]:
        ## Instantiate the vocabularies:
        if not self.s_rand is None:
            all_pool = [self.s_rand] + [self.s_reg for _ in range(self.motif_repeat)] + [self.s_reg]
        else:
            s_rand = Sound_pool.from_list(self.sound_pool.pick_norepeat_n(self.rand_size))
            s_reg = Sound_pool.from_list(s_rand.pick_norepeat_n(2))
            all_pool = [s_rand] + [s_reg for _ in range(self.motif_repeat)] + [s_reg]
        all_seq = [self.randSeq] + [self.regSeq for _ in range(self.motif_repeat)] + [self.regSeq]
        return all_pool, all_seq


@dataclass
class RandRegRand_LOT_Generalize(RandRegRand_LOT):

    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool],list[Sequence]]:
        ## Instantiate the vocabularies:
        ## In this case we want to change the tone used in the generalize sequence at every step
        # so we pick enough tone in a pool and probably forbid to take them...
        s_poolReg = Sound_pool.from_list(self.sound_pool.pick_norepeat_n(self.motif_repeat*2))
        s_rand = Sound_pool.from_list(s_poolReg.pick_norepeat_n(self.rand_size))
        s_poolReg.clear_picked() # clear the poolReg to be able to choose again from the self.motif_repeat*2
        s_regs = [Sound_pool.from_list(s_poolReg.pick_norepeat_n(2)) for _ in range(self.motif_repeat)]
        all_pool = [s_rand] + s_regs + [s_regs[-1]]
        all_seq = [self.randSeq] + [self.regSeq for _ in range(self.motif_repeat)] + [self.randSeqEnd]
        return all_pool,all_seq


@dataclass
class RandRegRand_LOT_Generalize_deviant(RandRegRand_LOT_deviant):
    s_reg: list[list[Sound]] = field(default=None)

    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool],list[Sequence]]:

        if not self.s_rand is None:
            all_pool = [self.s_rand] + self.s_reg
        else:
            ## Instantiate the vocabularies:
            ## In this case we want to change the tone used in the generalize sequence at every step
            # so we pick enough tone in a pool and probably forbid to take them...
            s_poolReg = Sound_pool.from_list(self.sound_pool.pick_norepeat_n(self.motif_repeat*2))
            s_rand = Sound_pool.from_list(s_poolReg.pick_norepeat_n(self.rand_size))
            s_poolReg.clear_picked() # clear the poolReg to be able to choose again from the self.motif_repeat*2
            s_regs = [Sound_pool.from_list(s_poolReg.pick_norepeat_n(2)) for _ in range(self.motif_repeat+1)]
            all_pool = [s_rand] + s_regs
        all_seq = [self.randSeq] + [self.regSeq for _ in range(self.motif_repeat)] + [self.devSeq]
        return all_pool,all_seq

@dataclass
class RandRegRand_LOT_Generalize_orig(RandRegRand_LOT_deviant):
    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool],list[Sequence]]:
        if not self.s_rand is None:
            all_pool = [self.s_rand] + self.s_reg
        else:
            ## Instantiate the vocabularies:
            ## In this case we want to change the tone used in the generalize sequence at every step
            # so we pick enough tone in a pool and probably forbid to take them...
            s_poolReg = Sound_pool.from_list(self.sound_pool.pick_norepeat_n(self.motif_repeat*2))
            s_rand = Sound_pool.from_list(s_poolReg.pick_norepeat_n(self.rand_size))
            s_poolReg.clear_picked() # clear the poolReg to be able to choose again from the self.motif_repeat*2
            s_regs = [Sound_pool.from_list(s_poolReg.pick_norepeat_n(2)) for _ in range(self.motif_repeat+1)]
            all_pool = [s_rand] + s_regs
        all_seq = [self.randSeq] + [self.regSeq for _ in range(self.motif_repeat)] + [self.regSeq]
        return all_pool,all_seq


from cstim.sounds.perExperiment.sound_elements.segment_elements import SoundSegment
from typing import List
from pathlib import Path

@dataclass
class RandRegRand_LOT_otherStim(RandRegRand_LOT):
    """
        Saffran paradigm with different stimulis.
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
class RandRegRand_LOT_syllable(RandRegRand_LOT):
    """
        Saffran paradigm with different stimulis.
    """

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

@dataclass
class RandRegRand_LOT_MatchedRandom(RandRegRand_LOT):
    """
    Matched-random control:
    - same Rand-Reg-Rand structure
    - same 2-tone pool for middle section
    - middle section is random instead of LOT rule
    """
    reg_len: int = 16
    balanced_middle: bool = True  # keep 50/50 A/B in each middle block

    def _random_binary_pattern(self) -> list[int]:
        if not self.balanced_middle:
            return np.random.choice([0, 1], size=self.reg_len, replace=True).tolist()

        n0 = self.reg_len // 2
        n1 = self.reg_len - n0
        p = np.array([0] * n0 + [1] * n1, dtype=int)
        np.random.shuffle(p)
        return p.tolist()

    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool], list[Sequence]]:
        # Middle uses the same 2-tone pool as the algebraic condition (s_reg),
        # so the only difference between conditions is arrangement, not spectral diversity.
        if self.s_rand is not None:
            all_pool = [self.s_rand] + [self.s_reg for _ in range(self.motif_repeat)] + [self.s_rand]
        else:
            s_rand = Sound_pool.from_list(self.sound_pool.pick_norepeat_n(self.rand_size))
            s_reg = Sound_pool.from_list(s_rand.pick_norepeat_n(2))
            all_pool = [s_rand] + [s_reg for _ in range(self.motif_repeat)] + [s_rand]

        middle_seq = []
        for _ in range(self.motif_repeat):
            rp = RandomPattern(isi=self.isi, nb_unique_elements=2, len=self.reg_len)
            middle_seq.append(rp)

        all_seq = [self.randSeq] + middle_seq + [self.randSeqEnd]
        return all_pool, all_seq
