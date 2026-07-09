from cstim.sounds.perExperiment.sound_elements.sound_class import Sound,Sound_pool
from cstim.sounds.perExperiment.sound_elements.tones_elements import Silence
from dataclasses import dataclass,field
import numpy as np
from typing import Union


@dataclass(frozen=False)
class Sequence:
    """
    The basic sound sequences, consisting of a pattern, made of nb_unique_elements unique elements.
    We take from a sound_pool each element according to the pattern, and add silence if needed
    """
    pattern: list[int] = field(init=False)
    nb_unique_elements: int = field(init=False)
    isi: float = 0.0  # in seconds

    def __call__(self, spool: Sound_pool) -> list[Sound]:
        try:
            assert len(np.unique([s.samplerate for s in spool]))==1 # shared samplerate
        except:
            raise Exception("Sounds should have a similar samplerate to be used in a sequence")

        out_s = []
        for p in self.pattern:
            out_s += [spool[p]]
            if self.isi > 0:
                out_s += [Silence(samplerate=spool[p].samplerate,duration=self.isi)]
        return out_s