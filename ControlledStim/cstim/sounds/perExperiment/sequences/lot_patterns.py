import copy

import numpy as np
from cstim.sounds.perExperiment.sequences import Sequence
from dataclasses import dataclass,field
from typing import Iterable
## Note: I am not convinced this is the best way to do things...
## We generate all sound classes based on dictionnary

@dataclass
class LOT_p(Sequence):
    name: str = field(default_factory=str)
    deviant_pos: list[int] = field(init=False)
    def get_deviant_pattern(self) -> Iterable[list[int]]:
        for dp in self.deviant_pos:
            p = [p for p in self.pattern]
            p[dp] = 1 - p[dp]
            yield p

    def as_deviant_pattern(self, id_deviant: int):
        self.pattern[self.deviant_pos[id_deviant]] = 1 - self.pattern[self.deviant_pos[id_deviant]]
        self.name = self.name + "_deviant"
    def as_deviant_pattern_from_pos(self, id_pos: int):
        self.pattern[id_pos] = 1 - self.pattern[id_pos]
        self.name = self.name + "_deviant"

@dataclass
class LOT_repeat(LOT_p):
    name = "repeat"
    def __post_init__(self):
        self.pattern = list(np.concatenate([[0,0] for _ in range(8)]))
        self.deviant_pos = [8,11,12,14]
@dataclass
class LOT_alternate(LOT_p):
    name = "alternate"
    def __post_init__(self):
        self.pattern = list(np.concatenate([[0,1] for _ in range(8)]))
        self.deviant_pos = [8,11,13,14]

@dataclass
class LOT_pairs(LOT_p):
    name = "pairs"
    def __post_init__(self):
        self.pattern = list(np.concatenate([[0,0,1,1] for _ in range(4)]))
        self.deviant_pos = [9,10,13,14]

@dataclass
class LOT_quadruplets(LOT_p):
    name = "quadruplets"
    def __post_init__(self):
        self.pattern =  [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]
        self.deviant_pos = [8,11,12,14]

@dataclass
class LOT_pairsAndAlt1(LOT_p):
    name = "pairsAndAlt1"
    def __post_init__(self):
        self.pattern = [0,0,1,1,0,1,0,1,0,0,1,1,0,1,0,1]
        self.deviant_pos = [9,10,13,14]

@dataclass
class LOT_shrinking(LOT_p):
    name = "shrinking"
    def __post_init__(self):
        self.pattern = [0,0,0,0,1,1,1,1,0,0,1,1,0,1,0,1]
        self.deviant_pos = [9,10,13,14]

@dataclass
class LOT_pairsAndAlt2(LOT_p):
    name = "pairsAndAlt2"
    def __post_init__(self):
        self.pattern = [0,1,0,0,1,1,0,1,0,1,0,0,1,1,0,1]
        self.deviant_pos = [9,11,12,14]
@dataclass
class LOT_threetwo(LOT_p):
    name = "threetwo"
    def __post_init__(self):
        self.pattern = [0,0,0,1,1,0,1,1,0,0,0,1,1,0,1,1]
        self.deviant_pos = [8,10,12,14]

@dataclass
class LOT_centermirror(LOT_p):
    name = "centermirror"
    def __post_init__(self):
        self.pattern = [0,0,0,0,1,1,0,1,0,1,0,0,1,1,1,1]
        self.deviant_pos = [8,11,12,14]

@dataclass
class LOT_complex(LOT_p):
    name = "complex"
    def __post_init__(self):
        self.pattern = [0,1,0,0,0,1,1,1,1,0,1,1,0,0,0,1]
        self.deviant_pos = [8,11,13,14]
orig_lot_patterns = {"repeat": LOT_repeat,
               "alternate": LOT_alternate,
               "pairs": LOT_pairs,
               "quadruplets": LOT_quadruplets,
               "pairsAndAlt1": LOT_pairsAndAlt1,
               "shrinking": LOT_shrinking,
               "pairsAndAlt2": LOT_pairsAndAlt2,
               "threetwo": LOT_threetwo,
               "centermirror": LOT_centermirror,
               "complex": LOT_complex}

lot_patterns = copy.deepcopy(orig_lot_patterns)
new_lot_patterns = {}
def lambdaseq(X, Y, i):
    seq = [[X, X, X, Y, Y, Y, X, X, Y, Y, X, Y, Y, X, Y, Y],
           [X, X, X, Y, Y, Y, X, X, Y, Y, X, Y, X, Y, X, Y],
           [X, X, X, Y, Y, Y, X, X, Y, Y, X, Y, X, X, Y, Y],
           [X, Y, X, Y, X, Y, X, X, Y, Y, X, X, X, Y, Y, Y],
           [X, X, Y, Y, X, X, X, Y, Y, Y, X, X, X, X, Y, Y],
           [X, Y, Y, Y, X, Y, X, X, X, Y, X, Y, Y, Y, X, Y],
           [X, Y, Y, X, Y, X, X, Y, X, Y, Y, X, Y, X, X, Y],
           [X, X, X, Y, X, X, Y, X, X, Y, X, X, Y, X, X, X],
           [X, Y, Y, Y, Y, X, Y, X, X, X, X, Y, X, Y, Y, X],
           [X, X, Y, Y, Y, X, X, X, X, Y, Y, Y, X, X, Y, X]]
    return seq[i]

for id_newlot in range(10):
    @dataclass
    class newLOT(LOT_p):
        name = "newLOT-"+str(id_newlot)
        id_newlot = id_newlot
        def __post_init__(self):
            self.pattern = lambdaseq(0,1,self.id_newlot)
            self.deviant_pos = [8,11,13,14]
    lot_patterns["newLOT-"+str(id_newlot)] = newLOT
    new_lot_patterns["newLOT-"+str(id_newlot)] = newLOT