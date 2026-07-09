import numpy as np

from cstim.sounds.perExperiment.sequences import Sequence
from dataclasses import dataclass,field

## Question: what is the advantage of subclassing the Sequence object?
# instead of using it directly ? --> adding further property

@dataclass
class Local_Deviant(Sequence):
    def __post_init__(self):
        self.pattern = [0,0,0,0,1]
        self.nb_unique_elements = 2

@dataclass
class Local_Standard(Sequence):
    def __post_init__(self):
        self.pattern = [0,0,0,0,0]
        self.nb_unique_elements = 1

@dataclass
class ToneList(Sequence):
    cycle : int = 5
    def __post_init__(self):
        self.pattern = list(range(self.cycle))
        self.nb_unique_elements = self.cycle

@dataclass
class RandomPattern(Sequence):
    len : int = 1
    nb_unique_elements : int = 1
    def __post_init__(self):
        self.pattern  = np.random.choice(np.arange(self.nb_unique_elements),self.len,replace=True)

@dataclass
class FullCommunityGraph(Sequence):
    nb_nodes : int = 12
    walk_length : int = 16
    graph : np.ndarray = field(init=False)
    def __post_init__(self):
        ## The graph is represented as an adjency matrix:
        community = np.ones((self.nb_nodes//2,self.nb_nodes//2))
        community[np.diag_indices(self.nb_nodes//2)]=False
        graphTop = np.concatenate([community,np.zeros_like(community)],axis=1)
        graphBot = np.concatenate([np.zeros_like(community),community],axis=1)
        graph = np.concatenate([graphTop,graphBot],axis=0)
        ## Add the link between the community:
        graph[0,self.nb_nodes//2]=True
        graph[self.nb_nodes//2,0]=True
        graph[self.nb_nodes//2-1,-1]=True
        graph[-1,self.nb_nodes//2-1]=True
        self.graph = graph
        startWalk = np.random.randint(0,self.nb_nodes,1)[0]
        randomWalk = [startWalk]
        for _ in range(self.walk_length-1):
            possibilities = np.where(self.graph[randomWalk[-1]])[0]
            randomWalk += [np.random.choice(possibilities,1)[0]]
        self.pattern = randomWalk

    def out_community_step(self) -> None:
        ## Given the current pattern, extend it by a step from the community to another:
        if self.pattern[-1]<self.nb_nodes//2:
            target = np.where(np.logical_not(self.graph[self.pattern[-1]][self.nb_nodes//2:]))[0]+self.nb_nodes//2
        else:
            target = np.where(np.logical_not(self.graph[self.pattern[-1]][:self.nb_nodes//2]))[0]
        assert len(target)>0
        pick_target = np.random.choice(target,1)[0]
        self.pattern.append(pick_target)

    def in_community_step(self) -> None:
        ## Given the current pattern, extend it by a step within the community:
        if self.pattern[-1]<self.nb_nodes//2:
            target = np.where(np.logical_not(self.graph[self.pattern[-1]][:self.nb_nodes//2]))[0]
        else:
            target = np.where(np.logical_not(self.graph[self.pattern[-1]][self.nb_nodes//2:]))[0]+self.nb_nodes//2
        assert len(target)>0
        pick_target = np.random.choice(target,1)[0]
        self.pattern.append(pick_target)

@dataclass
class SparseCommunityGraph(FullCommunityGraph):
    nb_nodes : int = 12
    walk_length : int = 16
    graph : np.ndarray = field(init=False)

    def __post_init__(self):
        ## The graph is represented as an adjency matrix:
        communityA = np.array([[0,0,1,1,1,0],[0,0,1,1,1,1],[1,1,0,1,1,0],[1,1,1,0,0,1],[1,1,1,0,0,1],[0,1,0,1,1,0]],dtype=bool)
        communityB = np.array([[0,1,1,0,1,0],[1,0,1,1,1,0],[1,1,0,1,0,1],[0,1,1,0,1,1],[1,1,0,1,0,1],[0,0,1,1,1,0]],dtype=bool)
        graphTop = np.concatenate([communityA,np.zeros_like(communityA)],axis=1)
        graphBot = np.concatenate([np.zeros_like(communityB),communityB],axis=1)
        graph = np.concatenate([graphTop,graphBot],axis=0)
        ## Add the link between the community:
        graph[0,self.nb_nodes//2]=True
        graph[self.nb_nodes//2,0]=True
        graph[self.nb_nodes//2-1,-1]=True
        graph[-1,self.nb_nodes//2-1]=True

        startWalk = np.random.randint(0,self.nb_nodes,1)[0]
        randomWalk = [startWalk]
        for _ in range(self.walk_length-1):
            possibilities = np.where(graph[randomWalk[-1]])[0]
            randomWalk += [np.random.choice(possibilities,1)[0]]
        self.pattern = randomWalk

@dataclass
class HighSparseCommunityGraph(FullCommunityGraph):
    nb_nodes : int = 12
    walk_length : int = 16
    graph : np.ndarray = field(init=False)

    def __post_init__(self):
        self.pattern = list(range(self.cycle))
        self.nb_unique_elements = self.cycle


@dataclass
class WordStream(Sequence):
    size_words: int = 3
    nb_words: int = 4
    len : int = 42 # the total number of words in the simulation

    def __post_init__(self):
        # A stream that is made of an alternation of "words" each constitued of size_word elements
        # for a total of nb_words*size_words elements (default: 4*3 = 12)

        if self.nb_words < 1:
            raise ValueError("WordStream requires nb_words >= 1.")
        if self.len < 0:
            raise ValueError("WordStream requires len >= 0.")
        if self.len > 1 and self.nb_words < 2:
            raise ValueError(
                "Cannot generate a stream of length > 1 without adjacent repeats when nb_words < 2."
            )

        # Sample word sequence while forbidding immediate repeats
        word_pattern = np.empty(self.len, dtype=int)
        if self.len > 0:
            word_pattern[0] = np.random.randint(self.nb_words)
            for i in range(1, self.len):
                prev = word_pattern[i - 1]
                candidates = np.delete(np.arange(self.nb_words), prev)
                word_pattern[i] = np.random.choice(candidates)

        self.nb_unique_elements = self.nb_words * self.size_words

        # Expand each word ID into its syllable IDs
        self.pattern = np.repeat(word_pattern, self.size_words) * self.size_words
        for i in range(self.size_words):
            self.pattern[i::self.size_words] = self.pattern[::self.size_words] + i
