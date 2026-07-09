import pandas as pd
from cstim.sounds.perExperiment.sequences import Sequence
from cstim.sounds.perExperiment.sequences.patterns import WordStream
from cstim.sounds.perExperiment.sound_elements import EnglishSyllable,FrenchSyllable,SoundSegment,Bip

from cstim.sounds.perExperiment.sound_elements import Sound_pool,Sound
from cstim.sounds.perExperiment.protocols.ProtocolGeneration import Protocol_independentTrial
from cstim.sounds.perExperiment.sound_elements import ramp_sound,normalize_sound,Silence
from dataclasses import dataclass,field
import numpy as np
from typing import Union,Tuple,List
from pathlib import Path

@dataclass
class Saffran(Protocol_independentTrial):
    name : str = "Saffran"
    sequence_isi : float = 0
    isi : float = 0
    duration_tone : float = 0.2
    samplerate : int = 16000
    motif_repeat : int = 42
    nb_words : int = 4
    size_words : int = 3
    # tones_fs : Union[list[list[float]],np.ndarray] = field(default_factory=list)
    # s_reg : list[Sound] = field(default=None)


    def words_sample(self,syllables):
        ### Sample randomly one of the sequence:
        # In the original experiment the words have no syllable in common so we respect that here.
        words = []
        to_remove = []
        for _ in range(self.nb_words):
            syl = np.random.choice(np.setdiff1d(range(len(syllables)),to_remove),self.size_words,replace=False)
            words += [[syllables[s] for s in syl]]
            to_remove += [syl]
        words = np.array(words)
        return words


    def __post_init__(self):
        self.name = self.name
        syllables = np.array([["t","u"],["p","i"],["r","o"],["b","i"],["d","a"],["k","u"],
                     ["g","o"],["l","a"],["b","u"],["p","a"],["d","o"],["t","i"]])
        self.syllables = ["".join(e) for e in syllables]


    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool],list[Sequence]]:
        words = self.words_sample(self.syllables)
        sounds_words = [[EnglishSyllable(name=s,syllable=s,samplerate=self.samplerate,duration=self.duration_tone) for s in w] for w in words]
        self.sound_pool = Sound_pool.from_list(np.concatenate(sounds_words))

        regSeq = WordStream(nb_words=self.nb_words, size_words=self.size_words, len=self.motif_repeat)

        all_pool = [self.sound_pool]
        all_seq = [regSeq]
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
        df_info = pd.DataFrame.from_dict({"pattern":[all_seq[0].pattern]})
        return (all_sound,nb_element,df_info)


@dataclass
class Saffran_StressClue(Saffran):
    """
        Saffran paradigm with added stress on the first syllable.
    """
    name : str = "Saffran_stressed"

    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool],list[Sequence]]:
        words = self.words_sample(self.syllables)
        sounds_words = [[EnglishSyllable(name=s,syllable=s,samplerate=self.samplerate,duration=self.duration_tone) for s in w] for w in words]

        # Generate stress with a slower reading speed and increased duration:
        for w in range(len(sounds_words)):
            sounds_words[w][0] = EnglishSyllable(name=sounds_words[w][0].name,
                                                 syllable=sounds_words[w][0].syllable,samplerate=self.samplerate,duration=2*self.duration_tone,
                                                 speed=0.5*160)

        self.sound_pool = Sound_pool.from_list(np.concatenate(sounds_words))

        regSeq = WordStream(nb_words=self.nb_words, size_words=self.size_words, len=self.motif_repeat)

        all_pool = [self.sound_pool]
        all_seq = [regSeq]
        return all_pool,all_seq

@dataclass
class Saffran_otherStim(Saffran):
    """
        Saffran paradigm with different stimulis.
    """
    sound_paths : List[Union[str,Path]] = ""
    start: list[float] = 0
    stop: list[float] = 0.05

    def __post_init__(self):
        self.name = self.name
        syllables =   np.array([["t","u"],["p","i"],["r","o"],["b","i"],["d","a"],["k","u"],
                     ["g","o"],["l","a"],["b","u"],["p","a"],["d","o"],["t","i"]])
        self.syllables = ["".join(e) for e in syllables]

        self.mapping = {s:(self.sound_paths[ide],self.start[ide],self.stop[ide]) for ide,s in enumerate(self.syllables)}


    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool],list[Sequence]]:
        words = self.words_sample(self.syllables)
        sounds_words = [[SoundSegment(name=s,filename=self.mapping[s][0],start=self.mapping[s][1],stop=self.mapping[s][2]) for s in w] for w in words]
        self.sound_pool = Sound_pool.from_list(np.concatenate(sounds_words))

        regSeq = WordStream(nb_words=self.nb_words, size_words=self.size_words, len=self.motif_repeat)

        all_pool = [self.sound_pool]
        all_seq = [regSeq]
        return all_pool,all_seq


@dataclass
class Saffran_Tones(Saffran):
    """
        Saffran paradigm with tones stimulis.
    """
    tones : Union[list[float],np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        self.name = self.name
        syllables =   np.array([["t","u"],["p","i"],["r","o"],["b","i"],["d","a"],["k","u"],
                     ["g","o"],["l","a"],["b","u"],["p","a"],["d","o"],["t","i"]])
        self.syllables = ["".join(e) for e in syllables]
        self.mapping = {s:[self.tones[ide]] for ide,s in enumerate(self.syllables)}


    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool],list[Sequence]]:
        words = self.words_sample(self.syllables)
        sounds_words = [[Bip(name=s,duration=self.duration_tone,fs=self.mapping[s]) for s in w] for w in words]
        self.sound_pool = Sound_pool.from_list(np.concatenate(sounds_words))

        regSeq = WordStream(nb_words=self.nb_words, size_words=self.size_words, len=self.motif_repeat)
        all_pool = [self.sound_pool]
        all_seq = [regSeq]
        return all_pool,all_seq

@dataclass
class Saffran_MVariations(Saffran):
    """
    Generate n base sequences, each with m variants that differ only in the final word.
    - m/2 variants end with a true word
    - m/2 variants end with a part-word (cross-boundary pseudo-word)
    """
    name: str = "Saffran_MVariations"
    m_variations: int = 8  # must be even

    def _sample_word_ids_no_adjacent(self, n_words: int) -> np.ndarray:
        if n_words <= 0:
            return np.array([], dtype=int)
        if n_words > 1 and self.nb_words < 2:
            raise ValueError("Need at least 2 words to avoid adjacent repeats.")

        out = np.empty(n_words, dtype=int)
        out[0] = np.random.randint(self.nb_words)
        for i in range(1, n_words):
            prev = out[i - 1]
            candidates = np.delete(np.arange(self.nb_words), prev)
            out[i] = np.random.choice(candidates)
        return out

    def _word_to_ids(self, word_id: int) -> np.ndarray:
        start = word_id * self.size_words
        return np.arange(start, start + self.size_words, dtype=int)

    def _words_to_pattern(self, word_ids: np.ndarray) -> np.ndarray:
        pattern = np.empty(len(word_ids) * self.size_words, dtype=int)
        for i, w in enumerate(word_ids):
            pattern[i * self.size_words:(i + 1) * self.size_words] = self._word_to_ids(int(w))
        return pattern

    def _sample_partword_ids(self, stream_word_ids: np.ndarray) -> tuple[np.ndarray, dict]:
        # pick a boundary between adjacent words in the stream
        b = np.random.randint(0, len(stream_word_ids) - 1)
        left_w = int(stream_word_ids[b])
        right_w = int(stream_word_ids[b + 1])

        left_ids = self._word_to_ids(left_w)
        right_ids = self._word_to_ids(right_w)

        # split in 1..size_words-1: take final `split` syllables from left and rest from right
        split = np.random.randint(1, self.size_words)
        part_ids = np.concatenate([left_ids[-split:], right_ids[: self.size_words - split]])

        meta = {
            "boundary_left_word_id": left_w,
            "boundary_right_word_id": right_w,
            "split": int(split),
        }
        return part_ids, meta

    def _pattern_to_sounds(self, pattern: np.ndarray, spool: Sound_pool) -> list[Sound]:
        all_sound = []
        for pid in pattern:
            s = ramp_sound(spool[int(pid)], cosine_rmp_length=0.005)
            all_sound.append(s)
            if self.isi > 0:
                all_sound.append(Silence(samplerate=self.samplerate, duration=self.isi))
        return all_sound

    def generate(self, n_trial: int, output_dir: Union[str, Path]) -> pd.DataFrame:
        """
        n_trial = number of base sequences (n)
        each base sequence generates m_variations files
        """
        if self.m_variations % 2 != 0:
            raise ValueError("m_variations must be even.")
        if self.m_variations < 2:
            raise ValueError("m_variations must be >= 2.")
        if self.motif_repeat < 2:
            raise ValueError("motif_repeat must be >= 2 to build part-words from boundaries.")

        name_trials, wav_paths, mask_info_path = [], [], []
        sound_durations, sound_info_paths, number_elements = [], [], []
        trial_infos = []

        for seq_id in range(n_trial):
            # 1) sample lexicon for this base sequence
            words = self.words_sample(self.syllables)
            sounds_words = [
                [EnglishSyllable(name=s, syllable=s, samplerate=self.samplerate, duration=self.duration_tone) for s in w]
                for w in words
            ]
            spool = Sound_pool.from_list(np.concatenate(sounds_words))

            # 2) base stream of words (no identical adjacent words)
            stream_word_ids = self._sample_word_ids_no_adjacent(self.motif_repeat)

            # keep context fixed, vary only the final word/triplet
            context_word_ids = stream_word_ids[:-1]
            context_pattern = self._words_to_pattern(context_word_ids)

            # 3) build endings: m/2 true words + m/2 part-words
            half = self.m_variations // 2
            endings = []

            for _ in range(half):
                wid = int(np.random.randint(self.nb_words))
                endings.append({
                    "ending_type": "word",
                    "ending_ids": self._word_to_ids(wid),
                    "ending_word_id": wid,
                    "boundary_left_word_id": None,
                    "boundary_right_word_id": None,
                    "split": None,
                })

            for _ in range(half):
                part_ids, part_meta = self._sample_partword_ids(stream_word_ids)
                endings.append({
                    "ending_type": "partword",
                    "ending_ids": part_ids,
                    "ending_word_id": None,
                    "boundary_left_word_id": part_meta["boundary_left_word_id"],
                    "boundary_right_word_id": part_meta["boundary_right_word_id"],
                    "split": part_meta["split"],
                })

            endings = list(np.random.permutation(endings))

            # 4) save each variation
            for var_id, e in enumerate(endings):
                full_pattern = np.concatenate([context_pattern, e["ending_ids"]])
                all_sound = self._pattern_to_sounds(full_pattern, spool)

                nb_element = int(np.sum([type(s) != Silence for s in all_sound]))
                name = f"{self.name}_seq-{seq_id}_var-{var_id}"

                sd_out = self._savetrial(all_sound, output_dir, name)

                name_trials.append(name)
                wav_paths.append(str(Path("sounds") / f"{name}.wav"))
                mask_info_path.append(None)
                sound_durations.append(sd_out.shape[0] / self.samplerate)
                sound_info_paths.append(str(Path("sound_info") / f"{name}.csv"))
                number_elements.append(nb_element)

                info = pd.DataFrame.from_dict({
                    "seq_id": [seq_id],
                    "var_id": [var_id],
                    "m_variations": [self.m_variations],
                    "ending_type": [e["ending_type"]],
                    "ending_word_id": [e["ending_word_id"]],
                    "boundary_left_word_id": [e["boundary_left_word_id"]],
                    "boundary_right_word_id": [e["boundary_right_word_id"]],
                    "split": [e["split"]],
                    "size_words": [self.size_words],
                    "nb_words": [self.nb_words],
                    "motif_repeat": [self.motif_repeat],
                })
                trial_infos.append(info)

        df = pd.DataFrame()
        df["name"] = name_trials
        df["wav_path"] = wav_paths
        df["mask_info_path"] = mask_info_path
        df["duration"] = sound_durations
        df["sound_info_path"] = sound_info_paths
        df["number_element"] = number_elements
        df = df.join(pd.concat(trial_infos).set_axis(df.index))
        df.to_csv(Path(output_dir) / "trials.csv", index=False)
        return df
