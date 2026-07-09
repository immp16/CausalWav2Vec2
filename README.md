# CausalWav2Vec2
This repository contains the implementation for a causal version of wav2vec2. Additionally, it provides a script to train the model on a specified auditory dataset. The repository also contains some implementations of relevant paradigms in statistical learning to evaluate the model's in-context learning capacity. The weights for all the pre-trained models have been uploaded to *Hugging Face*. They can be accessed on <https://huggingface.co/immp23/wav2vec2causal/>

# Acknowledgements
The code is based on Pierre Orhan's implementation of *wav2vec2* and the corresponding paradigms aimed at testing the model's ability to extract auditory regularities from novel sequences. For more information, check out <https://github.com/PierreOrhan/>.

# Example
This section contains a use case of *wav2vec2* acquiring regularities in Al Roumi et al. (2023)'s algebraic-pattern detection paradigm. First, download both *SurpriseProbing* and *ControllStim* and install both packages:

```
pip install -e /content/SurpriseProbing
pip install -e /content/ControlledStim
```

Now, generate the stimuli (binary-tone sequences of different levels of complexity):

```
import copy
import numpy as np
import pandas as pd
from pathlib import Path

# Correctly select the location of the package you just installed, otherwise
# the import will give an error.
from ControlledStim.cstim.sounds.perExperiment.protocols.AlRoumi2023 import (
    RandRegRand_LOT,
    RandRegRand_LOT_MatchedRandom,
)

def generate_yoked_repeat_vs_random_separate(
    n_trial: int,
    lot_seq: str,
    output_root: str,
    tones_fs,
    rand_size: int = 16,
    motif_repeat: int = 6,
    reg_len: int = 16,
    isi: float = 0.2,
    sequence_isi: float = 0.0,
    duration_tone: float = 0.05,
    samplerate: int = 16000,
    seed: int = 1234,
):
    rng = np.random.default_rng(seed)

    root = Path(output_root)
    lot_seq_dir = root / lot_seq
    random_dir = root / "matched_random"
    lot_seq_dir.mkdir(parents=True, exist_ok=True)
    random_dir.mkdir(parents=True, exist_ok=True)

    rows_lot_seq = []
    rows_random = []
    rows_all = []

    for k in range(n_trial):
        # Fresh protocol instances per trial (avoid carry-over state)
        p_lot_seq = RandRegRand_LOT(
            lot_seq=lot_seq,
            tones_fs=tones_fs,
            rand_size=rand_size,
            motif_repeat=motif_repeat,
            isi=isi,
            sequence_isi=sequence_isi,
            duration_tone=duration_tone,
            samplerate=samplerate,
            name=f"RandRegRand_LOT_{lot_seq}",
        )

        p_random = RandRegRand_LOT_MatchedRandom(
            lot_seq=lot_seq,  # Label only; middle is randomized in this class
            tones_fs=tones_fs,
            rand_size=rand_size,
            motif_repeat=motif_repeat,
            reg_len=reg_len,
            isi=isi,
            sequence_isi=sequence_isi,
            duration_tone=duration_tone,
            samplerate=samplerate,
            name="RandRegRand_LOT_matched_random",
        )

        # 1) Shared pools (yoking)
        s_rand, s_reg = p_lot_seq.samplePool()
        p_lot_seq.fixPoolSampled(copy.deepcopy(s_rand), copy.deepcopy(s_reg))
        p_random.fixPoolSampled(copy.deepcopy(s_rand), copy.deepcopy(s_reg))

        # 2) Shared post-random pattern (yoking)
        shared_post = rng.integers(0, rand_size, size=rand_size, endpoint=False).tolist()
        p_lot_seq.randSeqEnd.pattern = shared_post.copy()
        p_random.randSeqEnd.pattern = shared_post.copy()

        # 3) Generate + save repeat
        name_lot_seq = f"{lot_seq}_trial-{k}"
        snd_lot_seq, n_lot_seq, info_lot_seq = p_lot_seq._trial()
        dur_lot_seq = p_lot_seq._savetrial(snd_lot_seq, lot_seq_dir, name_lot_seq).shape[0] / p_lot_seq.samplerate

        row_lot_seq = {
            "name": name_lot_seq,
            "pair_id": k,
            "condition": lot_seq,
            "wav_path": str(Path("sounds") / f"{name_lot_seq}.wav"),
            "sound_info_path": str(Path("sound_info") / f"{name_lot_seq}.csv"),
            "duration": dur_lot_seq,
            "number_element": n_lot_seq,
        }
        if isinstance(info_lot_seq, pd.DataFrame) and len(info_lot_seq) > 0:
            for c in info_lot_seq.columns:
                row_lot_seq[c] = info_lot_seq.iloc[0][c]
        rows_lot_seq.append(row_lot_seq)

        # 4) Generate + save matched random
        name_rnd = f"matched_random_trial-{k}"
        snd_rnd, n_rnd, info_rnd = p_random._trial()
        dur_rnd = p_random._savetrial(snd_rnd, random_dir, name_rnd).shape[0] / p_random.samplerate

        row_rnd = {
            "name": name_rnd,
            "pair_id": k,
            "condition": "matched_random",
            "wav_path": str(Path("sounds") / f"{name_rnd}.wav"),
            "sound_info_path": str(Path("sound_info") / f"{name_rnd}.csv"),
            "duration": dur_rnd,
            "number_element": n_rnd,
        }
        if isinstance(info_rnd, pd.DataFrame) and len(info_rnd) > 0:
            for c in info_rnd.columns:
                row_rnd[c] = info_rnd.iloc[0][c]
        rows_random.append(row_rnd)

    # Save per-condition metadata
    df_repeat = pd.DataFrame(rows_lot_seq)
    df_random = pd.DataFrame(rows_random)
    df_repeat.to_csv(lot_seq_dir / "trials.csv", index=False)
    df_random.to_csv(random_dir / "trials.csv", index=False)

    # Save combined metadata (paths relative to output_root)
    for r in rows_lot_seq:
        rr = r.copy()
        rr["wav_path"] = str(Path("repeat") / rr["wav_path"])
        rr["sound_info_path"] = str(Path("repeat") / rr["sound_info_path"])
        rows_all.append(rr)

    for r in rows_random:
        rr = r.copy()
        rr["wav_path"] = str(Path("matched_random") / rr["wav_path"])
        rr["sound_info_path"] = str(Path("matched_random") / rr["sound_info_path"])
        rows_all.append(rr)

    df_all = pd.DataFrame(rows_all)
    df_all.to_csv(root / "trials.csv", index=False)

    return df_repeat, df_random, df_all

tones_fs = np.logspace(np.log(222), np.log(2000), 20, base=np.exp(1))
df_rep, df_rnd, df_all = generate_yoked_repeat_vs_random_separate(
    n_trial=5,
    lot_seq = alg_pattern,
    output_root="/content/randregrand_lot",
    tones_fs=tones_fs,
    rand_size=16,
    motif_repeat=6,
    reg_len=16,
    isi=0.06,
    sequence_isi=0,
    duration_tone=0.10,
    samplerate=16000,
    seed=1234,
)
```

We will now generate masks for each tone so that the model can predict them. That way, we will measure how certain/uncertain the model was when predicting each tone:

```
from ControlledStim.cstim.sounds.experimentsClass.element_masking import mask_and_latent
import os

mask_and_latent(f"/content/randregrand_lot/{alg_pattern}", causal = True)
mask_and_latent(f"/content/randregrand_lot/matched_random", causal = True)
```

Optionally, sync negatives across regular and matched random trial pairs:

```
import zarr as zr

def sync_negatives(root="/content/randregrand_lot", algebraic_pattern = 'repeat', k=2):
    root = Path(root)
    seq = algebraic_pattern
    seq_trials = pd.read_csv(root / seq / "trials.csv")
    rnd_trials = pd.read_csv(root / "matched_random" / "trials.csv")

    # Build pair_id -> row lookup
    seq_map = {int(r["pair_id"]): r for _, r in seq_trials.iterrows()}
    rnd_map = {int(r["pair_id"]): r for _, r in rnd_trials.iterrows()}

    common_pairs = sorted(set(seq_map.keys()) & set(rnd_map.keys()))
    print("pairs to sync:", len(common_pairs))

    n_ok = 0
    for pid in common_pairs:
        r_seq = seq_map[pid]
        rrnd = rnd_map[pid]

        seq_mask_path = root / seq / r_seq["mask_info_path"]
        rnd_mask_path = root / "matched_random" / rrnd["mask_info_path"]

        zg_seq = zr.open_group(seq_mask_path, mode="r")
        zg_rnd = zr.open_group(rnd_mask_path, mode="a")

        neg_seq = zg_seq["sampled_negative_indices"][:]

        # Sanity: shapes must match
        if "sampled_negative_indices" not in zg_rnd:
            raise RuntimeError(f"Missing sampled_negative_indices in {rnd_mask_path}")
        if zg_rnd["sampled_negative_indices"].shape != neg_seq.shape:
            raise RuntimeError(
                f"Shape mismatch for pair {pid}: "
                f"repeat={neg_seq.shape}, random={zg_rnd['sampled_negative_indices'].shape}"
            )

        # Overwrite random negatives with repeat negatives
        del zg_rnd["sampled_negative_indices"]
        zg_rnd.create_array(
            "sampled_negative_indices",
            data=neg_seq,
            chunks=(1, neg_seq.shape[1], neg_seq.shape[2]),
            overwrite=True,
        )
        n_ok += 1

    print("Synced pairs:", n_ok)

# Remember to put the directory where the audio files were stored.
sync_negatives("/content/randregrand_lot", alg_pattern)
```

After downloading one of the checkpoints, we will load it:

```
import torch
import tqdm
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor

# Remember to correctly set the package location
from SurpriseProbing.surprise_probing.ANN.models import Wav2vec2_forLoss_ConstrainedMask

config = Wav2Vec2Config.from_pretrained('patrickvonplaten/wav2vec2-base-v2')

extractor = Wav2Vec2FeatureExtractor.from_pretrained('patrickvonplaten/wav2vec2-base-v2')
extractor.return_attention_mask = True # Or false if in Orhan
#extractor.do_normalize = False
extractor.to_json_file('./extractor.json')

checkpoint = torch.load(f'/content/wav2vec2_fma_causal_100k.checkpoint', weights_only=False, map_location='cuda:0')
model = Wav2vec2_forLoss_ConstrainedMask(
        config,
        'izei'
    ).eval()
missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
model.to('cuda')
```

Now, we will load data into *wav2vec2* and compute contrastive loss for each tone in the sequences. Feel free to change the matched randoms for the algebraic pattern you chose:

```
from SurpriseProbing.surprise_probing.probe.analysers.utils import load_ANNdataset_withMask, load_ANNdataset
from torch.utils.data import DataLoader

ds = load_ANNdataset_withMask(pathlib.Path('/content/randregrand_lot/matched_random'), partially_causal = False)

# Make sure you get inside the folder corresponding to the algebraic-pattern files or the matched-random files! Otherwise, the library that loads audios will not find the files.
%cd /content/randregrand_lot/matched_random/

batch_size = 1
path_config =  Path("/content/")
path_preprocessor = path_config / "extractor.json"
data_collator = Wav2vec2_forLoss_ConstrainedMask.get_collator(file_configPreprocessor = path_preprocessor)
dataLoader = DataLoader(
    ds,
    batch_size=batch_size,
    collate_fn=data_collator,
    shuffle=False,
)

mainloss_name = Wav2vec2_forLoss_ConstrainedMask.get_mainloss_name()

rows = []
model.eval()

for row_id, batch in enumerate(dataLoader):
    inputs = {k: v.to(model.device) for k, v in batch.items() if k != "meta"}

    with torch.no_grad():
        out = model(**inputs)

    rows.append({
        "id": batch['meta'][0][4], # Allows knowing which contrastive loss value corresponds to which audio file. Note that depending on how
                                   # the 'meta' attribute is set, this field could require some modifications.
        "row_id": row_id,
        "loss": float(out[mainloss_name].item()),
    })

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv("/content/experiment_matched_random_fma.csv", index=False)
```

Depending on how the files are stored, it might be the case that this code needs to be adapted to each user's settings, but this would be the general idea and use case of *wav2vec* acquiring auditory regularities after self-sueprvised pre-training.

# Contact
If you find any bug or would like to contact me, you can do so by sending an email to izeimujika@hotmail.com.
