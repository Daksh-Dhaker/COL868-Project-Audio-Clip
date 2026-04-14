import os
import warnings
import multiprocessing as mp

import tqdm
import librosa
import soundfile as sf

import numpy as np
import pandas as pd

import torch.utils.data as td

import sklearn.model_selection as skms

import utils.transforms as transforms

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional


class UrbanSound8K(td.Dataset):

    def __init__(self,
                 root: str,
                 sample_rate: int = 22050,
                 train: bool = True,
                 fold: Optional[int] = None,
                 mono: bool = False,
                 transform_audio=None,
                 target_transform=None,
                 **_):

        super(UrbanSound8K, self).__init__()

        self.root = root
        self.sample_rate = sample_rate
        self.train = train
        self.random_split_seed = None

        if fold is None:
            fold = 1

        if not (1 <= fold <= 10):
            raise ValueError(f'Expected fold in range [1, 10], got {fold}')

        self.fold = fold
        self.folds_to_load = set(range(1, 11))

        if self.fold not in self.folds_to_load:
            raise ValueError(f'fold {fold} does not exist')

        if self.train:
            # if in training mode, keep all but test fold
            self.folds_to_load -= {self.fold}
        else:
            # if in evaluation mode, keep the test samples only
            self.folds_to_load -= self.folds_to_load - {self.fold}

        self.mono = mono

        self.transform = transform_audio
        self.target_transform = target_transform

        self.data: Dict[str, Dict[str, Any]] = dict()
        self.indices = dict()
        self.load_data()

        self.class_idx_to_label = dict()
        for row in self.data.values():
            idx = row['target']
            label = row['category']
            self.class_idx_to_label[idx] = label
        self.label_to_class_idx = {lb: idx for idx, lb in self.class_idx_to_label.items()}

    @staticmethod
    def _load_worker(fn: str, path_to_file: str, sample_rate: int, mono: bool = False) -> Tuple[str, int, np.ndarray]:
        # Always load as mono for consistent shape [1, N]
        wav, sample_rate_ = librosa.load(path_to_file, sr=sample_rate, mono=True)
        wav = wav[np.newaxis, :]  # shape [1, N]
        # Truncate or pad to 4 seconds (like before)
        wav = wav[:, :sample_rate * 4]
        # Scale to match previous behavior
        wav = transforms.scale(wav, wav.min(), wav.max(), -32768.0, 32767.0)
        return fn, sample_rate, wav.astype(np.float32)

    def load_data(self):
        # Robust loader: only add files that exist, like zero_shot_eval.py
        meta_path = os.path.join(self.root, 'metadata', 'UrbanSound8K.csv')
        audio_root = os.path.join(self.root, 'audio')
        if not os.path.exists(meta_path):
            alt_path = os.path.join(self.root, 'UrbanSound8K.csv')
            if os.path.exists(alt_path):
                meta_path = alt_path
                audio_root = self.root
            else:
                raise FileNotFoundError(f"UrbanSound8K.csv not found in 'metadata/' or root of {self.root}")
        meta = pd.read_csv(meta_path, sep=',', index_col='slice_file_name')

        self.data = dict()
        for fn, row in meta.iterrows():
            fold = int(row['fold'])
            path = os.path.join(audio_root, f'fold{fold}', fn)
            if os.path.exists(path):
                self.data[fn] = path, self.sample_rate, self.mono

        files_to_load = list()
        if self.random_split_seed is not None:
            skf = skms.StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_split_seed)
            for fold_idx, (train_ids, test_ids) in enumerate(skf.split(
                    np.zeros(len(meta)), meta['classID'].values.astype(int)
            ), 1):
                if fold_idx == self.fold:
                    ids = train_ids if self.train else test_ids
                    filenames = meta.iloc[ids].index
                    files_to_load.extend([fn for fn in filenames if fn in self.data])
                    break
        else:
            for fn, row in meta.iterrows():
                if int(row['fold']) in self.folds_to_load and fn in self.data:
                    files_to_load.append(fn)

        self.data = {fn: vals for fn, vals in self.data.items() if fn in files_to_load}
        self.indices = {idx: fn for idx, fn in enumerate(self.data)}

        num_processes = os.cpu_count()
        warnings.filterwarnings('ignore')
        with mp.Pool(processes=num_processes) as pool:
            tqdm.tqdm.write(f'Loading {self.__class__.__name__} (train={self.train})')
            for fn, sample_rate, wav in pool.starmap(
                func=self._load_worker,
                iterable=[(fn, path, sr, mono) for fn, (path, sr, mono) in self.data.items()],
                chunksize=int(np.ceil(len(meta) / num_processes)) or 1
            ):
                self.data[fn] = {
                    'audio': wav,
                    'sample_rate': sample_rate,
                    'target': meta.loc[fn, 'classID'],
                    'category': meta.loc[fn, 'class'].replace('_', ' ').strip(' '),
                    'background': bool(meta.loc[fn, 'salience'] - 1)
                }

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        if not (0 <= index < len(self)):
            raise IndexError

        audio: np.ndarray = self.data[self.indices[index]]['audio']
        target: str = self.data[self.indices[index]]['category']

        if self.transform is not None:
            audio = self.transform(audio)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return audio, None, [target]

    def __len__(self) -> int:
        return len(self.data)
