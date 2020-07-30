import pathlib

import torch
import torch.nn as nn
import torch.utils.data
import torchaudio

from torch.optim import Adam

from stft import STFT

DATA_DIR = pathlib.Path('~/data').expanduser()
EXPERIMENTS_DIR = pathlib.Path('./experiments')
experiment_name = 'trainable_fft'


class Collate(object):
    """Audio truncating batch collator"""

    def __init__(self, duration=1.0, sample_rate=22050):
        self.duration = duration
        self.sample_rate = sample_rate

    def __call__(self, batch):
        n_samples = int(self.duration * self.sample_rate)
        wav_truncated = torch.FloatTensor(len(batch), n_samples)
        for i in range(len(batch)):
            wav = batch[i][0]
            assert wav.shape[-1] >= n_samples
            wav_truncated[i, :] = wav[0, :n_samples]
        return wav_truncated


dataset = torchaudio.datasets.LJSPEECH(DATA_DIR)
dataset = torch.utils.data.Subset(dataset, range(100))

# get sample rate from first LJSpeech utterance
# elem (waveform, sample_rate, transcript, normalized_transcript)
sample_rate = dataset[0][1]
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True,
                                         collate_fn=Collate(sample_rate=sample_rate))

stft_deterministic = STFT(filter_length=256, hop_length=128, win_length=256)
stft_model = STFT(filter_length=256, hop_length=128, win_length=256, trainable=True)

criterion = nn.MSELoss()
optimizer = Adam(stft_model.parameters(), lr=1e-1)

n_epoch = 100
torch.save(stft_model.state_dict(), EXPERIMENTS_DIR.joinpath(f'{experiment_name}_{-1}'))
for epoch in range(n_epoch):

    for i, batch in enumerate(dataloader):
        stft_model.zero_grad()
        targ = stft_deterministic(batch)
        pred = stft_model(batch)
        loss = criterion(pred, targ)
        loss.backward()
        optimizer.step()
        print(f'iter: {i}, loss: {loss}')
    torch.save(stft_model.state_dict(), EXPERIMENTS_DIR.joinpath(f'{experiment_name}_{epoch}'))
