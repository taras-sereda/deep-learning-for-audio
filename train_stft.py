import torch
import torch.nn as nn
import torch.utils.data
import torchaudio

from torch.optim import Adam

from stft import STFT


class Collate(object):
    def __call__(self, batch):
        n_samples = 22050
        wav_truncated = torch.FloatTensor(len(batch), n_samples)
        for i in range(len(batch)):
            wav = batch[i][0]
            assert wav.shape[-1] >= n_samples
            wav_truncated[i, :] = wav[0, :n_samples]
        return wav_truncated


dataset = torchaudio.datasets.LJSPEECH('./data')
dataset = torch.utils.data.Subset(dataset, range(100))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=Collate())

stft_deterministic = STFT(filter_length=256, hop_length=128, win_length=256)
stft_model = STFT(filter_length=256, hop_length=128, win_length=256, trainable=True)

criterion = nn.MSELoss()
optimizer = Adam(stft_model.parameters(), lr=1e-1)

n_epoch = 100
torch.save(stft_model.state_dict(), f'./experiments/trainable_fft_{-1}')
for epoch in range(n_epoch):

    for i, batch in enumerate(dataloader):
        stft_model.zero_grad()
        targ = stft_deterministic(batch)
        pred = stft_model(batch)
        loss = criterion(pred, targ)
        loss.backward()
        optimizer.step()
        print(f'iter: {i}, loss: {loss}')
    torch.save(stft_model.state_dict(), f'./experiments/trainable_fft_{epoch}')
