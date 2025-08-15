# models/losses_combined.py
import torch
import torch.nn as nn

##新增20250813
class MultiResolutionSTFTLoss(nn.Module):
    """HiFi-GAN风格的多分辨率STFT损失"""
    def __init__(self,
                 fft_sizes=[512, 1024, 2048],
                 hop_sizes=[128, 256, 512],
                 win_lengths=[512, 1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.windows = [torch.hann_window(wl) for wl in win_lengths]

    def forward(self, x_pred, x_gt):
        sc_loss, mag_loss = 0.0, 0.0
        for fs, hs, wl, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths, self.windows):
            win = win.to(x_pred.device)
            X_pred = torch.stft(x_pred, fs, hop_length=hs, win_length=wl, window=win,
                                return_complex=True)
            X_gt = torch.stft(x_gt, fs, hop_length=hs, win_length=wl, window=win,
                              return_complex=True)

            mag_pred = torch.abs(X_pred)
            mag_gt = torch.abs(X_gt)

            sc = torch.norm(mag_gt - mag_pred, p='fro') / (torch.norm(mag_gt, p='fro') + 1e-9)
            mag = torch.mean(torch.abs(mag_gt - mag_pred))
            sc_loss += sc
            mag_loss += mag
        return (sc_loss + mag_loss) / len(self.fft_sizes)


class ISTFTConsistencyLoss(nn.Module):
    """预测波形的 STFT→ISTFT→STFT 一致性约束"""
    def __init__(self, fft_size=1024, hop_size=256, win_length=1024):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = torch.hann_window(win_length)

    def forward(self, x_pred, x_gt):
        win = self.window.to(x_pred.device)
        # STFT
        X_pred = torch.stft(x_pred, self.fft_size, hop_length=self.hop_size,
                            win_length=self.win_length, window=win, return_complex=True)
        # ISTFT恢复
        wav_recon = torch.istft(X_pred, self.fft_size, hop_length=self.hop_size,
                                win_length=self.win_length, window=win)
        # 再次STFT
        X_recon = torch.stft(wav_recon, self.fft_size, hop_length=self.hop_size,
                             win_length=self.win_length, window=win, return_complex=True)
        X_gt = torch.stft(x_gt, self.fft_size, hop_length=self.hop_size,
                          win_length=self.win_length, window=win, return_complex=True)

        mag_recon = torch.abs(X_recon)
        mag_gt = torch.abs(X_gt)
        return torch.mean(torch.abs(mag_recon - mag_gt))


class CombinedLoss(nn.Module):
    """Mel + Waveform L1 + MR-STFT + ISTFT"""
    def __init__(self, vocoder, alpha=1.0, beta=1.0, gamma=1.0, delta=0.5):
        super().__init__()
        self.vocoder = vocoder
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.l1 = nn.L1Loss()
        self.mrstft = MultiResolutionSTFTLoss()
        self.istft = ISTFTConsistencyLoss()

    def forward(self, mel_pred, mel_gt, wav_gt):
        # Vocoder解码得到波形预测
        wav_pred = self.vocoder(mel_pred)  # shape: [B, T]

        # 各项损失
        loss_mel = self.l1(mel_pred, mel_gt)
        loss_wave = self.l1(wav_pred, wav_gt)
        loss_mrstft = self.mrstft(wav_pred, wav_gt)
        loss_istft = self.istft(wav_pred, wav_gt)

        # 加权总损失
        total = (self.alpha * loss_mel +
                 self.beta * loss_wave +
                 self.gamma * loss_mrstft +
                 self.delta * loss_istft)

        # 返回总loss和明细
        return total, {
            "loss_total": total.item(),
            "loss_mel": loss_mel.item(),
            "loss_wave": loss_wave.item(),
            "loss_mrstft": loss_mrstft.item(),
            "loss_istft": loss_istft.item()
        }