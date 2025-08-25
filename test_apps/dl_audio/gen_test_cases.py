#!/usr/bin/env python3
"""
generate_fbank_bin_v2.py
把 config + fbank 矩阵一次性写入 fbank_cases.bin
"""
import struct

import numpy as np
import torchaudio
import torchaudio.compliance.kaldi as kaldi

WAV_FILE = "test_cases/test.wav"
FBANK_FILE = "test_cases/test_fbank.bin"
SPECT_FILE = "test_cases/test_spectrogram.bin"
MFCC_FILE = "test_cases/test_mfcc.bin"

test_configs = [
    dict(
        frame_shift=10.0,
        frame_length=25.0,
        num_mel_bins=26,
        num_ceps=13,
        use_power=True,
        use_log_fbank=True,
        low_freq=0.0,
        high_freq=0.0,
        preemphasis_coefficient=0.97,
        remove_dc_offset=True,
        window_type="hanning",
    ),
    dict(
        frame_shift=10.0,
        frame_length=25.0,
        num_mel_bins=26,
        num_ceps=13,
        use_power=True,
        use_log_fbank=False,
        low_freq=0.0,
        high_freq=8000.0,
        preemphasis_coefficient=0.0,
        remove_dc_offset=True,
        window_type="hamming",
    ),
    dict(
        frame_shift=32.0,
        frame_length=32.0,
        num_mel_bins=80,
        num_ceps=30,
        use_power=True,
        use_log_fbank=True,
        low_freq=50.0,
        high_freq=7000.0,
        preemphasis_coefficient=0.97,
        remove_dc_offset=False,
        window_type="povey",
    ),
    dict(
        frame_shift=16.0,
        frame_length=32.0,
        num_mel_bins=48,
        num_ceps=11,
        use_power=False,
        use_log_fbank=False,
        low_freq=0.0,
        high_freq=0.0,
        preemphasis_coefficient=0.0,
        remove_dc_offset=False,
        window_type="rectangular",
    ),
    dict(
        frame_shift=32.0,
        frame_length=50.0,
        num_mel_bins=32,
        num_ceps=23,
        use_power=True,
        use_log_fbank=True,
        low_freq=100.0,
        high_freq=6000.0,
        preemphasis_coefficient=0.97,
        remove_dc_offset=False,
        window_type="rectangular",
    ),
]


def pack_window(wt: str) -> bytes:
    """把 window_type 填充成 16 字节，0 结尾"""
    b = wt.encode("ascii")
    return b[:15].ljust(16, b"\0")


def save_test_cases(f, cfg, mat):
    mat = mat.numpy().astype(np.float32)
    T, D = mat.shape
    print(cfg, T, D)

    # 写 config
    f.write(
        struct.pack(
            "<fffff",
            cfg["frame_shift"],
            cfg["frame_length"],
            cfg["low_freq"],
            cfg["high_freq"],
            cfg["preemphasis_coefficient"],
        )
    )
    f.write(
        struct.pack(
            "<IIIII",
            int(cfg["num_mel_bins"]),
            int(cfg["num_ceps"]),
            int(cfg["use_power"]),
            int(cfg["use_log_fbank"]),
            int(cfg["remove_dc_offset"]),
        )
    )
    f.write(pack_window(cfg["window_type"]))
    f.write(struct.pack("<II", T, D))
    # 写矩阵
    f.write(mat.tobytes())


def test_fbank(output_file: str = FBANK_FILE):
    waveform, sr = torchaudio.load(WAV_FILE)
    assert sr == 16000
    waveform = waveform[0]
    print(waveform.shape)

    with open(output_file, "wb") as f:
        f.write(struct.pack("<II", 0xFBA5FBA5, len(test_configs)))  # header
        for cfg in test_configs:
            mat = kaldi.fbank(
                waveform.unsqueeze(0),
                sample_frequency=sr,
                num_mel_bins=cfg["num_mel_bins"],
                frame_shift=cfg["frame_shift"],
                frame_length=cfg["frame_length"],
                use_power=cfg["use_power"],
                use_log_fbank=cfg["use_log_fbank"],
                low_freq=cfg["low_freq"],
                high_freq=cfg["high_freq"],
                preemphasis_coefficient=cfg["preemphasis_coefficient"],
                window_type=cfg["window_type"],
                remove_dc_offset=cfg["remove_dc_offset"],
                use_energy=False,
            )
            save_test_cases(f, cfg, mat)
    print(f"Wrote {len(test_configs)} cases with config -> {output_file}")


def test_spectrogram(output_file):
    waveform, sr = torchaudio.load(WAV_FILE)
    assert sr == 16000
    waveform = waveform[0]
    print(waveform.shape, waveform[0], waveform[1000], waveform.dtype)

    with open(output_file, "wb") as f:
        f.write(struct.pack("<II", 0xFBA5FBA5, len(test_configs)))  # header
        for cfg in test_configs:
            mat = kaldi.spectrogram(
                waveform.unsqueeze(0),
                sample_frequency=sr,
                frame_shift=cfg["frame_shift"],
                frame_length=cfg["frame_length"],
                preemphasis_coefficient=cfg["preemphasis_coefficient"],
                window_type=cfg["window_type"],
                remove_dc_offset=cfg["remove_dc_offset"],
                energy_floor=0.0,
            )
            cfg["use_log_fbank"] = True
            cfg["use_power"] = True
            save_test_cases(f, cfg, mat)
    print(f"Wrote {len(test_configs)} cases with config -> {output_file}")


def test_mfcc(output_file: str = FBANK_FILE):
    waveform, sr = torchaudio.load(WAV_FILE)
    assert sr == 16000
    waveform = waveform[0]
    print(waveform.shape)

    with open(output_file, "wb") as f:
        f.write(struct.pack("<II", 0xFBA5FBA5, len(test_configs)))  # header
        for cfg in test_configs:
            mat = kaldi.mfcc(
                waveform.unsqueeze(0),
                sample_frequency=sr,
                num_mel_bins=cfg["num_mel_bins"],
                num_ceps=cfg["num_ceps"],
                frame_shift=cfg["frame_shift"],
                frame_length=cfg["frame_length"],
                low_freq=cfg["low_freq"],
                high_freq=cfg["high_freq"],
                preemphasis_coefficient=cfg["preemphasis_coefficient"],
                window_type=cfg["window_type"],
                remove_dc_offset=cfg["remove_dc_offset"],
                use_energy=False,
            )
            cfg["use_log_fbank"] = True
            cfg["use_power"] = True
            save_test_cases(f, cfg, mat)
    print(f"Wrote {len(test_configs)} cases with config -> {output_file}")


if __name__ == "__main__":
    test_fbank(FBANK_FILE)
    test_spectrogram(SPECT_FILE)
    test_mfcc(MFCC_FILE)
    # test_kaldifeat(FBANK_FILE)
