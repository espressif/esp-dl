# ESP-DL Audio Processing Module

The ESP-DL Audio Processing Module is a C++ library designed for audio signal processing, particularly focused on speech feature extraction. It provides implementations of common audio processing algorithms optimized for ESP platforms.

## Features

- WAV file decoding
- Speech feature extractionincluding:
  - Fbank (Filter Bank) features
  - MFCC (Mel-Frequency Cepstral Coefficients)
  - Spectrogram
- Support for various window functions (Hanning, Hamming, Povey, etc.)
- Configurable parameters for feature extraction
- Optimized for ESP platforms with memory allocation capabilities
- Aligned with Kaldi's implementation [(torchaudio.compliance.kaldi)](https://docs.pytorch.org/audio/stable/compliance.kaldi.html).

## Directory Structure

```
audio/
├── common/              # Common audio processing utilities
│   ├── dl_audio_common.cpp/hpp  # Common audio functions and definitions
│   └── dl_audio_wav.cpp/hpp     # WAV file decoding utilities
└── speech_features/     # Speech feature extraction algorithms
    ├── dl_fbank.cpp/hpp         # Fbank (Filter Bank) feature extraction
    ├── dl_mfcc.cpp/hpp          # MFCC (Mel-Frequency Cepstral Coefficients)
    ├── dl_spectrogram.cpp/hpp   # Spectrogram feature extraction
    └── dl_speech_features.cpp/hpp # Base class for speech features
```

## Common Audio Utilities

### WAV Decoding
The module provides functionality to decode WAV audio files into raw PCM data.

```cpp
#include "dl_audio_wav.hpp"

dl::audio::dl_audio_t *audio = dl::audio::decode_wav(wav_data, data_len);
```

### Audio Common Functions
Provides common audio processing functions such as:
- Window function generation (Hanning, Hamming, Blackman, etc.)
- Mel filterbank initialization
- Pre-emphasis filtering
- FFT-related operations

## Speech Feature Extraction

All speech feature extraction classes inherit from the `SpeechFeatureBase` class, which provides a common interface.

### Configuration
Speech feature extraction can be configured using the `SpeechFeatureConfig` structure:

```cpp
dl::audio::SpeechFeatureConfig config;
config.sample_rate = 16000;
config.frame_length = 25;  // ms
config.frame_shift = 10;   // ms
config.num_mel_bins = 26;
config.window_type = dl::audio::WinType::HANNING;
```

### Fbank (Filter Bank)
Extracts filter bank features from audio signals.

```cpp
#include "dl_fbank.hpp"

dl::audio::Fbank fbank(config);
// Process audio data
std::vector<int> shape = fbank.get_output_shape(audio_length);
float *output_features = (float*) malloc(shape[0] * shape[1]);
fbank.process(audio_data, audio_length, output_features);
```

### MFCC (Mel-Frequency Cepstral Coefficients)
Extracts MFCC features, which are commonly used in speech recognition.

```cpp
#include "dl_mfcc.hpp"

dl::audio::MFCC mfcc(config);
// Process audio data
std::vector<int> shape = mfcc.get_output_shape(audio_length);
float *output_features = (float*) malloc(shape[0] * shape[1]);
mfcc.process(audio_data, audio_length, output_features);
```

### Spectrogram
Computes spectrogram features aligned with torchaudio.compliance.kaldi.spectrogram.

```cpp
#include "dl_spectrogram.hpp"

dl::audio::Spectrogram spectrogram(config);
// Process audio data
std::vector<int> shape = mfcc.get_output_shape(audio_length);
float *output_features = (float*) malloc(shape[0] * shape[1]);
spectrogram.process(audio_data, audio_length, output_features);
```


## License

MIT License