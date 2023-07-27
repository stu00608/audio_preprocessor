import numpy as np
import soundfile as sf
import os
import shutil
from audio_preprocessor.resample import audio_resampler
from audio_preprocessor.slicer import audio_slicer
from audio_preprocessor.splitter import audio_splitter
from audio_preprocessor.whisper_transcriber import whisper_transcriber_pipeline
from pathlib import Path

TEST_DIR = Path(__file__).parent / "test_data"


def generate_sine_wave(freq, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return 0.5 * np.sin(2 * np.pi * freq * t)


def setup_module(module):
    os.makedirs(TEST_DIR, exist_ok=True)
    sf.write(TEST_DIR / "test.wav", generate_sine_wave(440, 5, 22050), 22050)
    sf.write(TEST_DIR / "test.flac", generate_sine_wave(440, 5, 22050), 22050)


def teardown_module(module):
    shutil.rmtree(TEST_DIR)


def test_resample():
    output_dir = TEST_DIR / "resample_output"
    os.makedirs(output_dir, exist_ok=True)
    try:
        audio_resampler(TEST_DIR, output_dir, 44100, True, 1)
        for file in list(output_dir.glob('**/*.flac')) + list(output_dir.glob('**/*.wav')):
            data, sample_rate = sf.read(file)
            assert sample_rate == 44100
    finally:
        shutil.rmtree(output_dir)


def test_slicer():
    output_dir = TEST_DIR / "slicer_output"
    os.makedirs(output_dir, exist_ok=True)
    try:
        audio_slicer(TEST_DIR, output_dir, -40, 5000, 300, 10, 500, 1)
        # Test that files are created in the output directory
        assert len(list(output_dir.glob('**/*.wav'))) > 0
    finally:
        shutil.rmtree(output_dir)


def test_split():
    output_dir = TEST_DIR / "split_output"
    os.makedirs(output_dir, exist_ok=True)
    try:
        audio_splitter(TEST_DIR, output_dir, 1, 1)
        # Test that files are created in the output directory
        assert len(list(output_dir.glob('**/*.flac'))) + \
            len(list(output_dir.glob('**/*.wav'))) > 0
        # Test that the files are of the correct length, remember the last segment may be shorter
        for file in list(output_dir.glob('**/*.flac')) + list(output_dir.glob('**/*.wav')):
            data, sample_rate = sf.read(file)
            assert len(data) / sample_rate <= 1

    finally:
        shutil.rmtree(output_dir)


def test_whisper():
    output_dir = TEST_DIR / "whisper_output"
    os.makedirs(output_dir, exist_ok=True)
    try:
        text, lang = whisper_transcriber_pipeline(
            "./examples/", output_dir, "base", "")
        # Test that files are created in the output directory
        assert len(list(output_dir.glob('**/*.txt'))) > 0
        # Read the text file and check that it contains the correct text
        # expected_text = "柔らかいウールの方が あらゆうるより 効果で そのどちらとも 内論性の人口線より 情答である。"
        # with open(output_dir / "test.txt", "r") as f:
        #     assert f.read().strip("\n") == expected_text
    finally:
        shutil.rmtree(output_dir)
