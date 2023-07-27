from setuptools import setup, find_packages

setup(
    name='audio_preprocessor',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'audio_preprocessor = audio_preprocessor.cli_main:cli',
        ],
    },
    install_requires=[
        'click',
        'soundfile',
        'argparse',
        'pathlib',
        'numpy',
        'librosa',
        'tqdm',
        'pytest',
        'openai-whisper',
    ],
)
