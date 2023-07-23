import os
import librosa
import soundfile as sf
import multiprocessing
from pathlib import Path
from multiprocessing import Manager
from audio_preprocessor.utils import print_progress_bar


def resample_audio(input_file_path, output_file_path, target_sample_rate, overwrite, pbar, total):
    # Check if file already exists and overwrite is False
    if not overwrite and os.path.exists(output_file_path):
        print(
            f"File {output_file_path} already exists and overwrite is set to False. Skipping.")
        return

    try:
        data, sample_rate = sf.read(input_file_path)
        if sample_rate == target_sample_rate:
            print(
                f"Sample rate of file {input_file_path} is already {sample_rate}. Skipping.")
            return
        # Resampling
        resampled_data = librosa.resample(
            data, orig_sr=sample_rate, target_sr=target_sample_rate)
    except Exception as e:
        print(f"Error while processing file {input_file_path}: {e}")
        return

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    sf.write(output_file_path, resampled_data, target_sample_rate)

    # Update the progress bar
    pbar.value += 1
    print_progress_bar(pbar.value, total.value,
                       prefix='Progress:', suffix='Complete', length=50)


def audio_resampler(input_dir, output_dir, sample_rate, overwrite, n_workers):
    input_path = Path(input_dir)
    output_dir = Path(output_dir)

    if input_path.is_file():
        files_to_process = [input_path]
    else:
        files_to_process = list(input_path.glob(
            '**/*.flac')) + list(input_path.glob('**/*.wav'))

    with Manager() as manager:
        pbar = manager.Value('i', 0)
        total = manager.Value('i', len(files_to_process))

        with multiprocessing.Pool(n_workers) as pool:
            pool.starmap(resample_audio, [(str(file), str(output_dir / file.relative_to(
                input_path)), sample_rate, overwrite, pbar, total) for file in files_to_process])
            pool.close()
            pool.join()

    print("\nDone!")
