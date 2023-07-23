import os
import multiprocessing
from pathlib import Path
import soundfile as sf
from audio_preprocessor.utils import print_progress_bar


def split_audio(input_file_path, output_dir, time_interval, pbar, total):
    # Read the audio file
    data, sample_rate = sf.read(input_file_path)
    total_samples = len(data)
    total_segments = total_samples // (sample_rate * time_interval) + 1

    for i in range(total_segments):
        start_sample = i * sample_rate * time_interval
        end_sample = min((i + 1) * sample_rate * time_interval, total_samples)

        # Get the segment data
        segment_data = data[start_sample:end_sample]

        # Check if the segment has any audio data
        if len(segment_data) > 0 and segment_data.any():
            # Save the segment
            segment_file_path = os.path.join(
                output_dir, f"{Path(input_file_path).stem}_{i}{Path(input_file_path).suffix}")
            os.makedirs(os.path.dirname(segment_file_path), exist_ok=True)
            sf.write(segment_file_path, segment_data, sample_rate)

    # Update the progress bar
    pbar.value += 1
    print_progress_bar(pbar.value, total.value,
                       prefix='Progress:', suffix='Complete', length=50)


def audio_splitter(input_dir, output_dir, time_interval, n_workers):
    # Get the list of audio files
    input_path = Path(input_dir)
    output_dir = Path(output_dir)

    if input_path.is_file():
        files_to_process = [input_path]
    else:
        files_to_process = list(input_path.glob(
            '**/*.flac')) + list(input_path.glob('**/*.wav'))

    # Initialize the progress bar
    with multiprocessing.Manager() as manager:
        pbar = manager.Value('i', 0)
        total = manager.Value('i', len(files_to_process))

        # Process the files
        with multiprocessing.Pool(n_workers) as pool:
            pool.starmap(split_audio, [
                         (str(file), output_dir, time_interval, pbar, total) for file in files_to_process])
            pool.close()
            pool.join()

    print("\nDone!")
