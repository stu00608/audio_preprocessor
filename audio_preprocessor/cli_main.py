import click
import multiprocessing
from audio_preprocessor.resample import audio_resampler
from audio_preprocessor.slicer import audio_slicer
from audio_preprocessor.splitter import audio_splitter


@click.group()
def cli():
    pass


@cli.command()
@click.option('--input_dir', required=True, help='Directory containing audio files to process or single audio file path')
@click.option('--output_dir', required=True, help='Directory to output processed files')
@click.option('--sample_rate', type=int, required=True, help='Target sample rate')
@click.option('--overwrite', is_flag=True, help='Whether to overwrite existing files in output_dir')
@click.option('--n_workers', type=int, default=multiprocessing.cpu_count(), help='Number of worker processes')
def resample(input_dir, output_dir, sample_rate, overwrite, n_workers):
    audio_resampler(input_dir, output_dir, sample_rate, overwrite, n_workers)


cli.add_command(resample)


@click.command()
@click.option('--input_dir', required=True, help='The audio folder/file to be sliced')
@click.option('--output_dir', required=True, help='Output directory of the sliced audio clips')
@click.option('--db_thresh', type=float, default=-40, help='The dB threshold for silence detection')
@click.option('--min_length', type=int, default=5000, help='The minimum milliseconds required for each sliced audio clip')
@click.option('--min_interval', type=int, default=300, help='The minimum milliseconds for a silence part to be sliced')
@click.option('--hop_size', type=int, default=10, help='Frame length in milliseconds')
@click.option('--max_sil_kept', type=int, default=500, help='The maximum silence length kept around the sliced clip, presented in milliseconds')
@click.option('--n_workers', type=int, default=multiprocessing.cpu_count(), help='Number of worker processes')
def slicer(input_dir, output_dir, db_thresh, min_length, min_interval, hop_size, max_sil_kept, n_workers):
    audio_slicer(input_dir, output_dir, db_thresh, min_length,
                 min_interval, hop_size, max_sil_kept, n_workers)


cli.add_command(slicer)


@click.command()
@click.option('--input_dir', required=True, help='Directory containing audio files to process or single audio file path')
@click.option('--output_dir', required=True, help='Directory to output processed files')
@click.option('--time_interval', type=int, default=1800, help='Time in seconds for each segment')
@click.option('--n_workers', type=int, default=multiprocessing.cpu_count(), help='Number of worker processes')
def split(input_dir, output_dir, time_interval, n_workers):
    audio_splitter(input_dir, output_dir, time_interval, n_workers)


cli.add_command(split)

if __name__ == "__main__":
    cli()
