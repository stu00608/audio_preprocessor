import os
from tqdm import tqdm


def whisper_transcriber(input_dir, output_dir, whisper_size, language):
    import torch
    import whisper

    assert whisper_size in ["tiny", "base", "small",
                            "medium", "large"], "Invalid whisper size name."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(whisper_size, device=device)

    audio_paths = []
    if os.path.isdir(input_dir):
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".wav") or file.endswith(".flac") or file.endswith(".mp3"):
                    audio_paths.append(os.path.join(root, file))
    else:
        audio_paths = [input_dir]

    for audio_file in tqdm(audio_paths):
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(audio_file)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # detect the spoken language
        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        print(f"Detected language: {detected_language}")

        if language != "" and detected_language != language:
            print(
                f"Detected language is not {language}, skipping this audio file.")
            continue

        # decode the audio
        options = whisper.DecodingOptions(
            fp16=True if model.device.type == "cuda" else False)
        result = whisper.decode(model, mel, options)

        # print the recognized text
        print(result.text)

        # Export transcribed text to a text file in output_dir
        os.makedirs(output_dir, exist_ok=True)
        ouput_file_name = os.path.basename(
            audio_file).rsplit(".", maxsplit=1)[0] + ".txt"
        with open(os.path.join(output_dir, ouput_file_name), "w") as f:
            f.write(result.text)

        return result.text, detected_language
