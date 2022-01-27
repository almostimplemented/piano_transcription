import torch
import os
import sys
import argparse
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
sys.path.insert(1, os.path.join(sys.path[0], 'pytorch'))
from pathlib import Path
from inference import PianoTranscription
from utilities import load_audio

sample_rate = 16000

def audio_file_paths(root):
  return [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.mp3') or f.endswith('.wav')]

def transcribe_file(transcriptor, infile, output_root):
    '''Transcribes an audio file into MIDI using the provided transcriptor

       Args:
           transcriptor: the PianoTranscription instance
           infile: the audio file to be transcribed
           output_root: filesystem location to write MIDI file beneath

       Returns:
           A dictionary of metadata about the transcription
    '''

    # Load audio
    (audio, _) = load_audio(infile, sr=sample_rate, mono=True)
    if audio is None:
        print('Failed to read infile: ', infile)
        return

    # Compute output path
    audiofile_basename = os.path.basename(infile)
    midifile_basename = os.path.splitext(audiofile_basename)[0] + '.mid'
    midi_path = os.path.join(output_root, midifile_basename)

    # Run inference
    return transcriptor.transcribe(audio, midi_path)

def transcribe(args):
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    audio_input_path = args.audio_input_path
    midi_output_path = args.midi_output_path
    post_processor_type = args.post_processor_type
    segment_samples = sample_rate * 10  

    # Transcriptor
    transcriptor = PianoTranscription(model_type, device=device, 
        checkpoint_path=checkpoint_path, segment_samples=segment_samples, 
        post_processor_type=post_processor_type)

    # ensure output path
    Path(midi_output_path).mkdir(parents=True, exist_ok=True)

    stats_dict = {
        'offset_by_offset':0,
        'offset_by_frame':0,
        'offset_by_frame_no_offset':0, 
        'offset_by_max_duration':0,
    }

    # iterate over the files
    for audio_file in audio_file_paths(audio_input_path):
        print(audio_file)
        transcribed_dict = transcribe_file(transcriptor, audio_file, midi_output_path)
        for k, v in transcribed_dict['stats_dict'].items():
            print(k, " : ", v)
            stats_dict[k] += v
    
    print('stats_dict:')
    print(stats_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--audio_input_path', type=str, required=True)
    parser.add_argument('--midi_output_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='Note_pedal')
    parser.add_argument('--post_processor_type', type=str, default='regression', choices=['onsets_frames', 'regression'])
    parser.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()
    transcribe(args)