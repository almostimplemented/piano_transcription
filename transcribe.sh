# ============ Transcribe directory of audio files to MIDI with a trained model  ============
CHECKPOINT_PATH="/jmain02/home/J2AD007/txk47/axe90-txk47/Projects/piano_transcription/aug_combined_model_0.pth"
AUDIO_DIR="/jmain02/home/J2AD007/txk47/axe90-txk47/Projects/piano_transcription/small_maestro"
MIDI_OUTPUT_DIR="/jmain02/home/J2AD007/txk47/axe90-txk47/Projects/piano_transcription/small_maestro_midi"

python3 transcribe.py --checkpoint_path="$CHECKPOINT_PATH" --audio_input_path="$AUDIO_DIR" --midi_output_path="$MIDI_OUTPUT_DIR" --cuda
