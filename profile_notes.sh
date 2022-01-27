# ============ Train piano transcription system from scratch ============
# MAESTRO dataset directory. Users need to download MAESTRO dataset into this folder.
DATASET_DIR="/jmain02/home/J2AD007/txk47/shared/datasets/maestro-v3.0.0"

# Modify to your workspace
WORKSPACE="/jmain02/home/J2AD007/txk47/axe90-txk47/Projects/piano_transcription"

python3 -m torch.utils.bottleneck pytorch/main.py train --workspace=$WORKSPACE --model_type='Regress_onset_offset_frame_velocity_CRNN' --loss_type='regress_onset_offset_frame_velocity_bce' --augmentation='aug' --max_note_shift=3 --batch_size=12 --learning_rate=0.5e-3 --reduce_iteration=10000 --resume_iteration=0 --early_stop=100 --cuda
