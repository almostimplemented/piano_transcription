# ============ Train piano transcription system from scratch ============
# MAESTRO dataset directory. Users need to download MAESTRO dataset into this folder.
DATASET_DIR="/jmain02/home/J2AD007/txk47/shared/datasets/maestro-v3.0.0"

# Modify to your workspace
WORKSPACE="/jmain02/home/J2AD007/txk47/axe90-txk47/Projects/piano_transcription"

# Pack audio files to hdf5 format for training
python3 utils/features.py pack_maestro_dataset_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

python3 pytorch/main.py train --workspace=$WORKSPACE --model_type='Regress_pedal_CRNN' --loss_type='regress_pedal_bce' --augmentation='aug' --max_note_shift=3 --batch_size=12 --learning_rate=5e-4 --reduce_iteration=10000 --resume_iteration=0 --early_stop=300000 --cuda
