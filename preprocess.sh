# ============ Train piano transcription system from scratch ============
# MAESTRO dataset directory. Users need to download MAESTRO dataset into this folder.
DATASET_DIR="/jmain02/home/J2AD007/txk47/shared/datasets/maestro-v3.0.0"

# Modify to your workspace
WORKSPACE="/jmain02/home/J2AD007/txk47/axe90-txk47/Projects/piano_transcription"

# Pack audio files to hdf5 format for training
python3 utils/features.py pack_maestro_dataset_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE
