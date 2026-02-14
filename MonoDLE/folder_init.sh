# folder init
mkdir log
cd log

parent_folders=("checkpoints" "stats" "test" "train")

for parent_folder in "${parent_folders[@]}"; do
    mkdir -p "$parent_folder"
done

cd ../
mkdir outputs
cd outputs

parent_folders=("train" "val" "test")

for parent_folder in "${parent_folders[@]}"; do
    mkdir -p "$parent_folder"
done

cd ../
