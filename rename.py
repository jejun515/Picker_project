import os

HOME_DIR = "/home/hyunjong"

def rename_imgs():
    data_dir = os.path.join(HOME_DIR, "tracker_data")
    label_dir = os.path.join(HOME_DIR, "tracker_label")
    data_files = [x.split(".")[0] for x in os.listdir(data_dir)]

    unfound = []

    for label_file in os.listdir(label_dir):
        if label_file.split(".")[0] not in data_files:
            unfound.append(label_file)
    
    for item in sorted(unfound):
        print(f"{item} NOT found in data directory")

if __name__ == "__main__":
    rename_imgs()