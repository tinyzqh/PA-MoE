import os
import numpy as np

BITRATE_LEVELS = 6

COOKED_TRACE_FOLDER = os.path.join(os.getcwd(), "src/envs/trace/")


def load_trace(bandwidth_model=None):
    cooked_trace_folder = COOKED_TRACE_FOLDER
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []

        with open(file_path, "rb") as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]) * 2)

        # Check whether the bandwidth meets requirements
        if bandwidth_model == "low":
            if np.mean(cooked_bw) < 1 * 2:
                all_cooked_time.append(cooked_time)
                all_cooked_bw.append(cooked_bw)
                all_file_names.append(cooked_file)
        elif bandwidth_model == "high":
            if np.mean(cooked_bw) > 1.5 * 2:
                all_cooked_time.append(cooked_time)
                all_cooked_bw.append(cooked_bw)
                all_file_names.append(cooked_file)
        else:
            assert bandwidth_model == "hybrid", "BandWidth Error!"
            all_cooked_time.append(cooked_time)
            all_cooked_bw.append(cooked_bw)
            all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names


def load_video_size():
    video_size = {}  # in bytes
    for bitrate in range(BITRATE_LEVELS):
        video_size[bitrate] = []
        VIDEO_SIZE_FILE = os.path.join(os.getcwd(), "src/envs/envivio/video_size_")
        with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
            for line in f:
                video_size[bitrate].append(int(line.split()[0]))
    return video_size


if __name__ == "__main__":
    a = load_trace("high")  # 76, # 51, low:high = 6:4
    print(len(a[0]))
