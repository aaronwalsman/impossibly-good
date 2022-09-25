import matplotlib.pyplot as plt
import json
import argparse
import glob
import os
import statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", help="pattern of folders to evaluate on", required=True)
    parser.add_argument("--outputs", help="where to put the output plots", required=True)

    args = parser.parse_args()
    folders = glob.glob(os.path.expanduser(args.inputs))

    for folder in folders:
        frame_num = []
        avg_r = []
        first = True
        for seed in os.listdir(folder):
            eval_log_path = os.path.join(folder, seed, "eval_log.json")
            with open(eval_log_path, 'r') as f:
                data = json.load(f)
            for idx, frame in enumerate(data):
                if first:
                    frame_num.append(frame['num_frame'])
                    avg_r.append([frame['return_stats']['mean']])
                else:
                    avg_r[idx].append(frame['return_stats']['mean'])
            first = False
        
        std_r_above = [statistics.mean(a)+statistics.pstdev(a) for a in avg_r]
        std_r_below = [statistics.mean(a)-statistics.pstdev(a) for a in avg_r]
        avg_r = [statistics.mean(a) for a in avg_r]
        
        plt.plot(frame_num, avg_r, label=folder.split("/")[-2].split("v0_")[-1])
        plt.fill_between(frame_num, std_r_above, std_r_below, alpha=0.3)
    
    plt.title(label=folder.split("/")[-2].split("_")[0])
    plt.xlabel("Frame number")
    plt.ylabel("Cumulative reward")
    plt.ylim([0, 1])
    plt.legend()
    os.makedirs(args.outputs, exist_ok=True)
    plt.savefig(args.outputs + "performance.jpg")
        