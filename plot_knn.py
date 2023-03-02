import numpy as np
import matplotlib.pyplot as plt
import os
file_name = "result.txt"

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    if "can" in s:
        return int(s.replace("can_", ""))
    elif "knn" in s:
        return int(s.replace("knn_", ""))
    elif "mlp" in s:
        return int(s.replace("mlp_", ""))

result_dict = {"neighbors": [], "time": [], "acc": [] }
files = os.listdir("./exp_results")
files.sort(key=alphanum_key)
for d1 in files:
    if "knn" not in d1:
        continue
    
    d1_full = os.path.join("./exp_results", d1)
    with open(os.path.join(d1_full, file_name)) as f:
        for l in f.read().split("\n"):
            if "best_val" in l:
                result_dict["acc"].append(float(l.replace("best_val: ", "")))
            if "inference_time" in l:
                result_dict["time"].append(float(l.replace("inference_time: ", "")) / 60.0)
            if "num_neighbors" in l:
                result_dict["neighbors"].append(int(l.replace("num_neighbors: ", "")))

time_data = {}
acc_data = {}
for i in range(len(result_dict["neighbors"])):
    time_data[str(result_dict["neighbors"][i])] = result_dict["time"][i]
    acc_data[str(result_dict["neighbors"][i])] = result_dict["acc"][i]
  
# accuracy
k = list(acc_data.keys())
values = list(acc_data.values())

fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(k, values, color ='green',
        width = 0.4)

xlocs, xlabs = plt.xticks()
 
plt.xlabel("number of neighbors")
plt.ylabel("accuracy")
plt.title("KNN: #neighbors V.S. accuracy")
plt.ylim(0.9, 1.0)
for i, v in enumerate(values):
    plt.text(xlocs[i] - 0.25, v+0.005, "{:.4f}".format(v))
plt.savefig("./knn-k-vs-acc")
plt.show()

# time
k = list(time_data.keys())
values = list(time_data.values())

fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(k, values, color ='green',
        width = 0.4)
 
plt.xlabel("number of neighbors")
plt.ylabel("time (min)")
plt.title("KNN: #neighbors V.S. prediction time")
plt.ylim(2.0, 6.0)
for i, v in enumerate(values):
    plt.text(xlocs[i] - 0.25, v+0.009, "{:.5f}".format(v))
plt.savefig("./knn-k-vs-time")
plt.show()