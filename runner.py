import time
import os

print("start")
# epoch_time * 60 * epoch
wait = 35 * 60 * 21
time.sleep(wait)
# os.system("python main.py")

print("run")
epoch = [4]
models = [29]  #27, , 27, 27 , 5, 7, 0, 1 , 25, 25, 25, 25
for ind in range(len(models)):
    model = models[ind]
    for data in ["CASIA", "JS_COLUMBIA", "NIST16", "COVERAGE_AUG"][:1]:  #, "IMD2020" , "JS_COLUMBIA_ORI"
        os.system(f"python finetune.py {model} {data} 1.0 1")
        wait = 60 * 1
        # time.sleep(wait)
        os.system(f"python evaluate.py {model} {data} 1.0 0")
