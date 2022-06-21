# IMPORTANT: The process of error analysis is fully documented in `Analysis.ipynb`

import pandas as pd
from os import system, name

def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')

dataframe = pd.read_json("default_test.json")


incomprehensible = 0
no_ans = 0
wrong_amt_qa = 0
wrong_ans = 0
repeating_qa = 0

for index, data in dataframe.iterrows():
    clear()
    print("SOURCE:", data["Source"])
    print("PREDS:", data["Predictions"])
    print("LABELS:", data["Labels"])
    print("---------")
    print("0: Correct")
    print("1: Incomprehensible")
    print("2: No Answer")
    print("3: Wrong Amount Questions")
    print("4: Wrong Answers")
    print("5: Repeating QA")
    try:
        cmd = int(input("Number: "))
    except:
        cmd = int(input("Number: "))

    if cmd == 0:
        pass

    elif cmd == 1:
        incomprehensible += 1
    
    elif cmd == 2:
        no_ans += 1
    
    elif cmd == 3:
        wrong_amt_qa += 1
    
    elif cmd == 4:
        wrong_ans += 1
    
    elif cmd == 5:
        repeating_qa += 1
    
    with open("error_result.txt", mode="w") as f:
        f.write(f"Incomprehensible: {incomprehensible}\n")
        f.write(f"No Answer: {no_ans}\n")
        f.write(f"Wrong Amount QA: {wrong_amt_qa}\n")
        f.write(f"Wrong Answer: {wrong_ans}\n")
        f.write(f"Repeating: {repeating_qa}")


print(f"Incomprehensible: {incomprehensible}")
print(f"No Answer: {no_ans}")
print(f"Wrong Amount QA: {wrong_amt_qa}")
print(f"Wrong Answer: {wrong_ans}")
print(f"Repeating: {repeating_qa}")