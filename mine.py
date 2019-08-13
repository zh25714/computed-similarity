import random
f = open("data/cli/clinical_train.txt","r")
lines = f.readlines()
tab = [lines[0]]
lines = lines[1:]
random.shuffle(lines)
test = tab + lines[:160]
train = tab + lines[160:]
f_train = open("data/cli/train.txt","w")
f_test = open("data/cli/test.txt","w")
f_train.writelines(train)
f_test.writelines(test)
print(test)