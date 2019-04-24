import os


with open('user.txt', 'r') as user:
    use = []
    use = user.read()
    print(use)
print("done")

data = []

for i in range(len(use)):
    if(use[i] == '\n'):
        continue
    else:
        data.append(int(use[i]))
print(data)
