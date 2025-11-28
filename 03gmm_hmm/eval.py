label_file = '../data/label/test/test_label.txt'
result_file = './exp/data/test/result.txt'
labels = {}

with open(label_file, 'r') as lf:
     for line in lf:
         id_, lab = line.strip().split()
         labels[id_] = lab

     n = 0
     h = 0

     with open(result_file, 'r') as rf:
         lines = rf.readlines()
         for i in range(0, len(lines), 2):
             if i + 1 >= len(lines):
                 break
                
             x = lines[i].strip().split()[0]
                
             if "Result = " in lines[i + 1]:
                 x_h = lines[i + 1].strip().split("Result = ")[1]
             else:
                 continue
                
             n += 1

             if x in labels and labels[x] == x_h:
                 h += 1

     acc = (h / n) * 100
     print(f"============ Results Analysis ============")
     print(f"Test: {result_file}")
     print(f"True: {label_file}")
     print(f"Accuracy: {acc:.2f}%")
     print(f"Hits: {h}, Total: {n}")
     print(f"==========================================")
        
