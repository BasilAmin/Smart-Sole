import csv
file_path = "c:/Users/basil/Documents/PROJECTS/SMART__SOLE/Smart Sole Software/DATA/walking.csv"
line_count = 0
with open(file_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
            line_count += 1
print(line_count)