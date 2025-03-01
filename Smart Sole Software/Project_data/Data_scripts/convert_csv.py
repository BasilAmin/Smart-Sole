import ast
import csv

input_file = 'Heel_avoidance_dynamic.txt' 
output_file = 'Heel_avoidance_dynamic.csv'


expanded_data = []
with open(input_file, 'r', encoding='utf-8-sig') as file:  
    for line_num, line in enumerate(file, start=1): 
        try:
            parsed_line = ast.literal_eval(line.strip())
            for row in parsed_line:
                expanded_data.append(row)  
        except Exception as e:
            print(f"Error processing line {line_num}: {e}")
            continue 
with open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(expanded_data)

print(f"Data successfully converted to CSV: {output_file}")
