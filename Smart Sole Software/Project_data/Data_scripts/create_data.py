import numpy as np
import pandas as pd
import random
LABELS = {
    'Standing': 0,
    'Sitting': 1,
    'Walking': 2,
    'limping': 3,
    'heel_avoidance_stationary': 4,
    'heel_avoidance_dynamic': 5,
    'LateralArch_avoidance_stationary': 6,
    'LateralArch_avoidance_dynamic': 7,
}

walkingdf = pd.read_csv("walking.csv")
standingdf = pd.read_csv("standing.csv")
sittingdf = pd.read_csv("sitting.csv")
limpingdf = pd.read_csv("Limping.csv")
Lat_pressure_stationarydf = pd.read_csv("Lateral_arch_pressure_stationary.csv")
Lat_pressure_dynamicdf = pd.read_csv("Lateral_arch_pressure_dynamic.csv")
Heel_avoidance_stationarydf = pd.read_csv("Heel_avoidance_stationary.csv")
Heel_avoidance_dynamicdf = pd.read_csv("Heel_avoidance_dynamic.csv")

def chunk(df, chunk_size=10):
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk.values.flatten())
    return chunks

walking_chunks = chunk(walkingdf)
standing_chunks = chunk(standingdf)
sitting_chunks = chunk(sittingdf)
limping_chunks = chunk(limpingdf)
Lat_pressure_stationary_chunks = chunk(Lat_pressure_stationarydf)
Lat_pressure_dynamic_chunks = chunk(Lat_pressure_dynamicdf)
Heel_avoidance_stationary_chunks = chunk(Heel_avoidance_stationarydf)
Heel_avoidance_dynamic_chunks = chunk(Heel_avoidance_dynamicdf)
def generate_behaviour_data(behavior, chunks, min_chunks=5, max_chunks=30):
    label = LABELS[behavior]
    num_chunks = random.randint(min_chunks, max_chunks)
    behaviour_data = []
    for _ in range(num_chunks):
        chunk_data = random.choice(chunks)
        labeled_chunk = list(chunk_data)
        labeled_chunk.append(label)
        behaviour_data.append(labeled_chunk)
    return behaviour_data
behaviors = [
    ('Walking', walking_chunks),
    ('Standing', standing_chunks),
    ('Sitting', sitting_chunks),
    ('limping', limping_chunks),
    ('heel_avoidance_stationary', Heel_avoidance_stationary_chunks),
    ('heel_avoidance_dynamic', Heel_avoidance_dynamic_chunks),
    ('LateralArch_avoidance_stationary', Lat_pressure_stationary_chunks),
    ('LateralArch_avoidance_dynamic', Lat_pressure_dynamic_chunks),
]

final_dataset = []
for behaviour_name, chunks in behaviors:
    behaviour_data = generate_behaviour_data(behaviour_name, chunks)
    final_dataset.extend(behaviour_data)

random.shuffle(final_dataset)

num_features = len(final_dataset[0]) - 1 
columns = []
for i in range(10): 
    columns.append(f'heel_{i}')
    columns.append(f'lat_{i}')
    columns.append(f'metatarsal_{i}')
columns.append('label')
final_df = pd.DataFrame(final_dataset, columns=columns)
final_df.to_csv('Training_set2.csv', index=False)
print("Data successfully created and saved to ProcessedTrainingSet.csv")
