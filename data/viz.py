import json
import matplotlib.pyplot as plt

# Load your JSON data from a file
with open('prompts-with-output.json', 'r') as file:
    data = json.load(file)

# Extract output lengths
output_lengths = [item['output_length'] for item in data]

# Plotting the output length distribution
plt.figure(figsize=(10, 6))
plt.hist(output_lengths, bins=20, color='blue', alpha=0.7)
plt.title('Distribution of Output Lengths')
plt.xlabel('Output Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
