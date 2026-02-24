import re

input_path = "../data/computer_fundamental.txt"
output_path = "../data/computer_fundamental_clean.txt"

with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

# Normalize spacing
text = re.sub(r"\n{3,}", "\n\n", text)

# Remove repeated page markers if needed (keep them for now)
# text = re.sub(r"--- Page \d+ ---", "", text)

text = text.strip()

with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Cleaned text saved to", output_path)
print("Total characters:", len(text))