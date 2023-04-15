import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str)
parser.add_argument('--output_file', type=str)

args, _ = parser.parse_known_args()
input_file = args.input_file
output_file = args.output_file

with open(input_file) as file:
    file_content = json.load(file)

cells = file_content["cells"]

with open(output_file, 'w') as py_file:
    for cell in cells:
        if cell["cell_type"] == "code":
            source = cell["source"]
            for line in source:
                py_file.write(line)
            py_file.write('\n')