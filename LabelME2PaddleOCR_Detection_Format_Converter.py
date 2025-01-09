import os
import json

def convert_labelme_to_paddleocr(labelme_folder, output_file):
    # Create a list to store the formatted data
    formatted_data = []

    # Iterate through all files in the specified folder
    for filename in os.listdir(labelme_folder):
        if filename.endswith('.json'):
            # Construct the full file path
            json_file_path = os.path.join(labelme_folder, filename)
            
            with open(json_file_path, 'r') as file:
                labelme_data = json.load(file)
                
                # Extract the image file name (assumes the same name as the json)
                img_file_name = os.path.splitext(filename)[0] + '.jpg'  # Change to .png if needed
                img_file_path = f'images/{img_file_name}'  # Assuming the format is "images/img.jpg"

                # Process each shape in the LabelMe JSON
                annotations = []
                for shape in labelme_data['shapes']:
                    transcription = shape['label']
                    points = shape['points']

                    # Format points as required by PaddleOCR
                    formatted_points = [[float(point[0]), float(point[1])] for point in points]

                    annotations.append({
                        "transcription": transcription,
                        "points": formatted_points
                    })
                
                # Properly format the annotations as a JSON string
                formatted_annotation = json.dumps(annotations, ensure_ascii=False)
                
                # Append the formatted entry to the data list
                formatted_data.append(f"{img_file_path}\t{formatted_annotation}")

    # Write the formatted data to the output file
    with open(output_file, 'w', encoding='utf-8') as output:
        for line in formatted_data:
            output.write(line + '\n')

# Usage
labelme_folder = './pin_code_text/test/detection/labels'  # Change this to your LabelMe folder path
output_file = './pin_code_text/test/detection/test.txt'  # Change this to your desired output file name
convert_labelme_to_paddleocr(labelme_folder, output_file)