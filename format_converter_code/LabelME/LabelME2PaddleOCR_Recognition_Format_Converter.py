import os
import json
import cv2
import numpy as np

def crop_and_convert_labelme_to_paddleocr(labelme_json_folder, output_cropped_folder, output_txt_path):
    if not os.path.exists(output_cropped_folder):
        os.makedirs(output_cropped_folder)

    with open(output_txt_path, 'w') as output_file:
        for json_file in os.listdir(labelme_json_folder):
            if json_file.endswith('.json'):
                json_path = os.path.join(labelme_json_folder, json_file)
                
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                    # Load the image
                    image_path = os.path.join(labelme_json_folder, data['imagePath'])
                    image = cv2.imread(image_path)
                    
                    for idx, shape in enumerate(data['shapes']):
                        label = shape['label']
                        points = shape['points']
                        
                        # Convert to integer coordinates
                        points = np.array(points, dtype=np.int32)
                        
                        # Get the bounding box
                        x_min = min(points[:, 0])
                        y_min = min(points[:, 1])
                        x_max = max(points[:, 0])
                        y_max = max(points[:, 1])
                        
                        # Crop the image
                        cropped_image = image[y_min:y_max, x_min:x_max]
                        
                        # Save the cropped image
                        cropped_image_name = f"{os.path.splitext(json_file)[0]}_{idx}.jpg"
                        cropped_image_path = os.path.join(output_cropped_folder, cropped_image_name)
                        cv2.imwrite(cropped_image_path, cropped_image)
                        
                        # Write to the output text file
                        if label:
                            output_line = f"{cropped_image_path}\t\"{label}\"\n"
                            output_file.write(output_line)

if __name__ == "__main__":
    labelme_folder = './pin_code_text/test/detection/labels'  # Change to your folder
    output_cropped_folder = './pin_code_text/test/recognition/crop_images'  # Change to your output folder
    output_file = './pin_code_text/test/recognition/test.txt'  # Change to your output file
    crop_and_convert_labelme_to_paddleocr(labelme_folder, output_cropped_folder, output_file)
    print(f"Conversion and cropping completed! Output saved to {output_file} and cropped images to {output_cropped_folder}.")
