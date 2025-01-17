import os
import yaml
import cv2
import numpy as np

# Function to read class_id to label mapping from data.yaml
def load_class_labels(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def crop_and_convert_yolov8_to_paddleocr(yolov8_folder, output_cropped_folder, output_txt_path, image_folder, yaml_path, image_width, image_height):
    # Load the class_id to label mapping
    class_id_to_label = load_class_labels(yaml_path)

    # Ensure the output cropped images folder exists
    if not os.path.exists(output_cropped_folder):
        os.makedirs(output_cropped_folder)

    # Ensure the image folder exists, if not create it
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    with open(output_txt_path, 'w') as output_file:
        for filename in os.listdir(yolov8_folder):
            if filename.endswith('.txt'):
                txt_path = os.path.join(yolov8_folder, filename)
                image_file_name = os.path.splitext(filename)[0] + '.jpg'  # Change to .png if needed
                image_path = os.path.join(image_folder, image_file_name)
                image = cv2.imread(image_path)

                with open(txt_path, 'r') as f:
                    for idx, line in enumerate(f.readlines()):
                        class_id, x_center, y_center, width, height = map(float, line.split())
                        x_center *= image_width
                        y_center *= image_height
                        width *= image_width
                        height *= image_height

                        # Get the bounding box
                        x_min = int(x_center - (width / 2))
                        y_min = int(y_center - (height / 2))
                        x_max = int(x_center + (width / 2))
                        y_max = int(y_center + (height / 2))

                        # Crop the image
                        cropped_image = image[y_min:y_max, x_min:x_max]

                        # Save the cropped image
                        cropped_image_name = f"{os.path.splitext(filename)[0]}_{idx}.jpg"
                        cropped_image_path = os.path.join(output_cropped_folder, cropped_image_name)

                        # Convert to forward slashes
                        cropped_image_path = cropped_image_path.replace("\\", "/")

                        cv2.imwrite(cropped_image_path, cropped_image)

                        # Get the actual label from the class_id_to_label list
                        label = class_id_to_label[int(class_id)] if int(class_id) < len(class_id_to_label) else "unknown"

                        # Write to the output text file
                        if label:
                            output_line = f"{cropped_image_path}\t\"{label}\"\n"
                            output_file.write(output_line)

if __name__ == "__main__":
    folders = ["train", "valid", "test"]
    colors = ['rgb']
    sizes = [320]
    for folder in folders:
        for color in colors:
            for size in sizes:
                yolov8_folder = f'Carrier_number_{size}x{size}_{color}/{folder}/det/labels'  # Change to your folder
                output_cropped_folder = f'Carrier_number_{size}x{size}_{color}/{folder}/rec/images'  # Change to your output folder
                output_file = f'Carrier_number_{size}x{size}_{color}/{folder}/rec/labels.txt'  # Change to your output file
                image_folder = f'Carrier_number_{size}x{size}_{color}/{folder}/det/images'  # Folder containing your images
                yaml_path = f'Carrier_number_{size}x{size}_{color}/data.yaml'  # Path to your data.yaml file

                crop_and_convert_yolov8_to_paddleocr(yolov8_folder, output_cropped_folder, output_file, image_folder, yaml_path, image_width=size, image_height=size)
                print(f"Conversion and cropping completed! Output saved to {output_file} and cropped images to {output_cropped_folder}.")
