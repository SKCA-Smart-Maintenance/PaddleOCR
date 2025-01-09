import os
import json

def convert_yolov8_to_paddleocr(yolov8_folder, output_file, image_folder, image_width, image_height):
    """
    Convert YOLOv8 annotation format to PaddleOCR annotation format.
    
    Args:
    - yolov8_folder: Path to folder containing YOLO txt annotation files
    - output_file: Path to save the converted annotations
    - image_folder: Path to folder containing corresponding images
    - image_width: Width of the images
    - image_height: Height of the images
    """
    formatted_data = []
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process each text file in the YOLOv8 annotations folder
    for filename in os.listdir(yolov8_folder):
        if filename.endswith('.txt'):
            # Construct the full file paths
            txt_file_path = os.path.join(yolov8_folder, filename)
            
            # Try both .jpg and .png extensions
            image_file_name_jpg = os.path.splitext(filename)[0] + '.jpg'
            image_file_name_png = os.path.splitext(filename)[0] + '.png'
            
            # Check which image file exists
            if os.path.exists(os.path.join(image_folder, image_file_name_jpg)):
                image_file_name = image_file_name_jpg
            elif os.path.exists(os.path.join(image_folder, image_file_name_png)):
                image_file_name = image_file_name_png
            else:
                print(f"Warning: No image found for {filename}")
                continue
            
            # Construct image file path with forward slashes
            img_file_path = os.path.normpath(os.path.join(image_folder, image_file_name)).replace("\\", "/")
            
            # Read and convert annotations
            try:
                with open(txt_file_path, 'r') as file:
                    annotations = []
                    
                    for line in file:
                        # Split and convert annotation values
                        parts = line.strip().split()
                        
                        # Ensure we have 5 parts (class_id, x_center, y_center, width, height)
                        if len(parts) != 5:
                            print(f"Warning: Skipping invalid annotation in {filename}")
                            continue
                        
                        # Convert strings to floats
                        class_id, x_center, y_center, width, height = map(float, parts)
                        
                        # Scale coordinates to image dimensions
                        x_center *= image_width
                        y_center *= image_height
                        width *= image_width
                        height *= image_height
                        
                        # Calculate corner points of the bounding box
                        x_min = x_center - (width / 2)
                        y_min = y_center - (height / 2)
                        x_max = x_center + (width / 2)
                        y_max = y_center + (height / 2)
                        
                        # Use class ID as transcription
                        transcription = str(int(class_id))
                        
                        # Format points for PaddleOCR
                        formatted_points = [
                            [x_min, y_min],  # top-left
                            [x_max, y_min],  # top-right
                            [x_max, y_max],  # bottom-right
                            [x_min, y_max]   # bottom-left
                        ]
                        
                        annotations.append({
                            "transcription": transcription,
                            "points": formatted_points
                        })
                    
                    # Convert annotations to JSON
                    formatted_annotation = json.dumps(annotations, ensure_ascii=False)
                    formatted_data.append(f"{img_file_path}\t{formatted_annotation}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Write converted annotations to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as output:
            for line in formatted_data:
                output.write(line + '\n')
        print(f"Conversion complete. Output saved to {output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")

# Example usage
def main(project_name, folders, colors, sizes):    
    for folder in folders:
        for color in colors:
            for size in sizes:
                # Construct paths
                yolov8_folder = f'{project_name}_{size}x{size}_{color}/{folder}/det/labels'
                output_file = f'{project_name}_{size}x{size}_{color}/{folder}/det/labels.txt'
                image_folder = f'{project_name}_{size}x{size}_{color}/{folder}/det/images'
                
                # Convert annotations
                convert_yolov8_to_paddleocr(
                    yolov8_folder, 
                    output_file, 
                    image_folder, 
                    image_width=size, 
                    image_height=size
                )

if __name__ == "__main__":
    project_name = 'Carrier_number'
    folders = ['train', 'valid', 'test']
    sizes = [320]
    colors = ['rgb']
    main(project_name, folders, colors, sizes)