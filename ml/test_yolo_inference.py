from ultralytics import YOLO
import cv2
import urllib.request
import os

def test_inference():
    print("Loading lightweight YOLOv8n model...")
    # This will automatically download yolov8n.pt if it doesn't exist
    model = YOLO('yolov8n.pt')

    print("Downloading a sample test image...")
    # URL of a sample traffic image
    image_url = 'https://ultralytics.com/images/bus.jpg'
    
    # Save the downloaded image in our models directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_image_dir = os.path.join(base_dir, 'models', 'test_images')
    os.makedirs(test_image_dir, exist_ok=True)
    
    img_path = os.path.join(test_image_dir, 'sample_traffic.jpg')
    urllib.request.urlretrieve(image_url, img_path)
    print(f"Sample image saved to {img_path}")

    print("Running YOLO inference...")
    # Run the model on the image
    results = model(img_path)

    print("Saving the annotated image...")
    # The results object contains a lot of data. We can plot the bounding boxes directly.
    # We take the first result (since we gave it one image)
    res = results[0]
    
    # Render the image with boxes
    annotated_img = res.plot()
    
    # Save the output image
    output_path = os.path.join(test_image_dir, 'sample_traffic_detected.jpg')
    cv2.imwrite(output_path, annotated_img)
    print(f"Annotated image saved successfully to {output_path}!")
    
    # Print the objects detected
    print("\n--- Objects Detected ---")
    
    # We iterate over every detected box
    for box in res.boxes:
        class_id = int(box.cls[0])           # Get internal class ID
        class_name = model.names[class_id]   # Convert ID to human readable string
        confidence = float(box.conf[0])      # Get confidence score
        
        print(f"Found a {class_name} formatting with {confidence*100:.1f}% confidence")

if __name__ == "__main__":
    test_inference()
