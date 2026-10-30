
import cv2
from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# Read input image
image = cv2.imread("motor .jpg")

if image is None:
    print("Error: Image not found!")
    exit()

# Perform object detection
results = model(image)

# Process detection results
for result in results:
    for box in result.boxes:

        # Get detected class
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Detect motorcycle only
        if class_name == "motorcycle":

            # Get bounding box coordinates
            x_min, y_min, x_max, y_max = map(
                int, box.xyxy[0]
            )

            # Get confidence score
            confidence = float(box.conf[0])

            # Print detection information
            print("Object: Pelanggar K4L")
            print(f"x-min: {x_min}")
            print(f"y-min: {y_min}")
            print(f"x-max: {x_max}")
            print(f"y-max: {y_max}")
            print(f"Confidence: {confidence:.4f}")
            print("------------------------")

            # Draw bounding box
            cv2.rectangle(
                image,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

            # Change label
            label = f"Pelanggar K4L: {confidence:.2f}"

            # Draw label
            cv2.putText(
                image,
                label,
                (x_min, max(y_min - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

# Save output image
cv2.imwrite("output.jpg", image)

print("Detection completed!")
print("Output saved as output.jpg")

# Display result
cv2.imshow("Motorcycle Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
