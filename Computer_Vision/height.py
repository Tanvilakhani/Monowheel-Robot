import cv2

# Variables to store points and drawing state
start_point = None
end_point = None
drawing = False
bounding_box = None

def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing, bounding_box

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing when left mouse button is pressed
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        # Update the end point as the mouse moves
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        # Stop drawing when the left mouse button is released
        drawing = False
        end_point = (x, y)
        bounding_box = (start_point, end_point)

        # Draw the final rectangle
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow("Select Object", image)

# Load the image
image_path = "Computer_Vision/IMG_2951.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Display the image and set the callback
cv2.imshow("Select Object", image)
cv2.setMouseCallback("Select Object", draw_rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate the size of the object in pixels if a bounding box was drawn
if bounding_box:
    x1, y1 = bounding_box[0]
    x2, y2 = bounding_box[1]

    # Calculate width and height in pixels
    object_width_px = abs(x2 - x1)
    object_height_px = abs(y2 - y1)

    print(f"Object width in pixels: {object_width_px}")
    print(f"Object height in pixels: {object_height_px}")

else:
    print("No bounding box was selected.")
