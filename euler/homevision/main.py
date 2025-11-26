import cv2
import matplotlib.pyplot as plt


FILL_THRESHOLD = 0.25  # Increasing may lead to missing slightly filled check boxes

class CheckboxChecker:

    def __init__(self, image_path: str):
        # not the best for forward-compatibility, but for this project it's fine
        self.image_path = image_path

    def preprocess_image(self, image):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive thresholding to highlight checkboxes clearly
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            25, 10
        )
        return thresh

    def find_checkbox_contours(self, thresh_img):
        # Find contours from thresholded image
        contours, _ = cv2.findContours(
            thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        checkbox_contours = []
        for cnt in contours:
            # Approximate contours to polygons + bounding rect
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Checkbox contour should be rectangular with 4 vertices and specific size range
            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)

                # TODO: Adjust these values based on the image resolution and checkbox sizes
                if 15 < w < 50 and 15 < h < 50:
                    checkbox_contours.append((x, y, w, h))
                
        return checkbox_contours

    def is_checkbox_filled(self, checkbox_roi, threshold=FILL_THRESHOLD):
        total_pixels = checkbox_roi.size
        filled_pixels = cv2.countNonZero(checkbox_roi)
        fill_ratio = filled_pixels / total_pixels

        return fill_ratio > threshold

    def display_image(self, image):
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Checkbox Detection (Green=Filled, Red=Empty)')
        plt.axis('off')
        plt.show()

    def detect_checkboxes(self):
        image = cv2.imread(self.image_path)
        processed_image = self.preprocess_image(image)
        checkbox_contours = self.find_checkbox_contours(processed_image)

        checkboxes_detected = []

        for (x, y, w, h) in checkbox_contours:
            # Crop the checkbox area from thresholded image to check if filled
            # ROI stands for Region of Interest, or the portion in pixels that comprise the checkbox
            checkbox_roi = processed_image[y:y+h, x:x+w]
            filled = self.is_checkbox_filled(checkbox_roi)
            checkboxes_detected.append({
                "location": (x, y, w, h),
                "filled": filled
            })
            # Draw bounding boxes and status on original image for visualization
            color = (0, 255, 0) if filled else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

        # TODO: Save the results to a structured output (e.g., JSON file)

        # Display results
        self.display_image(image)

        return checkboxes_detected


if __name__ == "__main__":
    input_image_path = "images/sample-section-mod.webp"
    checker = CheckboxChecker(input_image_path)
    results = checker.detect_checkboxes()

    # TODO: Implement detailed logging and error handling for production usage
