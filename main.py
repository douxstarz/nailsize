from flask import Flask, request, render_template, redirect
import cv2
import tensorflow as tf
import numpy as np



app = Flask(__name__)

def load_frozen_graph(frozen_graph_filename):
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

frozen_graph_path = "./model/export_model_008/frozen_inference_graph.pb"
model = load_frozen_graph(frozen_graph_path)

nail_shapes = {
    'smalmd': {
        0: 14,
        1: 13,
        2: 12,
        3: 11,
        4: 10.7,
        5: 10,
        6: 9.7,
        7: 9,
        8: 8.7,
        9: 8
    },
    'mdalmd': {
        0: 15,
        1: 13,
        2: 12,
        3: 11,
        4: 10.5,
        5: 10,
        6: 9.5,
        7: 9,
        8: 8.5,
        9: 8
    },
    'mdcoff': {
        0: 12.7,
        1: 11.7,
        2: 10.7,
        3: 10,
        4: 9.5,
        5: 9,
        6: 8.5,
        7: 8,
        8: 7.7,
        9: 7.5
    },
    'mdsqr': {
        0: 13,
        1: 12,
        2: 11,
        3: 10.5,
        4: 10,
        5: 9.5,
        6: 9,
        7: 8.5,
        8: 8,
        9: 7.5
    }
}

preset_sizes = {
    'XS': [3, 6, 5, 7, 9],
    'S': [2, 5, 4, 6, 9],
    'M': [1, 5, 4, 6, 8],
    'L': [0, 4, 3, 5, 7]
}


@app.route ('/')
def home():
    return render_template('index.html')

@app.route('/upload' , methods=['POST'])
def upload_image():
    hand_file = request.files.get('hand_image')
    thumb_file = request.files.get('thumb_image')
    nail_shape = request.form.get('nail_shape')

    if hand_file and thumb_file:
        hand_image = cv2.imdecode(np.frombuffer(hand_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        thumb_image = cv2.imdecode(np.frombuffer(thumb_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        diameter_pixels = measure_quarter_with_hough(thumb_image)
        
        if diameter_pixels is None:
            return render_template('results.html', closest_preset_size=None, nail_shape=nail_shape, error="quarter not detected")

        mm_to_pixel_conversion_factor = 24.26 / diameter_pixels

        sizes = detect_nail_sizes(hand_image, thumb_image)
        sizes_in_mm = [convert_to_mm(size, mm_to_pixel_conversion_factor) for size in sizes if size is not None]
        
        closest_preset_size = map_to_preset_sizes(sizes_in_mm, nail_shape)
        
        return render_template('results.html', closest_preset_size=closest_preset_size, nail_shape=nail_shape, sizes_in_mm=sizes_in_mm)

def detect_nail_sizes(hand_image, thumb_image):
    
    image_resized = cv2.resize(hand_image, (300,300))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)
            
    with tf.compat.v1.Session(graph=model) as sess:
        input_tensor = model.get_tensor_by_name("image_tensor:0")
        boxes_tensor = model.get_tensor_by_name("detection_boxes:0")
        scores_tensor = model.get_tensor_by_name("detection_scores:0")

        (boxes, scores) = sess.run([boxes_tensor, scores_tensor], feed_dict={input_tensor: image_expanded})
    sizes = []
    hand_height, hand_width = hand_image.shape[:2]
    output_image = hand_image.copy()
    
    for box, score in zip(boxes[0], scores[0]):
        if score >= 0.6:
            startY, startX, endY, endX = box

            startX = int(startX * hand_width)
            startY = int(startY * hand_height)
            endX = int(endX * hand_width)
            endY = int(endY * hand_height)

            nail_width = endX - startX
            sizes.append(nail_width)

            cv2.rectangle(output_image, (startX, startY), (endX, endY),(0, 255, 0), 2)
            
            
            
    cv2.imwrite("detect_nails.jpg", output_image)
    return sizes 


def measure_quarter_with_hough(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("grayscale.jpg", gray)  # Save grayscale for inspection

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    cv2.imwrite("enhanced_contrast.jpg", enhanced_gray)  # Save enhanced contrast

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced_gray, (9, 9), 0)
    cv2.imwrite("blurred.jpg", blurred)  # Save blurred image

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=30, param2=15, minRadius=90, maxRadius=200)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, radius) in circles:
            diameter_pixels = radius * 2
            print(f"Detected circle with diameter: {diameter_pixels} pixels at position ({x}, {y})")

            # Draw the detected circle
            output_image = image.copy()
            cv2.circle(output_image, (x, y), radius, (255, 0, 0), 2)
            cv2.imwrite("detected_quarter_hough.jpg", output_image)  # Save final detection image

            return diameter_pixels

    print("No quarter detected.")
    return None


def enhance_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)



def is_quarter(contour, diameter_pixels):
   
   

   area = cv2.contourArea(contour)
   perimeter = cv2.arcLength(contour, True)
   if perimeter == 0:
       return False
   
   circularity = 4 * np.pi * (area / (perimeter ** 2))

   if 0.7 <= circularity <= 1.3:
       print(f"Detected object has diameter: {diameter_pixels} pixels and circularity: {circularity}")
       return True  # Detected quarter
   else:
        print(f"Rejected object with diameter: {diameter_pixels} pixels and circularity: {circularity}")
        return False
   

def convert_to_mm(pixel_measurement, mm_to_pixel_conversion_factor):
    
    return pixel_measurement * mm_to_pixel_conversion_factor

def map_to_preset_sizes(measurements, nail_shape):
    best_fit = None
    smallest_diff = float('inf')

    if nail_shape in nail_shapes:
        shape_sizes = nail_shapes[nail_shape]

    for size_label, size_numbers in preset_sizes.items():
        preset_mm_values = [shape_sizes[i] for i in size_numbers]

        total_diff = sum(abs(m-p) for m, p in zip(measurements, preset_mm_values))

        if total_diff < smallest_diff:
            smallest_diff = total_diff
            best_fit = size_label
    
    return best_fit


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)