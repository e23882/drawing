import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import json

# --- Configuration ---
CONFIG_FILE = 'detect_config.json'

# --- Functions ---
def select_background_image():
    """Opens a file dialog to select an image and returns its path."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a Background Image",
        filetypes=(("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
    )
    return file_path

def nothing(x):
    """Dummy function for trackbar creation."""
    pass

def load_settings():
    """Loads settings from a JSON file."""
    defaults = {
        'fg_width': 50, 'fg_x': 50, 'fg_y': 50, 'fg_alpha': 70, 'fg_rotate': 0,
        'bg_width': 100, 'bg_x': 50, 'bg_y': 50, 'zoom_factor': 3
    }
    if not os.path.exists(CONFIG_FILE):
        return defaults
    try:
        with open(CONFIG_FILE, 'r') as f:
            settings = json.load(f)
            # Ensure all keys are present, use default if a key is missing
            for key in defaults:
                if key not in settings:
                    settings[key] = defaults[key]
            return settings
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading settings: {e}. Using defaults.")
        return defaults

def save_settings(settings):
    """Saves settings to a JSON file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
    except IOError as e:
        print(f"Error saving settings: {e}")

# --- Main Program ---

# 0. Load previous settings or defaults
settings = load_settings()

# 1. Select background image
background_image_path = select_background_image()
if not background_image_path:
    print("No image selected. Exiting program.")
    exit()

# 2. Initialize Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get webcam frame dimensions and aspect ratio (this will be our fixed canvas size)
ret, frame = cap.read()
if not ret:
    print("Error: Could not read initial frame from webcam.")
    cap.release()
    exit()
canvas_h, canvas_w, _ = frame.shape

# 3. Load background image and get its aspect ratio
background_original = cv2.imread(background_image_path)
if background_original is None:
    print(f"Error: Could not read the background image at '{background_image_path}'")
    cap.release()
    exit()
bg_aspect_ratio = background_original.shape[0] / background_original.shape[1]

# 4. Create effect assets and flags
background_gray = cv2.cvtColor(background_original, cv2.COLOR_BGR2GRAY)
_, background_binary = cv2.threshold(background_gray, 127, 255, cv2.THRESH_BINARY)
background_binary_bgr = cv2.cvtColor(background_binary, cv2.COLOR_GRAY2BGR)
background_effect = cv2.addWeighted(background_original, 0.5, background_binary_bgr, 0.5, 0)
binarize_mode = False # Flag to toggle the binarize effect
show_crosshair = False # Flag to toggle the crosshair
edge_detection_mode = False # Flag to toggle edge detection on foreground

# 5. Create UI Windows
main_window = 'Webcam Live'
bg_controls_window = 'Background Controls'
fg_controls_window = 'Foreground Controls'
magnifier_window = 'Magnifier'

cv2.namedWindow(main_window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(main_window, canvas_w, canvas_h)
cv2.namedWindow(magnifier_window)


cv2.namedWindow(bg_controls_window)
cv2.resizeWindow(bg_controls_window, 300, 150)
cv2.moveWindow(bg_controls_window, 100, 100)

cv2.namedWindow(fg_controls_window)
cv2.resizeWindow(fg_controls_window, 300, 250)
cv2.moveWindow(fg_controls_window, canvas_w + 200, 100)

# Create trackbars using loaded settings
cv2.createTrackbar('Width % ', bg_controls_window, settings['bg_width'], 100, nothing)
cv2.createTrackbar('X Pos % ', bg_controls_window, settings['bg_x'], 100, nothing)
cv2.createTrackbar('Y Pos % ', bg_controls_window, settings['bg_y'], 100, nothing)

cv2.createTrackbar('Width % ', fg_controls_window, settings['fg_width'], 100, nothing)
cv2.createTrackbar('X Pos % ', fg_controls_window, settings['fg_x'], 100, nothing)
cv2.createTrackbar('Y Pos % ', fg_controls_window, settings['fg_y'], 100, nothing)
cv2.createTrackbar('Alpha % ', fg_controls_window, settings['fg_alpha'], 100, nothing)
cv2.createTrackbar('Rotate', fg_controls_window, settings['fg_rotate'], 3, nothing)

cv2.createTrackbar('Zoom', magnifier_window, settings['zoom_factor'], 15, nothing)
cv2.setTrackbarMin('Zoom', magnifier_window, 2)


# --- Mouse Callback for Magnifier ---
final_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
def mouse_zoom(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        magnifier_size = 200
        zoom_factor = settings.get('zoom_factor', 3)

        # Define the region to crop from the source
        try:
            crop_size = int(magnifier_size / zoom_factor)
        except ZeroDivisionError:
            crop_size = int(magnifier_size / 2) # Fallback

        # Calculate the top-left corner of the crop region, centered around the mouse
        crop_x1 = x - crop_size // 2
        crop_y1 = y - crop_size // 2
        
        # Ensure the crop region is within the bounds of the final_canvas
        crop_x1 = np.clip(crop_x1, 0, final_canvas.shape[1] - crop_size)
        crop_y1 = np.clip(crop_y1, 0, final_canvas.shape[0] - crop_size)
        
        crop_x2 = crop_x1 + crop_size
        crop_y2 = crop_y1 + crop_size
        
        # Crop the region and resize it (magnify)
        roi = final_canvas[crop_y1:crop_y2, crop_x1:crop_x2]
        if roi.size > 0:
            magnified_roi = cv2.resize(roi, (magnifier_size, magnifier_size), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(magnifier_window, magnified_roi)

cv2.setMouseCallback(main_window, mouse_zoom)


# --- Main Loop ---
while cv2.getWindowProperty(main_window, cv2.WND_PROP_VISIBLE) >= 1:
    # --- Get trackbar positions ---
    settings['fg_width'] = cv2.getTrackbarPos('Width % ', fg_controls_window)
    settings['fg_x'] = cv2.getTrackbarPos('X Pos % ', fg_controls_window)
    settings['fg_y'] = cv2.getTrackbarPos('Y Pos % ', fg_controls_window)
    settings['fg_alpha'] = cv2.getTrackbarPos('Alpha % ', fg_controls_window)
    settings['fg_rotate'] = cv2.getTrackbarPos('Rotate', fg_controls_window)

    settings['bg_width'] = cv2.getTrackbarPos('Width % ', bg_controls_window)
    settings['bg_x'] = cv2.getTrackbarPos('X Pos % ', bg_controls_window)
    settings['bg_y'] = cv2.getTrackbarPos('Y Pos % ', bg_controls_window)
    
    settings['zoom_factor'] = cv2.getTrackbarPos('Zoom', magnifier_window)

    # --- Create blank canvas ---
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # --- Process and draw background ---
    bg_to_draw = background_effect if binarize_mode else background_original
    
    if settings['bg_width'] > 0:
        bg_w = int(canvas_w * (settings['bg_width'] / 100.0)); bg_h = int(bg_w * bg_aspect_ratio)
        if bg_h > canvas_h: bg_h = canvas_h; bg_w = int(bg_h / bg_aspect_ratio)
        if bg_w > 0 and bg_h > 0:
            scaled_bg = cv2.resize(bg_to_draw, (bg_w, bg_h))
            max_x = canvas_w - bg_w; max_y = canvas_h - bg_h
            bg_x = int(max_x * (settings['bg_x'] / 100.0)); bg_y = int(max_y * (settings['bg_y'] / 100.0))
            canvas[bg_y:bg_y+bg_h, bg_x:bg_x+bg_w] = scaled_bg

    # --- Process and draw foreground (webcam) ---
    ret, frame = cap.read()
    if not ret: break

    # Apply rotation
    if settings['fg_rotate'] == 1: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif settings['fg_rotate'] == 2: frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif settings['fg_rotate'] == 3: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if edge_detection_mode: frame = cv2.cvtColor(cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 100, 200), cv2.COLOR_GRAY2BGR)

    frame_aspect_ratio = frame.shape[0] / frame.shape[1]

    if settings['fg_width'] > 0:
        fg_w = int(canvas_w * (settings['fg_width'] / 100.0)); fg_h = int(fg_w * frame_aspect_ratio)
        if fg_h > canvas_h: fg_h = canvas_h; fg_w = int(fg_h / frame_aspect_ratio)
        if fg_w > 0 and fg_h > 0:
            scaled_fg = cv2.resize(frame, (fg_w, fg_h))
            max_x = canvas_w - fg_w; max_y = canvas_h - fg_h
            fg_x = int(max_x * (settings['fg_x'] / 100.0)); fg_y = int(max_y * (settings['fg_y'] / 100.0))
            alpha = settings['fg_alpha'] / 100.0; beta = 1.0 - alpha
            roi = canvas[fg_y:fg_y+fg_h, fg_x:fg_x+fg_w]
            if roi.shape[:2] == scaled_fg.shape[:2]:
                blended_roi = cv2.addWeighted(scaled_fg, alpha, roi, beta, 0.0)
                canvas[fg_y:fg_y+fg_h, fg_x:fg_x+fg_w] = blended_roi

    if show_crosshair: # Draw crosshair on top of everything
        cv2.line(canvas, (0, canvas_h // 2), (canvas_w, canvas_h // 2), (0, 0, 255), 1)
        cv2.line(canvas, (canvas_w // 2, 0), (canvas_w // 2, canvas_h), (0, 0, 255), 1)

    final_canvas = canvas.copy()
    cv2.imshow(main_window, canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('z'): break
    elif key == ord('q'): binarize_mode = not binarize_mode
    elif key == ord('w'): show_crosshair = not show_crosshair
    elif key == ord('e'): edge_detection_mode = not edge_detection_mode
    elif key == ord('1'):
        settings['fg_alpha'] = 25
        cv2.setTrackbarPos('Alpha % ', fg_controls_window, 25)
    elif key == ord('2'):
        settings['fg_alpha'] = 50
        cv2.setTrackbarPos('Alpha % ', fg_controls_window, 50)
    elif key == ord('3'):
        settings['fg_alpha'] = 75
        cv2.setTrackbarPos('Alpha % ', fg_controls_window, 75)
    elif key == ord('4'):
        settings['fg_alpha'] = 100
        cv2.setTrackbarPos('Alpha % ', fg_controls_window, 100)

# --- Cleanup ---
save_settings(settings) # Save settings on exit
cap.release()
cv2.destroyAllWindows()
