import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

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

# --- Main Program ---

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
# frame_aspect_ratio = frame.shape[0] / frame.shape[1] # height / width

# 3. Load background image and get its aspect ratio
background_original = cv2.imread(background_image_path)
if background_original is None:
    print(f"Error: Could not read the background image at '{background_image_path}'")
    cap.release()
    exit()
bg_aspect_ratio = background_original.shape[0] / background_original.shape[1]

# 4. Create UI Windows
main_window = 'Webcam Live'
bg_controls_window = 'Background Controls'
fg_controls_window = 'Foreground Controls'

cv2.namedWindow(main_window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(main_window, canvas_w, canvas_h)

cv2.namedWindow(bg_controls_window)
cv2.resizeWindow(bg_controls_window, 300, 150)
cv2.moveWindow(bg_controls_window, 100, 100) # Position on left

cv2.namedWindow(fg_controls_window)
cv2.resizeWindow(fg_controls_window, 300, 250) # Increased height for new trackbar
cv2.moveWindow(fg_controls_window, canvas_w + 200, 100) # Position on right

# Create trackbars for Background Image in its own window
cv2.createTrackbar('Width % ', bg_controls_window, 100, 100, nothing)
cv2.createTrackbar('X Pos % ', bg_controls_window, 50, 100, nothing)
cv2.createTrackbar('Y Pos % ', bg_controls_window, 50, 100, nothing)

# Create trackbars for Foreground (Webcam) in its own window
cv2.createTrackbar('Width % ', fg_controls_window, 50, 100, nothing)
cv2.createTrackbar('X Pos % ', fg_controls_window, 50, 100, nothing)
cv2.createTrackbar('Y Pos % ', fg_controls_window, 50, 100, nothing)
cv2.createTrackbar('Alpha % ', fg_controls_window, 70, 100, nothing)
cv2.createTrackbar('Rotate', fg_controls_window, 0, 3, nothing) # 0:0, 1:90, 2:180, 3:270

# --- Main Loop ---
while cv2.getWindowProperty(main_window, cv2.WND_PROP_VISIBLE) >= 1:
    # --- Get trackbar positions ---
    fg_width_p = cv2.getTrackbarPos('Width % ', fg_controls_window)
    fg_x_p = cv2.getTrackbarPos('X Pos % ', fg_controls_window)
    fg_y_p = cv2.getTrackbarPos('Y Pos % ', fg_controls_window)
    fg_alpha_p = cv2.getTrackbarPos('Alpha % ', fg_controls_window)
    rotation = cv2.getTrackbarPos('Rotate', fg_controls_window)

    bg_width_p = cv2.getTrackbarPos('Width % ', bg_controls_window)
    bg_x_p = cv2.getTrackbarPos('X Pos % ', bg_controls_window)
    bg_y_p = cv2.getTrackbarPos('Y Pos % ', bg_controls_window)

    # --- Create blank canvas ---
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # --- Process and draw background ---
    if bg_width_p > 0:
        bg_w = int(canvas_w * (bg_width_p / 100.0)); bg_h = int(bg_w * bg_aspect_ratio)
        if bg_h > canvas_h: bg_h = canvas_h; bg_w = int(bg_h / bg_aspect_ratio)
        if bg_w > 0 and bg_h > 0:
            scaled_bg = cv2.resize(background_original, (bg_w, bg_h))
            max_x = canvas_w - bg_w; max_y = canvas_h - bg_h
            bg_x = int(max_x * (bg_x_p / 100.0)); bg_y = int(max_y * (bg_y_p / 100.0))
            canvas[bg_y:bg_y+bg_h, bg_x:bg_x+bg_w] = scaled_bg

    # --- Process and draw foreground (webcam) ---
    ret, frame = cap.read()
    if not ret: break

    # Apply rotation
    if rotation == 1:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 2:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 3:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # if rotation is 0, do nothing (original orientation)
    
    # Recalculate aspect ratio after rotation
    frame_aspect_ratio = frame.shape[0] / frame.shape[1]

    if fg_width_p > 0:
        fg_w = int(canvas_w * (fg_width_p / 100.0)); fg_h = int(fg_w * frame_aspect_ratio)
        if fg_h > canvas_h: fg_h = canvas_h; fg_w = int(fg_h / frame_aspect_ratio)
        if fg_w > 0 and fg_h > 0:
            scaled_fg = cv2.resize(frame, (fg_w, fg_h))
            max_x = canvas_w - fg_w; max_y = canvas_h - fg_h
            fg_x = int(max_x * (fg_x_p / 100.0)); fg_y = int(max_y * (fg_y_p / 100.0))
            alpha = fg_alpha_p / 100.0; beta = 1.0 - alpha
            roi = canvas[fg_y:fg_y+fg_h, fg_x:fg_x+fg_w]
            
            # Ensure ROI and scaled_fg have the same dimensions
            if roi.shape[:2] == scaled_fg.shape[:2]:
                blended_roi = cv2.addWeighted(scaled_fg, alpha, roi, beta, 0.0)
                canvas[fg_y:fg_y+fg_h, fg_x:fg_x+fg_w] = blended_roi

    # --- Display result ---
    cv2.imshow(main_window, canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()