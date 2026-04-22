import cv2
import numpy as np
import math
import pygame
import csv
import json
import os
from collections import deque
import sys
from datetime import datetime

#Configuration
VIDEO_SOURCE = 'exp.MP4' 
RESIZE_FACTOR = 1.0 
DISPLAY_SCALE = 0.5 
SPIKE_SENSITIVITY = 15 
SMOOTHING_WINDOW = 15 
OUTPUT_CSV_FILE = datetime.now().strftime("%H%M%S.csv")
CONFIG_FILE = 'tracker_config.json'

MAX_SPRING_TRAVEL = 40  # Max pixels the pivot is allowed to vibrate from its origin
SKELETON_TOLERANCE = 5  # Max pixels the dot is allowed to drift from the physical radius

COORD_SMOOTHING = 0.15  

BG_COLOR = (25, 25, 30)
TEXT_COLOR = (240, 240, 240)
HIGHLIGHT_COLOR = (0, 255, 100)
LOCKED_COLOR = (150, 150, 150)
PADDING = 20
IMG_Y_OFFSET = 60 

STATE_AIRFOIL = 0
STATE_LED = 1
STATE_PIVOT = 2
STATE_LEAD = 3
STATE_TRACKING = 4

def create_tracker():
    try:
        return cv2.TrackerCSRT_create()
    except AttributeError:
        return cv2.legacy.TrackerCSRT_create()

def get_center(bbox):
    x, y, w, h = [int(v) for v in bbox]
    return (x + w // 2, y + h // 2)

def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p1[1] - p2[1] 
    return math.degrees(math.atan2(dy, dx))

def cv2_to_pygame(cv_img):
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(np.swapaxes(rgb_img, 0, 1))

def get_clamped_rect(p1, p2, img_w, img_h):
    x1, y1 = p1
    x2, y2 = p2
    x = max(0, min(x1, x2))
    y = max(0, min(y1, y2))
    w = min(img_w - x, abs(x2 - x1))
    h = min(img_h - y, abs(y2 - y1))
    return (int(x), int(y), int(w), int(h))

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    ret, first_frame = cap.read()
    if not ret: return

    first_frame = cv2.resize(first_frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    base_w, base_h = first_frame.shape[1], first_frame.shape[0]
    
    pygame.init()
    
    init_win_w = int(base_w * DISPLAY_SCALE) + (PADDING * 2)
    init_win_h = int(base_h * DISPLAY_SCALE) + IMG_Y_OFFSET + PADDING
    screen = pygame.display.set_mode((init_win_w, init_win_h))
    pygame.display.set_caption("PivotPoint")
    
    font = pygame.font.SysFont("arial", 20, bold=True)
    small_font = pygame.font.SysFont("arial", 14)
    clock = pygame.time.Clock()

    state = STATE_AIRFOIL
    running = True

    airfoil_roi = led_roi = pivot_roi = lead_roi = None
    airfoil_first_crop = None
    
    initial_pivot_pt = (0, 0)
    initial_lead_pt = (0, 0)
    airfoil_radius = 0.0
    smooth_pt1 = [0.0, 0.0]
    smooth_pt2 = [0.0, 0.0]
    
    trackers = []
    initial_angle = 0
    pivot_w, pivot_h, lead_w, lead_h = 0, 0, 0, 0
    ambient_redness = None
    aoa_history = deque(maxlen=SMOOTHING_WINDOW)

    csv_file = csv_writer = None
    drawing = False
    start_pos = current_roi = (0, 0, 0, 0) 
    bg_cache = None
    last_state = -1
    
    has_led_glowed_once = False

    instructions = {
        STATE_AIRFOIL: "Draw box around ENTIRE Airfoil. Press ENTER.",
        STATE_LED: "Draw a TIGHT box around the RED LED. Press ENTER.",
        STATE_PIVOT: "Draw Central Pivot (Bolt). Press ENTER.",
        STATE_LEAD: "Draw Leading Edge Dot. Press ENTER."
    }

    # This func auto-loads saved config. Delete tracker_config.json if you need to draw new bboxes.
    if os.path.exists(CONFIG_FILE):
        print(f"Found {CONFIG_FILE}. Loading previous bounding boxes...")
        with open(CONFIG_FILE, 'r') as f: config = json.load(f)
            
        airfoil_roi, led_roi = tuple(config['airfoil']), tuple(config['led'])
        pivot_roi, lead_roi = tuple(config['pivot']), tuple(config['lead'])
        
        ax, ay, aw, ah = airfoil_roi
        airfoil_first_crop = first_frame[ay:ay+ah, ax:ax+aw].copy()
        pivot_w, pivot_h, lead_w, lead_h = pivot_roi[2], pivot_roi[3], lead_roi[2], lead_roi[3]

        trackers = [create_tracker(), create_tracker()]
        trackers[0].init(airfoil_first_crop, pivot_roi)
        trackers[1].init(airfoil_first_crop, lead_roi)

        initial_pivot_pt = get_center(pivot_roi)
        initial_lead_pt = get_center(lead_roi)
        initial_angle = calculate_angle(initial_pivot_pt, initial_lead_pt)
        
        smooth_pt1 = list(initial_pivot_pt)
        smooth_pt2 = list(initial_lead_pt)
        airfoil_radius = math.hypot(initial_lead_pt[0] - initial_pivot_pt[0], initial_lead_pt[1] - initial_pivot_pt[1])
        
        disp_aw, disp_ah = int(aw * DISPLAY_SCALE), int(ah * DISPLAY_SCALE)
        disp_lw, disp_lh = int(disp_aw * 0.25), int(disp_ah * 0.25)
        screen = pygame.display.set_mode((disp_aw + (PADDING * 2), disp_ah + disp_lh + IMG_Y_OFFSET + (PADDING * 3) + 40))

        csv_file = open(OUTPUT_CSV_FILE, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Timestamp_ms', 'Frame_Number', 'AoA_deg', 'LED_Status'])

        state = STATE_TRACKING
    else:
        print("No config file found. Please set up the tracking boxes manually.")


    try:
        while running:
            if state <= STATE_LED:
                cur_img_w, cur_img_h = base_w, base_h
            elif state <= STATE_LEAD and airfoil_first_crop is not None:
                cur_img_w, cur_img_h = airfoil_first_crop.shape[1], airfoil_first_crop.shape[0]

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_q, pygame.K_ESCAPE]: running = False
                    
                    elif event.key == pygame.K_RETURN and state < STATE_TRACKING:
                        if current_roi[2] > 5 and current_roi[3] > 5: 
                            if state == STATE_AIRFOIL:
                                airfoil_roi, state = current_roi, STATE_LED
                            elif state == STATE_LED:
                                led_roi = current_roi
                                ax, ay, aw, ah = airfoil_roi
                                airfoil_first_crop = first_frame[ay:ay+ah, ax:ax+aw].copy()
                                screen = pygame.display.set_mode((int(aw*DISPLAY_SCALE)+PADDING*2, int(ah*DISPLAY_SCALE)+IMG_Y_OFFSET+PADDING))
                                state = STATE_PIVOT
                            elif state == STATE_PIVOT:
                                pivot_roi, state = current_roi, STATE_LEAD
                            elif state == STATE_LEAD:
                                lead_roi = current_roi
                                pivot_w, pivot_h = pivot_roi[2], pivot_roi[3]
                                lead_w, lead_h = lead_roi[2], lead_roi[3]
                                
                                trackers = [create_tracker(), create_tracker()]
                                trackers[0].init(airfoil_first_crop, pivot_roi)
                                trackers[1].init(airfoil_first_crop, lead_roi)

                                initial_pivot_pt = get_center(pivot_roi)
                                initial_lead_pt = get_center(lead_roi)
                                initial_angle = calculate_angle(initial_pivot_pt, initial_lead_pt)
                                
                                smooth_pt1 = list(initial_pivot_pt)
                                smooth_pt2 = list(initial_lead_pt)
                                airfoil_radius = math.hypot(initial_lead_pt[0] - initial_pivot_pt[0], initial_lead_pt[1] - initial_pivot_pt[1])
                                
                                with open(CONFIG_FILE, 'w') as f:
                                    json.dump({'airfoil': airfoil_roi, 'led': led_roi, 'pivot': pivot_roi, 'lead': lead_roi}, f)
                                
                                aw, ah = airfoil_roi[2], airfoil_roi[3]
                                d_aw, d_ah = int(aw*DISPLAY_SCALE), int(ah*DISPLAY_SCALE)
                                screen = pygame.display.set_mode((d_aw+PADDING*2, d_ah+int(d_ah*0.25)+IMG_Y_OFFSET+PADDING*3+40))

                                csv_file = open(OUTPUT_CSV_FILE, mode='w', newline='')
                                csv_writer = csv.writer(csv_file)
                                csv_writer.writerow(['Timestamp_ms', 'Frame_Number', 'AoA_deg', 'LED_Status'])

                                state = STATE_TRACKING
                            
                            current_roi, drawing = (0, 0, 0, 0), False

                elif event.type == pygame.MOUSEBUTTONDOWN and state < STATE_TRACKING:
                    drawing = True
                    start_pos = (event.pos[0]-PADDING)/DISPLAY_SCALE, (event.pos[1]-IMG_Y_OFFSET)/DISPLAY_SCALE
                elif event.type == pygame.MOUSEMOTION and drawing and state < STATE_TRACKING:
                    cur_pos = (event.pos[0]-PADDING)/DISPLAY_SCALE, (event.pos[1]-IMG_Y_OFFSET)/DISPLAY_SCALE
                    current_roi = get_clamped_rect(start_pos, cur_pos, cur_img_w, cur_img_h)
                elif event.type == pygame.MOUSEBUTTONUP:
                    drawing = False

            screen.fill(BG_COLOR)

            #UI
            if state < STATE_TRACKING:
                screen.blit(font.render(instructions[state], True, TEXT_COLOR), (PADDING, PADDING))
                
                if state != last_state:
                    img = first_frame if state <= STATE_LED else airfoil_first_crop
                    bg_cache = pygame.transform.smoothscale(cv2_to_pygame(img), (int(img.shape[1]*DISPLAY_SCALE), int(img.shape[0]*DISPLAY_SCALE)))
                    last_state = state
                screen.blit(bg_cache, (PADDING, IMG_Y_OFFSET))

                if state == STATE_LED and airfoil_roi:
                    rx, ry, rw, rh = [int(v*DISPLAY_SCALE) for v in airfoil_roi]
                    pygame.draw.rect(screen, LOCKED_COLOR, (rx+PADDING, ry+IMG_Y_OFFSET, rw, rh), 2)
                elif state == STATE_LEAD and pivot_roi:
                    rx, ry, rw, rh = [int(v*DISPLAY_SCALE) for v in pivot_roi]
                    pygame.draw.rect(screen, LOCKED_COLOR, (rx+PADDING, ry+IMG_Y_OFFSET, rw, rh), 2)

                if current_roi[2] > 0:
                    rx, ry, rw, rh = [int(v*DISPLAY_SCALE) for v in current_roi]
                    pygame.draw.rect(screen, HIGHLIGHT_COLOR, (rx+PADDING, ry+IMG_Y_OFFSET, rw, rh), 2)

        
            elif state == STATE_TRACKING:
                if not cap.isOpened(): break
                ret, frame = cap.read()
                if not ret: break

                frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

                ax, ay, aw, ah = airfoil_roi
                lx, ly, lw, lh = led_roi
                
                airfoil_frame = frame[ay:ay+ah, ax:ax+aw]
                led_frame = frame[ly:ly+lh, lx:lx+lw]
                display_led = led_frame.copy()

                success1, bbox1 = trackers[0].update(airfoil_frame)
                success2, bbox2 = trackers[1].update(airfoil_frame)
                
                if success1 and success2:
                    raw_pt1 = get_center(bbox1)
                    raw_pt2 = get_center(bbox2) 
                   
                    pivot_drift = math.hypot(raw_pt1[0] - initial_pivot_pt[0], raw_pt1[1] - initial_pivot_pt[1])
                    if pivot_drift > MAX_SPRING_TRAVEL:
                        clamped_angle = math.atan2(raw_pt1[1] - initial_pivot_pt[1], raw_pt1[0] - initial_pivot_pt[0])
                        raw_pt1 = (
                            int(initial_pivot_pt[0] + MAX_SPRING_TRAVEL * math.cos(clamped_angle)),
                            int(initial_pivot_pt[1] + MAX_SPRING_TRAVEL * math.sin(clamped_angle))
                        )
                        trackers[0].init(airfoil_frame, (raw_pt1[0]-pivot_w//2, raw_pt1[1]-pivot_h//2, pivot_w, pivot_h))

                    angle_rad = math.atan2(raw_pt1[1] - raw_pt2[1], raw_pt2[0] - raw_pt1[0])
                    perfect_pt2 = (
                        int(raw_pt1[0] + airfoil_radius * math.cos(angle_rad)),
                        int(raw_pt1[1] - airfoil_radius * math.sin(angle_rad))
                    )
                    
                    dot_drift = math.hypot(raw_pt2[0] - perfect_pt2[0], raw_pt2[1] - perfect_pt2[1])
                    if dot_drift > SKELETON_TOLERANCE:
                        raw_pt2 = perfect_pt2
                        trackers[1].init(airfoil_frame, (raw_pt2[0]-lead_w//2, raw_pt2[1]-lead_h//2, lead_w, lead_h))

                    smooth_pt1[0] = smooth_pt1[0] * (1 - COORD_SMOOTHING) + raw_pt1[0] * COORD_SMOOTHING
                    smooth_pt1[1] = smooth_pt1[1] * (1 - COORD_SMOOTHING) + raw_pt1[1] * COORD_SMOOTHING
                    
                    smooth_pt2[0] = smooth_pt2[0] * (1 - COORD_SMOOTHING) + raw_pt2[0] * COORD_SMOOTHING
                    smooth_pt2[1] = smooth_pt2[1] * (1 - COORD_SMOOTHING) + raw_pt2[1] * COORD_SMOOTHING

                    final_pt1 = (int(smooth_pt1[0]), int(smooth_pt1[1]))
                    final_pt2 = (int(smooth_pt2[0]), int(smooth_pt2[1]))

                    current_raw_angle = calculate_angle(final_pt1, final_pt2)
                    raw_aoa = current_raw_angle - initial_angle
                    raw_aoa = (raw_aoa + 180) % 360 - 180 

                    aoa_history.append(raw_aoa)
                    smoothed_aoa = sum(aoa_history) / len(aoa_history)
                    led_on = False 

                    if led_frame.size > 0:
                        current_redness = np.mean(led_frame[:, :, 2]) 
                        if ambient_redness is None: ambient_redness = current_redness
                        if current_redness < ambient_redness + 5: ambient_redness = ambient_redness * 0.9 + current_redness * 0.1

                        if current_redness > (ambient_redness + SPIKE_SENSITIVITY):
                            led_on = True
                            has_led_glowed_once = True # --- TRIGGER FLAG UPON FIRST GLOW ---
                            
                            cv2.rectangle(display_led, (0, 0), (lw-1, lh-1), (0, 0, 255), 4)
                            cv2.putText(airfoil_frame, "LED ON", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    

                    if not has_led_glowed_once or led_on:
                        csv_writer.writerow([f"{timestamp_ms:.2f}", frame_num, f"{smoothed_aoa:.4f}", "ON" if led_on else "OFF"])
                        cv2.putText(airfoil_frame, f"AoA: {smoothed_aoa:.2f} deg", (final_pt2[0]+15, final_pt2[1]), 0, 0.7, (255, 255, 0), 2)

                    line_length = math.hypot(final_pt2[0] - final_pt1[0], final_pt2[1] - final_pt1[1])
                    base_x = int(final_pt1[0] + line_length * math.cos(math.radians(initial_angle)))
                    base_y = int(final_pt1[1] - line_length * math.sin(math.radians(initial_angle))) 
                    
                    cv2.line(airfoil_frame, final_pt1, (base_x, base_y), (200, 200, 200), 2) 
                    cv2.line(airfoil_frame, final_pt1, final_pt2, (255, 0, 0), 2) 
                    cv2.rectangle(airfoil_frame, (raw_pt1[0]-pivot_w//2, raw_pt1[1]-pivot_h//2), (raw_pt1[0]+pivot_w//2, raw_pt1[1]+pivot_h//2), (50, 50, 50), 1)
                    cv2.rectangle(airfoil_frame, (raw_pt2[0]-lead_w//2, raw_pt2[1]-lead_h//2), (raw_pt2[0]+lead_w//2, raw_pt2[1]+lead_h//2), (50, 50, 50), 1)
                    cv2.circle(airfoil_frame, final_pt1, 4, (0, 255, 0), -1) 
                    cv2.circle(airfoil_frame, final_pt2, 4, (0, 255, 0), -1)

                else:
                    cv2.putText(airfoil_frame, "TRACKING LOST", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    if not has_led_glowed_once:
                        csv_writer.writerow([f"{timestamp_ms:.2f}", frame_num, "NaN", "UNKNOWN"])

                # Draw UI
                d_aw, d_ah = int(aw * DISPLAY_SCALE), int(ah * DISPLAY_SCALE)
                d_lw, d_lh = int(d_aw * 0.25), int(d_ah * 0.25)
                
                screen.blit(font.render("Analysing AoA", True, TEXT_COLOR), (PADDING, PADDING))
                screen.blit(font.render("LED Status", True, TEXT_COLOR), (PADDING, IMG_Y_OFFSET + d_ah + 10))

                screen.blit(pygame.transform.scale(cv2_to_pygame(airfoil_frame), (d_aw, d_ah)), (PADDING, IMG_Y_OFFSET))
                screen.blit(pygame.transform.scale(cv2_to_pygame(display_led), (d_lw, d_lh)), (PADDING, IMG_Y_OFFSET + d_ah + 40))

            pygame.display.flip()
            clock.tick(60) 
            
    finally:
        cap.release()
        if csv_file: csv_file.close()
        pygame.quit()

if __name__ == "__main__":
    main()
