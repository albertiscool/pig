import cv2
import os

input_dir = r"c:\Users\User\Desktop\final_pig_v0\scratch\sample_frames_black_pig"
output_dir = r"c:\Users\User\Desktop\final_pig_v0\output_llm_cropped_faces_black_pig"
os.makedirs(output_dir, exist_ok=True)

# LLM Visual Grounding Coordinates (Estimated by Gemini Multimodal Vision)
# Format: { 'frame_name': { 'face': [x1, y1, x2, y2], 'eyes': [[x1,y1,x2,y2], ...], 'snout': [x1,y1,x2,y2], 'ears': [[x1,y1,x2,y2], ...] } }
annotations = {
    "llm_test_frame_01.jpg": {
        "face": [350, 200, 700, 600],
        "eyes": [[500, 350, 550, 380]],
        "snout": [600, 450, 700, 550],
        "ears": [[400, 200, 500, 300], [550, 220, 650, 320]]
    },
    "llm_test_frame_02.jpg": {
        "face": [620, 120, 800, 400],
        "eyes": [[700, 250, 730, 290]],
        "snout": [750, 320, 800, 390],
        "ears": [[680, 150, 800, 230]]
    },
    "llm_test_frame_03.jpg": {
        "face": [650, 130, 820, 420],
        "eyes": [[720, 260, 750, 300]],
        "snout": [770, 330, 820, 400],
        "ears": [[700, 160, 810, 240]]
    },
    "llm_test_frame_04.jpg": {
        "face": [300, 140, 740, 650],
        "eyes": [[460, 410, 510, 460]],
        "snout": [600, 370, 740, 520],
        "ears": [[340, 310, 450, 450], [480, 140, 620, 300]]
    },
    "llm_test_frame_05.jpg": {
        "face": [500, 130, 730, 470],
        "eyes": [[600, 270, 630, 300]],
        "snout": [630, 370, 720, 470],
        "ears": [[540, 130, 690, 240]]
    }
}

for i in range(1, 6):
    filename = f"llm_test_frame_{i:02d}.jpg"
    filepath = os.path.join(input_dir, filename)
    if not os.path.exists(filepath):
        continue
        
    img = cv2.imread(filepath)
    anno = annotations.get(filename)
    if not anno: continue

    # Draw Face (Green)
    fx1, fy1, fx2, fy2 = anno["face"]
    
    # Crop Face with some margin
    margin = 40
    crop_x1 = max(0, fx1 - margin)
    crop_y1 = max(0, fy1 - margin)
    crop_x2 = min(img.shape[1], fx2 + margin)
    crop_y2 = min(img.shape[0], fy2 + margin)
    
    face_crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    
    # Adjust coordinates to cropped image
    def adjust(box):
        return [box[0] - crop_x1, box[1] - crop_y1, box[2] - crop_x1, box[3] - crop_y1]

    # Draw Eyes (Blue)
    for eye in anno["eyes"]:
        ex1, ey1, ex2, ey2 = adjust(eye)
        cv2.rectangle(face_crop, (ex1, ey1), (ex2, ey2), (255, 0, 0), 2)
        cv2.putText(face_crop, "Left/Right Eye (LLM)", (ex1, max(15, ey1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    # Draw Snout (Yellow)
    sx1, sy1, sx2, sy2 = adjust(anno["snout"])
    cv2.rectangle(face_crop, (sx1, sy1), (sx2, sy2), (0, 255, 255), 2)
    cv2.putText(face_crop, "Nose ROI (LLM)", (sx1, max(15, sy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Draw Ears (Red)
    for ear in anno["ears"]:
        erx1, ery1, erx2, ery2 = adjust(ear)
        cv2.rectangle(face_crop, (erx1, ery1), (erx2, ery2), (0, 0, 255), 2)
        cv2.putText(face_crop, "Ear Root (LLM)", (erx1, max(15, ery1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Save
    out_path = os.path.join(output_dir, f"llm_face_black_pig_{i:02d}.jpg")
    cv2.imwrite(out_path, face_crop)
    print(f"Saved LLM Crop to {out_path}")
