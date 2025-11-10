from ultralytics import YOLO

model = YOLO('D:/Desktop/IA_barthos/runs/detect/train44/weights/best.pt')

results = model.track(source='./Vídeos/S2_T2_VF.MP4',show=True, tracker="bytetrack.yaml")