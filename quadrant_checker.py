import cv2
from ultralytics import YOLO

DETECT_TARGET = [0, 1]

def main():
    model = YOLO('/home/rokey/hj/Picker_project/webcam_final.pt') 

    cap = cv2.VideoCapture(2)

    # 웹캠이 제대로 열렸는지 확인
    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없습니다.")
        exit()

    # 3. 실시간 루프 시작
    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        # 이미지 프레임 중심 좌표 구하기
        frame_size = frame.shape[:2]
        frame_cx, frame_cy = frame_size[1] // 2, frame_size[0] // 2

        # 프레임을 읽는 데 실패하면 루프 종료
        if not ret:
            print("경고: 프레임을 읽을 수 없습니다.")
            break

        # 4. YOLO 객체 탐지 수행
        results = model.predict(frame, classes=DETECT_TARGET, conf=0.5, verbose=False)

        annotated_frame = frame.copy()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(val) for val in box.xyxy[0].tolist()]
                conf = round(box.conf[0].item(), 2)
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id]
                
                # 바운딩 박스 중심 계산
                xc = int((x1 + x2) / 2)
                yc = int((y1 + y2) / 2)

                color = (0, 255, 0)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f'{cls_name} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(annotated_frame, (xc, yc), 4, (0, 0, 255), -1)
                cv2.circle(annotated_frame, (frame_cx, frame_cy), 4, (0, 255, 255), -1)

                if (xc >= frame_cx) and (yc <= frame_cy):
                    quadrant = 1
                elif (xc <= frame_cx) and (yc <= frame_cy):
                    quadrant = 2
                elif (xc <= frame_cx) and (yc >= frame_cy):
                    quadrant = 3
                else:
                    quadrant = 4

                print(f"Car is now at quadrant{quadrant}.")

        # 6. 결과 시각화 윈도우 표시
        cv2.imshow("Quadrant Checker", annotated_frame)

        # 7. 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 8. 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    print("실시간 탐지 종료.")

if __name__ == "__main__":
    main()