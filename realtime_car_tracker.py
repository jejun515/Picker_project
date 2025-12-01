from ultralytics import YOLO
import cv2
import time
import numpy as np


class RealTimeCarTracker:
    def __init__(self):
        model_path = "./balloons_3cls_params.pt"
        cam_source = 2  # usb 웹캠

        self.model = YOLO(model_path)
        
        self.cap = cv2.VideoCapture(cam_source)
        if not self.cap.isOpened():
            raise IOError(f"웹캠 소스 {cam_source}를 열 수 없습니다.")

        print(f"✅ YOLOv8 모델 로드 완료: {model_path}")
        print(f"✅ Car 클래스 추적을 시작합니다.")

    def run_tracking(self):
        """웹캠에서 실시간으로 객체 추적을 실행합니다."""

        tracking_config = 'bytetrack.yaml'
        
        # FPS 계산용 변수
        # prev_time = time.time()
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("프레임을 읽을 수 없습니다. 스트림 종료.")
                break

            # ----------------------------------------------------
            # 1. YOLO 모델 추적 실행
            # ----------------------------------------------------
            results = self.model.track(
                frame, 
                persist=True, 
                tracker=tracking_config,
                classes=[0],    # car 클래스
                verbose=False
            )

            # ----------------------------------------------------
            # 2. 추적 결과 처리 및 시각화
            # ----------------------------------------------------
            
            current_frame = results[0].orig_img # 원본 이미지 가져오기
            
            # 감지된 객체 정보 (바운딩 박스, ID, 클래스 등)
            boxes = results[0].boxes
            
            # 추적 ID가 존재하는 경우에만 처리
            if boxes.id is not None:
                # ID, 바운딩 박스 좌표, 클래스 정보를 numpy 배열에서 리스트로 변환
                track_ids = boxes.id.tolist()
                xyxy = boxes.xyxy.tolist()
                
                # 각 감지된 car 객체에 대해 반복
                for i, track_id in enumerate(track_ids):
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    
                    # 텍스트 오버레이
                    label = f"Car ID: {track_id}"
                    
                    # 바운딩 박스와 ID 표시
                    cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        current_frame, 
                        label, 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        (0, 255, 0), 
                        2
                    )
            
            # ----------------------------------------------------
            # 3. FPS 표시 및 화면 출력
            # ----------------------------------------------------
            
            # FPS 계산
            # current_time = time.time()
            # fps = 1 / (current_time - prev_time)
            # prev_time = current_time
            
            # cv2.putText(
            #     current_frame, 
            #     f"FPS: {fps:.2f}", 
            #     (10, 30), 
            #     cv2.FONT_HERSHEY_SIMPLEX, 
            #     1, 
            #     (255, 0, 0), 
            #     2
            # )
            
            cv2.imshow("YOLOv8 Real-Time Car Tracking", current_frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 4. 종료 시 자원 해제
        self.cap.release()
        cv2.destroyAllWindows()
        print("프로그램을 종료합니다.")

# =======================================================
# 메인 실행
# =======================================================

if __name__ == '__main__':
    try:
        tracker = RealTimeCarTracker()
        tracker.run_tracking()
    except Exception as e:
        print(f"오류 발생: {e}")