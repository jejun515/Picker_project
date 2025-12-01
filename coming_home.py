import rclpy
from rclpy.node import Node
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator, TurtleBot4Directions

def main(args=None):
    rclpy.init(args=args)
    
    navigator = TurtleBot4Navigator()

    # 1. 내비게이션 시스템 활성화 대기 (필수)
    print("⏳ 내비게이션 시스템 연결 중...")
    navigator.waitUntilNav2Active()
    print("✅ 연결 완료! 복귀 시퀀스를 시작합니다.")

    # 2. 도킹 전 대기 장소 설정 (x=-0.3, y=-0.3)
    # 로봇이 도킹 스테이션을 바라보도록(NORTH) 설정
    staging_pose = navigator.getPoseStamped([-0.3, -0.3], TurtleBot4Directions.NORTH)

    print(f"🚀 복귀 시작! {[-0.3, -0.3]} 지점으로 이동합니다.")
    
    # 3. 무조건 이동 시작
    result = navigator.goToPose(staging_pose)

    # 4. 도착 후 도킹 시도
    if result:
        print("📍 도착 완료. 도킹을 시도합니다...")
        navigator.dock()
        
        # 결과 출력 (선택 사항)
        if navigator.getDockedStatus():
             print("🎉 도킹 성공! 충전 시작.")
        else:
             print("⚠️ 도킹 실패. 다시 시도하거나 위치를 확인하세요.")
    else:
        print("❌ 이동 실패! 경로가 막혀있거나 로봇이 길을 잃었습니다.")

    rclpy.shutdown()

if __name__ == '__main__':
    main()