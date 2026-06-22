# 2026-HAI-ProjectTeam4
Project Name TBD, Team 4, Project(Model Implementation), 2026 HAI S1
# HAI-2026-1-TEAM-4---TRAINING
- parser.py 프로토타입 : rawsource 폴더의 .osu파일 source 폴더에 변환
- requirement : ffmpeg 8, python 3.14.3
- crawling_beatmap.py 사용법 : 
    0. 그냥 실행해본 후 정상실행이 안된다면.
    1. "크롬"으로 osu.ppy.sh에 로그인한다.
    2. f12로 개발자 도구를 킨 후 적당한 mania 4key 비트맵 하나를 다운로드한다. 
    3. network에서 download이름의 쿠키를 찾은 후 요청 헤더에서 쿠키를 복사해서 소스코드의 osu_session_value에 붙여넣는다.
- 03/26 TODO:
    1. 변속곡 대응할 수 있게 parser 수정 또는 변속곡 제거
    2. lstm, transfomer 모델로 데모 하나씩 만들어오고, 설명할수 있을만큼 공부해오기.
    3. parser.py output의 메타데이터에 beatmapsetID 추가.
    4. ogg대응
- 04/02 TODO
    1. 너무 짦은 HOLD 고치기 (EX. 2->4)
    2. 잘못된 HOLD 처리 (EX. 2->3->3->2->3->4)
이상의 TODO는 완료.

- 04/30 회의
    1. 

