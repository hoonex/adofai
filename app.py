import streamlit as st
import librosa
import numpy as np
import json
import tempfile
import os
import google.generativeai as genai

# 1. Gemini API 설정 (Streamlit Secrets 활용)
# Streamlit 대시보드의 Advanced settings -> Secrets에 GEMINI_API_KEY="네_키" 입력
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-2.5-flash') # 빠른 처리를 위해 Flash 모델 사용

st.set_page_config(page_title="ADOFAI AI Generator", page_icon="🧊")
st.title("🧊 얼불춤(ADOFAI) AI 맵 자동 생성기")
st.write("음악 파일을 올리면 AI가 비트를 분석하고 고퀄리티 맵을 텍스트 기반으로 찍어냅니다.")

# 파일 업로더
uploaded_file = st.file_uploader("음악 파일 업로드 (모든 파일 가능)", type=None)


if uploaded_file is not None:
    st.info("🎵 음악 분석 및 맵 생성 중... (AI가 기믹을 디자인하고 있습니다)")
    
    with st.spinner("비트 추출 및 타일 연산 중..."):
        # 임시 파일로 저장하여 librosa가 읽을 수 있게 처리
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            # 2. 오디오 수학적 분석 (BPM & Onset)
            y, sr = librosa.load(tmp_file_path)
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            
            # BPM이 배열 형태로 반환될 수 있으므로 스칼라 값으로 변환
            bpm_value = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
            
            # 3. 타일 배치 알고리즘 (기본 변환 공식)
            # 가장 기본적인 1비트 = 1타일(직진 0도 또는 90도 꺾임) 로직
            # 추후 이 부분을 엇박자 감지 로직으로 고도화 가능
            angle_data = [0] # 첫 시작 타일
            for i in range(1, len(beat_times)):
                time_diff = beat_times[i] - beat_times[i-1]
                # 간격이 일정하면 직진(0), 짧아지면(엇박) 꺾기(90) - 예시 로직
                if time_diff < (60 / bpm_value) * 0.8:
                    angle_data.append(90)
                else:
                    angle_data.append(0)

            # 4. AI 기믹 디자인 (Gemini API 호출)
            prompt = f"""
            너는 얼불춤(A Dance of Fire and Ice) 맵 제작 장인이야.
            내가 분석한 노래의 BPM은 {bpm_value:.1f}이고, 총 비트 수는 {len(beat_times)}개야.
            이 곡에 어울리는 맵의 기본 설정(Settings)을 JSON 형식으로만 출력해줘.
            JSON의 키값은 "trackColor", "backgroundColor", "hitsound", "bgImage" 를 포함해야 해.
            오직 JSON 코드만 반환해.
            """
            
            response = model.generate_content(prompt)
            # 응답에서 JSON 파싱 (마크다운 백틱 제거)
            ai_settings_text = response.text.replace("```json", "").replace("```", "").strip()
            
            try:
                ai_settings = json.loads(ai_settings_text)
            except:
                # 파싱 실패 시 기본값 (안전 장치)
                ai_settings = {"trackColor": "ff0000", "backgroundColor": "000000"}

            # 5. 최종 .adofai JSON 구조 조립
            adofai_map = {
                "pathData": "", # angleData를 쓰면 pathData는 비워둠
                "angleData": angle_data,
                "settings": {
                    "version": 11,
                    "artist": "AI Generated",
                    "song": uploaded_file.name,
                    "author": "ADOFAI AI",
                    "separateCountdownTime": True,
                    "previewImage": "",
                    "previewIcon": "",
                    "previewIconColor": "0082ba",
                    "previewSongStart": 0,
                    "previewSongDuration": 10,
                    "seizureWarning": False,
                    "levelDesc": "AI가 자동으로 분석하고 생성한 맵입니다.",
                    "levelTags": "",
                    "artistPermission": "",
                    "artistLinks": "",
                    "syncTrack": 0,
                    "timeSignature": 4,
                    "volume": 100,
                    "overlayColor": "000000",
                    "layer1Image": "",
                    "layer1Position": [0, 0],
                    "layer1Size": [100, 100],
                    "bpm": bpm_value,
                    "offset": 0,
                    "pitch": 100,
                    **ai_settings # AI가 생성한 설정 덮어쓰기
                },
                "actions": [] # 추후 카메라 흔들림 등 이벤트 자동화 추가 위치
            }

            # JSON을 텍스트로 변환 (들여쓰기 없이 압축하면 용량이 줄어듦)
            map_json_str = json.dumps(adofai_map)

            st.success("✨ 맵 생성이 완료되었습니다!")
            st.write(f"- **분석된 BPM:** {bpm_value:.1f}")
            st.write(f"- **생성된 타일 수:** {len(angle_data)}개")
            
            # 다운로드 버튼
            st.download_button(
                label="📥 .adofai 맵 파일 다운로드",
                data=map_json_str,
                file_name=f"AI_Generated_{uploaded_file.name}.adofai",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"에러가 발생했습니다: {e}")
        finally:
            # 서버 용량 관리를 위해 임시 파일 삭제
            os.remove(tmp_file_path)
