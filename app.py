import streamlit as st
import librosa
import numpy as np
import json
import tempfile
import os
import zipfile
import io
import requests
import random
import string
from pathlib import Path
import google.generativeai as genai

# 1. Gemini API Setup
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"API Key 설정 에러: Secrets에 GEMINI_API_KEY가 있는지 확인해줘. ({e})")
    st.stop()

st.set_page_config(page_title="ADOFAI AI Generator Pro", page_icon="🧊")
st.title("🧊 얼불춤(ADOFAI) AI 맵 자동 생성기 PRO")
st.write("음원 분석 후 서버에 업로드하여 고퀄리티 다이렉트 URL을 발급합니다.")

uploaded_file = st.file_uploader("음악 파일 업로드 (MP3, WAV, OGG 등)", type=None)

if uploaded_file is not None:
    st.info("🎵 오디오 심층 분석 및 스마트 채보 생성 중... (조금만 기다려줘!)")
    
    with st.spinner("처리 중... (조금만 기다려줘!)"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            # 2. 오디오 심층 분석
            y, sr = librosa.load(tmp_file_path)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            # 노래의 평균 BPM 및 하이라이트 구간 분석용 RMS 에너지 추출
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            bpm_value = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
            
            # 에너지(Volume) 기반으로 꺾기 난이도 결정
            rms = librosa.feature.rms(y=y)
            mean_energy = np.mean(rms)
            times_rms = librosa.times_like(rms)

            # 3. 스마트 타일 배치 알고리즘 (Dynamnic Angle Engine V2)
            # ADOFAI 표준 꺾기 각도 리스트
            angles = [0, 45, 90, 135, 180, 225, 270, 315]
            
            angle_data = [0] # 시작 타일은 East
            for i in range(1, len(onset_times)):
                current_time = onset_times[i]
                
                # 해당 타임스탬프의 에너지 확인
                energy_idx = np.argmin(np.abs(times_rms - current_time))
                current_energy = rms[0, energy_idx]

                # Decide next angle based on music intensity
                # 노래가 하이라이트로 터지면 화려한 꺾기 패턴 적용
                if current_energy > mean_energy * 1.5:
                    # 복잡한 구간: 45도, 135도 및 엇박자 감지 시 직각 섞음
                    if i % 4 == 0:
                        next_abs_angle = random.choice([45, 135, 225, 315]) # Acute turns
                    else:
                        next_abs_angle = random.choice([0, 90, 180, 270]) # Grid turns
                else:
                    # 조용한 구간: 직선 또는 부드러운 직각 위주
                    if i % 8 == 0:
                        next_abs_angle = random.choice([90, 270]) 
                    else:
                        next_abs_angle = 0 # Mostly straight
                
                angle_data.append(next_abs_angle)

            # 4. Gemini AI - 고대비 색상 디자인
            prompt = f"""
            너는 리듬게임 A Dance of Fire and Ice 맵 디자이너야.
            BPM {bpm_value:.1f}인 곡의 분위기에 맞는 맵 트랙 색상과 배경 색상을 HEX 코드로 디자인해줘.
            반드시 배경색과 트랙색이 명확하게 구분되는 고대비(High Contrast)로 설정해야 해.
            반드시 아래 2개의 키만 포함된 순수 JSON 객체 하나만 출력해:
            "trackColor", "backgroundColor"
            """
            
            try:
                response = model.generate_content(prompt)
                ai_settings_text = response.text.replace("```json", "").replace("```", "").strip()
                ai_settings = json.loads(ai_settings_text)
                color_track = ai_settings.get("trackColor", "ffffff") # 기본 밝은색
                color_bg = ai_settings.get("backgroundColor", "000000") # 기본 어두운색
            except:
                color_track = "ffffff"
                color_bg = "000000"

            # 5. 노래 재생 문제 해결: 파일명 안전화 (Sanitization)
            uploaded_file_path = Path(uploaded_file.name)
            file_extension = uploaded_file_path.suffix.lower()
            safe_audio_filename = f"song{file_extension}" # song.wav 또는 song.mp3로 고정

            # 6. 얼불춤 유니티 엔진 완벽 호환 순정 세팅
            settings_block = {
                "version": 11,
                "artist": "AI Generated Pro",
                "specialArtistType": "None",
                "artistPermission": "",
                "song": safe_audio_filename, # JSON에서도 song.wav를 부르도록 설정
                "author": "ADOFAI AI",
                "separateCountdownTime": "Enabled",
                "previewImage": "",
                "previewIcon": "",
                "previewIconColor": "0082ba",
                "previewSongStart": 0,
                "previewSongDuration": 10,
                "seizureWarning": "Disabled",
                "levelDesc": "AI Generated High Quality Map",
                "levelTags": "",
                "artistLinks": "",
                "difficulty": 1,
                "requiredMods": [],
                "bpm": bpm_value,
                "volume": 100,
                "offset": 0,
                "pitch": 100,
                "hitsound": "Kick",
                "hitsoundVolume": 100,
                "countdownTicks": 4,
                "trackColorType": "Single",
                "trackColor": color_track,
                "secondaryTrackColor": "ffffff",
                "trackColorAnimDuration": 2,
                "trackColorPulse": "None",
                "trackPulseLength": 10,
                "trackStyle": "Standard",
                "trackAnimation": "None",
                "beatsAhead": 3,
                "beatsBehind": 4,
                "backgroundColor": color_bg,
                "showDefaultBGIfNoImage": "Enabled",
                "bgImage": "",
                "bgImageColor": "ffffff",
                "parallax": [100, 100],
                "bgDisplayMode": "FitToScreen",
                "lockRot": "Disabled",
                "loopBG": "Disabled",
                "unscaledSize": 100,
                "relativeTo": "Player",
                "position": [0, 0],
                "rotation": 0,
                "zoom": 100,
                "bgVideo": "",
                "loopVideo": "Disabled",
                "vidOffset": 0,
                "floorIconOutlines": "Disabled",
                "stickToFloors": "Enabled",
                "planetEase": "Linear",
                "planetEaseParts": 1,
                "planetEasePartBehavior": "Mirror",
                "defaultTextColor": "ffffff",
                "defaultTextShadowColor": "00000050",
                "congratulationsText": "",
                "perfectText": "",
                "strictBehavior": "Ignore",
                "zoomText": "Disabled"
            }

            settings_json = json.dumps(settings_block, ensure_ascii=False)
            angle_json = json.dumps(angle_data)
            
            # 최종 .adofai 문자열 조립
            adofai_str = f"""{{
    "angleData": {angle_json},
    "settings": {settings_json},
    "actions": [],
    "decorations": []
}}"""

            # 7. ZIP 압축 및 노래 파일명 변경 적용
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                # 맵 데이터 저장
                zip_file.writestr("level.adofai", adofai_str)
                
                # 음악 데이터 저장: 한글/특수문자 방지를 위해 safe_audio_filename으로 저장
                uploaded_file.seek(0)
                zip_file.writestr(safe_audio_filename, uploaded_file.read())

            st.success("✨ 고퀄리티 맵 완성! 다운로드 링크 생성 중...")

            # 8. Filebin 서버 업로드 (Unity 엔진 호환)
            bin_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
            # 인코딩 에러 방지를 위해 패키지 이름도 영어로 고정
            upload_url = f"https://filebin.net/{bin_id}/AI_Pro_Map_Package.zip"
            
            headers = {"Content-Type": "application/zip"}
            upload_res = requests.post(upload_url, data=zip_buffer.getvalue(), headers=headers)
            
            if upload_res.status_code == 201:
                st.write("### 🔗 고퀄리티 ADOFAI 다이렉트 URL (PRO):")
                st.code(upload_url, language="text")
                st.write("이 링크를 얼불춤에 붙여넣으면 고대비 색상과 스마트 채보가 적용된 맵이 열리고, 노래도 무조건 나올 거야!")
            else:
                st.error(f"서버 업로드 실패. 상태 코드: {upload_res.status_code}")

            # Backup manual download
            st.download_button(
                label="📦 수동 다운로드(.zip)",
                data=zip_buffer.getvalue(),
                file_name="AI_Pro_Map.zip",
                mime="application/zip"
            )

        except Exception as e:
            st.error(f"에러가 발생했어: {e}")
        finally:
            os.remove(tmp_file_path)
