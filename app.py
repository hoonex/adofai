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
import soundfile as sf
import google.generativeai as genai

# 1. Gemini API Setup
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"API Key 설정 에러: {e}")
    st.stop()

st.set_page_config(page_title="ADOFAI AI Generator Pro", page_icon="🧊")
st.title("🧊 얼불춤(ADOFAI) AI 맵 자동 생성기 PRO")
st.write("모바일 완벽 호환 OGG 인코딩 & 부드러운 채보 알고리즘 적용")

uploaded_file = st.file_uploader("음악 파일 업로드", type=None)

if uploaded_file is not None:
    st.info("🎵 오디오 분석 및 순정 OGG 인코딩 중... (시간이 조금 걸릴 수 있어!)")
    
    with st.spinner("처리 중..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            # 2. 오디오 로드 (모바일 유니티 에러 방지를 위해 OGG로 재인코딩할 준비)
            y_stereo, sr = librosa.load(tmp_file_path, sr=None, mono=False)
            
            # 분석용 모노 변환
            if y_stereo.ndim > 1:
                y_mono = librosa.to_mono(y_stereo)
                audio_export = y_stereo.T # soundfile 저장을 위한 형태 변환
            else:
                y_mono = y_stereo
                audio_export = y_stereo

            # 3. 비트 및 에너지 분석
            onset_frames = librosa.onset.onset_detect(y=y_mono, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr)
            bpm_value = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
            
            rms = librosa.feature.rms(y=y_mono)
            mean_energy = np.mean(rms)
            times_rms = librosa.times_like(rms)

            # 4. 스마트 타일 배치 (Delta Angle Engine - 부드럽고 예쁜 곡선형 맵)
            current_angle = 0
            angle_data = [current_angle]
            pattern_dir = 1 # 1은 우회전, -1은 좌회전 패턴
            
            for i in range(1, len(onset_times)):
                time_diff = onset_times[i] - onset_times[i-1]
                current_time = onset_times[i]
                
                energy_idx = np.argmin(np.abs(times_rms - current_time))
                current_energy = rms[0, energy_idx]
                
                # 맵이 한쪽으로만 감기지 않게 가끔 방향 전환
                if random.random() < 0.15:
                    pattern_dir *= -1
                    
                if current_energy > mean_energy * 1.3:
                    # 음악이 터지는 하이라이트 구간
                    if time_diff < (60 / bpm_value) * 0.6: 
                        delta = 90 * pattern_dir  # 화려한 계단식 지그재그
                    else:
                        delta = 45 * pattern_dir  # 큼직한 소용돌이 곡선
                else:
                    # 음악이 잔잔한 구간
                    if i % 4 == 0:
                        delta = 45 * pattern_dir  # 살짝 꺾어주기
                    else:
                        delta = 0  # 직진
                        
                # 절대 각도로 변환해서 저장
                current_angle = (current_angle + delta) % 360
                angle_data.append(int(current_angle))

            # 5. 디자인 설정
            prompt = f"""
            너는 A Dance of Fire and Ice 맵 디자이너야.
            BPM {bpm_value:.1f}의 곡에 어울리는 트랙 색상과 배경 색상을 디자인해.
            반드시 아래 2개의 키만 포함된 순수 JSON 객체 하나만 출력해:
            "trackColor", "backgroundColor"
            """
            try:
                response = model.generate_content(prompt)
                ai_settings_text = response.text.replace("```json", "").replace("```", "").strip()
                ai_settings = json.loads(ai_settings_text)
                color_track = ai_settings.get("trackColor", "ffffff")
                color_bg = ai_settings.get("backgroundColor", "000000")
            except:
                color_track = "ffffff"
                color_bg = "000000"

            # 무조건 OGG 파일로 고정
            safe_audio_filename = "song.ogg"

            # 6. 유니티 호환 세팅
            settings_block = {
                "version": 11,
                "artist": "AI Generator Pro",
                "specialArtistType": "None",
                "artistPermission": "",
                "song": safe_audio_filename,
                "author": "ADOFAI AI",
                "separateCountdownTime": "Enabled",
                "previewImage": "",
                "previewIcon": "",
                "previewIconColor": "0082ba",
                "previewSongStart": 0,
                "previewSongDuration": 10,
                "seizureWarning": "Disabled",
                "levelDesc": "High Quality AI Map",
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
            
            adofai_str = f"""{{
    "angleData": {angle_json},
    "settings": {settings_json},
    "actions": [],
    "decorations": []
}}"""

            # 7. 오디오 OGG 변환 및 ZIP 압축
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                # 맵 텍스트 추가
                zip_file.writestr("level.adofai", adofai_str)
                
                # 오디오를 순수 OGG로 인코딩해서 추가 (메타데이터 싹 지움)
                audio_buffer = io.BytesIO()
                sf.write(audio_buffer, audio_export, sr, format='OGG')
                zip_file.writestr(safe_audio_filename, audio_buffer.getvalue())

            st.success("✨ 고퀄리티 맵 & OGG 사운드 변환 완료!")

            # 8. Filebin 업로드
            bin_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
            upload_url = f"https://filebin.net/{bin_id}/AI_Pro_Map_Package.zip"
            
            headers = {"Content-Type": "application/zip"}
            upload_res = requests.post(upload_url, data=zip_buffer.getvalue(), headers=headers)
            
            if upload_res.status_code == 201:
                st.write("### 🔗 얼불춤(ADOFAI) 다이렉트 URL:")
                st.code(upload_url, language="text")
                st.write("이제 음악 무조건 나오고, 트랙도 훨씬 자연스럽게 꼬불꼬불 그려질 거야!")
            else:
                st.error(f"서버 업로드 실패. 상태 코드: {upload_res.status_code}")

        except Exception as e:
            st.error(f"에러가 발생했어: {e}")
        finally:
            os.remove(tmp_file_path)
