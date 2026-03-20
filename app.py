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
import google.generativeai as genai

# 1. Gemini API Setup
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-2.5-flash')

st.set_page_config(page_title="ADOFAI AI Generator", page_icon="🧊")
st.title("🧊 얼불춤(ADOFAI) URL 자동 생성기")
st.write("음원 분석 후 서버에 업로드하여 다이렉트 URL을 발급합니다.")

uploaded_file = st.file_uploader("음악 파일 업로드 (모든 확장자 가능)", type=None)

if uploaded_file is not None:
    st.info("🎵 오디오 분석 및 패키지 생성 중...")
    
    with st.spinner("처리 중... (조금만 기다려줘!)"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            # 2. 오디오 분석
            y, sr = librosa.load(tmp_file_path)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm_value = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
            
            angle_data = [0]
            for i in range(1, len(onset_times)):
                time_diff = onset_times[i] - onset_times[i-1]
                if time_diff < (60 / bpm_value) * 0.5:
                    angle_data.append(90)
                else:
                    angle_data.append(0)

            # 3. AI 설정 디자인 (색상만 안전하게 받아옴)
            prompt = f"""
            너는 A Dance of Fire and Ice 맵 디자이너야.
            BPM {bpm_value:.1f}의 곡 분위기에 어울리는 맵 트랙 색상과 배경 색상을 HEX 코드로 짜줘.
            반드시 아래 2개의 키만 포함된 순수 JSON 객체 하나만 출력해:
            "trackColor", "backgroundColor"
            """
            response = model.generate_content(prompt)
            ai_settings_text = response.text.replace("```json", "").replace("```", "").strip()
            
            try:
                ai_settings = json.loads(ai_settings_text)
                color_track = ai_settings.get("trackColor", "ff0000")
                color_bg = ai_settings.get("backgroundColor", "000000")
            except:
                color_track = "ff0000"
                color_bg = "000000"

            audio_filename = uploaded_file.name

            # 4. 얼불춤 유니티 엔진 완벽 호환 순정 세팅 (Enum 에러 방지용 강제 고정값)
            settings_block = {
                "version": 11,
                "artist": "AI Generated",
                "specialArtistType": "None",
                "artistPermission": "",
                "song": audio_filename,
                "author": "ADOFAI AI",
                "separateCountdownTime": "Enabled",
                "previewImage": "",
                "previewIcon": "",
                "previewIconColor": "0082ba",
                "previewSongStart": 0,
                "previewSongDuration": 10,
                "seizureWarning": "Disabled",
                "levelDesc": "AI Generated Map",
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
                "showDefaultBGIfNoImage": "True",
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
            
            # 5. 최종 .adofai 문자열 조립
            adofai_str = f"""{{
    "angleData": {angle_json},
    "settings": {settings_json},
    "actions": [],
    "decorations": []
}}"""

            # 6. ZIP 압축
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                zip_file.writestr("level.adofai", adofai_str)
                uploaded_file.seek(0)
                zip_file.writestr(audio_filename, uploaded_file.read())

            st.success("✨ 맵 완성! 다운로드 링크 생성 중...")

            # 7. Filebin 서버 업로드
            bin_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
            safe_filename = "AI_Map_Package.zip"
            upload_url = f"https://filebin.net/{bin_id}/{safe_filename}"
            
            headers = {"Content-Type": "application/zip"}
            upload_res = requests.post(upload_url, data=zip_buffer.getvalue(), headers=headers)
            
            if upload_res.status_code == 201:
                download_link = upload_url
                st.write("### 🔗 얼불춤(ADOFAI) 다이렉트 URL:")
                st.code(download_link, language="text")
                st.write("이제 Enum(Value cannot be null) 에러가 발생하지 않을 거야!")
            else:
                st.error(f"서버 업로드에 실패했어. 상태 코드: {upload_res.status_code}")

            st.download_button(
                label="📦 수동 다운로드(.zip)",
                data=zip_buffer.getvalue(),
                file_name=f"AI_Map.zip",
                mime="application/zip"
            )

        except Exception as e:
            st.error(f"에러가 발생했어: {e}")
        finally:
            os.remove(tmp_file_path)
