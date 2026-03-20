import streamlit as st
import librosa
import numpy as np
import json
import tempfile
import os
import zipfile
import io
import requests
from pathlib import Path
import google.generativeai as genai

# --- [초기 설정] Secrets에서 Gemini API 키 불러오기 ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"🚨 Gemini API Key 에러! 스트림릿 Secrets 설정을 확인해줘. ({e})")
    st.stop()

st.set_page_config(page_title="ADOFAI AI Generator Pro", page_icon="🧊")
st.title("🧊 얼불춤(ADOFAI) AI 창의적 맵 생성기")
st.write("원본 오디오 유지 + 제미나이 디자인 + 지나간 타일 삭제 적용 버전")

uploaded_file = st.file_uploader("음악 파일 업로드 (모든 확장자 가능)", type=None)

if uploaded_file is not None:
    st.info("🎵 오디오 심층 분석 및 제미나이 디자인 컨설팅 중... (10~15초 소요)")
    
    with st.spinner("제미나이가 맵을 디자인하고 채보를 그리는 중..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            uploaded_file.seek(0)
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            # 1. 램 소모 최소화를 위한 가벼운 분석
            y, sr = librosa.load(tmp_file_path, sr=22050, mono=True)
            
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm_value = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)

            # 2. 제미나이에게 디자인 의뢰
            prompt = f"""
            You are an expert ADOFAI level designer.
            Analyze music params: BPM {bpm_value:.1f}.
            The fixed trackColor is '#debb7b' (Gold).
            Generate complementary high-contrast settings strictly.
            Output ONLY valid JSON with these 2 keys:
            "backgroundColor": (HEX code, high contrast with #debb7b),
            "hitsound": (Choose ONE exactly: 'Kick', 'Snare', 'Clap')
            """
            
            try:
                response = model.generate_content(prompt)
                ai_settings_text = response.text.replace("```json", "").replace("```", "").strip()
                ai_settings = json.loads(ai_settings_text)
                
                color_bg = ai_settings.get("backgroundColor", "000000")
                chosen_hitsound = ai_settings.get("hitsound", "Kick")
                
                # [강력한 안전장치 추가] 제미나이가 이상한 소리를 골라도 무조건 기본음으로 필터링!
                valid_hitsounds = ["Kick", "Snare", "Clap"]
                if chosen_hitsound not in valid_hitsounds:
                    chosen_hitsound = "Kick"
                
                if color_bg.lower().replace('#','') == 'debb7b':
                    color_bg = "000000" 
            except:
                color_bg = "000000"
                chosen_hitsound = "Kick"

            # 3. 곡선 & 지그재그 혼합 스마트 채보 알고리즘
            angle_data = [0]
            current_angle = 0
            curve_dir = 45

            for i in range(1, len(onset_times)):
                time_diff = onset_times[i] - onset_times[i-1]
                
                if time_diff < (60 / bpm_value) * 0.4:
                    current_angle = (current_angle + 90) % 360
                elif time_diff > (60 / bpm_value) * 1.5:
                    current_angle = (current_angle + 90 * (curve_dir // 45)) % 360
                    curve_dir *= -1
                else:
                    current_angle = (current_angle + curve_dir) % 360
                    if i % 8 == 0:
                        curve_dir *= -1
                
                angle_data.append(int(current_angle))

            # 4. 원본 오디오 확장자 유지
            file_extension = Path(uploaded_file.name).suffix.lower()
            if file_extension not in ['.mp3', '.wav', '.ogg']:
                file_extension = '.mp3'
            safe_audio_filename = f"song{file_extension}"

            # 5. 얼불춤 유니티 호환 순정 세팅
            settings_block = {
                "version": 11,
                "artist": "AI Generated",
                "song": safe_audio_filename, 
                "author": "ADOFAI AI",
                "bpm": bpm_value,
                "hitsound": chosen_hitsound, 
                "trackColor": "debb7b", 
                "beatsBehind": 0, 
                "backgroundColor": color_bg, 
                "specialArtistType": "None", "artistPermission": "", 
                "separateCountdownTime": "Enabled", "previewImage": "", "previewIcon": "",
                "previewIconColor": "0082ba", "previewSongStart": 0, "previewSongDuration": 10,
                "seizureWarning": "Disabled", "levelDesc": "High Quality AI Map",
                "levelTags": "", "artistLinks": "", "difficulty": 1, "requiredMods": [],
                "volume": 100, "offset": 0, "pitch": 100, "hitsoundVolume": 100,
                "countdownTicks": 4, "trackColorType": "Single", "secondaryTrackColor": "ffffff",
                "trackColorAnimDuration": 2, "trackColorPulse": "None", "trackPulseLength": 10,
                "trackStyle": "Standard", "trackAnimation": "None", "beatsAhead": 5,
                "showDefaultBGIfNoImage": "Enabled", "bgImage": "", "bgImageColor": "ffffff",
                "parallax": [100, 100], "bgDisplayMode": "FitToScreen", "lockRot": "Disabled",
                "loopBG": "Disabled", "unscaledSize": 100, "relativeTo": "Player",
                "position": [0, 0], "rotation": 0, "zoom": 100, "bgVideo": "",
                "loopVideo": "Disabled", "vidOffset": 0, "floorIconOutlines": "Disabled",
                "stickToFloors": "Enabled", "planetEase": "Linear", "planetEaseParts": 1,
                "planetEasePartBehavior": "Mirror", "defaultTextColor": "ffffff",
                "defaultTextShadowColor": "00000050", "strictBehavior": "Ignore", "zoomText": "Disabled"
            }

            settings_json = json.dumps(settings_block, ensure_ascii=False)
            angle_json = json.dumps(angle_data)
            
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
                zip_file.writestr(safe_audio_filename, uploaded_file.read())

            st.success("✨ 제미나이의 디자인과 스마트 채보가 결합된 맵 완성!")

            # 7. Tmpfiles.org 다이렉트 링크 발급
            upload_res = requests.post(
                "https://tmpfiles.org/api/v1/upload", 
                files={"file": ("AI_Pro_Map.zip", zip_buffer.getvalue(), "application/zip")}
            )
            
            if upload_res.status_code == 200:
                original_url = upload_res.json()['data']['url']
                direct_download_url = original_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
                
                st.write("### 🔗 제미나이 탑재 얼불춤 다이렉트 URL:")
                st.code(direct_download_url, language="text")
                st.write("이제 Hitsound 에러는 절대 안 날 거야!")
            else:
                st.error(f"서버 업로드 실패. 상태 코드: {upload_res.status_code}")

        except Exception as e:
            st.error(f"에러가 발생했어: {e}")
        finally:
            os.remove(tmp_file_path)
