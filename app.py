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
import subprocess
import google.generativeai as genai

# 1. Gemini API Setup
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"API Key Error: {e}")
    st.stop()

st.set_page_config(page_title="ADOFAI AI Generator Pro", page_icon="🧊")
st.title("🧊 얼불춤(ADOFAI) AI 맵 자동 생성기 PRO")
st.write("메모리 최적화 & FFmpeg OGG 다이렉트 변환 탑재 버전")

uploaded_file = st.file_uploader("음악 파일 업로드 (MP3, WAV 등)", type=None)

if uploaded_file is not None:
    st.info("🎵 오디오 최적화 및 스마트 채보 분석 중... (서버 뻗음 방지 적용됨)")
    
    with st.spinner("Processing..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        ogg_path = tmp_file_path + ".ogg"

        try:
            # 2. FFmpeg Direct Conversion (Bypasses Python RAM limit)
            try:
                subprocess.run(
                    ["ffmpeg", "-i", tmp_file_path, "-map", "0:a", "-c:a", "libvorbis", "-b:a", "128k", ogg_path, "-y"],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            except FileNotFoundError:
                st.error("🚨 FFmpeg is missing! Please create a `packages.txt` file in your GitHub with the word `ffmpeg` in it.")
                st.stop()
            except subprocess.CalledProcessError as e:
                st.error(f"Audio conversion failed: {e.stderr.decode()}")
                st.stop()

            # 3. Lightweight Audio Load (Low Sample Rate, Mono) to save memory
            y_mono, sr = librosa.load(ogg_path, sr=22050, mono=True)

            # 4. Beat and Energy Analysis
            onset_frames = librosa.onset.onset_detect(y=y_mono, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr)
            bpm_value = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
            
            rms = librosa.feature.rms(y=y_mono)
            mean_energy = np.mean(rms)
            times_rms = librosa.times_like(rms)

            # 5. Smart Tile Engine (Delta Curve Algorithm)
            current_angle = 0
            angle_data = [current_angle]
            pattern_dir = 1 
            
            for i in range(1, len(onset_times)):
                time_diff = onset_times[i] - onset_times[i-1]
                current_time = onset_times[i]
                
                energy_idx = np.argmin(np.abs(times_rms - current_time))
                current_energy = rms[0, energy_idx]
                
                if random.random() < 0.15:
                    pattern_dir *= -1
                    
                if current_energy > mean_energy * 1.3:
                    if time_diff < (60 / bpm_value) * 0.6: 
                        delta = 90 * pattern_dir  # Zig-zag for fast beats
                    else:
                        delta = 45 * pattern_dir  # Swirl curves for drops
                else:
                    if i % 4 == 0:
                        delta = 45 * pattern_dir
                    else:
                        delta = 0  # Straight line
                        
                current_angle = (current_angle + delta) % 360
                angle_data.append(int(current_angle))

            # 6. AI Color Design
            prompt = f"""
            You are an ADOFAI map designer.
            Design trackColor and backgroundColor for a {bpm_value:.1f} BPM song.
            Output ONLY a pure JSON object with these 2 keys. Ensure high contrast.
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

            safe_audio_filename = "song.ogg"

            # 7. Engine Settings
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

            # 8. Create Final ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                zip_file.writestr("level.adofai", adofai_str)
                with open(ogg_path, "rb") as f:
                    zip_file.writestr(safe_audio_filename, f.read())

            st.success("✨ Processing Complete! Uploading...")

            # 9. Upload to Filebin
            bin_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
            upload_url = f"https://filebin.net/{bin_id}/AI_Pro_Map.zip"
            
            headers = {"Content-Type": "application/zip"}
            upload_res = requests.post(upload_url, data=zip_buffer.getvalue(), headers=headers)
            
            if upload_res.status_code == 201:
                st.write("### 🔗 ADOFAI Direct URL:")
                st.code(upload_url, language="text")
                st.write("메모리 에러 해결 완료! 이제 맵이랑 음악이 제대로 출력될 거야.")
            else:
                st.error(f"Upload failed: {upload_res.status_code}")

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            if os.path.exists(ogg_path):
                os.remove(ogg_path)
