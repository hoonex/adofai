import streamlit as st
import librosa
import numpy as np
import json
import tempfile
import os
import zipfile
import io
from pathlib import Path

st.set_page_config(page_title="ADOFAI Perfect Sync V15", page_icon="🧊")
st.title("🧊 얼불춤(ADOFAI) 완벽 싱크 생성기")
st.write("오디오 원본 100% 보존! 노래 무조건 나옵니다.")

uploaded_file = st.file_uploader("음악 파일 업로드 (MP3, OGG, WAV)", type=None)

if uploaded_file is not None:
    st.info("🎵 박자 정밀 분석 중... (약 10초)")
    
    with st.spinner("싱크 계산 및 맵 패키징 중..."):
        # 1. 원본 파일 훼손 없이 바이트(Byte) 단위로 안전하게 보관
        uploaded_file.seek(0)
        raw_audio_bytes = uploaded_file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(raw_audio_bytes)
            tmp_file_path = tmp_file.name

        try:
            # 2. 오디오 로드 및 타격점 스캔
            y, sr = librosa.load(tmp_file_path, sr=22050, mono=True)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm_value = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)

            # 3. 완벽 싱크 맵핑 알고리즘
            angle_data = [0]
            if len(onset_times) > 0:
                offset_ms = int(onset_times[0] * 1000)
                theoretical_time = onset_times[0]
            else:
                offset_ms = 0
                theoretical_time = 0

            prev_abs_angle = 0

            for i in range(1, len(onset_times)):
                actual_time = onset_times[i]
                delta_t = actual_time - theoretical_time
                
                # 너무 짧은 노이즈 비트 무시
                if delta_t < 0.06:
                    continue

                ideal_travel_angle = delta_t * bpm_value * 3.0
                snapped_travel = round(ideal_travel_angle / 15.0) * 15
                
                # 박자 간격이 길 때 180도(직진) 타일로 보정
                while snapped_travel > 360:
                    angle_data.append(int(prev_abs_angle))
                    theoretical_time += 180.0 / (bpm_value * 3.0)
                    snapped_travel -= 180
                
                if snapped_travel < 15:
                    snapped_travel = 15

                # 얼불춤 절대 각도 계산
                next_abs_angle = (prev_abs_angle + snapped_travel - 180) % 360
                angle_data.append(int(next_abs_angle))

                theoretical_time += snapped_travel / (bpm_value * 3.0)
                prev_abs_angle = next_abs_angle

            # 4. 원본 파일의 확장자 그대로 사용
            ext = Path(uploaded_file.name).suffix.lower()
            if ext not in ['.mp3', '.wav', '.ogg']: ext = '.mp3'
            safe_audio_filename = f"song{ext}"

            # 5. V15 완벽 호환 세팅 (이전 작동 버전과 동일)
            settings_block = {
                "version": 15, "artist": "Sync Engine", "specialArtistType": "None", "artistPermission": "",
                "song": safe_audio_filename, "author": "ADOFAI Sync AI", "separateCountdownTime": True,
                "previewImage": "", "previewIcon": "", "previewIconColor": "003f52", "previewSongStart": 0,
                "previewSongDuration": 10, "seizureWarning": False, "levelDesc": "100% Perfect Sync Map",
                "levelTags": "", "artistLinks": "", "speedTrialAim": 0, "difficulty": 1, "requiredMods": [],
                "songFilename": safe_audio_filename, 
                "bpm": bpm_value, "volume": 100, "offset": offset_ms, "pitch": 100,
                "hitsound": "Kick", "hitsoundVolume": 100, "countdownTicks": 4, "songURL": "", "tileShape": "Long",
                "trackColorType": "Single", "trackColor": "debb7b", "secondaryTrackColor": "ffffff",
                "trackColorAnimDuration": 2, "trackColorPulse": "None", "trackPulseLength": 10,
                "trackStyle": "Standard", "trackTexture": "", "trackTextureScale": 1, "trackGlowIntensity": 100,
                "trackAnimation": "None", "beatsAhead": 5, "trackDisappearAnimation": "None",
                "beatsBehind": 0,
                "backgroundColor": "000000", "showDefaultBGIfNoImage": True, "showDefaultBGTile": True,
                "defaultBGTileColor": "101121", "defaultBGShapeType": "Default", "defaultBGShapeColor": "ffffff",
                "bgImage": "", "bgImageColor": "ffffff", "parallax": [100, 100], "bgDisplayMode": "FitToScreen",
                "imageSmoothing": True, "lockRot": False, "loopBG": False, "scalingRatio": 100,
                "relativeTo": "Player", "position": [0, 0], "rotation": 0, "zoom": 100, "pulseOnFloor": True,
                "bgVideo": "", "loopVideo": False, "vidOffset": 0, "floorIconOutlines": False,
                "stickToFloors": True, "planetEase": "Linear", "planetEaseParts": 1, "planetEasePartBehavior": "Mirror",
                "defaultTextColor": "ffffff", "defaultTextShadowColor": "00000050", "congratsText": "",
                "perfectText": "", "legacyFlash": False, "legacyCamRelativeTo": False, "legacySpriteTiles": False,
                "legacyTween": False, "disableV15Features": False
            }

            adofai_str = json.dumps({"angleData": angle_data, "settings": settings_block, "actions": [], "decorations": []}, ensure_ascii=False)

            # 6. ZIP 패키징 (오디오 훼손 없이 원본 바이트 그대로 투입!)
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("level.adofai", adofai_str)
                zip_file.writestr(safe_audio_filename, raw_audio_bytes) # ★ 원본 100% 삽입

            st.success("✨ 맵 완성! (오디오 손실 제로)")
            
            st.download_button(
                label="📦 .zip 맵 다운로드", 
                data=zip_buffer.getvalue(), 
                file_name="AI_Sync_Map.zip",
                mime="application/zip"
            )

        except Exception as e:
            st.error(f"오류 발생: {e}")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
