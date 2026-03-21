import streamlit as st
import librosa
import numpy as np
import json
import tempfile
import os
import zipfile
import io
from pathlib import Path

st.set_page_config(page_title="ADOFAI Perfect Sync Generator", page_icon="🧊")
st.title("🧊 얼불춤(ADOFAI) 완벽 싱크(Sync) 생성기")
st.write("오직 노래의 박자에만 100% 맞춰서 타일을 배열하는 수학적 맵핑 엔진입니다.")

uploaded_file = st.file_uploader("음악 파일 업로드 (MP3, WAV, OGG)", type=None)

if uploaded_file is not None:
    st.info("🎵 오디오 정밀 타격점(Onset) 분석 중... (약 10초)")
    
    with st.spinner("박자에 맞는 정확한 각도를 수학적으로 계산하는 중..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            uploaded_file.seek(0)
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            # 1. 오디오 로드 및 타격점(Onset) 정확히 스캔
            y, sr = librosa.load(tmp_file_path, sr=22050, mono=True)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm_value = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)

            # 2. 완벽 싱크(Perfect Sync) 맵핑 알고리즘
            # 타격점 간의 시간을 계산해, 정확히 그 시간에 도달하는 '이동 각도(Travel Angle)'를 산출
            angle_data = [0]
            
            if len(onset_times) > 0:
                offset_ms = int(onset_times[0] * 1000) # 첫 비트의 시작점을 Offset으로 지정 (밀리초)
                theoretical_time = onset_times[0]
            else:
                offset_ms = 0
                theoretical_time = 0

            prev_abs_angle = 0

            for i in range(1, len(onset_times)):
                actual_time = onset_times[i]
                delta_t = actual_time - theoretical_time
                
                # 너무 짧은 노이즈 비트 무시 (초당 16비트 이상의 말도 안 되는 간격 방지)
                if delta_t < 0.06:
                    continue

                # 해당 시간(delta_t)을 소모하기 위해 필요한 얼불춤 타일의 이동 각도 계산
                ideal_travel_angle = delta_t * bpm_value * 3.0
                
                # 얼불춤에서 사용 가능한 15도 단위로 반올림 (스냅)
                snapped_travel = round(ideal_travel_angle / 15.0) * 15
                
                # 예외 처리: 각도가 너무 작거나 클 경우 제한
                if snapped_travel < 15:
                    snapped_travel = 15
                elif snapped_travel > 900:
                    snapped_travel = 900
                elif snapped_travel % 360 == 0:
                    snapped_travel = 360 # 0도가 되면 타일이 겹치므로 360도(한 바퀴)로 처리

                # 얼불춤 절대 각도 산출 공식 (180 + 이전 절대 각도 - 이동 각도)
                next_abs_angle = (180 + prev_abs_angle - snapped_travel) % 360
                angle_data.append(int(next_abs_angle))

                # 다음 타일 계산을 위해 현재 타일을 쳤을 때의 '이론적 시간' 누적
                # (오차가 누적되지 않도록 소수점까지 계속 더해줌)
                theoretical_time += snapped_travel / (bpm_value * 3.0)
                prev_abs_angle = next_abs_angle

            # 3. 오디오 불순물 클리닝 (모바일 묵음 버그 방지)
            ext = Path(uploaded_file.name).suffix.lower()
            if ext not in ['.mp3', '.wav', '.ogg']: ext = '.mp3'
            safe_audio_filename = f"song{ext}"

            if ext == '.mp3':
                try:
                    from mutagen.mp3 import MP3
                    audio_cleaner = MP3(tmp_file_path)
                    audio_cleaner.delete() 
                    audio_cleaner.save()
                except Exception:
                    pass

            # 4. 순정 세팅 적용 (길 색상 debb7b, 밟으면 즉시 사라짐 유지)
            settings_block = {
                "version": 11, "artist": "Sync Engine", "song": safe_audio_filename, 
                "author": "ADOFAI Sync AI", "bpm": bpm_value, "hitsound": "Kick", 
                "trackColor": "debb7b", "beatsBehind": 0, "backgroundColor": "000000", 
                "offset": offset_ms, # 정확한 오프셋 적용!
                "separateCountdownTime": "Enabled", "difficulty": 1, "volume": 100, 
                "pitch": 100, "hitsoundVolume": 100, "countdownTicks": 4, 
                "trackColorType": "Single", "trackStyle": "Standard", "beatsAhead": 5,
                "showDefaultBGIfNoImage": "Enabled", "bgDisplayMode": "FitToScreen",
                "lockRot": "Disabled", "loopBG": "Disabled", "unscaledSize": 100,
                "relativeTo": "Player", "position": [0, 0], "rotation": 0, "zoom": 100,
                "stickToFloors": "Enabled", "planetEase": "Linear", "planetEaseParts": 1
            }

            adofai_str = json.dumps({"angleData": angle_data, "settings": settings_block, "actions": [], "decorations": []}, ensure_ascii=False)

            # 5. ZIP 다운로드 패키징
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("level.adofai", adofai_str)
                with open(tmp_file_path, "rb") as f:
                    zip_file.writestr(safe_audio_filename, f.read())

            st.success("✨ 완벽한 박자 동기화 맵이 생성되었습니다! 아래 버튼을 눌러 다운로드하세요.")
            
            st.download_button(
                label="📦 .zip 맵 패키지 다운로드", 
                data=zip_buffer.getvalue(), 
                file_name="AI_Perfect_Sync_Map.zip",
                mime="application/zip"
            )

        except Exception as e:
            st.error(f"오류 발생: {e}")
        finally:
            os.remove(tmp_file_path)
