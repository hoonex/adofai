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
from pathlib import Path

st.set_page_config(page_title="ADOFAI Art Chart Engine", page_icon="🧊", layout="wide")
st.title("🧊 얼불춤(ADOFAI) 아트 채보 마스터 엔진")
st.write("별, 다이아몬드, 번개 등 기하학적 아트 채보(Art Chart) 알고리즘 무제한 적용")

uploaded_file = st.file_uploader("음악 파일 업로드 (MP3 권장)", type=None)

if uploaded_file is not None:
    st.info("🎵 오디오 정밀 스캔 및 기하학 아트 채보 설계 중... (코드 길이 제한 해제됨)")
    
    with st.spinner("아트 채보 렌더링 및 V15 시각 효과 연출 중..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            uploaded_file.seek(0)
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            # ==========================================
            # 1. 오디오 심층 분석 (Onset & Energy)
            # ==========================================
            y, sr = librosa.load(tmp_file_path, sr=22050, mono=True)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm_value = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)

            rms = librosa.feature.rms(y=y)[0]
            mean_energy = np.mean(rms)
            times_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

            # ==========================================
            # 2. 아트 채보(Art Chart) 매크로 사전
            # ==========================================
            # 각도는 이전 타일 기준 '상대 각도(Delta)'입니다.
            ART_MACROS = {
                "star": [144, 144, 144, 144, 144],          # 5각 별 그리기
                "diamond": [45, 90, 90, 90, 45],            # 다이아몬드 형태
                "hexagon_loop": [60, 60, 60, 60, 60, 60],   # 완벽한 육각형 궤도
                "lightning": [45, 315, 45, 315, 45, 315],   # 번개 지그재그 (V자 연속)
                "tresillo": [135, 135, 90],                 # 근본 엇박자 트레실로
                "stairway": [90, 270, 90, 270, 90, 270],    # 상승 계단
                "spiral": [45, 45, 45, 45, 45, 45]          # 소용돌이
            }

            # ==========================================
            # 3. 패턴 제너레이터 & 시각 효과(Actions) 엔진
            # ==========================================
            angle_data = [0]
            actions = []
            current_angle = 0
            
            pattern_queue = [] 
            is_highlight = False

            # V15 기본 세팅 (배경은 검정, 길은 debb7b 고정)
            actions.append({"floor": 1, "eventType": "SetSpeed", "speedType": "Bpm", "beatsPerMinute": bpm_value, "bpmMultiplier": 1, "angleOffset": 0})
            actions.append({"floor": 1, "eventType": "ColorTrack", "trackColorType": "Single", "trackColor": "debb7b", "secondaryTrackColor": "ffffff", "trackColorAnimDuration": 2, "trackColorPulse": "None", "trackPulseLength": 10, "trackStyle": "Standard", "trackTexture": "", "trackTextureScale": 1, "trackGlowIntensity": 100, "justThisTile": False})

            for i in range(1, len(onset_times)):
                current_time = onset_times[i]
                time_diff = onset_times[i] - onset_times[i-1]
                
                energy_idx = np.argmin(np.abs(times_rms - current_time))
                current_energy = rms[energy_idx]
                floor_number = i + 1 

                # 하이라이트(Drop) 진입 -> 화려한 아트 채보 발동
                if current_energy > mean_energy * 1.6 and not is_highlight:
                    is_highlight = True
                    # 아트 채보가 시작될 때 화면이 줌아웃되며(ScalePlanets) 전체 그림을 보여주도록 카메라 연출 추가
                    actions.append({"floor": floor_number, "eventType": "ShakeScreen", "duration": 1, "strength": 120, "intensity": 100, "ease": "Linear", "fadeOut": True, "angleOffset": 0, "eventTag": ""})
                    actions.append({"floor": floor_number, "eventType": "Flash", "duration": 1, "plane": "Background", "startColor": "ffffff", "startOpacity": 80, "endColor": "ffffff", "endOpacity": 0, "angleOffset": 0, "ease": "Linear", "eventTag": ""})
                    actions.append({"floor": floor_number, "eventType": "Bloom", "enabled": True, "threshold": 50, "intensity": 150, "color": "ffffff", "duration": 0, "ease": "Linear", "angleOffset": 0, "eventTag": ""})
                    actions.append({"floor": floor_number, "eventType": "MoveCamera", "duration": 2, "position": [None, None], "zoom": 150, "angleOffset": 0, "ease": "OutQuad", "eventTag": ""}) # 줌아웃으로 그림 보여주기
                    actions.append({"floor": floor_number, "eventType": "CustomBackground", "color": "220000", "bgImage": "", "imageColor": "ffffff", "parallax": [100, 100], "bgDisplayMode": "FitToScreen", "imageSmoothing": True, "lockRot": False, "loopBG": False, "scalingRatio": 100, "angleOffset": 0, "eventTag": ""})

                # 하이라이트 종료
                elif current_energy < mean_energy * 0.9 and is_highlight:
                    is_highlight = False
                    actions.append({"floor": floor_number, "eventType": "MoveCamera", "duration": 2, "position": [None, None], "zoom": 100, "angleOffset": 0, "ease": "InOutSine", "eventTag": ""}) # 카메라 줌 원상복구
                    actions.append({"floor": floor_number, "eventType": "CustomBackground", "color": "000000", "bgImage": "", "imageColor": "ffffff", "parallax": [100, 100], "bgDisplayMode": "FitToScreen", "imageSmoothing": True, "lockRot": False, "loopBG": False, "scalingRatio": 100, "angleOffset": 0, "eventTag": ""})

                # --- 🎨 아트 채보 매크로 주입 (Art Chart Injection) ---
                if len(pattern_queue) > 0:
                    delta = pattern_queue.pop(0)
                else:
                    if is_highlight:
                        # 하이라이트에서는 복잡한 그림(아트 채보)을 그림
                        art_shape = random.choice(["star", "diamond", "hexagon_loop", "lightning"])
                        pattern_queue = ART_MACROS[art_shape].copy()
                        
                        # 아트 채보가 시작되는 타일에 타일 반짝임 효과(RecolorTrack) 추가
                        actions.append({"floor": floor_number, "eventType": "RecolorTrack", "startTile": [0, "ThisTile"], "endTile": [len(pattern_queue), "ThisTile"], "trackColorType": "Single", "trackColor": "ffffff", "secondaryTrackColor": "ffffff", "trackColorAnimDuration": 0.5, "trackColorPulse": "None", "trackPulseLength": 10, "trackStyle": "Standard", "trackGlowIntensity": 200, "angleOffset": 0, "ease": "Linear", "eventTag": ""})
                    
                    elif current_energy > mean_energy * 1.0: 
                        # 중간 빌드업 구간
                        art_shape = random.choice(["tresillo", "stairway", "spiral"])
                        pattern_queue = ART_MACROS[art_shape].copy()
                        if art_shape == "spiral":
                            actions.append({"floor": floor_number, "eventType": "Twirl"})
                    
                    else: 
                        # 잔잔한 구간: 직진 또는 부드러운 커브
                        if time_diff > (60 / bpm_value) * 1.5:
                            pattern_queue = [90] # 긴 공백 후 직각 꺾기
                        else:
                            pattern_queue = [0] * random.randint(2, 5) # 2~5칸 직진

                    delta = pattern_queue.pop(0) if len(pattern_queue) > 0 else 0

                # 매우 짧은 간격(동타) 예외 처리
                if time_diff < (60 / bpm_value) * 0.15:
                    delta = 15

                # 절대 각도로 계산 (이전 각도 + 델타 각도)
                current_angle = (current_angle + delta) % 360
                angle_data.append(int(current_angle))

            # ==========================================
            # 4. 오디오 불순물 클리닝 (안드로이드 버그 원천 차단)
            # ==========================================
            ext = Path(uploaded_file.name).suffix.lower()
            if ext not in ['.mp3', '.wav', '.ogg']: ext = '.mp3'
            safe_audio_filename = f"song{ext}"

            if ext == '.mp3':
                try:
                    from mutagen.mp3 import MP3
                    audio_cleaner = MP3(tmp_file_path)
                    audio_cleaner.delete() 
                    audio_cleaner.save()
                except Exception as e:
                    print(f"Mutagen 스킵됨: {e}")

            # ==========================================
            # 5. V15 완벽 호환 Settings (요청사항 100% 반영)
            # ==========================================
            settings_block = {
                "version": 15, "artist": "Art Chart Engine", "specialArtistType": "None", "artistPermission": "",
                "song": safe_audio_filename, "author": "ADOFAI AI", "separateCountdownTime": True,
                "previewImage": "", "previewIcon": "", "previewIconColor": "003f52", "previewSongStart": 0,
                "previewSongDuration": 10, "seizureWarning": False, "levelDesc": "Art Chart & VFX Map",
                "levelTags": "", "artistLinks": "", "speedTrialAim": 0, "difficulty": 1, "requiredMods": [],
                "songFilename": safe_audio_filename, "bpm": bpm_value, "volume": 100, "offset": 0, "pitch": 100,
                "hitsound": "Kick", "hitsoundVolume": 100, "countdownTicks": 4, "songURL": "", "tileShape": "Long",
                "trackColorType": "Single", "trackColor": "debb7b", "secondaryTrackColor": "ffffff",
                "trackColorAnimDuration": 2, "trackColorPulse": "None", "trackPulseLength": 10,
                "trackStyle": "Standard", "trackTexture": "", "trackTextureScale": 1, "trackGlowIntensity": 100,
                "trackAnimation": "None", "beatsAhead": 5, "trackDisappearAnimation": "None",
                "beatsBehind": 0,  # 지나간 타일 칼같이 삭제
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

            adofai_dict = {
                "angleData": angle_data,
                "settings": settings_block,
                "actions": actions,
                "decorations": []
            }
            adofai_str = json.dumps(adofai_dict, ensure_ascii=False)

            # ==========================================
            # 6. ZIP 패키징 & 다이렉트 링크 (Uguu.se)
            # ==========================================
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("level.adofai", adofai_str)
                with open(tmp_file_path, "rb") as f:
                    zip_file.writestr(safe_audio_filename, f.read())

            st.success("✨ 아트 채보와 연출이 결합된 마스터 맵이 완성되었습니다!")
            
            uguu_url = "https://uguu.se/api.php?d=upload-tool"
            files = {"file": ("AI_Art_Map.zip", zip_buffer.getvalue(), "application/zip")}
            upload_res = requests.post(uguu_url, files=files)
            
            if upload_res.status_code == 200:
                direct_link = upload_res.text.strip()
                st.write("### 🔗 ADOFAI V15 아트 채보 URL:")
                st.code(direct_link, language="text")
                st.info("이 링크 하나면 별, 다이아몬드 패턴과 함께 화면이 흔들리는 V15 액션을 볼 수 있습니다.")
            else:
                st.error("업로드 서버 통신 실패")

            st.download_button("📦 수동 다운로드", data=zip_buffer.getvalue(), file_name="AI_Art_Map.zip")

        except Exception as e:
            st.error(f"오류 발생: {e}")
        finally:
            os.remove(tmp_file_path)
