import streamlit as st
import librosa
import numpy as np
import json
import tempfile
import os
import zipfile
import io
from pathlib import Path

st.set_page_config(page_title="ADOFAI True Deterministic Engine", page_icon="🧊", layout="wide")
st.title("🧊 얼불춤(ADOFAI) 진(眞) 마스터피스 엔진")
st.write("주사위(Random) 0%! 오직 주파수(음색)와 에너지 데이터로만 맵을 스케치합니다.")

class TrueDeterministicMapGenerator:
    def __init__(self, audio_path, raw_bytes, filename):
        self.audio_path = audio_path
        self.raw_bytes = raw_bytes
        self.filename = filename
        
        self.bpm = 0
        self.offset_ms = 0
        self.angle_data = [0]
        self.actions = []
        
        # 상태 추적
        self.current_abs_angle = 0
        self.theoretical_time = 0.0
        self.current_floor = 1
        
    def analyze_audio(self):
        # 1. 오디오 로드 및 타격음 분리 (HPSS)
        y, sr = librosa.load(self.audio_path, sr=22050, mono=True)
        y_harm, y_perc = librosa.effects.hpss(y, margin=2.0)
        
        # 2. 박자 추출
        tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)
        self.bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        
        onset_frames = librosa.onset.onset_detect(y=y_perc, sr=sr, backtrack=True)
        self.onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # 3. [진짜 AI의 핵심 1] 음의 밝기/높낮이(Spectral Centroid) 분석
        # 이 데이터가 타일이 왼쪽으로 꺾일지 오른쪽으로 꺾일지 결정함
        centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        self.times_centroids = librosa.frames_to_time(np.arange(len(centroids)), sr=sr)
        self.centroid_profile = centroids
        self.median_centroid = np.median(centroids) # 곡 전체의 평균 음색
        
        # 4. [진짜 AI의 핵심 2] 볼륨 에너지(RMS) 분석 -> 시각 효과 결정
        rms = librosa.feature.rms(y=y)[0]
        self.times_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        self.energy_profile = rms
        self.mean_energy = np.mean(rms)
        
        # 노이즈 필터링 (너무 짧은 따닥 소리 방지)
        self.clean_onsets = self._filter_onsets(self.onset_times, self.bpm)
        if len(self.clean_onsets) > 0:
            self.offset_ms = int(self.clean_onsets[0] * 1000)
            self.theoretical_time = self.clean_onsets[0]

    def _filter_onsets(self, onsets, bpm):
        min_time = (60.0 / bpm) / 4.0 
        filtered = [onsets[0]] if len(onsets) > 0 else []
        for t in onsets[1:]:
            if t - filtered[-1] >= min_time:
                filtered.append(t)
        return filtered

    def _get_audio_features(self, current_time):
        """현재 타격점의 에너지(볼륨)와 센트로이드(음색) 수치를 정확히 가져옴"""
        idx_rms = np.argmin(np.abs(self.times_rms - current_time))
        energy = self.energy_profile[idx_rms]
        
        idx_cent = np.argmin(np.abs(self.times_centroids - current_time))
        centroid = self.centroid_profile[idx_cent]
        
        return energy, centroid

    def generate_map_logic(self):
        # 기본 세팅 (요청하신 debb7b 고정)
        self.actions.append({"floor": 1, "eventType": "SetSpeed", "speedType": "Bpm", "beatsPerMinute": self.bpm, "bpmMultiplier": 1, "angleOffset": 0})
        self.actions.append({"floor": 1, "eventType": "ColorTrack", "trackColorType": "Single", "trackColor": "debb7b", "secondaryTrackColor": "ffffff", "trackColorAnimDuration": 2, "trackColorPulse": "None", "trackPulseLength": 10, "trackStyle": "Standard", "trackTexture": "", "trackTextureScale": 1, "trackGlowIntensity": 100, "justThisTile": False})
        
        is_highlight = False

        for i in range(1, len(self.clean_onsets)):
            actual_time = self.clean_onsets[i]
            delta_t = actual_time - self.theoretical_time
            
            if delta_t < 0.05: continue
            
            energy, centroid = self._get_audio_features(actual_time)
            self.current_floor += 1
            
            # --- 🎬 100% 데이터 기반 시각 효과 (하이라이트 자동 감지) ---
            if energy > self.mean_energy * 1.5 and not is_highlight:
                is_highlight = True
                self.actions.append({"floor": self.current_floor, "eventType": "ShakeScreen", "duration": 1, "strength": 120, "intensity": 100, "ease": "Linear", "fadeOut": True, "angleOffset": 0, "eventTag": ""})
                self.actions.append({"floor": self.current_floor, "eventType": "Flash", "duration": 1, "plane": "Background", "startColor": "ffffff", "startOpacity": 70, "endColor": "ffffff", "endOpacity": 0, "angleOffset": 0, "ease": "Linear", "eventTag": ""})
                self.actions.append({"floor": self.current_floor, "eventType": "CustomBackground", "color": "220505", "bgImage": "", "imageColor": "ffffff", "parallax": [100, 100], "bgDisplayMode": "FitToScreen", "imageSmoothing": True, "lockRot": False, "loopBG": False, "scalingRatio": 100, "angleOffset": 0, "eventTag": ""})
            elif energy < self.mean_energy * 0.9 and is_highlight:
                is_highlight = False
                self.actions.append({"floor": self.current_floor, "eventType": "CustomBackground", "color": "000000", "bgImage": "", "imageColor": "ffffff", "parallax": [100, 100], "bgDisplayMode": "FitToScreen", "imageSmoothing": True, "lockRot": False, "loopBG": False, "scalingRatio": 100, "angleOffset": 0, "eventTag": ""})

            # --- 📐 완벽 싱크를 위한 타일 이동 각도 (Magnitude) 계산 ---
            ideal_travel_angle = delta_t * (self.bpm / 60.0) * 180.0
            snapped_travel = round(ideal_travel_angle / 15.0) * 15
            
            # 박자가 길게 비면 직진 타일로 채움 (싱크 누수 완벽 방어)
            while snapped_travel > 360:
                self.angle_data.append(int(self.current_abs_angle)) 
                self.theoretical_time += 180.0 / (self.bpm / 60.0 * 180.0)
                snapped_travel -= 180
                self.current_floor += 1
                
            if snapped_travel < 15: snapped_travel = 15

            # --- 🧠 주파수(음색) 기반 방향성(Direction) 결정 (No Random) ---
            # snapped_travel 값 자체는 박자이기 때문에 바꿀 수 없음 (바꾸면 싱크가 나감)
            # 단, 타일을 오른쪽으로 꺾을지(기본 각도), 왼쪽으로 꺾을지(대칭 각도)를 결정함
            
            if snapped_travel == 180 or snapped_travel == 360:
                final_travel = snapped_travel # 직진이나 제자리는 방향 전환 의미 없음
            else:
                # 소리가 전체 평균보다 '높고 날카로우면' 기본 궤도 유지
                # 소리가 '낮고 묵직하면' 궤도를 반대로 꺾음 (360 - 각도)
                if centroid > self.median_centroid:
                    final_travel = snapped_travel
                else:
                    final_travel = 360 - snapped_travel
            
            # 최종 절대 각도 산출
            next_abs_angle = (self.current_abs_angle + final_travel - 180) % 360
            self.angle_data.append(int(next_abs_angle))
            
            # 수학적 시간 누적
            self.theoretical_time += snapped_travel / (self.bpm / 60.0 * 180.0)
            self.current_abs_angle = next_abs_angle

    def build_json(self):
        settings_block = {
            "version": 15, "artist": "Data-Driven Engine", "specialArtistType": "None", "artistPermission": "",
            "song": self.filename, "author": "ADOFAI True AI", "separateCountdownTime": True,
            "previewImage": "", "previewIcon": "", "previewIconColor": "003f52", "previewSongStart": 0,
            "previewSongDuration": 10, "seizureWarning": False, "levelDesc": "No Random, 100% Deterministic",
            "levelTags": "", "artistLinks": "", "speedTrialAim": 0, "difficulty": 1, "requiredMods": [],
            "songFilename": self.filename, 
            "bpm": self.bpm, "volume": 100, "offset": self.offset_ms, "pitch": 100,
            "hitsound": "Kick", "hitsoundVolume": 100, "countdownTicks": 4, "songURL": "", "tileShape": "Long",
            "trackColorType": "Single", "trackColor": "debb7b", "secondaryTrackColor": "ffffff",
            "trackColorAnimDuration": 2, "trackColorPulse": "None", "trackPulseLength": 10,
            "trackStyle": "Standard", "trackTexture": "", "trackTextureScale": 1, "trackGlowIntensity": 100,
            "trackAnimation": "None", "beatsAhead": 5, "trackDisappearAnimation": "None",
            "beatsBehind": 0, # 요청하신 밟으면 즉시 사라짐 고정
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

        return json.dumps({
            "angleData": self.angle_data,
            "settings": settings_block,
            "actions": self.actions,
            "decorations": []
        }, ensure_ascii=False)

# ==========================================
# Streamlit UI 및 실행부
# ==========================================
uploaded_file = st.file_uploader("음악 파일 업로드 (MP3, WAV, OGG)", type=None)

if uploaded_file is not None:
    st.info("🎵 오디오의 주파수 파형과 에너지를 정밀 스캔 중... (No Random)")
    
    with st.spinner("오직 음원 데이터로만 맵을 스케치하고 있습니다..."):
        uploaded_file.seek(0)
        raw_audio_bytes = uploaded_file.read()
        
        ext = Path(uploaded_file.name).suffix.lower()
        if ext not in ['.mp3', '.wav', '.ogg']: ext = '.mp3'
        safe_audio_filename = f"song{ext}"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(raw_audio_bytes)
            tmp_file_path = tmp_file.name

        try:
            # 진(眞) 마스터피스 엔진 가동 (랜덤 0%)
            engine = TrueDeterministicMapGenerator(tmp_file_path, raw_audio_bytes, safe_audio_filename)
            engine.analyze_audio()
            engine.generate_map_logic()
            adofai_str = engine.build_json()

            # ZIP 패키징 (원본 파일 100% 보존)
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("level.adofai", adofai_str)
                zip_file.writestr(safe_audio_filename, raw_audio_bytes)

            st.success("✨ 주파수와 볼륨 데이터 100% 반영! 진정한 AI 맵 생성 완료.")
            
            st.download_button(
                label="📦 .zip 데이터 기반 맵 다운로드", 
                data=zip_buffer.getvalue(), 
                file_name="AI_True_Data_Map.zip",
                mime="application/zip"
            )

        except Exception as e:
            st.error(f"오류 발생: {e}")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
