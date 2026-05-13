import streamlit as st
import librosa
import numpy as np
import json
import tempfile
import os
import zipfile
import io
import math
from pathlib import Path

st.set_page_config(page_title="ADOFAI True Deterministic Engine v2", page_icon="🧊", layout="wide")
st.title("🧊 얼불춤(ADOFAI) 진(眞) 마스터피스 엔진 v2.0")
st.markdown("""
**주사위(Random) 0%!** 단순한 1차원 데이터 매핑을 넘어, **공간 지각(Collision Tracking)**과 **기하학적 패턴(Pattern Dictionary)**을 결합하여 진짜 인간 고수처럼 맵을 스케치합니다.
""")

# ==========================================
# [모듈 1] 2D 공간 지각 시스템 (Collision Tracker)
# ==========================================
class GridTracker:
    def __init__(self):
        # 15도 단위의 각도를 x, y 벡터로 변환 (대략적인 충돌 계산용)
        self.angle_to_vector = {
            0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1),
            45: (1, 1), 135: (-1, 1), 225: (-1, -1), 315: (1, -1)
        }
        self.visited = set()
        self.visited.add((0, 0))
        self.current_pos = (0, 0)
        self.current_abs_angle = 0
        
    def check_collision(self, next_angle_diff):
        """다음에 이동할 타일이 기존 타일과 겹치는지 확인"""
        test_abs_angle = (self.current_abs_angle + next_angle_diff - 180) % 360
        # 45도 단위에 근사화시켜서 좌표 이동 (간이 계산)
        snapped_angle = min(self.angle_to_vector.keys(), key=lambda k: abs(k - test_abs_angle))
        dx, dy = self.angle_to_vector[snapped_angle]
        
        next_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)
        return next_pos in self.visited
        
    def move(self, next_angle_diff):
        """실제 이동 적용"""
        self.current_abs_angle = (self.current_abs_angle + next_angle_diff - 180) % 360
        snapped_angle = min(self.angle_to_vector.keys(), key=lambda k: abs(k - self.current_abs_angle))
        dx, dy = self.angle_to_vector[snapped_angle]
        
        self.current_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)
        self.visited.add(self.current_pos)
        return self.current_abs_angle

# ==========================================
# [모듈 2] 인간형 패턴 딕셔너리 (Phrase Macro)
# ==========================================
class PatternDictionary:
    """사람이 자주 사용하는 예쁜 기하학적 맵핑 패턴들"""
    PATTERNS = {
        "straight": [180], # 직진
        "stairs_right": [270, 270], # 계단 (우회전 위주)
        "stairs_left": [90, 90],    # 계단 (좌회전 위주)
        "zigzag": [135, 225],       # 지그재그
        "square_loop": [270, 270, 270, 270], # 사각형 (회전용)
        "magic_circle": [225, 225, 225, 225, 225, 225, 225, 225] # 8각 마법진
    }
    
    @staticmethod
    def get_pattern(intensity, centroid, bpm):
        """곡의 분위기(에너지, 음색)에 맞춰 패턴을 선택"""
        if intensity == "low":
            return PatternDictionary.PATTERNS["straight"]
        elif intensity == "medium":
            if centroid > 2000: # 음색이 높을 때
                return PatternDictionary.PATTERNS["stairs_right"]
            else:
                return PatternDictionary.PATTERNS["stairs_left"]
        elif intensity == "high":
            if bpm > 150: # 빠르고 신날 때
                return PatternDictionary.PATTERNS["zigzag"]
            else:
                return PatternDictionary.PATTERNS["square_loop"]
        else: # 폭발적인 드랍 구간
            return PatternDictionary.PATTERNS["magic_circle"]

# ==========================================
# [모듈 3] 진(眞) 마스터피스 엔진
# ==========================================
class TrueDeterministicMapGenerator:
    def __init__(self, audio_path, raw_bytes, filename):
        self.audio_path = audio_path
        self.raw_bytes = raw_bytes
        self.filename = filename
        
        self.bpm = 0
        self.offset_ms = 0
        self.angle_data = [0]
        self.actions = []
        
        self.grid = GridTracker()
        self.theoretical_time = 0.0
        self.current_floor = 1
        
    def analyze_audio(self):
        st.toast("1/4: HPSS 오디오 분리 및 비트 트래킹 시작...")
        y, sr = librosa.load(self.audio_path, sr=22050, mono=True)
        y_harm, y_perc = librosa.effects.hpss(y, margin=2.0)
        
        tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)
        self.bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        
        st.toast("2/4: Onset(타격점) 정밀 분석 중...")
        onset_frames = librosa.onset.onset_detect(y=y_perc, sr=sr, backtrack=True)
        self.onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        st.toast("3/4: 스펙트럼 센트로이드(음색) 및 RMS(에너지) 계산 중...")
        self.centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        self.rms = librosa.feature.rms(y=y)[0]
        self.times = librosa.frames_to_time(np.arange(len(self.rms)), sr=sr)
        
        self.mean_energy = np.mean(self.rms)
        self.mean_centroid = np.mean(self.centroids)
        
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

    def _get_audio_context(self, current_time):
        """현재 시간의 에너지 수준과 음색을 평가"""
        idx = np.argmin(np.abs(self.times - current_time))
        energy = self.rms[idx]
        centroid = self.centroids[idx]
        
        if energy > self.mean_energy * 2.0:
            intensity = "drop"
        elif energy > self.mean_energy * 1.2:
            intensity = "high"
        elif energy > self.mean_energy * 0.8:
            intensity = "medium"
        else:
            intensity = "low"
            
        return intensity, energy, centroid

    def generate_map_logic(self):
        st.toast("4/4: 기하학적 맵 스케치 및 시각 효과(VFX) 연출 시작...")
        
        # [초기 세팅]
        self.actions.append({"floor": 1, "eventType": "SetSpeed", "speedType": "Bpm", "beatsPerMinute": self.bpm, "bpmMultiplier": 1, "angleOffset": 0})
        self.actions.append({"floor": 1, "eventType": "ColorTrack", "trackColorType": "Glow", "trackColor": "00D4FF", "secondaryTrackColor": "005dff", "trackColorAnimDuration": 2, "trackColorPulse": "Forward", "trackPulseLength": 10, "trackStyle": "Neon"})
        self.actions.append({"floor": 1, "eventType": "CustomBackground", "color": "000000", "bgDisplayMode": "FitToScreen"})

        is_highlight = False
        pattern_queue = [] # 실행할 패턴을 담아두는 큐

        for i in range(1, len(self.clean_onsets)):
            actual_time = self.clean_onsets[i]
            delta_t = actual_time - self.theoretical_time
            
            if delta_t < 0.03: continue
            
            intensity, energy, centroid = self._get_audio_context(actual_time)
            self.current_floor += 1
            
            # --- 🎬 [VFX Director] 연출가 로직 ---
            if intensity in ["high", "drop"] and not is_highlight:
                is_highlight = True
                # 하이라이트 진입 시 카메라 무빙 & 화면 흔들림 & 플래시
                self.actions.append({"floor": self.current_floor, "eventType": "MoveCamera", "duration": 2, "relativeTo": "Player", "position": [0, 0], "rotation": 15, "zoom": 120, "angleOffset": 0, "ease": "OutCubic"})
                self.actions.append({"floor": self.current_floor, "eventType": "ShakeScreen", "duration": 2, "strength": 50, "intensity": 50, "fadeOut": True})
                self.actions.append({"floor": self.current_floor, "eventType": "Flash", "duration": 1, "plane": "Background", "startColor": "ffffff", "startOpacity": 50, "endColor": "000000", "endOpacity": 0})
                self.actions.append({"floor": self.current_floor, "eventType": "SetFilter", "filter": "Aberration", "enabled": True, "intensity": 45})
            
            elif intensity in ["low", "medium"] and is_highlight:
                is_highlight = False
                # 하이라이트 종료 시 카메라 원상복구
                self.actions.append({"floor": self.current_floor, "eventType": "MoveCamera", "duration": 2, "relativeTo": "Player", "position": [0, 0], "rotation": 0, "zoom": 100, "angleOffset": 0, "ease": "InCubic"})
                self.actions.append({"floor": self.current_floor, "eventType": "SetFilter", "filter": "Aberration", "enabled": False, "intensity": 0})

            # --- 📐 [Grid Tracking] 타일 배치 로직 ---
            ideal_travel_angle = delta_t * (self.bpm / 60.0) * 180.0
            snapped_travel = round(ideal_travel_angle / 15.0) * 15
            if snapped_travel < 15: snapped_travel = 15

            # 롱 노트 처리 (긴 공백)
            while snapped_travel > 360:
                self.grid.move(180) # 직진
                self.angle_data.append(self.grid.current_abs_angle)
                self.theoretical_time += 180.0 / (self.bpm / 60.0 * 180.0)
                snapped_travel -= 180
                self.current_floor += 1

            # 패턴 큐가 비어있으면 분위기에 맞는 패턴 새로 가져오기
            if not pattern_queue:
                pattern_queue = PatternDictionary.get_pattern(intensity, centroid, self.bpm).copy()
            
            # 패턴에서 다음 꺾일 각도(diff) 꺼내기
            next_diff = pattern_queue.pop(0)
            
            # 충돌 감지 로직 (Collision Detection)
            if self.grid.check_collision(next_diff):
                # 충돌이 예상되면 Twirl(소용돌이) 이벤트를 넣고 각도를 반대로 비틂
                self.actions.append({"floor": self.current_floor, "eventType": "Twirl"})
                next_diff = 360 - next_diff # 대칭 이동으로 회피

            # 최종 격자 이동 및 각도 기록
            new_abs_angle = self.grid.move(next_diff)
            self.angle_data.append(int(new_abs_angle))
            
            self.theoretical_time += snapped_travel / (self.bpm / 60.0 * 180.0)

    def build_json(self):
        settings_block = {
            "version": 15, "artist": "Data-Driven Engine v2", "specialArtistType": "None",
            "song": self.filename, "author": "ADOFAI True AI", "separateCountdownTime": True,
            "seizureWarning": False, "levelDesc": "Spatial Awareness & VFX Directed",
            "difficulty": 5, "songFilename": self.filename, 
            "bpm": self.bpm, "volume": 100, "offset": self.offset_ms, "pitch": 100,
            "hitsound": "Hat", "hitsoundVolume": 100, "countdownTicks": 4,
            "trackColorType": "Glow", "trackColor": "00D4FF", "secondaryTrackColor": "005dff",
            "trackColorAnimDuration": 2, "trackColorPulse": "Forward", "trackPulseLength": 10,
            "trackStyle": "Neon", "trackAnimation": "None", "beatsAhead": 4, "trackDisappearAnimation": "Fade",
            "beatsBehind": 2, "backgroundColor": "000000", "showDefaultBGIfNoImage": True,
            "legacyFlash": False
        }

        return json.dumps({
            "angleData": self.angle_data,
            "settings": settings_block,
            "actions": self.actions,
            "decorations": []
        }, ensure_ascii=False)

# ==========================================
# Streamlit UI
# ==========================================
uploaded_file = st.file_uploader("음악 파일 업로드 (MP3, WAV, OGG)", type=None)

if uploaded_file is not None:
    with st.spinner("AI가 음악의 기하학적 형태를 스케치하는 중..."):
        uploaded_file.seek(0)
        raw_audio_bytes = uploaded_file.read()
        
        ext = Path(uploaded_file.name).suffix.lower()
        if ext not in ['.mp3', '.wav', '.ogg']: ext = '.mp3'
        safe_audio_filename = f"song{ext}"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(raw_audio_bytes)
            tmp_file_path = tmp_file.name

        try:
            # 진(眞) 엔진 v2 가동
            engine = TrueDeterministicMapGenerator(tmp_file_path, raw_audio_bytes, safe_audio_filename)
            engine.analyze_audio()
            engine.generate_map_logic()
            adofai_str = engine.build_json()

            # ZIP 패키징
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("level.adofai", adofai_str)
                zip_file.writestr(safe_audio_filename, raw_audio_bytes)

            st.success("✨ 공간 충돌 방지 및 카메라 연출이 적용된 맵 생성 완료!")
            
            # 맵 데이터 미리보기 제공
            with st.expander("생성된 Angle Data 미리보기 (첫 100개 타일)"):
                st.write(engine.angle_data[:100])
                st.caption(f"총 타일 개수: {len(engine.angle_data)}개")
                st.caption(f"총 생성된 카메라/VFX 이벤트 수: {len(engine.actions)}개")

            st.download_button(
                label="📦 .zip 진(眞) 엔진 맵 다운로드", 
                data=zip_buffer.getvalue(), 
                file_name="ADOFAI_Masterpiece_Map.zip",
                mime="application/zip"
            )

        except Exception as e:
            st.error(f"오류 발생: {e}")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
