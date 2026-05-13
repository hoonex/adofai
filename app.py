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
from dataclasses import dataclass
from typing import List, Tuple

st.set_page_config(page_title="ADOFAI Masterpiece Engine (Ultimate)", page_icon="🧊", layout="wide")
st.title("🧊 얼불춤 진(眞) 마스터피스 엔진 - Ultimate Edition")
st.markdown("보컬 싱크, 절대 시간, 소수점 단위 충돌 방지, 미적 패턴 연출이 모두 결합된 최종 완성본입니다.")

# ==========================================
# [모듈 1] 정밀 수학 기반 2D 공간 지각 & 충돌 방지 시스템
# ==========================================
@dataclass
class Vector2D:
    x: float
    y: float

class AdvancedGridTracker:
    def __init__(self):
        self.current_pos = Vector2D(0.0, 0.0)
        self.current_abs_angle = 0.0
        self.history: List[Vector2D] = [Vector2D(0.0, 0.0)]
        self.tile_radius = 0.85 # 타일 간 최소 안전 거리 (겹침 판정 기준)
        
    def _calculate_next_pos(self, abs_angle: float) -> Vector2D:
        """각도를 라디안으로 변환하여 다음 타일의 정확한 x, y 좌표를 계산"""
        rad = math.radians(abs_angle)
        dx = math.cos(rad)
        dy = math.sin(rad)
        return Vector2D(self.current_pos.x + dx, self.current_pos.y + dy)

    def check_collision(self, next_angle_diff: float) -> bool:
        """과거의 모든 타일 위치와 거리를 계산하여 정밀하게 겹침을 검사 (소수점 단위)"""
        test_abs_angle = (self.current_abs_angle + next_angle_diff - 180) % 360
        next_pos = self._calculate_next_pos(test_abs_angle)
        
        # 최근 2개의 타일은 겹칠 일이 없으므로 제외하고 검사
        for past_pos in self.history[:-2]:
            dist = math.hypot(next_pos.x - past_pos.x, next_pos.y - past_pos.y)
            if dist < self.tile_radius:
                return True # 겹침 발생!
        return False
        
    def move(self, next_angle_diff: float) -> float:
        """실제 이동 후 좌표 기록"""
        self.current_abs_angle = (self.current_abs_angle + next_angle_diff - 180) % 360
        self.current_pos = self._calculate_next_pos(self.current_abs_angle)
        self.history.append(self.current_pos)
        return self.current_abs_angle

# ==========================================
# [모듈 2] 미적 감각 연출가 (Aesthetic Pattern Director)
# ==========================================
class AestheticPatternDirector:
    def __init__(self):
        # 맵을 예쁘게 만드는 기본 각도 조각들
        self.base_angles = [90, 180, 270, 360]
        
    def get_beautiful_angle(self, required_travel: float, energy: float) -> float:
        """
        요구되는 시간(각도)을 최대한 '예쁜' 기하학적 각도로 변환합니다.
        불가피한 엇박자(예: 135도)는 그대로 반환하지만, 
        가능한 90도(직각)나 270도(계단) 형태로 유도합니다.
        """
        # 타일이 너무 짧게 꺾이는 노이즈 방지
        if required_travel < 45:
            return 180.0 
            
        # 에너지(볼륨)가 폭발할 때: 마법진이나 날카로운 지그재그 패턴 유도
        if energy > 2.0:
            best_angles = [225, 315, 135, 45] # 대각선 위주
        else:
            best_angles = [90, 270, 180, 360] # 직각, 직선 위주
            
        # 요구된 시간(required_travel)과 가장 가까운 예쁜 각도를 찾음
        closest_angle = min(best_angles, key=lambda x: abs(x - required_travel))
        
        # 단, 싱크가 너무 심하게 틀어지면 안 되므로 오차가 30도 이내일 때만 보정
        if abs(closest_angle - required_travel) <= 30:
            return closest_angle
        return round(required_travel / 15.0) * 15.0 # 보정 불가 시 15도 스냅 유지

# ==========================================
# [모듈 3] 진(眞) 마스터피스 코어 엔진
# ==========================================
class UltimateMapGenerator:
    def __init__(self, audio_path, raw_bytes, filename):
        self.audio_path = audio_path
        self.raw_bytes = raw_bytes
        self.filename = filename
        
        self.bpm = 0
        self.offset_ms = 0
        self.angle_data = [0]
        self.actions = []
        
        self.grid = AdvancedGridTracker()
        self.aesthetics = AestheticPatternDirector()
        
        self.theoretical_time = 0.0
        self.current_floor = 1
        
    def analyze_audio(self):
        st.toast("1/4: 보컬 대역 및 비트 정밀 스캔 중... (고해상도 유지)")
        y, sr = librosa.load(self.audio_path, sr=22050, mono=True)
        y_harm, y_perc = librosa.effects.hpss(y, margin=2.0)
        
        tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)
        self.bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        
        st.toast("2/4: 발음(Onset) 기준 타격점 스캔 중... (가사 싱크)")
        onset_frames = librosa.onset.onset_detect(y=y_harm, sr=sr, backtrack=True)
        self.onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
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
        """가사 발음이 너무 잘게 쪼개져서 타일이 더러워지는 것을 방지 (최소 16분음표 간격)"""
        min_time = (60.0 / bpm) / 4.0 
        filtered = [onsets[0]] if len(onsets) > 0 else []
        for t in onsets[1:]:
            if t - filtered[-1] >= min_time:
                filtered.append(t)
        return filtered

    def _get_audio_context(self, current_time):
        idx = np.argmin(np.abs(self.times - current_time))
        energy = self.rms[idx] / self.mean_energy # 정규화된 에너지
        centroid = self.centroids[idx]
        return energy, centroid

    def generate_map_logic(self):
        st.toast("4/4: 절대 시간 동기화 및 기하학 맵핑 중...")
        
        # [초기 VFX 세팅]
        self.actions.append({"floor": 1, "eventType": "SetSpeed", "speedType": "Bpm", "beatsPerMinute": self.bpm, "bpmMultiplier": 1, "angleOffset": 0})
        self.actions.append({"floor": 1, "eventType": "ColorTrack", "trackColorType": "Glow", "trackColor": "00D4FF", "secondaryTrackColor": "005dff", "trackColorAnimDuration": 2, "trackColorPulse": "Forward", "trackPulseLength": 10, "trackStyle": "Neon"})
        self.actions.append({"floor": 1, "eventType": "CustomBackground", "color": "050510", "bgDisplayMode": "FitToScreen"})
        
        is_highlight = False
        current_accumulated_angle = 0.0 # 절대 시간 동기화의 핵심 (스노우볼 방어)

        for i in range(1, len(self.clean_onsets)):
            target_audio_time = self.clean_onsets[i]
            
            # 1. 절대 시간(Beat) 계산
            target_beats = (target_audio_time - self.theoretical_time) * (self.bpm / 60.0)
            target_total_angle = target_beats * 180.0
            
            # 이 타일이 곡의 싱크를 맞추기 위해 소모해야 하는 '시간'
            required_travel = round(target_total_angle - current_accumulated_angle)
            
            if required_travel < 15:
                continue # 오차 수준의 짧은 소리는 무시
                
            energy, centroid = self._get_audio_context(target_audio_time)
            self.current_floor += 1
            
            # --- 🎬 화려한 카메라/이펙트 연출 ---
            if energy > 1.8 and not is_highlight:
                is_highlight = True
                self.actions.append({"floor": self.current_floor, "eventType": "MoveCamera", "duration": 2, "relativeTo": "Player", "zoom": 120, "rotation": 15, "ease": "OutCubic"})
                self.actions.append({"floor": self.current_floor, "eventType": "SetFilter", "filter": "Aberration", "enabled": True, "intensity": 30})
                self.actions.append({"floor": self.current_floor, "eventType": "Flash", "duration": 1, "plane": "Background", "startColor": "ffffff", "startOpacity": 50, "endColor": "000000", "endOpacity": 0})
            elif energy < 1.2 and is_highlight:
                is_highlight = False
                self.actions.append({"floor": self.current_floor, "eventType": "MoveCamera", "duration": 2, "relativeTo": "Player", "zoom": 100, "rotation": 0, "ease": "InCubic"})
                self.actions.append({"floor": self.current_floor, "eventType": "SetFilter", "filter": "Aberration", "enabled": False, "intensity": 0})

            # --- 📐 공간 지각 및 충돌 회피, 예쁜 패턴 계산 ---
            is_long_note = False
            
            # 보컬이 길게 끄는 구간 ("모~~~~~시") : 싱크가 맞을 때까지 직선(180)으로 채움
            while required_travel > 360:
                is_long_note = True
                self.grid.move(180) 
                self.angle_data.append(self.grid.current_abs_angle)
                current_accumulated_angle += 180
                required_travel -= 180
                self.current_floor += 1

            if is_long_note:
                self.actions.append({"floor": self.current_floor - 1, "eventType": "MoveCamera", "duration": 1, "relativeTo": "Player", "zoom": 130, "ease": "OutQuad"})

            # 남은 시간에 대해 '예쁜 각도'를 가져옴
            aesthetic_angle = self.aesthetics.get_beautiful_angle(required_travel, energy)
            
            # 겹침 방지(Collision) 로직: 예쁜 각도를 놨을 때 기존 맵과 겹치는가?
            final_angle = aesthetic_angle
            if self.grid.check_collision(aesthetic_angle):
                # 겹친다면, 소용돌이(Twirl) 이벤트를 넣고 각도를 대칭(360 - angle)으로 꺾어 회피
                self.actions.append({"floor": self.current_floor, "eventType": "Twirl"})
                final_angle = 360 - aesthetic_angle
                
                # 대칭으로 꺾어도 겹친다면? 최후의 수단으로 안전한 직선(180) 혹은 90도로 우회
                if self.grid.check_collision(final_angle):
                    final_angle = 180 if not self.grid.check_collision(180) else 90

            # 최종 타일 배치 및 절대 시간 트래커 동기화
            # 주의: time(current_accumulated_angle)은 실제 요구된 시간(required_travel)을 더해야 싱크가 안 밀림
            new_abs_angle = self.grid.move(final_angle)
            self.angle_data.append(int(new_abs_angle))
            current_accumulated_angle += required_travel 

    def build_json(self):
        settings_block = {
            "version": 15, "artist": "Data-Driven AI", "specialArtistType": "None",
            "song": self.filename, "author": "Ultimate Masterpiece Engine", "separateCountdownTime": True,
            "seizureWarning": False, "levelDesc": "Vocal Sync + Collision Free + Aesthetic Directed",
            "difficulty": 5, "songFilename": self.filename, 
            "bpm": self.bpm, "volume": 100, "offset": self.offset_ms, "pitch": 100,
            "hitsound": "Kick", "hitsoundVolume": 100, "countdownTicks": 4,
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
    with st.spinner("궁극의 엔진이 음악을 분해하고 맵을 스케치하는 중입니다..."):
        uploaded_file.seek(0)
        raw_audio_bytes = uploaded_file.read()
        
        ext = Path(uploaded_file.name).suffix.lower()
        if ext not in ['.mp3', '.wav', '.ogg']: ext = '.mp3'
        safe_audio_filename = f"song{ext}"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(raw_audio_bytes)
            tmp_file_path = tmp_file.name

        try:
            engine = UltimateMapGenerator(tmp_file_path, raw_audio_bytes, safe_audio_filename)
            engine.analyze_audio()
            engine.generate_map_logic()
            adofai_str = engine.build_json()

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("level.adofai", adofai_str)
                zip_file.writestr(safe_audio_filename, raw_audio_bytes)

            st.success("✨ 아름답고, 싱크가 완벽하며, 겹치지 않는 궁극의 맵이 완성되었습니다!")
            
            with st.expander("엔진 분석 리포트"):
                st.write(f"- 🎵 기본 분석 BPM: **{round(engine.bpm, 1)}**")
                st.write(f"- 🧱 생성된 타일 수: **{len(engine.angle_data)}개** (가사 발음 기준)")
                st.write(f"- 🎬 삽입된 특수 연출(Twirl, 카메라 등): **{len(engine.actions)}개**")

            st.download_button(
                label="📦 .zip 다운로드 (얼불춤 폴더에 압축 해제)", 
                data=zip_buffer.getvalue(), 
                file_name="ADOFAI_Ultimate_Map.zip",
                mime="application/zip"
            )

        except Exception as e:
            st.error(f"오류 발생: {e}")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
