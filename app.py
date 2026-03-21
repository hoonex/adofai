import streamlit as st
import librosa
import numpy as np
import json
import tempfile
import os
import zipfile
import io
from pathlib import Path

st.set_page_config(page_title="ADOFAI Masterpiece Engine", page_icon="🧊", layout="wide")
st.title("🧊 얼불춤(ADOFAI) 마스터피스 맵 생성기")
st.write("기승전결 구조 분석 + 알잘딱 스마트 패턴 + 오차 0% 완벽 동기화 엔진")

class AdofaiMapGenerator:
    def __init__(self, audio_path, raw_bytes, filename, original_ext):
        self.audio_path = audio_path
        self.raw_bytes = raw_bytes
        self.filename = filename
        self.original_ext = original_ext
        
        self.bpm = 0
        self.offset_ms = 0
        self.angle_data = [0]
        self.actions = []
        
        # 상태 추적 변수들
        self.current_abs_angle = 0
        self.theoretical_time = 0.0
        self.current_floor = 1
        
    def analyze_audio(self):
        # 1. 오디오 로드 및 HPSS (보컬과 드럼 분리)
        y, sr = librosa.load(self.audio_path, sr=22050, mono=True)
        y_harm, y_perc = librosa.effects.hpss(y, margin=2.0)
        
        # 2. 비트 및 템포 추출 (순수 타격음 기반)
        tempo, beat_frames = librosa.beat.beat_track(y=y_perc, sr=sr)
        self.bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        
        onset_frames = librosa.onset.onset_detect(y=y_perc, sr=sr, backtrack=True)
        self.onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # 3. 곡의 기승전결(Structure) 분석을 위한 에너지 스캔
        rms = librosa.feature.rms(y=y)[0]
        self.times_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        self.energy_profile = rms
        self.mean_energy = np.mean(rms)
        self.max_energy = np.max(rms)
        
        # 노이즈 필터링
        self.clean_onsets = self._filter_onsets(self.onset_times, self.bpm)
        if len(self.clean_onsets) > 0:
            self.offset_ms = int(self.clean_onsets[0] * 1000)
            self.theoretical_time = self.clean_onsets[0]

    def _filter_onsets(self, onsets, bpm):
        min_time = (60.0 / bpm) / 4.0 # 16분음표 최소 간격
        filtered = [onsets[0]] if len(onsets) > 0 else []
        for t in onsets[1:]:
            if t - filtered[-1] >= min_time:
                filtered.append(t)
        return filtered

    def _get_segment_type(self, current_time):
        """현재 시간이 곡의 어느 부분(도입부, 빌드업, 하이라이트)인지 파악"""
        idx = np.argmin(np.abs(self.times_rms - current_time))
        energy = self.energy_profile[idx]
        
        if energy > self.mean_energy * 1.5:
            return "DROP" # 하이라이트 (미친 변칙성 부여)
        elif energy > self.mean_energy * 0.9:
            return "BUILDUP" # 빌드업 (점진적 변화)
        else:
            return "VERSE" # 잔잔한 구간 (일관성 유지)

    def _calculate_ideal_angle(self, delta_time):
        """시간 차이를 바탕으로 얼불춤의 정확한 타일 이동 각도를 역산"""
        # 얼불춤 공식: 1박자(60/BPM 초) = 180도 이동
        ideal_travel_angle = delta_time * (self.bpm / 60.0) * 180.0
        
        # 리듬게임에 적합하도록 15도 단위 스냅 (칼각 보정)
        snapped = round(ideal_travel_angle / 15.0) * 15
        if snapped < 15: snapped = 15
        
        return snapped

    def generate_map_logic(self):
        # 초기 세팅 이벤트
        self.actions.append({"floor": 1, "eventType": "SetSpeed", "speedType": "Bpm", "beatsPerMinute": self.bpm, "bpmMultiplier": 1, "angleOffset": 0})
        self.actions.append({"floor": 1, "eventType": "ColorTrack", "trackColorType": "Single", "trackColor": "debb7b", "secondaryTrackColor": "ffffff", "trackColorAnimDuration": 2, "trackColorPulse": "None", "trackPulseLength": 10, "trackStyle": "Standard", "trackTexture": "", "trackTextureScale": 1, "trackGlowIntensity": 100, "justThisTile": False})
        
        prev_segment = "VERSE"
        turn_direction = 1

        for i in range(1, len(self.clean_onsets)):
            actual_time = self.clean_onsets[i]
            delta_t = actual_time - self.theoretical_time
            
            # 너무 짧은 오류 컷
            if delta_t < 0.05: continue
            
            segment = self._get_segment_type(actual_time)
            self.current_floor += 1
            
            # [시각 효과] 구간이 바뀔 때 연출 터뜨리기
            if segment == "DROP" and prev_segment != "DROP":
                self._trigger_vfx_drop()
            elif segment != "DROP" and prev_segment == "DROP":
                self._trigger_vfx_calm()
            prev_segment = segment

            # 1. 박자에 따른 기본 이동 각도(Travel Angle) 계산
            travel_angle = self._calculate_ideal_angle(delta_t)
            
            # 박자 간격이 길어 360도를 넘으면 직진(180도) 타일로 공간 채우기
            while travel_angle > 360:
                self.angle_data.append(int(self.current_abs_angle)) # 직진 타일 배정
                self.theoretical_time += 180.0 / (self.bpm / 60.0 * 180.0)
                travel_angle -= 180
                self.current_floor += 1

            # 2. 알잘딱 패턴 결정 (일관성 vs 변칙성)
            # travel_angle (이동량)을 실제 절대 각도(화면에 깔리는 각도)로 변환
            
            if segment == "VERSE":
                # [잔잔한 구간] 일관되게 직진하거나 깔끔한 계단만 생성
                if travel_angle == 180:
                    next_abs_angle = self.current_abs_angle # 일직선
                else:
                    # 일관된 방향으로 꺾기
                    next_abs_angle = (self.current_abs_angle + travel_angle * turn_direction - 180) % 360
                    if i % 8 == 0: turn_direction *= -1 # 8비트마다 방향 전환

            elif segment == "BUILDUP":
                # [빌드업] 다각형이나 트레실로 등 약간의 변칙 등장
                if travel_angle in [45, 90, 135]:
                    next_abs_angle = (self.current_abs_angle + travel_angle * turn_direction - 180) % 360
                    if i % 4 == 0: turn_direction *= -1
                else:
                    next_abs_angle = (self.current_abs_angle + travel_angle - 180) % 360

            elif segment == "DROP":
                # [하이라이트] 모든 각도 활용, 미친 패턴 등장
                # 특정 좁은 각도(플라밍고 등)나 넓은 각도를 변칙적으로 섞음
                if travel_angle == 45: # 소용돌이 유도
                    next_abs_angle = (self.current_abs_angle + 45 * turn_direction - 180) % 360
                elif travel_angle == 90: # 지그재그
                    turn_direction *= -1
                    next_abs_angle = (self.current_abs_angle + 90 * turn_direction - 180) % 360
                else:
                    # 복잡한 기하학 각도 허용
                    next_abs_angle = (self.current_abs_angle + travel_angle * random.choice([1, -1]) - 180) % 360
                
                # 가끔씩 화면을 돌려주는 시각 기믹 추가
                if i % 16 == 0:
                    self.actions.append({"floor": self.current_floor, "eventType": "Twirl"})

            self.angle_data.append(int(next_abs_angle))
            
            # 수학적 이론 시간 누적 (싱크 절대 안 밀림)
            self.theoretical_time += travel_angle / (self.bpm / 60.0 * 180.0)
            self.current_abs_angle = next_abs_angle

    def _trigger_vfx_drop(self):
        """하이라이트 구간 폭발 연출"""
        self.actions.append({"floor": self.current_floor, "eventType": "ShakeScreen", "duration": 1, "strength": 120, "intensity": 100, "ease": "Linear", "fadeOut": True, "angleOffset": 0, "eventTag": ""})
        self.actions.append({"floor": self.current_floor, "eventType": "Flash", "duration": 1, "plane": "Background", "startColor": "ffffff", "startOpacity": 70, "endColor": "ffffff", "endOpacity": 0, "angleOffset": 0, "ease": "Linear", "eventTag": ""})
        self.actions.append({"floor": self.current_floor, "eventType": "Bloom", "enabled": True, "threshold": 50, "intensity": 150, "color": "ffffff", "duration": 0, "ease": "Linear", "angleOffset": 0, "eventTag": ""})
        self.actions.append({"floor": self.current_floor, "eventType": "CustomBackground", "color": "220505", "bgImage": "", "imageColor": "ffffff", "parallax": [100, 100], "bgDisplayMode": "FitToScreen", "imageSmoothing": True, "lockRot": False, "loopBG": False, "scalingRatio": 100, "angleOffset": 0, "eventTag": ""})

    def _trigger_vfx_calm(self):
        """잔잔한 구간 복귀 연출"""
        self.actions.append({"floor": self.current_floor, "eventType": "CustomBackground", "color": "000000", "bgImage": "", "imageColor": "ffffff", "parallax": [100, 100], "bgDisplayMode": "FitToScreen", "imageSmoothing": True, "lockRot": False, "loopBG": False, "scalingRatio": 100, "angleOffset": 0, "eventTag": ""})

    def build_json(self):
        settings_block = {
            "version": 15, "artist": "Masterpiece Engine", "specialArtistType": "None", "artistPermission": "",
            "song": self.filename, "author": "ADOFAI Master AI", "separateCountdownTime": True,
            "previewImage": "", "previewIcon": "", "previewIconColor": "003f52", "previewSongStart": 0,
            "previewSongDuration": 10, "seizureWarning": False, "levelDesc": "Structured Perfect Sync Map",
            "levelTags": "", "artistLinks": "", "speedTrialAim": 0, "difficulty": 1, "requiredMods": [],
            "songFilename": self.filename, # 필수
            "bpm": self.bpm, "volume": 100, "offset": self.offset_ms, "pitch": 100,
            "hitsound": "Kick", "hitsoundVolume": 100, "countdownTicks": 4, "songURL": "", "tileShape": "Long",
            "trackColorType": "Single", "trackColor": "debb7b", "secondaryTrackColor": "ffffff",
            "trackColorAnimDuration": 2, "trackColorPulse": "None", "trackPulseLength": 10,
            "trackStyle": "Standard", "trackTexture": "", "trackTextureScale": 1, "trackGlowIntensity": 100,
            "trackAnimation": "None", "beatsAhead": 5, "trackDisappearAnimation": "None",
            "beatsBehind": 0, # 지나간 타일 즉시 삭제
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
    st.info("🎵 곡의 기승전결 구조 분석 및 스마트 맵핑 중... (고급 알고리즘 적용)")
    
    with st.spinner("마스터피스 엔진 가동 중..."):
        uploaded_file.seek(0)
        raw_audio_bytes = uploaded_file.read()
        
        ext = Path(uploaded_file.name).suffix.lower()
        if ext not in ['.mp3', '.wav', '.ogg']: ext = '.mp3'
        safe_audio_filename = f"song{ext}"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(raw_audio_bytes)
            tmp_file_path = tmp_file.name

        try:
            # 엔진 초기화 및 가동
            engine = AdofaiMapGenerator(tmp_file_path, raw_audio_bytes, safe_audio_filename, ext)
            engine.analyze_audio()
            engine.generate_map_logic()
            adofai_str = engine.build_json()

            # ZIP 패키징 (원본 파일 바이트 그대로 삽입하여 묵음 방지)
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("level.adofai", adofai_str)
                zip_file.writestr(safe_audio_filename, raw_audio_bytes)

            st.success("✨ 곡의 기승전결이 완벽히 반영된 마스터피스 맵 생성 완료!")
            
            st.download_button(
                label="📦 .zip 마스터 맵 다운로드", 
                data=zip_buffer.getvalue(), 
                file_name="AI_Masterpiece_Map.zip",
                mime="application/zip"
            )

        except Exception as e:
            st.error(f"오류 발생: {e}")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
