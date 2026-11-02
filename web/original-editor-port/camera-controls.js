/* Camera/navigation for the browser chart viewport.
 * The game scene and scoring engine are not included. This is an independent
 * interactive camera bound to the imported PC editor's chart stage.
 */
(function () {
  'use strict';
  const boot = () => {
    const api = window.ADOFAIChartPort;
    const svg = document.getElementById('chartSvg');
    const stage = document.querySelector('.chart-stage');
    const head = document.querySelector('.chart-stage-head');
    if (!api || !svg || !stage || !head) return;

    const camera = {
      cx: 0, cy: 0, zoom: 1.15, ready: false, follow: true,
      setting: false, suppressClick: false, pointers: new Map(),
      lastPinch: null, dragged: false
    };
    const UNIT = 54;
    const MIN_ZOOM = 0.075;
    const MAX_ZOOM = 4;
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    const positions = () => api.state.positions || [];
    const hasChart = () => !!api.state.doc && positions().length > 0;
    const focused = (floor = api.state.selected) => positions()[clamp(floor || 0, 0, positions().length - 1)];
    const rect = () => svg.getBoundingClientRect();
    const viewport = () => {
      const r = rect();
      return {w: Math.max(1, r.width || 850), h: Math.max(1, r.height || 440)};
    };
    function setViewBox() {
      if (!hasChart() || !camera.ready) return;
      const {w, h} = viewport();
      const width = w / camera.zoom;
      const height = h / camera.zoom;
      const desired = [camera.cx - width / 2, camera.cy - height / 2, width, height]
        .map(v => Math.round(v * 1000) / 1000).join(' ');
      if (svg.getAttribute('viewBox') !== desired) {
        camera.setting = true;
        svg.setAttribute('viewBox', desired);
        camera.setting = false;
      }
      updateUi();
      drawMini();
    }
    function initCamera() {
      if (!hasChart()) return;
      if (!camera.ready) {
        const p = focused();
        camera.cx = p.x * UNIT;
        camera.cy = p.y * UNIT;
        camera.zoom = 1.15;
        camera.ready = true;
      }
      setViewBox();
    }
    function target(floor, shouldZoom = false) {
      if (!hasChart()) return;
      initCamera();
      const p = focused(floor);
      camera.cx = p.x * UNIT;
      camera.cy = p.y * UNIT;
      if (shouldZoom) camera.zoom = Math.max(camera.zoom, 0.85);
      setViewBox();
    }
    function zoomAt(factor, clientX, clientY) {
      if (!hasChart()) return;
      initCamera();
      const r = rect();
      const dx = clientX - (r.left + r.width / 2);
      const dy = clientY - (r.top + r.height / 2);
      const worldX = camera.cx + dx / camera.zoom;
      const worldY = camera.cy + dy / camera.zoom;
      camera.zoom = clamp(camera.zoom * factor, MIN_ZOOM, MAX_ZOOM);
      camera.cx = worldX - dx / camera.zoom;
      camera.cy = worldY - dy / camera.zoom;
      setViewBox();
    }
    function fit() {
      if (!hasChart()) return;
      const ps = positions();
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const p of ps) {
        minX = Math.min(minX, p.x * UNIT);
        maxX = Math.max(maxX, p.x * UNIT);
        minY = Math.min(minY, p.y * UNIT);
        maxY = Math.max(maxY, p.y * UNIT);
      }
      if (!Number.isFinite(minX)) return;
      const {w, h} = viewport();
      camera.cx = (minX + maxX) / 2;
      camera.cy = (minY + maxY) / 2;
      camera.zoom = clamp(
        Math.min(w / Math.max(160, maxX - minX + 160), h / Math.max(160, maxY - minY + 160)),
        MIN_ZOOM, MAX_ZOOM
      );
      camera.ready = true;
      camera.follow = false;
      setViewBox();
    }

    const toolbar = document.createElement('div');
    toolbar.id = 'chartCameraBar';
    toolbar.innerHTML =
      '<div class="camera-left"><strong>맵 카메라</strong>' +
      '<button type="button" id="cameraMinus" title="축소" aria-label="맵 축소">−</button>' +
      '<output id="cameraZoom" aria-live="off">115%</output>' +
      '<button type="button" id="cameraPlus" title="확대" aria-label="맵 확대">＋</button>' +
      '<button type="button" id="cameraCenter" title="선택한 타일로 이동">◎ 선택 위치</button>' +
      '<button type="button" id="cameraFit" title="전체 맵 보기">전체 보기</button>' +
      '<button type="button" id="cameraFollow" aria-pressed="true" title="연습 플레이 중 현재 타일 따라가기">● 플레이 추적 켬</button></div>' +
      '<span class="camera-hint">손가락 드래그: 이동 · 두 손가락: 확대/축소 · 마우스 휠: 확대/축소</span>';
    head.insertAdjacentElement('afterend', toolbar);

    const mini = document.createElement('canvas');
    mini.id = 'chartMinimap';
    mini.width = 260;
    mini.height = 150;
    mini.title = '미니맵을 눌러 해당 위치로 이동';
    mini.setAttribute('aria-label', '클릭하면 해당 위치로 이동하는 미니맵');
    mini.setAttribute('role', 'img');
    stage.appendChild(mini);

    const css = document.createElement('style');
    css.textContent = [
      '#chartCameraBar{position:relative;z-index:3;display:flex;justify-content:space-between;gap:10px;align-items:center;padding:7px 12px;background:#202c36;border-bottom:1px solid #475969;flex-wrap:wrap}',
      '.camera-left{display:flex;gap:5px;align-items:center;flex-wrap:wrap}',
      '.camera-left strong{font-size:12px;color:#cbd9e7;margin-right:4px}',
      '.camera-left button{background:#354350;color:#f1f5f9;border:1px solid #6b7c8f;border-radius:4px;min-height:36px;min-width:37px;padding:6px 10px;font-size:13px;cursor:pointer;touch-action:manipulation}',
      '.camera-left button:hover,.camera-left button:focus-visible{background:#4c6681}',
      '.camera-left button[aria-pressed=true]{background:#2f6251;border-color:#5b9b81}',
      '#cameraZoom{font-variant-numeric:tabular-nums;min-width:51px;text-align:center;font-size:12px;color:#f3e8bc}',
      '.camera-hint{font-size:11px;color:#a0b3c4}',
      '#chartSvg{touch-action:none!important;user-select:none;cursor:grab;min-height:360px}',
      '#chartSvg.is-panning{cursor:grabbing}',
      '#chartMinimap{position:absolute;right:13px;bottom:81px;width:173px;height:100px;z-index:6;background:#0a1420e8;border:1px solid #7891a7;border-radius:5px;box-shadow:0 2px 12px #0009;touch-action:none;cursor:crosshair}',
      '@media(max-width:640px){#chartCameraBar{padding:7px 8px}.camera-hint{width:100%;font-size:10px}#chartSvg{height:48vh;min-height:320px}#chartMinimap{width:116px;height:67px;bottom:86px}.camera-left button{min-height:39px;padding:5px 7px}}'
    ].join('\n');
    document.head.appendChild(css);

    const $ = id => document.getElementById(id);
    function updateUi() {
      $('cameraZoom').textContent = Math.round(camera.zoom * 100) + '%';
      $('cameraFollow').setAttribute('aria-pressed', String(camera.follow));
      $('cameraFollow').textContent = camera.follow ? '● 플레이 추적 켬' : '○ 플레이 추적 끔';
    }
    $('cameraPlus').addEventListener('click', () => {
      const r = rect(); zoomAt(1.3, r.left + r.width / 2, r.top + r.height / 2);
    });
    $('cameraMinus').addEventListener('click', () => {
      const r = rect(); zoomAt(1 / 1.3, r.left + r.width / 2, r.top + r.height / 2);
    });
    $('cameraCenter').addEventListener('click', () => target(api.state.selected, true));
    $('cameraFit').addEventListener('click', fit);
    $('cameraFollow').addEventListener('click', () => {
      camera.follow = !camera.follow;
      if (camera.follow && window.ADOFAIArchivePlayer?.state?.playing) {
        target(window.ADOFAIArchivePlayer.state.playerIndex, true);
      }
      updateUi();
    });
    svg.addEventListener('wheel', event => {
      if (!hasChart()) return;
      event.preventDefault();
      camera.follow = false;
      zoomAt(Math.exp(-clamp(event.deltaY, -400, 400) * 0.0015), event.clientX, event.clientY);
    }, {passive: false});

    function pinchSnapshot() {
      const p = Array.from(camera.pointers.values());
      if (p.length !== 2) return null;
      return {
        distance: Math.hypot(p[0].x-p[1].x, p[0].y-p[1].y),
        midX: (p[0].x+p[1].x)/2,
        midY: (p[0].y+p[1].y)/2
      };
    }
    svg.addEventListener('pointerdown', event => {
      if (!hasChart() || (event.pointerType === 'mouse' && event.button !== 0)) return;
      initCamera();
      camera.pointers.set(event.pointerId, {x:event.clientX,y:event.clientY});
      camera.lastPinch = pinchSnapshot();
      camera.dragged = false;
      if (event.pointerType !== 'mouse') {try {svg.setPointerCapture(event.pointerId);} catch (_) {}}
    });
    svg.addEventListener('pointermove', event => {
      if (!camera.pointers.has(event.pointerId)) return;
      const prev = camera.pointers.get(event.pointerId);
      const dx = event.clientX - prev.x;
      const dy = event.clientY - prev.y;
      camera.pointers.set(event.pointerId, {x:event.clientX,y:event.clientY});
      if (camera.pointers.size === 2) {
        const p = pinchSnapshot();
        if (camera.lastPinch && camera.lastPinch.distance > 2) {
          const ratio = p.distance / camera.lastPinch.distance;
          camera.cx -= (p.midX - camera.lastPinch.midX) / camera.zoom;
          camera.cy -= (p.midY - camera.lastPinch.midY) / camera.zoom;
          zoomAt(ratio, p.midX, p.midY);
        }
        camera.lastPinch = p;
        camera.follow = false;
        camera.dragged = true;
        camera.suppressClick = true;
      } else if (Math.hypot(dx, dy) > 0.15) {
        camera.cx -= dx / camera.zoom;
        camera.cy -= dy / camera.zoom;
        camera.follow = false;
        if (Math.hypot(dx, dy) > 2) camera.dragged = true;
        if (camera.dragged) camera.suppressClick = true;
        setViewBox();
      }
      if (camera.dragged) {
        if (event.pointerType === 'mouse' && !svg.hasPointerCapture(event.pointerId)) {
          try {svg.setPointerCapture(event.pointerId);} catch (_) {}
        }
        event.preventDefault();
        svg.classList.add('is-panning');
      }
    });
    const pointerEnd = event => {
      camera.pointers.delete(event.pointerId);
      camera.lastPinch = pinchSnapshot();
      if (camera.pointers.size === 0) {
        svg.classList.remove('is-panning');
        setTimeout(() => {camera.suppressClick = false; camera.dragged = false;}, 100);
      }
    };
    svg.addEventListener('pointerup', pointerEnd);
    svg.addEventListener('pointercancel', pointerEnd);
    svg.addEventListener('lostpointercapture', pointerEnd);
    svg.addEventListener('click', event => {
      if (camera.suppressClick) {
        event.preventDefault(); event.stopImmediatePropagation();
      }
    }, true);
    svg.addEventListener('dblclick', event => {
      event.preventDefault();
      camera.follow = false;
      zoomAt(1.6, event.clientX, event.clientY);
    });
    window.addEventListener('keydown', event => {
      if (!hasChart() || event.altKey || event.ctrlKey || event.metaKey) return;
      const e = event.target;
      if (e && (e.isContentEditable || /^(INPUT|TEXTAREA|SELECT|BUTTON)$/.test(e.tagName))) return;
      const move = 90 / camera.zoom;
      switch(event.key) {
        case 'ArrowLeft': camera.cx -= move; break;
        case 'ArrowRight': camera.cx += move; break;
        case 'ArrowUp': camera.cy -= move; break;
        case 'ArrowDown': camera.cy += move; break;
        case '+': case '=': {
          const r = rect();zoomAt(1.2,r.left+r.width/2,r.top+r.height/2);event.preventDefault();return;
        }
        case '-': {
          const r = rect();zoomAt(1/1.2,r.left+r.width/2,r.top+r.height/2);event.preventDefault();return;
        }
        case 'Home': target(api.state.selected,true);event.preventDefault();return;
        default: return;
      }
      camera.follow = false;setViewBox();event.preventDefault();
    });

    let miniBounds = null;
    function drawMini() {
      const ps = positions();
      if (!ps.length) return;
      const ctx = mini.getContext('2d');
      if (!ctx) return;
      let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
      for (const p of ps) {
        minX=Math.min(minX,p.x*UNIT);maxX=Math.max(maxX,p.x*UNIT);
        minY=Math.min(minY,p.y*UNIT);maxY=Math.max(maxY,p.y*UNIT);
      }
      const pad=12;
      const scale=Math.min((mini.width-pad*2)/Math.max(1,maxX-minX),(mini.height-pad*2)/Math.max(1,maxY-minY));
      const ww=(maxX-minX)*scale,hh=(maxY-minY)*scale;
      const ox=(mini.width-ww)/2-minX*scale, oy=(mini.height-hh)/2-minY*scale;
      miniBounds={scale,ox,oy};
      ctx.clearRect(0,0,mini.width,mini.height);
      ctx.fillStyle='#0b1522';ctx.fillRect(0,0,mini.width,mini.height);
      ctx.strokeStyle='#7594b6';ctx.lineWidth=2;ctx.beginPath();
      ps.forEach((p,i) => {
        const x=p.x*UNIT*scale+ox,y=p.y*UNIT*scale+oy;
        if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      });
      ctx.stroke();
      const p=focused(),sx=p.x*UNIT*scale+ox,sy=p.y*UNIT*scale+oy;
      ctx.fillStyle='#ffc164';ctx.beginPath();ctx.arc(sx,sy,4,0,Math.PI*2);ctx.fill();
      const {w,h}=viewport();
      ctx.strokeStyle='#8fe7b1';ctx.lineWidth=2;
      ctx.strokeRect((camera.cx-w/camera.zoom/2)*scale+ox,
                     (camera.cy-h/camera.zoom/2)*scale+oy,
                     w/camera.zoom*scale,h/camera.zoom*scale);
    }
    mini.addEventListener('pointerdown', event => {
      if (!hasChart() || !miniBounds) return;
      const r = mini.getBoundingClientRect();
      const px=(event.clientX-r.left)*mini.width/r.width;
      const py=(event.clientY-r.top)*mini.height/r.height;
      camera.cx=(px-miniBounds.ox)/miniBounds.scale;
      camera.cy=(py-miniBounds.oy)/miniBounds.scale;
      camera.follow=false;camera.ready=true;setViewBox();
      event.preventDefault();
    });
    const observer = new MutationObserver(() => {
      if (!camera.setting && hasChart()) initCamera();
    });
    observer.observe(svg,{attributes:true,attributeFilter:['viewBox']});
    const fileInput=$('chartFile');
    if (fileInput) fileInput.addEventListener('change', () => {camera.ready=false;}, true);
    const newButton=$('chartNew');
    if (newButton) newButton.addEventListener('click', () => {camera.ready=false;}, true);
    const resetOnLoad=api.loadObject;
    api.loadObject=function(){
      camera.ready=false;
      return resetOnLoad.apply(api,arguments);
    };
    if (typeof ResizeObserver !== 'undefined') {
      new ResizeObserver(() => setViewBox()).observe(svg);
    }
    api.camera = {
      get center(){return {x:camera.cx,y:camera.cy};},
      get zoom(){return camera.zoom;},
      get following(){return camera.follow;},
      get ready(){return camera.ready;},
      centerSelected:()=>target(api.state.selected,true),
      centerFloor:(floor)=>target(floor,true),
      fit,
      zoomBy:(factor)=>{const r=rect();zoomAt(factor,r.left+r.width/2,r.top+r.height/2);},
      followFloor:(floor)=>{if (camera.follow) target(floor,false);},
      setFollow:(v)=>{camera.follow=!!v;updateUi();}
    };
    initCamera();
    updateUi();
  };
  if (document.readyState==='loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
