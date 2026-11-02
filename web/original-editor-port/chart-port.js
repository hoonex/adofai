/* Browser chart-editing layer wired to actual PC scnEditor GameObject names.
 * This is a clean browser implementation; Unity C#/IL2CPP gameplay is not executed.
 * Path char map, relative offsets, and event/floor behavior are based on source
 * evidence and the pinned adofaiex/ADOFAI-JS chart parser.
 */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const charDir = Object.freeze({R:0,p:15,J:30,E:45,T:60,o:75,U:90,q:105,G:120,Q:135,H:150,W:165,L:180,x:195,N:210,Z:225,F:240,V:255,D:270,Y:285,B:300,C:315,M:330,A:345,'!':999});
  const relChar = Object.freeze({'5':72,'6':-72,'7':52,'8':-52,'9':-30,h:120,j:-120,t:60,y:300});
  const relNum = Object.freeze({'555':72,'666':-72,'777':52,'888':-52});
  const port = {doc:null,filename:'',selected:0,dirty:false,undo:[],redo:[],sceneNodes:new Map(),angles:[],positions:[],view:'chart',focus:'events',downloadUrl:null};
  const htmlEscape = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const clamp = (n,a,b) => Math.max(a,Math.min(b,n));
  const norm = a => ((a%360)+360)%360;
  const clone = value => JSON.parse(JSON.stringify(value));
  const assertChart = data => {
    if (!data || Array.isArray(data) || typeof data !== 'object') throw Error('차트 최상위는 JSON 객체여야 합니다.');
    if (typeof data.pathData !== 'string' && !Array.isArray(data.angleData)) throw Error('pathData 또는 angleData가 필요합니다.');
    if (data.angleData !== undefined && !Array.isArray(data.angleData)) throw Error('angleData는 배열이어야 합니다.');
    if (Array.isArray(data.angleData) && !data.angleData.every(x => typeof x === 'number' && Number.isFinite(x))) throw Error('angleData에 올바르지 않은 각도가 있습니다.');
    if (data.actions !== undefined && !Array.isArray(data.actions)) throw Error('actions는 배열이어야 합니다.');
    if (data.decorations !== undefined && !Array.isArray(data.decorations)) throw Error('decorations는 배열이어야 합니다.');
    if (data.settings !== undefined && (!data.settings || Array.isArray(data.settings) || typeof data.settings !== 'object')) throw Error('settings는 객체여야 합니다.');
    return data;
  };
  function repairJson(text) {
    const input = text.replace(/^\uFEFF/, '');
    let result = '', quoted = false, escape = false;
    for (let i = 0; i < input.length; i++) {
      const c = input[i];
      if (quoted) {
        if (escape) { result += c; escape = false; continue; }
        if (c === '\\') { result += c; escape = true; continue; }
        if (c === '"') { quoted = false; result += c; continue; }
        if (c.charCodeAt(0) < 32) { result += '\\u' + c.charCodeAt(0).toString(16).padStart(4,'0'); continue; }
        result += c; continue;
      }
      if (c === '"') { quoted = true; result += c; continue; }
      if (c === ',') {
        let k = i + 1;
        while (k < input.length && /\s/.test(input[k])) k++;
        if (input[k] === '}' || input[k] === ']') continue;
      }
      result += c;
    }
    return result;
  }
  const parseChart = raw => assertChart(JSON.parse(repairJson(raw)));
  function anglesFromPath(path) {
    const result = []; let previous = 0;
    for (const c of path) {
      if (Object.prototype.hasOwnProperty.call(charDir,c)) {
        if (c === '!') result.push(999);
        else { previous = charDir[c]; result.push(previous); }
      } else if (Object.prototype.hasOwnProperty.call(relChar,c)) {
        previous = norm(previous + relChar[c]); result.push(previous);
      } else { throw Error('알 수 없는 pathData 방향 문자: ' + c + ' (데이터 변환 중단)'); }
    }
    return result;
  }
  function anglesFromData(raw) {
    const result = []; let previous = 0;
    for (const a of raw) {
      if (Object.prototype.hasOwnProperty.call(relNum,String(a))) previous = norm(previous + relNum[String(a)]);
      else if (a === 999) { result.push(999); continue; }
      else previous = a;
      result.push(previous);
    }
    return result;
  }
  function getAngles() {
    if (!port.doc) return [];
    return Array.isArray(port.doc.angleData) ? anglesFromData(port.doc.angleData) : anglesFromPath(port.doc.pathData);
  }
  function rebuildPositions() {
    port.angles = getAngles();
    let x = 0, y = 0;
    port.positions = [{x,y,index:0}];
    for (let i = 0; i < port.angles.length; i++) {
      const a = port.angles[i];
      if (a !== 999) { x += Math.cos(a * Math.PI / 180); y -= Math.sin(a * Math.PI / 180); }
      port.positions.push({x,y,index:i+1});
    }
    port.selected = clamp(port.selected,0,port.angles.length);
  }
  function status(msg,kind='info') {
    const el = $('chartStatus'); if (!el) return;
    el.textContent = msg; el.dataset.kind = kind;
  }
  function originalId(name) {
    const found = Array.from(port.sceneNodes.values()).find(n => n.name === name && n.path.startsWith('levelEditorScene/'));
    return found?.id ?? null;
  }
  function linkButton(id,originalName,label) {
    const el=$(id); if(!el)return;
    const original=originalId(originalName);
    el.disabled = original === null;
    el.dataset.originalGameobjectId = original === null ? '' : String(original);
    el.title = original === null ? '원본 씬에 해당 GameObject를 찾지 못했습니다.' : 'PC scnEditor GameObject ID '+original+' : '+originalName;
    el.textContent=label;
  }
  function connectOriginalScene() {
    if (typeof state === 'undefined' || !Array.isArray(state.nodes)) return;
    port.sceneNodes = new Map(state.nodes.map(n => [n.id,n]));
    linkButton('chartOpen','buttonOpen','열기');
    linkButton('chartSave','buttonSave','저장');
    linkButton('chartSaveAs','buttonSaveAs','다른 이름 저장');
    linkButton('chartAdd','AddTile','타일 추가');
    linkButton('chartSettingsTab','settingsPanel','설정');
    linkButton('chartEventsTab','levelEventsPanel','이벤트');
    $('chartOriginal').textContent = '원본 씬 연결됨 · 버튼 '+['buttonOpen','buttonSave','buttonSaveAs','AddTile'].map(x=>x+':'+originalId(x)).join(' · ');
  }
  function snapshot() {
    port.undo.push({doc:clone(port.doc),selection:port.selected});
    if (port.undo.length > 25) port.undo.shift();
    port.redo.length = 0;
  }
  function applyMutation(mutator,description) {
    if (!port.doc) return;
    const previous = clone(port.doc),index=port.selected;
    try {
      snapshot(); mutator(); assertChart(port.doc); rebuildPositions();
      port.dirty=true; render(); status(description+' · 변경됨');
    } catch (e) {
      port.doc=previous; port.selected=index; port.undo.pop(); rebuildPositions(); render(); status(e.message,'error');
    }
  }
  function editedAngles() {
    // Only when an angle is deliberately edited: convert original pathData
    // to explicit angleData while preserving all other chart fields verbatim
    // as JSON values. Unknown path codes are rejected rather than guessed.
    const angles=getAngles();
    delete port.doc.pathData;
    port.doc.angleData=angles;
    return port.doc.angleData;
  }
  function editAngle(raw) {
    if (port.selected===0) { status('시작 타일(0)은 진행 방향이 없습니다.','error'); return; }
    const v=Number(raw);
    if (!Number.isFinite(v) || (v!==999 && (v<0 || v>=360))) { status('각도는 0 이상 360 미만 또는 미드스핀 999여야 합니다.','error'); return; }
    if (v === port.angles[port.selected-1]) return;
    applyMutation(() => { editedAngles()[port.selected-1]=v; }, '타일 '+port.selected+' 방향 '+v+'°');
  }
  function shiftEvents(index,delta,deleteEqual=false) {
    for (const listName of ['actions','decorations']) {
      const list=port.doc[listName];
      if (!Array.isArray(list)) continue;
      port.doc[listName] = list.filter(ev=> !deleteEqual || Number(ev.floor)!==index).map(ev=>{
        if (!ev || typeof ev!=='object' || typeof ev.floor!=='number') return ev;
        if (ev.floor >= index + (deleteEqual ? 1 : 0)) return {...ev,floor:ev.floor+delta};
        return ev;
      });
    }
  }
  function addTile() {
    if (!port.doc) return;
    applyMutation(() => {
      const insertion=port.selected;
      const current=editedAngles();
      const preferred=current.length ? current[Math.min(insertion,current.length-1)] : 0;
      current.splice(insertion,0,preferred === 999 ? 0 : preferred);
      shiftEvents(insertion+1,1,false);
      port.selected=insertion+1;
    },'타일 추가');
  }
  function deleteTile() {
    if (!port.doc || port.selected===0) { status('시작 타일 0은 삭제할 수 없습니다.','error'); return; }
    const index=port.selected;
    const attached=['actions','decorations'].flatMap(k=>Array.isArray(port.doc[k])?port.doc[k].filter(ev=>Number(ev.floor)===index):[]);
    if (attached.length && !confirm('타일 '+index+'에 연결된 이벤트/장식 '+attached.length+'개도 함께 삭제됩니다. 계속할까요?')) return;
    applyMutation(() => { editedAngles().splice(index-1,1); shiftEvents(index,-1,true); port.selected=index-1; },'타일 '+index+' 삭제');
  }
  function newChart() {
    if (port.dirty && !confirm('저장하지 않은 수정 사항을 버리고 새 차트를 만들까요?')) return;
    port.doc={angleData:[0,90,180,270],settings:{version:15,bpm:120,offset:0},actions:[],decorations:[]};
    port.filename='new-level.adofai'; port.selected=0;port.undo=[];port.redo=[];port.dirty=false;
    rebuildPositions();render();status('새 빈 차트 생성 · PC 원본 게임의 에셋은 포함되지 않습니다.');
  }
  async function openFile(file) {
    if (!file) return;
    if(port.dirty && !confirm('현재 변경 사항을 저장하지 않았습니다. 다른 차트를 열까요?')) return;
    try {
      const data = parseChart(await file.text());
      // Verify before replacing loaded document.
      const angles=Array.isArray(data.angleData)?anglesFromData(data.angleData):anglesFromPath(data.pathData);
      if(angles.length>50000)throw Error('50,000개 초과 타일은 현재 브라우저 UI에서 열 수 없습니다.');
      port.doc=data;port.filename=file.name;port.selected=0;port.dirty=false;port.undo=[];port.redo=[];
      rebuildPositions();render();status(file.name+' · '+(angles.length+1)+'개 타일 불러오기 완료');
    } catch (e) { status('파일 열기 실패: '+e.message,'error'); }
  }
  function saveChart(as=false) {
    if(!port.doc) return;
    let filename=port.filename||'level.adofai';
    if(as) { const proposed=prompt('저장할 .adofai 파일 이름',filename);if(proposed===null)return;filename=proposed.trim()||filename; }
    if(!/\.adofai$/i.test(filename)) filename += '.adofai';
    try {
      const data=JSON.stringify(port.doc,null,'\t')+'\n';
      const uri=URL.createObjectURL(new Blob([data],{type:'application/json;charset=utf-8'}));
      const a=document.createElement('a');a.href=uri;a.download=filename;document.body.append(a);a.click();a.remove();
      setTimeout(()=>URL.revokeObjectURL(uri),15000);
      port.filename=filename;port.dirty=false;render();status('저장 파일 생성: '+filename+' · 실제 Unity 플레이 검증과는 별개입니다.');
    } catch(e) {status('저장 실패: '+e.message,'error');}
  }
  function undo() {
    if (!port.undo.length) return;
    port.redo.push({doc:clone(port.doc),selection:port.selected});
    const p=port.undo.pop();port.doc=p.doc;port.selected=p.selection;port.dirty=true;rebuildPositions();render();status('되돌리기');
  }
  function redo() {
    if(!port.redo.length)return;
    port.undo.push({doc:clone(port.doc),selection:port.selected});
    const p=port.redo.pop();port.doc=p.doc;port.selected=p.selection;port.dirty=true;rebuildPositions();render();status('다시 실행');
  }
  function selectFloor(index) {port.selected=clamp(index,0,port.angles.length);render();}
  function renderChart() {
    const svg=$('chartSvg');if(!svg)return;
    if(!port.doc){svg.innerHTML='<text x="20" y="40" fill="#a9bcd1" font-size="18">원본 PC 씬 버튼으로 .adofai 차트를 열어주세요.</text>';svg.setAttribute('viewBox','0 0 860 470');return;}
    const tiles=port.positions;
    const focus=tiles[port.selected]||{x:0,y:0};
    let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
    // Keep drawings bounded on massive charts without hiding off-screen count.
    const visible=tiles.length<=2500?tiles:tiles.slice(Math.max(0,port.selected-1000),Math.min(tiles.length,port.selected+1001));
    visible.forEach(p=>{minX=Math.min(minX,p.x);maxX=Math.max(maxX,p.x);minY=Math.min(minY,p.y);maxY=Math.max(maxY,p.y)});
    const unit=54,margin=66;let x0=minX*unit-margin,y0=minY*unit-margin,w=Math.max((maxX-minX)*unit+margin*2,580),h=Math.max((maxY-minY)*unit+margin*2,380);
    svg.setAttribute('viewBox',[x0,y0,w,h].join(' '));
    const sx=p=>p.x*unit,sy=p=>p.y*unit;
    const points=visible.map(p=>sx(p)+','+sy(p)).join(' ');
    let html='<polyline points="'+points+'" stroke="#5d81aa" stroke-width="10" stroke-linejoin="round" stroke-linecap="round" fill="none" opacity=".70"/>';
    const eventFloors=new Set(['actions','decorations'].flatMap(k=>Array.isArray(port.doc[k])?port.doc[k].map(e=>Number(e.floor)):[]));
    visible.forEach(p=>{
      const selected=p.index===port.selected;
      const r=selected?26:20;
      const mid=p.index>0&&port.angles[p.index-1]===999;
      const fill=selected?'#f6b956':mid?'#9e76db':'#4d769e';
      html+='<g class="chart-floor" data-floor="'+p.index+'" tabindex="0" role="button" aria-label="타일 '+p.index+'">'+
        '<rect x="'+(sx(p)-r)+'" y="'+(sy(p)-r)+'" width="'+r*2+'" height="'+r*2+'" rx="5" fill="'+fill+'" stroke="'+(selected?'#fff0cb':'#c7e8ff')+'" stroke-width="2"/>'+
        '<text x="'+sx(p)+'" y="'+(sy(p)+4)+'" text-anchor="middle" font-size="12" font-weight="700" fill="#081522">'+p.index+'</text>'+
        (eventFloors.has(p.index)?'<circle cx="'+(sx(p)+r-3)+'" cy="'+(sy(p)-r+3)+'" r="6" fill="#f472b6" stroke="#ffffff" stroke-width="1"/>':'')+'</g>';
    });
    svg.innerHTML=html;
    for(const g of svg.querySelectorAll('[data-floor]')){
      const id=Number(g.dataset.floor);
      g.addEventListener('click',()=>selectFloor(id));
      g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectFloor(id)}});
    }
    $('chartSelection').textContent='타일 '+port.selected+' / '+(tiles.length-1)+' · 각도: '+(port.selected===0?'시작':port.angles[port.selected-1]+'°')+(tiles.length>2500?' · 주변 2,001개만 렌더링':'');
    $('chartAngle').value=port.selected===0?'':String(port.angles[port.selected-1]);
    $('chartAngle').disabled=port.selected===0;
    $('chartDelete').disabled=port.selected===0;
  }
  function renderEventEditor() {
    if(!port.doc)return;
    const list=[];
    for(const group of ['actions','decorations']){
      if(!Array.isArray(port.doc[group]))continue;
      port.doc[group].forEach((ev,index)=>{if(Number(ev?.floor)===port.selected)list.push({group,index,ev})});
    }
    const host=$('chartEventList');
    host.innerHTML=list.length ? list.map((item,n)=>'<div class="chart-event-row"><button data-event="'+n+'">'+htmlEscape(item.group==='actions'?'이벤트':'장식')+' #'+item.index+' · '+htmlEscape(item.ev?.eventType||'알 수 없음')+'</button></div>').join(''):'<p class="chart-help">선택한 타일에 이벤트가 없습니다.</p>';
    host.querySelectorAll('button[data-event]').forEach(b=>b.onclick=()=>{
      const item=list[Number(b.dataset.event)];
      $('chartEventJson').value=JSON.stringify(item.ev,null,2);$('chartEventIndex').value=item.group+':'+item.index;
    });
    const current=$('chartEventIndex').value;
    if(!list.some(i=>i.group+':'+i.index===current)){$('chartEventIndex').value='';$('chartEventJson').value='';}
  }
  function eventApply() {
    if(!port.doc)return;
    const key=$('chartEventIndex').value;
    if(!key){status('수정할 이벤트를 먼저 선택하세요.','error');return;}
    let data;
    try {data=JSON.parse(repairJson($('chartEventJson').value));if(!data||typeof data!=='object'||Array.isArray(data))throw Error('이벤트는 JSON 객체여야 합니다.');}
    catch(e){status('이벤트 JSON 오류: '+e.message,'error');return;}
    const [group,indexString]=key.split(':');const idx=Number(indexString);
    if(!['actions','decorations'].includes(group)||!port.doc[group]?.[idx]){status('이벤트를 찾을 수 없습니다.','error');return;}
    if(!Number.isInteger(data.floor)||data.floor<0||data.floor>port.angles.length){status('event.floor는 존재하는 타일 번호여야 합니다.','error');return;}
    applyMutation(()=>{port.doc[group][idx]=data},'이벤트 수정');
  }
  function eventAdd() {
    if(!port.doc)return;
    const type=prompt('추가할 이벤트 타입(eventType)을 입력하세요.','Twirl');
    if(type===null)return;
    const name=type.trim();if(!name){status('이벤트 타입은 비어 있을 수 없습니다.','error');return;}
    applyMutation(()=>{(port.doc.actions ||= []).push({floor:port.selected,eventType:name})},'이벤트 '+name+' 추가');
  }
  function eventDelete() {
    if(!port.doc)return;
    const key=$('chartEventIndex').value;if(!key){status('삭제할 이벤트를 선택하세요.','error');return;}
    const [group,idxString]=key.split(':'),idx=Number(idxString);
    if(!['actions','decorations'].includes(group)||!port.doc[group]?.[idx])return;
    if(!confirm('이벤트를 삭제할까요?'))return;
    applyMutation(()=>{port.doc[group].splice(idx,1);$('chartEventIndex').value='';$('chartEventJson').value=''},'이벤트 삭제');
  }
  function applySettings() {
    if(!port.doc)return;
    try{
      const value=JSON.parse(repairJson($('chartSettingsJson').value));
      if(!value||Array.isArray(value)||typeof value!=='object')throw Error('settings는 JSON 객체여야 합니다.');
      applyMutation(()=>{port.doc.settings=value},'설정 수정');
    }catch(e){status('설정 JSON 오류: '+e.message,'error')}
  }
  function applyRaw() {
    if(!port.doc)return;
    let data;
    try{data=parseChart($('chartRawJson').value);
      if(Array.isArray(data.angleData))anglesFromData(data.angleData);else anglesFromPath(data.pathData);
    }catch(e){status('전체 JSON 오류: '+e.message,'error');return;}
    applyMutation(()=>{port.doc=data;port.selected=clamp(port.selected,0,(data.angleData?.length??data.pathData.length))},'전체 JSON 수정');
  }
  function renderSide() {
    if(!port.doc){$('chartEventList').innerHTML='';return;}
    renderEventEditor();
    if(port.focus==='settings')$('chartSettingsJson').value=JSON.stringify(port.doc.settings||{},null,2);
    if(port.focus==='raw')$('chartRawJson').value=JSON.stringify(port.doc,null,2);
  }
  function changeTab(which) {
    port.focus=which;
    for(const key of ['events','settings','raw']){
      $('chartPanel'+key).hidden=key!==which;
      const btn=$('chart'+key[0].toUpperCase()+key.slice(1)+'Tab');if(btn)btn.setAttribute('aria-pressed',String(key===which));
    }
    renderSide();
  }
  function render() {
    if(!$('chartSvg'))return;
    $('chartFilename').textContent=port.doc?port.filename+(port.dirty?' *':''):'차트가 열리지 않았습니다.';
    $('chartSave').disabled=!port.doc||originalId('buttonSave')===null;
    $('chartSaveAs').disabled=!port.doc||originalId('buttonSaveAs')===null;
    $('chartAdd').disabled=!port.doc||originalId('AddTile')===null;
    $('chartUndo').disabled=!port.undo.length;
    $('chartRedo').disabled=!port.redo.length;
    renderChart();renderSide();
  }
  function createUi() {
    const host=document.createElement('section');host.id='chartPort';host.className='chart-port';
    host.innerHTML='<div class="chart-toolbar"><div class="chart-brand">SCN EDITOR <span>PC scene-linked chart port · 원본 동작 일부 이식</span></div><div class="chart-buttons">'+
      '<button id="chartOpen">원본 buttonOpen</button><button id="chartSave">원본 buttonSave</button><button id="chartSaveAs">Save As</button><button id="chartNew">새 차트</button><button id="chartAdd">원본 AddTile</button><button id="chartDelete">삭제</button><button id="chartUndo">↶ 되돌리기</button><button id="chartRedo">↷ 다시</button><button id="chartViewer">원본 씬 보기</button>'+
      '</div><input type="file" id="chartFile" accept=".adofai,.json,application/json" hidden><div id="chartOriginal" class="chart-original">원본 씬 연결 확인 중...</div></div>'+
      '<div id="chartWorkspace"><div class="chart-stage"><div class="chart-stage-head"><strong id="chartFilename">차트를 열어주세요</strong><span id="chartSelection">타일 선택</span></div><svg id="chartSvg" viewBox="0 0 860 470" aria-label="ADOFAI 타일 경로 편집" role="img"></svg><div class="chart-angle-bar"><label>선택 타일 방향 <input id="chartAngle" type="number" min="0" max="999" step="any" placeholder="각도"></label><button id="chartSetAngle">각도 적용</button><small>999 = Midspin · 표시용 경로는 기본 방향만 계산하며 PositionTrack 등 일부 효과는 반영되지 않습니다.</small></div></div>'+
      '<aside class="chart-side"><div class="chart-tabs"><button id="chartEventsTab" aria-pressed="true">이벤트</button><button id="chartSettingsTab">설정</button><button id="chartRawTab">전체 JSON</button></div><div id="chartPanelevents"><div id="chartEventList"></div><input id="chartEventIndex" type="hidden"><textarea id="chartEventJson" spellcheck="false" aria-label="선택한 이벤트 JSON"></textarea><div class="chart-buttons"><button id="chartEventApply">이벤트 적용</button><button id="chartEventAdd">+ 이벤트</button><button id="chartEventDelete">이벤트 삭제</button></div></div><div id="chartPanelsettings" hidden><textarea id="chartSettingsJson" spellcheck="false" aria-label="settings JSON"></textarea><button id="chartSettingsApply">설정 적용</button></div><div id="chartPanelraw" hidden><textarea id="chartRawJson" spellcheck="false" aria-label="전체 차트 JSON"></textarea><button id="chartRawApply">전체 JSON 적용</button></div></aside></div><div id="chartStatus" class="chart-status">원본 씬 API와 연결 중...</div>';
    const info=$('status');info.parentNode.insertBefore(host,info.nextSibling);
    const css=document.createElement('style');css.textContent=`
    .chart-port{border-bottom:1px solid #40576d;background:#101b2b;color:#e3edf8;padding:0 15px 11px}
    .chart-toolbar{padding:12px 3px}.chart-brand{font-size:15px;font-weight:800;letter-spacing:.08em;color:#a9d4ff;margin-bottom:11px}
    .chart-brand span{font-size:11px;font-weight:400;letter-spacing:0;color:#819ab8;margin-left:8px}
    .chart-buttons{display:flex;flex-wrap:wrap;gap:7px}.chart-buttons button,.chart-tabs button,.chart-angle-bar button,.chart-side button,#chartViewer{background:#1e3550;color:#dcecf9;border:1px solid #48627d;border-radius:5px;padding:7px 9px;font-size:12px;cursor:pointer}
    .chart-port button:hover:not(:disabled){background:#335779}.chart-port button:disabled{opacity:.35;cursor:not-allowed}.chart-original{font-size:10px;color:#7091b0;margin-top:9px;overflow-wrap:anywhere}
    #chartWorkspace{display:grid;grid-template-columns:minmax(330px,1fr) minmax(255px,320px);gap:12px}.chart-stage{min-width:0;border:1px solid #34516f;background:#111e30;border-radius:5px}
    .chart-stage-head{display:flex;justify-content:space-between;gap:10px;font-size:12px;padding:11px 13px;border-bottom:1px solid #314b67;color:#c6dcf2}
    #chartSvg{width:100%;height:clamp(300px,43vh,550px);display:block;background:repeating-linear-gradient(0deg,#19263a 0px,#19263a 32px,#203148 33px),#101c2e;touch-action:pan-x pan-y}
    .chart-floor{cursor:pointer}.chart-floor:focus-visible rect{stroke:#fff;stroke-width:5}
    .chart-angle-bar{padding:8px 11px;display:flex;gap:9px;align-items:center;flex-wrap:wrap;border-top:1px solid #314b67;font-size:12px}.chart-angle-bar input{width:88px;padding:6px;color:white;background:#101c2c;border:1px solid #45607e;border-radius:3px}.chart-angle-bar small{color:#9db3c8;line-height:1.5}
    .chart-side{border:1px solid #34516f;border-radius:5px;background:#111e30;min-width:0}.chart-tabs{display:flex;border-bottom:1px solid #34516f}.chart-tabs button{flex:1;border:none;border-radius:0;background:#1a2e45}.chart-tabs button[aria-pressed="true"]{background:#31587f;color:white}
    #chartEventList{max-height:130px;overflow:auto;padding:10px}.chart-event-row button{width:100%;text-align:left;margin-bottom:3px}.chart-side textarea{display:block;width:calc(100% - 18px);min-height:180px;margin:9px;background:#091321;color:#d8e8fa;border:1px solid #4b6884;border-radius:4px;font-family:ui-monospace,Consolas,monospace;font-size:12px;padding:10px;resize:vertical}
    .chart-side .chart-buttons{padding:0 9px 9px}.chart-side #chartPanelsettings,.chart-side #chartPanelraw{padding-bottom:11px}.chart-side #chartSettingsApply,.chart-side #chartRawApply{margin-left:9px}.chart-status{font-size:12px;padding:12px 3px;color:#9fc9f1;min-height:36px}.chart-status[data-kind="error"]{color:#fb9999}.chart-help{font-size:12px;color:#a9bdce}
    @media(max-width:800px){#chartWorkspace{grid-template-columns:1fr}.chart-stage{min-width:0}.chart-side textarea{min-height:180px}.chart-brand span{display:block;margin:4px 0}}
    `;document.head.append(css);
    $('chartFile').addEventListener('change',async e=>{await openFile(e.target.files[0]);e.target.value='';});
    $('chartOpen').onclick=()=>$('chartFile').click();$('chartSave').onclick=()=>saveChart();$('chartSaveAs').onclick=()=>saveChart(true);
    $('chartNew').onclick=newChart;$('chartAdd').onclick=addTile;$('chartDelete').onclick=deleteTile;
    $('chartUndo').onclick=undo;$('chartRedo').onclick=redo;
    $('chartSetAngle').onclick=()=>editAngle($('chartAngle').value);
    $('chartAngle').onkeydown=e=>{if(e.key==='Enter')editAngle($('chartAngle').value)};
    $('chartEventApply').onclick=eventApply;$('chartEventAdd').onclick=eventAdd;$('chartEventDelete').onclick=eventDelete;
    $('chartSettingsApply').onclick=applySettings;$('chartRawApply').onclick=applyRaw;
    $('chartEventsTab').onclick=()=>changeTab('events');$('chartSettingsTab').onclick=()=>changeTab('settings');$('chartRawTab').onclick=()=>changeTab('raw');
    $('chartViewer').onclick=()=>{$('chartPort').hidden=true;document.querySelector('.layout').scrollIntoView({behavior:'smooth'})};
    document.querySelector('.layout').insertAdjacentHTML('beforebegin','<button id="chartReturn" class="action" style="margin:11px 17px">차트 편집기로 돌아가기</button>');
    $('chartReturn').onclick=()=>{$('chartPort').hidden=false;$('chartPort').scrollIntoView({behavior:'smooth'})};
    window.addEventListener('adofai:scene-loaded',()=>{connectOriginalScene();render()});
    if(typeof state!=='undefined'&&state.nodes.length)connectOriginalScene();
    render();
  }
  // Test hooks never persist a chart or its original PC bytes to any server.
  window.ADOFAIChartPort={parseChart,anglesFromPath,anglesFromData,get state(){return port},loadObject(obj,name='test.adofai'){port.doc=assertChart(clone(obj));port.filename=name;port.selected=0;port.dirty=false;port.undo=[];port.redo=[];rebuildPositions();render()},saveObject(){return port.doc?clone(port.doc):null},selectFloor,editAngle,addTile,deleteTile,undo,redo};
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',createUi);else createUi();
})();