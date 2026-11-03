/* Web ADOFAI native single-origin addon.
 * The ORIGINAL game's JS, search, tile rendering, audio and play button run in
 * the current document. This script only provides local custom-level workflows.
 */
(() => {
'use strict';
const DB='adofai-web-editor-level-packages';
const ORIGIN_TAG='WEB ADOFAI · Native Custom';
const MAX_ZIP=250*1024*1024,MAX_UNPACKED=450*1024*1024;
const st={id:null,record:null,doc:null,original:null,dirty:false,history:[],redo:[],floor:1,busy:false,tab:'tile',assets:[],open:false};
let ui,shadow;
const $=id=>shadow?.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]||'&#39;'));
const clone=x=>JSON.parse(JSON.stringify(x));
const clamp=(n,a,b)=>Math.max(a,Math.min(n,b));
const key=id=>id+':'+Date.now()+':'+Math.random().toString(36).slice(2);
function say(t,error=false){const e=$('status');if(e){e.textContent=t;e.dataset.error=String(error)}}
function gameRoute(){return /^\/levels\/cache\//.test(location.pathname)}
function currentId(){if(!gameRoute())return null;try{return decodeURIComponent(location.pathname.split('/')[3]||'')}catch{return null}}
function line(msg){try{console.log('[ADOFAI native studio]',msg)}catch{}}
function chooseMode(active){
 st.open=active;
 $('drawer').hidden=!active;
 $('editorButton').setAttribute('aria-pressed',String(active));
 if(active){void loadCurrent()}
}
function validDoc(doc){
 if(!doc||typeof doc!=='object'||Array.isArray(doc))throw Error('ADOFAI JSON 객체가 아닙니다.');
 if(!Array.isArray(doc.angleData)&&typeof doc.pathData!=='string')throw Error('angleData 또는 pathData가 필요합니다.');
 if(doc.angleData&&!Array.isArray(doc.angleData))throw Error('angleData 배열 형식이 잘못됐습니다.');
 if(doc.angleData&&!doc.angleData.every(v=>typeof v==='number'&&Number.isFinite(v)))throw Error('각도 데이터가 잘못됐습니다.');
 for(const name of ['actions','decorations'])if(doc[name]&&!Array.isArray(doc[name]))throw Error(name+' 배열 형식 오류');
 return doc;
}
function tolerantParse(raw){
 try{return validDoc(JSON.parse(raw.replace(/^\uFEFF/,'')))}
 catch(e){if(!/Unexpected|JSON|token|control|property/i.test(String(e)))throw e;
   const fixed=raw.replace(/^\uFEFF/,'').replace(/,\s*([}\]])/g,'$1');
   return validDoc(JSON.parse(fixed));
 }
}
function norm(a){return ((a%360)+360)%360}
const dir={R:0,p:15,J:30,E:45,T:60,o:75,U:90,q:105,G:120,Q:135,H:150,W:165,L:180,x:195,N:210,Z:225,F:240,V:255,D:270,Y:285,B:300,C:315,M:330,A:345,'!':999};
const relative={'5':72,'6':-72,'7':52,'8':-52,'9':-30,h:120,j:-120,t:60,y:300};
function angles(doc=st.doc){
 if(!doc)return[];
 if(Array.isArray(doc.angleData))return doc.angleData.map(Number);
 let last=0;const a=[];
 for(const c of doc.pathData){if(Object.hasOwn(dir,c)){last=dir[c];a.push(last)}
   else if(Object.hasOwn(relative,c)){last=norm(last+relative[c]);a.push(last)}
   else throw Error('미지원 pathData 문자: '+c)}
 return a;
}
function explicitAngles(){if(!st.doc.angleData){st.doc.angleData=angles();delete st.doc.pathData}return st.doc.angleData}
function mutate(fn,message){
 if(!st.doc)return;
 const old=clone(st.doc),floor=st.floor;
 try{
   fn();validDoc(st.doc);angles(st.doc);
   st.history.push({doc:old,floor});if(st.history.length>30)st.history.shift();st.redo.length=0;
   st.dirty=true;render();say(message+' · 적용 버튼을 누르면 원본 게임에 반영됩니다.');
 }catch(e){st.doc=old;st.floor=floor;render();say(e.message,true)}
}
function length(){return st.doc?angles().length+1:0}
function moveAttached(from,delta,deleteAt=false){
 for(const listName of ['actions','decorations']){
   if(!Array.isArray(st.doc[listName]))continue;
   st.doc[listName]=st.doc[listName].filter(x=>!deleteAt||x?.floor!==from).map(x=>{
     if(x&&Number.isInteger(x.floor)&&x.floor>=from+(deleteAt?1:0))return {...x,floor:x.floor+delta};
     return x;
   });
 }
}
function addTile(){
 mutate(()=>{
  const arr=explicitAngles(),at=clamp(st.floor,0,arr.length);
  const v=arr[Math.max(0,Math.min(arr.length-1,at-1))]??0;
  arr.splice(at,0,v===999?0:v);moveAttached(at+1,1);st.floor=at+1;
 },'타일 추가');
}
function removeTile(){
 if(st.floor<=0)return say('시작 타일 0은 삭제할 수 없습니다.',true);
 const id=st.floor;
 const attached=['actions','decorations'].reduce((n,x)=>n+(st.doc[x]||[]).filter(a=>a.floor===id).length,0);
 if(attached&&!confirm('타일 '+id+'에 연결된 이벤트/장식 '+attached+'개도 삭제할까요?'))return;
 mutate(()=>{explicitAngles().splice(id-1,1);moveAttached(id,-1,true);st.floor=Math.max(0,id-1)},'타일 삭제');
}
function setAngle(){
 const raw=$('angle').value.trim(),v=Number(raw);
 if(!raw||!Number.isFinite(v)||(v!==999&&(v<0||v>=360)))return say('각도는 0~359 또는 미드스핀 999입니다.',true);
 if(st.floor===0)return say('시작 타일에는 진행 방향이 없습니다.',true);
 mutate(()=>{explicitAngles()[st.floor-1]=v},'타일 '+st.floor+' 각도 변경');
}
function setSettings(){
 const bpm=Number($('bpm').value),offset=Number($('offset').value);
 if(!Number.isFinite(bpm)||bpm<=0||!Number.isFinite(offset))return say('올바른 BPM 및 오프셋을 입력하세요.',true);
 mutate(()=>{st.doc.settings??={};Object.assign(st.doc.settings,{bpm,offset,song:$('song').value,artist:$('artist').value,songFilename:$('songfile').value})},'레벨 설정 변경');
}
function setEvent(){
 const idx=Number($('events').value),arr=st.doc.actions??[];
 if(!Number.isInteger(idx)||idx<0||idx>=arr.length)return say('이벤트를 선택하세요.',true);
 let obj;try{obj=JSON.parse($('eventJson').value)}catch(e){return say('이벤트 JSON 오류: '+e.message,true)}
 if(!obj||typeof obj!=='object'||Array.isArray(obj))return say('이벤트는 JSON 객체여야 합니다.',true);
 if(!Number.isInteger(obj.floor)||obj.floor<0||obj.floor>=length())return say('이벤트 floor 번호가 잘못됐습니다.',true);
 mutate(()=>{st.doc.actions[idx]=obj},'이벤트 수정');
}
function addEvent(){
 const type=prompt('추가할 이벤트 종류(eventType)','Twirl');
 if(!type)return;
 mutate(()=>{st.doc.actions??=[];st.doc.actions.push({floor:st.floor,eventType:type.trim()})},'이벤트 추가');
}
function deleteEvent(){
 const idx=Number($('events').value);if(!Number.isInteger(idx)||idx<0||idx>=(st.doc.actions?.length||0))return;
 mutate(()=>{st.doc.actions.splice(idx,1)},'이벤트 삭제');
}
function editDecoration(){
 const idx=Number($('decorations').value),arr=st.doc.decorations||[];
 if(!Number.isInteger(idx)||idx<0||idx>=arr.length)return say('장식을 선택하세요.',true);
 let entry;try{entry=JSON.parse($('decorationJson').value)}catch(e){return say('장식 JSON 오류: '+e.message,true)}
 if(!entry||typeof entry!=='object'||Array.isArray(entry)||!Number.isInteger(entry.floor)||entry.floor<0||entry.floor>=length())return say('올바른 장식 floor 값이 필요합니다.',true);
 mutate(()=>{st.doc.decorations[idx]=entry},'장식 수정');
}
function addDecoration(){
 mutate(()=>{st.doc.decorations??=[];st.doc.decorations.push({floor:st.floor,eventType:'AddDecoration',decorationImage:'',position:[0,0],scale:[100,100],rotation:0})},'장식 추가');
}
function deleteDecoration(){
 const idx=Number($('decorations').value);
 if(!Number.isInteger(idx)||idx<0||idx>=(st.doc.decorations||[]).length)return;
 mutate(()=>{st.doc.decorations.splice(idx,1)},'장식 삭제');
}
function applyJson(){
 let d;try{d=tolerantParse($('fullJson').value);angles(d)}catch(e){return say('차트 JSON 오류: '+e.message,true)}
 mutate(()=>{st.doc=d;st.floor=clamp(st.floor,0,angles(d).length)},'차트 JSON 수정');
}
function undoRedo(redo=false){
 const from=redo?st.redo:st.history,to=redo?st.history:st.redo;
 if(!from.length)return;
 to.push({doc:clone(st.doc),floor:st.floor});
 const p=from.pop();st.doc=p.doc;st.floor=p.floor;st.dirty=true;render();say(redo?'다시 실행':'실행 취소');
}
function safePath(p){const v=String(p||'').replace(/\\/g,'/').replace(/^\/+/,'');if(!v||v.includes('\0')||v.split('/').some(s=>s==='..'||s==='.')||v.startsWith('__MACOSX/'))return null;return v}
function mime(name){
 const ext=name.split('.').pop().toLowerCase();
 return ({adofai:'application/json',json:'application/json',ogg:'audio/ogg',mp3:'audio/mpeg',wav:'audio/wav',flac:'audio/flac',png:'image/png',jpg:'image/jpeg',jpeg:'image/jpeg',webp:'image/webp'})[ext]||'application/octet-stream';
}
function choosePrimary(names){
 const charts=names.filter(n=>/\.adofai$/i.test(n));
 if(!charts.length)throw Error('압축파일 안에 .adofai 차트가 없습니다.');
 charts.sort((a,b)=>{const score=n=>(/backup|autosave|bak|백업/i.test(n)?100:0)+n.split('/').length+(n.length/1000);return score(a)-score(b)});
 return charts[0];
}
function readDB(){
 return new Promise((resolve,reject)=>{
  const req=indexedDB.open(DB,1);
  req.onupgradeneeded=()=>{const db=req.result;if(!db.objectStoreNames.contains('packages'))db.createObjectStore('packages',{keyPath:'packageId'});if(!db.objectStoreNames.contains('fileBlobs'))db.createObjectStore('fileBlobs')};
  req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error);
 })
}
async function transactionRead(id){
 const db=await readDB();
 try{return await new Promise((resolve,reject)=>{
   const tx=db.transaction(['packages','fileBlobs'],'readonly'),store=tx.objectStore('packages'),q=store.get(id);
   q.onsuccess=()=>{
     const r=q.result;if(!r){resolve(null);return}
     const list=r.files||[],entries=[];
     for(const file of list){const get=tx.objectStore('fileBlobs').get(file.blobKey);get.onsuccess=()=>entries.push({meta:file,blob:get.result})}
     tx.oncomplete=()=>resolve({record:r,entries});
   };
   tx.onerror=()=>reject(tx.error);
 })}finally{db.close()}
}
async function storePackage({files,primary,displayName}){
 if(st.busy)throw Error('파일 작업이 진행 중입니다.');
 st.busy=true;
 try{
  const now=Date.now(),id='local:'+crypto.randomUUID(),entries=[];
  for(const file of files){
   const name=safePath(file.path);if(!name)continue;
   const blob=file.blob instanceof Blob?file.blob:new Blob([file.blob],{type:mime(name)});
   const blobKey=id+':'+now+':'+name;
   entries.push({blob,meta:{relativePath:name,name:name.split('/').at(-1),type:mime(name),size:blob.size,sourcePath:name,storage:'indexeddb',blobKey}});
  }
  const meta=entries.find(x=>x.meta.relativePath===primary);
  if(!meta)throw Error('원본 차트가 ZIP에서 발견되지 않았습니다.');
  const text=await meta.blob.text();validDoc(tolerantParse(text));
  const base=primary.slice(0,primary.lastIndexOf('/'));
  const record={version:1,packageId:id,source:'local',displayName:displayName||base||'Custom level',targetLevelPath:primary,levelBaseDir:base,files:entries.map(x=>x.meta),fileCount:entries.length,totalBytes:entries.reduce((n,x)=>n+x.blob.size,0),createdAt:now,updatedAt:now};
  const db=await readDB();
  try{await new Promise((resolve,reject)=>{
    const tx=db.transaction(['packages','fileBlobs'],'readwrite');
    const s=tx.objectStore('fileBlobs');for(const e of entries)s.put(e.blob,e.meta.blobKey);
    tx.objectStore('packages').put(record);
    tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);
  })}finally{db.close()}
  localStorage.setItem('adofaiStudio:lastPackage',id);
  sessionStorage.setItem('adofaiStudio:flash','불러오기 완료 · '+(angles(tolerantParse(text)).length+1)+'개 타일 · 파일 '+entries.length+'개');
  sessionStorage.setItem('adofaiStudio:autoEditor','1');
  location.assign('/levels/cache/'+encodeURIComponent(id));
 }finally{st.busy=false}
}
async function openFiles(files,folder=false){
 if(st.busy||!files?.length)return;
 try{
  const list=Array.from(files);let entries=[],primary='';
  const zip=list.find(x=>/\.zip$/i.test(x.name));
  if(zip){
   if(zip.size>MAX_ZIP)throw Error('250MB 이하 ZIP만 지원합니다.');
   if(!window.JSZip)throw Error('ZIP 라이브러리를 불러오지 못했습니다.');
   say('ZIP 압축을 분석하는 중…');
   const archive=await JSZip.loadAsync(await zip.arrayBuffer(),{checkCRC32:true});
   const all=Object.values(archive.files).filter(x=>!x.dir&&safePath(x.name));
   if(all.length>3000)throw Error('파일이 너무 많습니다.');
   const size=all.reduce((n,x)=>n+(x._data?.uncompressedSize||0),0);
   if(size>MAX_UNPACKED)throw Error('압축 해제 크기가 450MB를 초과합니다.');
   primary=choosePrimary(all.map(x=>x.name));
   for(const file of all){
    const path=safePath(file.name);
    const blob=new Blob([await file.async('uint8array')],{type:mime(path)});
    entries.push({path,blob});
   }
  }else{
   for(const f of list){
    const path=safePath((folder?f.webkitRelativePath:'')||f.name);
    if(path)entries.push({path,blob:f});
   }
   primary=choosePrimary(entries.map(x=>x.path));
  }
  if(!entries.length)throw Error('유효한 파일이 없습니다.');
  const chart=await entries.find(x=>x.path===primary).blob.text();
  const parsed=tolerantParse(chart),song=parsed.settings?.songFilename||'';
  const display=primary.split('/').slice(0,-1).join('/')||primary.split('/').pop().replace(/\.adofai$/i,'');
  // Flat charts are normalized to their own folder so relative song paths resolve.
  if(!primary.includes('/')){
   const prefix='CustomMap/';
   entries=entries.map(e=>({path:prefix+e.path,blob:e.blob}));primary=prefix+primary;
  }
  if(song&&!entries.some(x=>x.path===primary.slice(0,primary.lastIndexOf('/')+1)+song||x.path.endsWith('/'+song)))line('Referenced song not present: '+song);
  await storePackage({files:entries,primary,displayName:display});
 }catch(err){say('불러오기 실패: '+err.message,true);console.error(err)}
}
async function loadCurrent(){
 const id=currentId();if(!id)return say('먼저 ZIP이나 .adofai를 열어주세요.',true);
 if(st.id===id&&st.doc){render();return}
 try{
  say('원본 게임의 레벨 데이터를 읽는 중…');const pkg=await transactionRead(id);
  if(!pkg)throw Error('원본 게임 캐시에 레벨이 없습니다.');
  const info=pkg.entries.find(x=>x.meta.relativePath===pkg.record.targetLevelPath);
  if(!info?.blob)throw Error('원본 게임에서 차트 파일을 찾지 못했습니다.');
  const doc=tolerantParse(await info.blob.text());
  st.id=id;st.record=pkg.record;st.doc=doc;st.original=clone(doc);st.assets=pkg.entries;st.floor=1;st.history=[];st.redo=[];st.dirty=false;
  render();say('원본 게임 차트 로딩 완료 · '+length()+'타일 · 파일 '+pkg.entries.length+'개');
 }catch(e){say('차트 읽기 실패: '+e.message,true)}
}
async function saveToNative(reload){
 if(!st.doc||!st.id)return say('차트가 없습니다.',true);
 const pkg=await transactionRead(st.id);
 if(!pkg)return say('레벨 캐시가 없습니다.',true);
 const rec=pkg.record,idx=rec.files.findIndex(x=>x.relativePath===rec.targetLevelPath);
 if(idx<0)return say('차트 경로가 없습니다.',true);
 try{
  const str=JSON.stringify(st.doc,null,'\t')+'\n';
  const file=rec.files[idx],blob=new Blob([str],{type:'application/json'});
  file.size=blob.size;file.type='application/json';rec.totalBytes=rec.files.reduce((n,x)=>n+x.size,0);rec.updatedAt=Date.now();
  const db=await readDB();
  try{await new Promise((resolve,reject)=>{
   const tx=db.transaction(['packages','fileBlobs'],'readwrite');
   tx.objectStore('fileBlobs').put(blob,file.blobKey);
   tx.objectStore('packages').put(rec);
   tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);
  })}finally{db.close()}
  st.record=rec;st.dirty=false;st.original=clone(st.doc);say('원본 게임 저장소에 변경 사항 저장 완료');
  if(reload){sessionStorage.setItem('adofaiStudio:flash','차트 변경 사항이 원본 게임에 적용되었습니다. 아래 원본 Play로 플레이하세요.');location.reload()}
 }catch(e){say('적용 실패: '+e.message,true)}
}
async function exportZip(){
 if(!st.id)return say('열린 커스텀 레벨이 없습니다.',true);
 if(!window.JSZip)return say('ZIP 라이브러리 로드 실패',true);
 try{
  const pkg=await transactionRead(st.id);
  if(!pkg)throw Error('레벨 저장소를 찾을 수 없습니다.');
  const zip=new JSZip();
  for(const f of pkg.entries){
   const path=safePath(f.meta.relativePath);if(!path||!f.blob)continue;
   if(path===pkg.record.targetLevelPath&&st.doc)zip.file(path,JSON.stringify(st.doc,null,'\t')+'\n');
   else zip.file(path,f.blob);
  }
  say('음악·이미지·백업을 보존해 ZIP 생성 중…');
  const blob=await zip.generateAsync({type:'blob',compression:'DEFLATE',compressionOptions:{level:6}});
  const name=(pkg.record.displayName||'custom-map').replace(/[\\/:*?"<>|]/g,'-')+'.zip';
  download(blob,name);
  say('ZIP 저장 완료 · '+name+' · 기존 에셋 파일 보존');
 }catch(e){say('ZIP 저장 실패: '+e.message,true)}
}
function download(blob,name){const u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),10000)}
function downloadChart(){
 if(!st.doc)return;
 const filename=(st.record?.targetLevelPath||'level.adofai').split('/').pop();
 download(new Blob([JSON.stringify(st.doc,null,'\t')+'\n'],{type:'application/json'}),filename);
}
function rendererIssue(){
 if(!gameRoute())return '';
 if(!navigator.gpu)return '현재 브라우저에서 WebGPU를 사용할 수 없어 원본 게임 렌더러가 실행되지 않습니다.';
 const txt=(document.body?.innerText||'').slice(-5500);
 if(/WebGPU 장치 연결이 끊겨|device was lost|context is not configured|No compatible WebGPU adapter|WebGPU.*지원하지/i.test(txt))
  return '원본 Web ADOFAI의 WebGPU 렌더러가 중단됐습니다. PC·모바일 모두 이 문제가 발생하면 실제 플레이가 되지 않습니다.';
 return '';
}
function monitorRenderer(){
 const msg=rendererIssue(),badge=$('rendererIssue');
 if(badge){badge.hidden=!msg;badge.textContent=msg}
}
async function nativePlay(){
 if(!gameRoute())return say('먼저 커스텀 레벨 ZIP을 불러오세요.',true);
 monitorRenderer();const fault=rendererIssue();
 if(fault)return say(fault+' 자동 판정이 아닌 직접 플레이도 이 상태에서는 시작되지 않습니다.',true);
 if(st.dirty){sessionStorage.setItem('adofaiStudio:manualPlayRequested','1');await saveToNative(true);return;}
 chooseMode(false);
 const buttons=Array.from(document.querySelectorAll('button'));
 const b=buttons.find(x=>x.getAttribute('aria-label')==='Play'||x.title==='Play'||x.innerText.trim()==='Play');
 if(!b)return say('원본 게임이 준비되지 않았습니다. Play 버튼이 나타나야 직접 플레이할 수 있습니다.',true);
 b.click();say('직접 플레이 진입 요청 · 판정 입력은 키보드 스페이스 또는 화면 터치로 하세요.');
 setTimeout(()=>{
  const issue=rendererIssue();
  if(issue){monitorRenderer();say('플레이 시작 실패: '+issue,true);return}
  const control=Array.from(document.querySelectorAll('button')).find(x=>x.getAttribute('aria-label')==='Play'||x.title==='Play');
  if(control)say('원본 게임이 편집 모드로 되돌아왔습니다. 현재 판정 플레이가 시작되지 않았습니다.',true);
 },4800);
}
function render(){
 if(!st.doc)return;
 const n=length();
 st.floor=clamp(st.floor,0,n-1);
 $('name').textContent=st.record?.displayName||'Custom level';
 $('tileCount').textContent=n+' tiles';
 $('floor').max=String(n-1);$('floor').value=String(st.floor);
 $('angle').disabled=st.floor===0;$('angle').value=st.floor===0?'':String(angles()[st.floor-1]);
 const settings=st.doc.settings||{};
 for(const [id,value] of Object.entries({bpm:settings.bpm??120,offset:settings.offset??0,song:settings.song??'',artist:settings.artist??'',songfile:settings.songFilename??''}))$(id).value=String(value);
 const act=st.doc.actions||[];
 const select=$('events');
 const remembered=select.value;
 select.innerHTML=act.map((e,i)=>'<option value="'+i+'">'+i+': '+esc(e.eventType||'?')+' · floor '+esc(e.floor)+'</option>').join('');
 if(act.length){
  select.value=remembered&&Number(remembered)<act.length?remembered:'0';
  $('eventJson').value=JSON.stringify(act[Number(select.value)],null,2);
 }else $('eventJson').value='';
 const deco=st.doc.decorations||[],dSelect=$('decorations'),lastDecoration=dSelect.value;
 dSelect.innerHTML=deco.map((e,i)=>'<option value="'+i+'">'+i+': '+esc(e.eventType||'Decoration')+' · floor '+esc(e.floor)+'</option>').join('');
 if(deco.length){dSelect.value=lastDecoration&&Number(lastDecoration)<deco.length?lastDecoration:'0';$('decorationJson').value=JSON.stringify(deco[Number(dSelect.value)],null,2)}
 else $('decorationJson').value='';
 $('fullJson').value=JSON.stringify(st.doc,null,2);
 $('assetList').innerHTML=st.assets.map(x=>'<div>'+esc(x.meta.relativePath)+' <span>'+Math.round(x.meta.size/1024)+'KB</span></div>').join('');
 $('dirty').textContent=st.dirty?'● 수정됨':'✓ 저장됨';
 $('undo').disabled=!st.history.length;$('redo').disabled=!st.redo.length;
 $('saveZip').disabled=!st.id;$('exportChart').disabled=!st.doc;
}
function setTab(tab){
 st.tab=tab;
 for(const t of ['tile','events','decorations','settings','files','json']){$('tab-'+t).hidden=t!==tab;$('nav-'+t).dataset.active=String(t===tab)}
}
function build(){
 const host=document.createElement('div');host.id='adofai-native-studio-host';
 document.body.append(host);shadow=host.attachShadow({mode:'open'});
 shadow.innerHTML='<style>'+
 ':host{all:initial}*{box-sizing:border-box}button,input,textarea,select{font:inherit}button{cursor:pointer}'+
 '.bar{pointer-events:auto;position:fixed;bottom:12px;right:12px;z-index:2147483600;display:flex;gap:6px;padding:7px;border:1px solid #52758d;border-radius:11px;background:#111b29ec;box-shadow:0 6px 20px #0009;font:12px system-ui,sans-serif;color:#ecf7ff}'+
 '.bar button,.drawer button{background:#283c50;border:1px solid #55718a;color:#eaf5ff;border-radius:6px;padding:7px 9px;min-height:33px}.bar button:hover,.drawer button:hover{background:#396081}.bar button[aria-pressed=true]{background:#386d90}'+
 '.drawer{position:fixed;pointer-events:auto;z-index:2147483601;right:12px;bottom:71px;width:min(420px,calc(100vw - 24px));max-height:calc(100dvh - 98px);overflow:auto;padding:15px;border:1px solid #607e99;border-radius:12px;background:#111b29f8;color:#ecf4fc;box-shadow:0 7px 30px #000b;font:13px system-ui,sans-serif}.drawer[hidden]{display:none}.drawer header{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:11px}.drawer header strong{font-size:15px}.muted{color:#9fb7cc;font-size:11px}.head-row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.drawer h3{margin:11px 0 7px;font-size:13px}'+
 '.drawer input,.drawer select,.drawer textarea{background:#0a1523;color:#fff;border:1px solid #52718a;border-radius:5px;padding:8px;width:100%;min-width:0}.drawer textarea{min-height:135px;resize:vertical;font:12px/1.45 ui-monospace,monospace}.drawer label{display:block;font-size:11px;color:#b5cbe0;margin:9px 0 3px}.fields{display:grid;grid-template-columns:1fr 1fr;gap:9px}.actions{display:flex;gap:6px;flex-wrap:wrap;margin:9px 0}.actions button{flex:1}.nav{display:flex;gap:5px;overflow-x:auto;padding:6px 0;border-top:1px solid #324b61;border-bottom:1px solid #324b61}.nav button{font-size:11px;min-height:32px;white-space:nowrap;padding:6px 8px}.nav button[data-active=true]{background:#3378a4}.tab{padding-top:10px}.tab[hidden]{display:none}.assets{max-height:180px;overflow:auto}.assets div{font-size:11px;padding:5px 2px;border-bottom:1px solid #344354;word-break:break-all}.assets span{color:#9eafbf;float:right}.pc-menu{display:flex;gap:2px;flex-wrap:wrap;margin:0 0 12px;padding:4px;background:#0d1925;border-bottom:1px solid #456079}.pc-menu button{font-size:11px;padding:5px 7px;min-height:28px;background:transparent;border-color:transparent}.pc-menu button:hover{background:#2e506b}.compat{position:fixed;top:12px;left:50%;transform:translateX(-50%);max-width:min(640px,calc(100vw - 30px));z-index:2147483598;padding:9px 13px;background:#571d24ee;color:#fff;font:12px system-ui;border:1px solid #db6b73;border-radius:7px;text-align:center;pointer-events:none}.compat[hidden]{display:none}.footer{margin-top:12px;display:flex;gap:6px;flex-wrap:wrap}'+
 '#status{margin-top:11px;color:#a9d9bf;line-height:1.5;font-size:11px;min-height:20px}#status[data-error=true]{color:#ff9f9f}.flash{pointer-events:none;position:fixed;z-index:2147483600;left:10px;bottom:71px;max-width:min(400px,calc(100vw - 20px));padding:10px 13px;border-radius:8px;background:#112839ec;color:#e8f5ff;font:12px system-ui,sans-serif;box-shadow:0 5px 16px #0008}'+
 '@media(max-width:570px){.bar{top:7px;bottom:auto;left:auto;right:7px;max-width:210px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px;padding:4px}.bar button{font-size:10px;padding:6px 4px;min-width:0;min-height:30px}.drawer{left:7px;right:7px;top:81px;bottom:7px;width:auto;max-height:calc(100dvh - 88px)}.flash{bottom:8px}}'+
 '</style>'+
 '<div class="bar"><button id="open" type="button">📂 ZIP 열기</button><button id="editorButton" type="button" aria-pressed="false">✎ 편집</button><button id="play" type="button">▶ 직접 플레이</button><button id="saveZip" type="button">💾 ZIP 저장</button></div><div class="compat" id="rendererIssue" hidden></div>'+
 '<input id="fileInput" type="file" accept=".zip,.adofai,.json,.ogg,.mp3,.wav,.png,.jpg,.jpeg" multiple hidden>'+
 '<input id="folderInput" type="file" webkitdirectory directory multiple hidden>'+
 '<div class="drawer" id="drawer" hidden><div class="pc-menu" role="menubar"><button id="menuFile" type="button">FILE</button><button id="menuEdit" type="button">EDIT</button><button id="menuLevel" type="button">LEVEL</button><button id="menuEvents" type="button">EVENTS</button><button id="menuDecorations" type="button">DECORATIONS</button><button id="menuSettings" type="button">SETTINGS</button></div><header><div><strong id="name">커스텀 레벨 편집</strong><div class="muted"><span id="tileCount">—</span> · <span id="dirty">—</span></div></div><button id="close">닫기 ✕</button></header>'+
 '<div class="actions"><button id="folder">📁 폴더</button><button id="undo">↶ 취소</button><button id="redo">↷ 다시</button><button id="exportChart">.adofai</button></div>'+
 '<div class="nav"><button id="nav-tile" data-active="true">타일</button><button id="nav-events">이벤트</button><button id="nav-decorations">장식</button><button id="nav-settings">설정</button><button id="nav-files">파일</button><button id="nav-json">고급</button></div>'+
 '<div class="tab" id="tab-tile"><label>선택 타일 (0은 시작 타일)</label><input id="floor" type="number" min="0" value="1"><label>타일 각도 (0–359, 미드스핀 999)</label><input id="angle" type="number" min="0" max="999"><div class="actions"><button id="setAngle">각도 적용</button><button id="addTile">+ 타일</button><button id="removeTile">타일 삭제</button></div></div>'+
 '<div class="tab" id="tab-events" hidden><label>이벤트 목록</label><select id="events"></select><label>선택한 이벤트의 JSON</label><textarea id="eventJson" spellcheck="false"></textarea><div class="actions"><button id="setEvent">수정</button><button id="addEvent">+ 이벤트</button><button id="deleteEvent">삭제</button></div></div>'+
 '<div class="tab" id="tab-decorations" hidden><label>장식 목록</label><select id="decorations"></select><label>선택한 장식의 JSON</label><textarea id="decorationJson" spellcheck="false"></textarea><div class="actions"><button id="setDecoration">장식 수정</button><button id="addDecoration">+ 장식</button><button id="deleteDecoration">삭제</button></div></div><div class="tab" id="tab-settings" hidden><div class="fields"><div><label>BPM</label><input id="bpm" type="number"></div><div><label>오프셋(ms)</label><input id="offset" type="number"></div></div><label>곡 제목</label><input id="song"><label>아티스트</label><input id="artist"><label>음원 파일명</label><input id="songfile"><div class="actions"><button id="setSettings">설정 적용</button></div></div>'+
 '<div class="tab" id="tab-files" hidden><p class="muted">원본 음악·이미지·백업 파일은 ZIP 저장 시 그대로 보존됩니다.</p><div class="assets" id="assetList"></div><div class="actions"><button id="openAgain">다른 ZIP 열기</button><button id="folderAgain">폴더 가져오기</button></div></div>'+
 '<div class="tab" id="tab-json" hidden><p class="muted">원본에서 지원하지 않는 이벤트도 보존하려면 전체 JSON에서 수정할 수 있습니다.</p><textarea id="fullJson" spellcheck="false" style="min-height:240px"></textarea><div class="actions"><button id="applyJson">전체 JSON 적용</button></div></div>'+
 '<div class="footer"><button id="applyNative" type="button">⟳ 원본 게임에 적용</button><button id="nativePlay" type="button">▶ 실제 플레이</button></div><div id="status" role="status"></div></div>'+
 '<div id="flash" class="flash" hidden></div>';
 for(const [name,fn] of Object.entries({open:()=> $('fileInput').click(),editorButton:()=>chooseMode(!st.open),play:nativePlay,saveZip:exportZip,close:()=>chooseMode(false),folder:()=> $('folderInput').click(),openAgain:()=> $('fileInput').click(),folderAgain:()=> $('folderInput').click(),addTile,removeTile,setAngle,setSettings,setEvent,addEvent,deleteEvent,setDecoration:editDecoration,addDecoration,deleteDecoration,applyJson,undo:()=>undoRedo(false),redo:()=>undoRedo(true),exportChart:downloadChart,applyNative:()=>saveToNative(true),nativePlay})){
   const el=$(name);if(el)el.addEventListener('click',fn);
 }
 $('fileInput').addEventListener('change',e=>{void openFiles(e.target.files);e.target.value=''});
 $('folderInput').addEventListener('change',e=>{void openFiles(e.target.files,true);e.target.value=''});
 $('floor').addEventListener('change',e=>{st.floor=clamp(Number(e.target.value)||0,0,length()-1);render()});
 $('events').addEventListener('change',()=>{const idx=Number($('events').value);$('eventJson').value=JSON.stringify(st.doc.actions[idx],null,2)});
 $('decorations').addEventListener('change',()=>{const idx=Number($('decorations').value);$('decorationJson').value=JSON.stringify(st.doc.decorations[idx],null,2)});
 for(const name of ['tile','events','decorations','settings','files','json'])$('nav-'+name).addEventListener('click',()=>setTab(name));
 const menuMap={menuFile:'files',menuEdit:'tile',menuLevel:'tile',menuEvents:'events',menuDecorations:'decorations',menuSettings:'settings'};
 for(const [key,tab] of Object.entries(menuMap))$(key).addEventListener('click',()=>setTab(tab));
 const flash=sessionStorage.getItem('adofaiStudio:flash');if(flash){sessionStorage.removeItem('adofaiStudio:flash');const e=$('flash');e.textContent=flash;e.hidden=false;setTimeout(()=>e.hidden=true,7000)}
 setInterval(monitorRenderer,1200);
 if(sessionStorage.getItem('adofaiStudio:manualPlayRequested')==='1'){
  sessionStorage.removeItem('adofaiStudio:manualPlayRequested');
  let retries=0;
  const watch=setInterval(()=>{
   retries++;if(retries>20){clearInterval(watch);say('원본 게임의 판정 모드를 열 수 없습니다. WebGPU 또는 차트 로딩 상태를 확인하세요.',true);return}
   if(rendererIssue()){clearInterval(watch);monitorRenderer();say(rendererIssue(),true);return}
   const button=Array.from(document.querySelectorAll('button')).find(x=>x.getAttribute('aria-label')==='Play'||x.title==='Play');
   if(button){clearInterval(watch);button.click();say('직접 플레이 모드 진입 · 직접 스페이스/터치로 판정하세요. 자동 입력은 사용하지 않습니다.')}
  },500);
 }
 if(gameRoute()){
  if(sessionStorage.getItem('adofaiStudio:autoEditor')==='1'){
   sessionStorage.removeItem('adofaiStudio:autoEditor');chooseMode(true);
  } else void loadCurrent();
 }
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',build);else build();
window.ADOFAINativeAddon={get state(){return st},openFiles,loadCurrent,exportZip,saveToNative,angles,chooseMode};
})();