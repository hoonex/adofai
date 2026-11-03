/* Native Studio v2 — robust local archive, folder and resource manager.
 * The Web ADOFAI engine is NOT bundled; the renderer is the original site's iframe.
 * Browser-controlled chart/ZIP imports are not uploaded to this site's server.
 */
(()=>{'use strict';
const $=id=>document.getElementById(id),API=window.ADOFAINativeStudio;
if(!API){console.error('Native Studio core missing');return;}
const st=API.state;
const X={archive:null,sourceName:'',draftFiles:null,charts:[],assetUrls:[],importing:false};
const LIMITS={archive:250*1024*1024,expanded:250*1024*1024,entry:100*1024*1024,chart:18*1024*1024,entries:2500};
const isChart=n=>/\.adofai$/i.test(n),isSong=n=>/\.(ogg|mp3|wav|flac|m4a)$/i.test(n),isImg=n=>/\.(png|jpe?g|webp|gif)$/i.test(n);
const base=n=>n.split('/').at(-1)||n;
const fmt=n=>n<1024?n+' B':n<1024**2?(n/1024).toFixed(1)+' KB':(n/1024**2).toFixed(2)+' MB';
const pathClean=v=>API.relativePath(v);
function progress(v){$('loadProgress').textContent=v||'';}
function notice(s,fail=false){$('status').textContent=s;$('status').className='status '+(fail?'error':'ok');}
function nameRating(path){const n=path.toLowerCase();return (/backup|autosave|autosaved|backup|temp|old|백업|\.bak/i.test(n)?1e3:0)+(n.split('/').length*2)+(n.length/100);}
function choosePrimary(paths){return paths.slice().sort((a,b)=>nameRating(a)-nameRating(b)||a.localeCompare(b))[0]||'';}
function cleanMemory(){for(const u of X.assetUrls){URL.revokeObjectURL(u)}X.assetUrls=[];$('songPreview').pause();$('songPreview').hidden=true;$('songPreview').removeAttribute('src');$('imagePreview').hidden=true;$('imagePreview').removeAttribute('src');}
function setPreview(blob,type){const url=URL.createObjectURL(blob);X.assetUrls.push(url);if(type==='audio'){$('songPreview').src=url;$('songPreview').hidden=false;$('songPreview').load();}else{$('imagePreview').src=url;$('imagePreview').hidden=false;}}
function describeResources(){
 const holder=$('assetList');holder.replaceChildren();cleanMemory();const files=Array.from(st.files||[]);$('assetCount').textContent=files.length+'개';
 if(!files.length){const el=document.createElement('div');el.className='asset-empty';el.textContent='파일을 가져오면 .adofai와 음악·이미지를 여기서 관리할 수 있어.';holder.append(el);return;}
 const song=API.guessSongName(),candidateImage=files.find(([path])=>path.toLowerCase().endsWith('/'+String(st.doc?.settings?.bgImage||'').toLowerCase()))||files.find(([path])=>isImg(path));
 if(song&&st.files.has(song))setPreview(st.files.get(song),'audio');
 if(candidateImage)setPreview(candidateImage[1],'image');
 for(const [path,blob] of files){const row=document.createElement('div');row.className='asset-line';if(path===st.mainPath)row.classList.add('asset-line--chart');if(path===song)row.classList.add('asset-line--song');
 const title=document.createElement('div');title.className='name';title.title=path;title.textContent=(path===st.mainPath?'◆ ':isSong(path)?'♫ ':isImg(path)?'▣ ':'· ')+path;
 const right=document.createElement('div'),mini=document.createElement('span');mini.className='meta';mini.textContent=' '+fmt(blob.size)+' ';
 right.append(mini);
 if(isSong(path)){const select=document.createElement('button');select.type='button';select.textContent='선택';select.title='차트 음악으로 지정';select.disabled=path===song;select.onclick=()=>setSong(path);right.append(select);}
 if(isImg(path)){const preview=document.createElement('button');preview.type='button';preview.textContent='보기';preview.onclick=()=>{const p=$('imagePreview');p.hidden=true;const url=URL.createObjectURL(blob);X.assetUrls.push(url);p.src=url;p.hidden=false;};right.append(preview);const use=document.createElement('button');use.type='button';use.textContent='배경';use.onclick=()=>{API.mutation(()=>{st.doc.settings??={};st.doc.settings.bgImage=base(path);},'배경 이미지 지정');describeResources();};right.append(use);}
 if(path!==st.mainPath){const remove=document.createElement('button');remove.type='button';remove.textContent='×';remove.title=path+' 삭제';remove.onclick=()=>removeAsset(path);right.append(remove);}
 row.append(title,right);holder.append(row);}
}
function setSong(path){if(!st.doc||!st.files.has(path))return;API.mutation(()=>{st.doc.settings??={};st.doc.settings.songFilename=base(path)},'음악 파일 변경');describeResources();populateQuick();}
function removeAsset(path){if(path===st.mainPath)return; if(!confirm('이 파일을 차트 ZIP에서 제거할까요?\n'+path))return;st.files.delete(path);st.dirty=true;API.scheduleCommit();describeResources();API.render();notice('파일 삭제 예약: '+path);}
function populateQuick(){if(!st.doc)return;const x=st.doc.settings||{};$('quickBpm').value=x.bpm??120;$('quickOffset').value=x.offset??0;$('quickSong').value=x.song??'';$('quickAuthor').value=x.author??'';}
function setCurrentFileName(v){X.sourceName=v;}
function fillChartChoice(charts){X.charts=charts;const s=$('archiveCharts');s.replaceChildren();for(const path of charts){const o=document.createElement('option');o.value=path;o.textContent=(path===st.mainPath?'현재 · ':'')+path;s.append(o);}s.value=st.mainPath;$('archiveChoice').hidden=charts.length<2;}
async function activate(draft,path){
 const blob=draft.files.get(path);if(!blob)throw Error('선택한 차트를 찾지 못했습니다: '+path);
 if(blob.size>LIMITS.chart)throw Error('차트가 너무 큽니다: '+fmt(blob.size));
 const doc=API.validate(API.parseJson(await blob.text()));const a=API.angles(doc);if(a.length>100000)throw Error('타일 100,000개를 초과합니다.');
 st.doc=doc;st.mainPath=path;st.filename=draft.name;st.files=new Map(draft.files);st.originalZip=draft.zip||null;st.packageId='local:'+(crypto.randomUUID?.()||('v2-'+Date.now()+'-'+Math.random().toString(36).slice(2)));st.selected=Math.min(1,a.length);st.undo=[];st.redo=[];st.dirty=false;
 X.archive=draft.zip||null;X.draftFiles=new Map(draft.files);X.sourceName=draft.name;
 API.render();populateQuick();fillChartChoice(draft.charts);describeResources();progress('가져오기 완료 · 파일 '+st.files.size+'개 · 타일 '+(a.length+1)+'개 · 이벤트 '+(doc.actions?.length||0)+'개');
 notice('불러오기 성공: '+path+' · 음악 '+(API.guessSongName()?'연결':'없음'),'ok');
 await API.selectMode('editor');
}
async function fromEntries(entries,name,zip=null){
 if(!entries.length)throw Error('차트 파일이 없습니다.');if(entries.length>LIMITS.entries)throw Error('ZIP에 너무 많은 파일이 있습니다.');
 let total=0;const map=new Map();let i=0;
 for(const {path,blob} of entries){const safe=pathClean(path);if(!safe||safe.startsWith('__MACOSX/')||safe.startsWith('.'))continue;
  if(blob.size>LIMITS.entry)throw Error(safe+'는 파일 크기 제한 초과입니다.');total+=blob.size;if(total>LIMITS.expanded)throw Error('압축 해제 용량 250MB 제한을 초과했습니다.');
  if(map.has(safe))throw Error('중복 경로: '+safe);map.set(safe,blob);i++;if(i%30===0)progress(i+'/'+entries.length+'개 파일 분석 중…');
 }
 const charts=[...map.keys()].filter(isChart);if(!charts.length)throw Error('ZIP 또는 폴더에 .adofai가 없습니다.');
 const primary=choosePrimary(charts);charts.sort((a,b)=>a===primary?-1:b===primary?1:nameRating(a)-nameRating(b));const draft={files:map,zip,name,charts};
 await activate(draft,primary);
}
async function importZip(file){
 progress('ZIP 구조 검사 중…');if(!window.JSZip)throw Error('JSZip 라이브러리를 로드하지 못했습니다.');if(file.size>LIMITS.archive)throw Error('ZIP이 250MB를 초과합니다.');
 const zip=await JSZip.loadAsync(file,{checkCRC32:true});const objs=Object.values(zip.files).filter(x=>!x.dir);
 if(objs.length>LIMITS.entries)throw Error('ZIP 파일 항목 수가 너무 많습니다: '+objs.length);
 let sum=0;for(const v of objs){const size=v._data?.uncompressedSize||0;if(size>LIMITS.entry)throw Error(v.name+'는 압축 해제 크기가 너무 큽니다.');sum+=size;if(sum>LIMITS.expanded)throw Error('압축 해제 크기 합계가 250MB를 초과합니다.');}
 const entries=[];for(let i=0;i<objs.length;i++){
   const f=objs[i];const safe=pathClean(f.name);if(!safe)continue;const bytes=await f.async('uint8array');entries.push({path:safe,blob:new Blob([bytes],{type:API.mime(safe)})});
   if(i%15===0||i===objs.length-1)progress((i+1)+'/'+objs.length+'개 파일 추출 중…');
 }
 await fromEntries(entries,file.name,zip);
}
async function importDirectory(files){const entries=[];for(const file of files){const path=file.webkitRelativePath||file.relativePath||file.name;entries.push({path,blob:file});}await fromEntries(entries,files.length?base(files[0].webkitRelativePath||files[0].name).replace(/\.[^.]+$/,'')+'-folder.zip':'folder.zip');}
async function openAny(file){if(!file)return;if(X.importing)return;X.importing=true;
 if(st.dirty&&!confirm('수정한 차트를 저장하지 않았습니다. 새로운 파일로 전환할까요?')){X.importing=false;return;}
 try{if(/\.zip$/i.test(file.name))await importZip(file);
 else if(isChart(file.name)){const path=file.name.replace(/\.adofai$/i,'')+'/'+file.name;await fromEntries([{path,blob:file}],file.name);}
 else if(isSong(file.name)||isImg(file.name)){await addAssets([file]);}
 else throw Error('.zip, .adofai, 이미지 또는 음악 파일만 지원합니다.');
 }catch(e){notice('파일 가져오기 오류: '+e.message,true);progress('');console.error('ADOFAI import',e);}finally{X.importing=false;}}
async function switchMain(){if(!X.draftFiles)return;const target=$('archiveCharts').value;if(!target)return;
 if(st.dirty&&!confirm('현재 차트 수정 사항을 버리고 다른 차트를 열까요?'))return;
 try{const draft={files:new Map(st.files),name:st.filename,zip:st.originalZip,charts:X.charts};await activate(draft,target);}catch(e){notice('차트 변경 오류: '+e.message,true);}}
async function addAssets(files){if(!files?.length)return;if(!st.doc){notice('차트를 먼저 열어야 음악·이미지를 추가할 수 있습니다.',true);return;}
 const parent=st.mainPath.slice(0,st.mainPath.lastIndexOf('/')+1);let added=0;for(const file of files){if(file.size>LIMITS.entry){notice(file.name+' 파일이 너무 큽니다.',true);return;}const path=pathClean(parent+file.name);if(st.files.has(path)&&!confirm(path+' 파일을 교체할까요?'))continue;st.files.set(path,new Blob([file],{type:file.type||API.mime(path)}));added++;}
 if(!added)return;const firstAudio=Array.from(files).find(f=>isSong(f.name));if(firstAudio){st.doc.settings??={};st.doc.settings.songFilename=firstAudio.name;}
 st.dirty=true;API.render();describeResources();API.scheduleCommit();notice(added+'개 자산을 추가했습니다.');}
async function exportZip(){if(!st.doc)return notice('먼저 차트를 열어주세요.',true);
 try{notice('ZIP 저장 중…');const zip=new JSZip();for(const [path,blob] of st.files){if(path===st.mainPath)continue;zip.file(path,blob,{binary:true});}
 zip.file(st.mainPath,JSON.stringify(st.doc,null,2)+'\n');const out=await zip.generateAsync({type:'blob',compression:'DEFLATE',compressionOptions:{level:6}});
 const original=(st.filename||'New Level').replace(/\.(zip|adofai)$/i,'')||'Custom Level';const name=original+'-edited.zip';API.download(out,name);st.dirty=false;API.render();notice('ZIP 저장 완료: '+name+' · 음악/이미지/백업 파일 유지','ok');
 }catch(e){notice('ZIP 저장 실패: '+e.message,true);}}
function saveChart(){if(!st.doc)return;const file=base(st.mainPath)||'level.adofai';API.download(new Blob([JSON.stringify(st.doc,null,2)+'\n'],{type:'application/json;charset=utf-8'}),/\.adofai$/i.test(file)?file:file+'.adofai');notice('차트 파일 저장 완료: '+file);}
function quickSettings(){if(!st.doc)return;const bpm=Number($('quickBpm').value),offset=Number($('quickOffset').value);if(!Number.isFinite(bpm)||bpm<=0||bpm>100000)return notice('BPM은 0보다 크고 100000 이하여야 합니다.',true);if(!Number.isFinite(offset))return notice('오프셋 숫자가 잘못되었습니다.',true);
 API.mutation(()=>{st.doc.settings??={};Object.assign(st.doc.settings,{bpm,offset,song:$('quickSong').value,author:$('quickAuthor').value});},'빠른 설정 적용');}
function presetAngle(val){if(!st.doc)return;$('angle').value=val;if(Number($('floor').value)===0){notice('시작 타일에는 방향이 없습니다.',true);return;}$('setAngle').click();}
function presetEvent(type){if(!st.doc)return;const floor=st.selected;const bpm=Number(st.doc.settings?.bpm)||120;const obj={floor,eventType:type};
 if(type==='SetSpeed')Object.assign(obj,{speedType:'Bpm',beatsPerMinute:bpm});
 else if(type==='Pause'||type==='Hold')obj.duration=1;
 else if(type==='MoveTrack')Object.assign(obj,{startTile:[floor,'ThisTile'],endTile:[floor,'ThisTile'],duration:1,positionOffset:[0,0],rotationOffset:0,scale:[100,100]});
 API.mutation(()=>{(st.doc.actions??=[]).push(obj);},type+' 이벤트 추가');$('eventSelect').value='actions:'+(st.doc.actions.length-1);API.setPane('events');}
function wire(){
 // Override only file controls: original import was functional but unhelpful for multi-chart ZIPs.
 $('picker').onchange=e=>{void openAny(e.target.files?.[0]);e.target.value='';};
 $('openFolder').onclick=()=>$('folderPicker').click();$('folderPicker').onchange=e=>{const files=Array.from(e.target.files||[]);void importDirectory(files).catch(ex=>notice(ex.message,true));e.target.value='';};
 $('archiveSelect').onclick=switchMain;
 $('assetPicker').onchange=e=>{void addAssets(Array.from(e.target.files||[]));e.target.value='';};$('addAssets').onclick=()=>$('assetPicker').click();
 $('audioPicker').onchange=e=>{void addAssets(Array.from(e.target.files||[]));e.target.value='';};
 $('saveZip').onclick=()=>void exportZip();$('saveChart').onclick=saveChart;
 $('applyQuick').onclick=quickSettings;$('autoSync').onchange=e=>{st.autoSync=!!e.target.checked;if(st.autoSync&&st.dirty)API.scheduleCommit();notice(st.autoSync?'자동 반영 켜짐':'자동 반영 끔 — 상단 반영 버튼으로 적용하세요.');};
 for(const btn of document.querySelectorAll('[data-angle]'))btn.onclick=()=>presetAngle(Number(btn.dataset.angle));
 for(const btn of document.querySelectorAll('[data-event]'))btn.onclick=()=>presetEvent(btn.dataset.event);
 const drop=$('drop');drop.ondragover=e=>{e.preventDefault();drop.classList.add('dragover')};drop.ondragleave=()=>drop.classList.remove('dragover');drop.ondrop=e=>{e.preventDefault();drop.classList.remove('dragover');const files=e.dataTransfer?.files;if(!files?.length)return;const entries=Array.from(files);if(entries.length>1)void importDirectory(entries).catch(ex=>notice(ex.message,true));else void openAny(entries[0]);};drop.onclick=()=>$('picker').click();drop.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('picker').click();}};
 $('toggleDrawer').onclick=()=>{const tools=$('tools'),collapsed=tools.classList.toggle('collapsed');$('toggleDrawer').setAttribute('aria-expanded',String(!collapsed));};
 $('editTab').addEventListener('click',()=>{$('tools').classList.remove('collapsed');$('toggleDrawer').setAttribute('aria-expanded','true');});
 $('newLevel').addEventListener('click',()=>{queueMicrotask(()=>{if(st.doc){X.archive=null;X.draftFiles=null;X.charts=[];$('archiveChoice').hidden=true;populateQuick();describeResources();}})});
 window.addEventListener('keydown',e=>{const tag=e.target?.tagName||'';const editable=/INPUT|TEXTAREA|SELECT/.test(tag)||e.target?.isContentEditable;const mod=e.ctrlKey||e.metaKey;
  if(mod&&e.key.toLowerCase()==='o'){e.preventDefault();$('picker').click();}
  if(mod&&e.key.toLowerCase()==='s'){e.preventDefault();exportZip();}
  if(!editable&&e.key==='['){e.preventDefault();$('prevFloor').click();}
  if(!editable&&e.key===']'){e.preventDefault();$('nextFloor').click();}
 },true);
 window.addEventListener('beforeunload',e=>{if(st.dirty){e.preventDefault();e.returnValue='';}});
 window.addEventListener('pagehide',cleanMemory);
 describeResources();
}
window.ADOFAINativeStudioV2={openAny,importZip,importDirectory,addAssets,exportZip,saveChart,activate,switchMain,presetAngle,quickSettings,presetEvent,state:X,refresh:()=>{API.render();populateQuick();describeResources();}};
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',wire);else wire();
})();