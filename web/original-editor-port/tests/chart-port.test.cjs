const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const path=require('node:path');
const window={};
const document={readyState:'loading',addEventListener(){},getElementById(){return null}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname,'..','chart-port.js'),'utf8'),{
  window,document,console,JSON,Number,Math,Set,Map,Array,Object,Error,Boolean,String,Date,Infinity,NaN,parseInt
});
const p=window.ADOFAIChartPort;
let count=0;
function verify(label,fn){fn();count++;console.log('PASS',label)}
verify('BOM and relaxed trailing comma input',()=>{
  assert.equal(p.parseChart('\uFEFF{"pathData":"R!5","settings":{"name":"X",},}').settings.name,'X')
});
verify('midspin and relative direction',()=>assert.equal(JSON.stringify(p.anglesFromPath('R!5')),'[0,999,72]'));
verify('numeric relative direction after midspin',()=>assert.equal(JSON.stringify(p.anglesFromData([0,999,555])),'[0,999,72]'));
const source={pathData:'R!5',settings:{bpm:120,unfamiliar:{a:[1,2]}},actions:[
  {floor:2,eventType:'Twirl',futureKey:'keep'},
  {floor:3,eventType:'SetSpeed',speedType:'Bpm',beatsPerMinute:180}
],decorations:[{floor:2,eventType:'AddDecoration',foo:1}],customRoot:{valid:true}};
p.loadObject(source);
verify('implicit starting floor 0',()=>assert.equal(p.state.positions.length,4));
p.selectFloor(2);p.editAngle(45);
verify('angle edit explicitly converts pathData',()=>{
  const d=p.saveObject();assert.equal('pathData' in d,false);assert.equal(JSON.stringify(d.angleData),'[0,45,72]')
});
verify('unknown chart root retained',()=>assert.equal(p.saveObject().customRoot.valid,true));
verify('unknown event property retained',()=>assert.equal(p.saveObject().actions[0].futureKey,'keep'));
p.addTile();
verify('insert shifts only later action indices',()=>{
  const d=p.saveObject();assert.equal(d.actions[1].floor,4);assert.equal(d.decorations[0].floor,2)
});
p.undo();verify('undo restores structure',()=>{
  const d=p.saveObject();assert.equal(d.actions[1].floor,3);assert.equal(d.angleData.length,3)
});
p.redo();verify('redo restores structure',()=>{
  const d=p.saveObject();assert.equal(d.actions[1].floor,4);assert.equal(d.angleData.length,4)
});
p.selectFloor(1);p.deleteTile();
verify('deleting floor shifts subsequent actions',()=>{
  const d=p.saveObject();assert.equal(d.actions[0].floor,1);assert.equal(d.actions[1].floor,3)
});
p.loadObject(source);verify('untouched file structurally unchanged',()=>assert.deepEqual(JSON.parse(JSON.stringify(p.saveObject())),source));
verify('unsupported path direction fails closed',()=>assert.throws(()=>p.anglesFromPath('R?')));
p.loadObject({pathData:'R',angleData:[90,90],settings:{},actions:[]});
verify('PC format pathData precedence',()=>assert.equal(JSON.stringify(p.state.angles),'[0]'));
verify('BOM escaped string commas intact',()=>{
  const d=p.parseChart('{"pathData":"R","settings":{"title":"hello,}","json":"{,}"},"actions":[]}');
  assert.equal(d.settings.json,'{,}')
});
console.log('PASS COUNT',count);
