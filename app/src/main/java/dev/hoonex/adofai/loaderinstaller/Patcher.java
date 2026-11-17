package dev.hoonex.adofai.loaderinstaller;
import android.content.*;import android.content.pm.*;import java.io.*;import java.util.*;import java.util.zip.*;
import com.android.zipflinger.*;import org.jf.dexlib2.*;import org.jf.dexlib2.builder.*;import org.jf.dexlib2.builder.instruction.*;import org.jf.dexlib2.iface.*;import org.jf.dexlib2.immutable.*;import org.jf.dexlib2.immutable.reference.*;import org.jf.dexlib2.immutable.instruction.*;
final class Patcher{
 static final String PKG="com.fizzd.connectedworlds",ACT="Lcom/unity3d/player/UnityPlayerActivity;",LIB="lib/arm64-v8a/libadofailoader.so",BOOT="localLoaderBootstrap";
 static File original(Context c){return new File(c.getFilesDir(),"original");}
 static File output(Context c){return new File(c.getFilesDir(),"patched");}
 static void copy(File a,File b)throws Exception{try(InputStream i=new FileInputStream(a);OutputStream o=new FileOutputStream(b)){Progress.transfer(i,o,"APK 복사: "+a.getName(),a.length());}}
 static File[] apks(File d){File[] a=d.listFiles((f,n)->n.endsWith(".apk"));if(a==null)return new File[0];Arrays.sort(a);return a;}
 static boolean entry(File f,String s)throws Exception{try(ZipFile z=new ZipFile(f)){return z.getEntry(s)!=null;}}
 static void backup(Context c)throws Exception{
  File dst=original(c);if(new File(dst,"complete").isFile()){Progress.stage("기존 원본 백업 확인");File[] originalApks=apks(dst);if(originalApks.length==0)throw new IOException("원본 백업 파일 없음");PackageInfo originalInfo=c.getPackageManager().getPackageArchiveInfo(originalApks[0].getAbsolutePath(),0);if(originalInfo==null||!PKG.equals(originalInfo.packageName)||!"3.3.1".equals(originalInfo.versionName)||originalInfo.getLongVersionCode()!=300382L)throw new IOException("원본 백업 버전 확인 실패");for(File f:originalApks)if(entry(f,LIB))throw new IOException("원본 백업에 로더가 포함되어 있습니다.");return;}
  PackageInfo p=c.getPackageManager().getPackageInfo(PKG,0); if(!"com.android.vending".equals(c.getPackageManager().getInstallSourceInfo(PKG).getInstallingPackageName()))throw new IOException("Google Play에서 구매한 정품 게임을 먼저 설치하세요.");
  if(!"3.3.1".equals(p.versionName)||p.getLongVersionCode()!=300382L)throw new IOException("지원 게임 3.3.1 (300382) 필요");
  ArrayList<File> files=new ArrayList<>();files.add(new File(p.applicationInfo.sourceDir));if(p.applicationInfo.splitSourceDirs!=null)for(String s:p.applicationInfo.splitSourceDirs)files.add(new File(s));
  for(File f:files)if(entry(f,LIB))throw new IOException("이미 수정된 APK입니다. 원본 정품이 필요합니다.");
  dst.mkdirs();for(File f:apks(dst))f.delete();int i=0;for(File f:files){copy(f,new File(dst,String.format(java.util.Locale.US,"%03d.apk",i++)));}
  try(FileOutputStream o=new FileOutputStream(new File(dst,"complete"))){o.write(1);}
 }
 static void patch(Context c,File lib)throws Exception{
  backup(c);Release.validate(lib);
  File dst=output(c);dst.mkdirs();new File(dst,"complete").delete();new File(dst,"recipe-3").delete();for(File f:apks(dst))if(!f.delete())throw new IOException("이전 패치 삭제 실패");
  File tmp=new File(c.getCacheDir(),"work.apk"),dex=new File(c.getCacheDir(),"input.dex"),changed=new File(c.getCacheDir(),"output.dex");int found=0;boolean arm=false;
  SigningIdentity key=SigningIdentity.loadOrCreate();
  try{
   File[] sources=apks(original(c));int part=0;
   for(File src:sources){
    String step="("+(++part)+"/"+sources.length+") ";Progress.stage(step+"APK 복사 준비");copy(src,tmp);
    List<String> dexNames=new ArrayList<>();try(ZipFile z=new ZipFile(src)){Enumeration<? extends ZipEntry> e=z.entries();while(e.hasMoreElements()){String n=e.nextElement().getName();if(n.matches("classes([0-9]+)?\\.dex"))dexNames.add(n);}}
    Collections.sort(dexNames);
    for(String n:dexNames){
     Progress.stage(step+"DEX 분석 및 패치: "+n);
     try(ZipFile z=new ZipFile(src);InputStream in=z.getInputStream(z.getEntry(n));OutputStream out=new FileOutputStream(dex)){in.transferTo(out);}
     int bits=patchDex(dex,changed);
     if(bits!=0){if((found&bits)!=0)throw new IOException("대상 DEX 클래스 중복");found|=bits;try(ZipArchive z=new ZipArchive(tmp.toPath())){z.delete(n);FullFileSource source=new FullFileSource(changed.toPath(),n,0);source.align(4);z.add(source);}changed.delete();}
    }
    Progress.stage(step+"APK 라이브러리 삽입 및 정렬");
    if(entry(src,"lib/arm64-v8a/libil2cpp.so")){if(arm)throw new IOException("ARM64 라이브러리 중복");arm=true;try(ZipArchive z=new ZipArchive(tmp.toPath())){FullFileSource source=new FullFileSource(lib.toPath(),LIB,0);source.align(16384);z.add(source);}}
    Progress.stage(step+"APK 서명 및 검증: "+src.getName());SplitSigner.signAndVerify(tmp,new File(dst,src.getName()),key);
   }
   if(found!=RECIPE||!arm)throw new IOException("원작자 패치 대상 누락: Unity="+((found&UNITY)!=0)+", Client="+((found&CLIENT)!=0)+", Provider="+((found&PROVIDER)!=0)+", ARM64="+arm);
   Progress.stage("패치 완료 표시 기록");try(FileOutputStream o=new FileOutputStream(new File(dst,"recipe-3"))){o.write(1);}try(FileOutputStream o=new FileOutputStream(new File(dst,"complete"))){o.write(1);}
  }finally{tmp.delete();dex.delete();changed.delete();}
 }

 static final int UNITY=1,CLIENT=2,PROVIDER=4,RECIPE=7;
 static final String CLIENT_CLASS="Lcom/pairip/licensecheck/LicenseClient;",PROVIDER_CLASS="Lcom/pairip/licensecheck/LicenseContentProvider;";
 static boolean ready(Context c){File d=output(c);return new File(d,"complete").isFile()&&new File(d,"recipe-3").isFile()&&apks(d).length>0;}
 static int patchDex(File input,File output)throws Exception {
  DexFile d=DexFileFactory.loadDexFile(input,Opcodes.forApi(35));List<ClassDef> classes=new ArrayList<>();int found=0;
  for(ClassDef cl:d.getClasses()){
   String type=cl.getType();int kind=ACT.equals(type)?UNITY:CLIENT_CLASS.equals(type)?CLIENT:PROVIDER_CLASS.equals(type)?PROVIDER:0;
   if(kind==0){classes.add(cl);continue;}
   if((found&kind)!=0)throw new IOException("대상 DEX 클래스 중복: "+type);
   List<Method> methods=new ArrayList<>();boolean injected=false;
   for(Method m:cl.getMethods()){
    if(kind==UNITY&&BOOT.equals(m.getName()))throw new IOException("이미 패치된 원본");
    boolean match=kind==UNITY&&"onCreate".equals(m.getName())&&params(m,"Landroid/os/Bundle;")&&"V".equals(m.getReturnType())||
      kind==CLIENT&&"checkLicense".equals(m.getName())&&params(m,"Landroid/content/Context;")&&"V".equals(m.getReturnType())||
      kind==PROVIDER&&"onCreate".equals(m.getName())&&params(m)&&"Z".equals(m.getReturnType());
    if(!match){methods.add(m);continue;}
    if(injected||m.getImplementation()==null||((m.getAccessFlags()&8)!=0)!=(kind==CLIENT))throw new IOException("지원하지 않는 대상 메서드: "+type);
    injected=true;org.jf.dexlib2.iface.MethodImplementation impl;
    if(kind==UNITY){
      MutableMethodImplementation body=new MutableMethodImplementation(m.getImplementation());
      body.addInstruction(0,new BuilderInstruction35c(Opcode.INVOKE_STATIC,0,0,0,0,0,0,new ImmutableMethodReference(ACT,BOOT,Collections.emptyList(),"V")));impl=body;
    }else if(kind==CLIENT){
      impl=new ImmutableMethodImplementation(1,Collections.singletonList(new ImmutableInstruction10x(Opcode.RETURN_VOID)),Collections.emptyList(),Collections.emptyList());
    }else{
      impl=new ImmutableMethodImplementation(2,Arrays.asList(new ImmutableInstruction11n(Opcode.CONST_4,0,1),new ImmutableInstruction11x(Opcode.RETURN,0)),Collections.emptyList(),Collections.emptyList());
    }
    methods.add(new ImmutableMethod(type,m.getName(),m.getParameters(),m.getReturnType(),m.getAccessFlags(),m.getAnnotations(),m.getHiddenApiRestrictions(),impl));
   }
   if(!injected)throw new IOException("원작자 문서의 대상 메서드 없음: "+type);
   if(kind==UNITY){
    ImmutableMethodImplementation body=new ImmutableMethodImplementation(1,Arrays.asList(new ImmutableInstruction21c(Opcode.CONST_STRING,0,new ImmutableStringReference("adofailoader")),new ImmutableInstruction35c(Opcode.INVOKE_STATIC,1,0,0,0,0,0,new ImmutableMethodReference("Ljava/lang/System;","loadLibrary",Collections.singletonList("Ljava/lang/String;"),"V")),new ImmutableInstruction10x(Opcode.RETURN_VOID)),Collections.emptyList(),Collections.emptyList());
    methods.add(new ImmutableMethod(ACT,BOOT,Collections.emptyList(),"V",10,Collections.emptySet(),Collections.emptySet(),body));
   }
   classes.add(new ImmutableClassDef(type,cl.getAccessFlags(),cl.getSuperclass(),cl.getInterfaces(),cl.getSourceFile(),cl.getAnnotations(),cl.getFields(),methods));found|=kind;
  }
  if(found!=0)DexFileFactory.writeDexFile(output.getAbsolutePath(),new ImmutableDexFile(d.getOpcodes(),classes));
  return found;
 }
 static boolean params(Method m,String... types){if(m.getParameterTypes().size()!=types.length)return false;for(int i=0;i<types.length;i++)if(!types[i].contentEquals(m.getParameterTypes().get(i)))return false;return true;}
 static void export(File dir,OutputStream out)throws Exception{if(!new File(dir,"complete").isFile())throw new IOException("완료된 파일 없음");try(ZipOutputStream z=new ZipOutputStream(out)){z.setLevel(0);for(File f:apks(dir)){z.putNextEntry(new ZipEntry(f.getName()));try(InputStream in=new FileInputStream(f)){Progress.transfer(in,z,"백업 내보내기: "+f.getName(),f.length());}z.closeEntry();}}}
 static boolean signatureConflict(Context c,File d)throws Exception{
  File[] files=apks(d);if(files.length==0)throw new IOException("APK 세트 없음");PackageManager pm=c.getPackageManager();
  PackageInfo next=pm.getPackageArchiveInfo(files[0].getAbsolutePath(),PackageManager.GET_SIGNING_CERTIFICATES);
  if(next==null||!PKG.equals(next.packageName)||next.signingInfo==null)throw new IOException("APK 서명을 읽을 수 없습니다.");
  PackageInfo installed;try{installed=pm.getPackageInfo(PKG,PackageManager.GET_SIGNING_CERTIFICATES);}catch(PackageManager.NameNotFoundException e){return false;}
  return installed.signingInfo==null||!Arrays.equals(installed.signingInfo.getApkContentsSigners(),next.signingInfo.getApkContentsSigners());
 }
 static void install(Context c,File d)throws Exception{
  if(!new File(d,"complete").isFile()||(d.equals(output(c))&&!ready(c)))throw new IOException("새 버전에서 원본 백업으로 다시 패치하세요.");
  if(signatureConflict(c,d))throw new IOException("기존 게임과 서명 충돌: 세이브 백업 후 직접 삭제 필요");
  File[] files=apks(d);long bytes=0;for(File f:files)bytes+=f.length();
  PackageInstaller pi=c.getPackageManager().getPackageInstaller();PackageInstaller.SessionParams params=new PackageInstaller.SessionParams(PackageInstaller.SessionParams.MODE_FULL_INSTALL);params.setAppPackageName(PKG);params.setSize(bytes);params.setRequireUserAction(PackageInstaller.SessionParams.USER_ACTION_REQUIRED);
  int id=pi.createSession(params);try(PackageInstaller.Session session=pi.openSession(id)){int n=0;for(File f:files){String step="설치 파일 전송 ("+(++n)+"/"+files.length+"): "+f.getName();try(InputStream in=new FileInputStream(f);OutputStream out=session.openWrite(f.getName(),0,f.length())){Progress.transfer(in,out,step,f.length());session.fsync(out);}}Progress.stage("Android 설치 승인 요청");android.app.PendingIntent pending=android.app.PendingIntent.getBroadcast(c,id,new Intent(c,InstallReceiver.class),android.app.PendingIntent.FLAG_UPDATE_CURRENT|android.app.PendingIntent.FLAG_MUTABLE);session.commit(pending.getIntentSender());}catch(Exception e){pi.abandonSession(id);throw e;}
 }
}