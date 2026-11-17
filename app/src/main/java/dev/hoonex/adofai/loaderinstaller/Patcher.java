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
  File dst=original(c);if(new File(dst,"complete").isFile()){Progress.stage("기존 원본 백업 확인");if(apks(dst).length==0)throw new IOException("원본 백업 파일 없음");return;}
  PackageInfo p=c.getPackageManager().getPackageInfo(PKG,0);
  if(!"3.3.1".equals(p.versionName)||p.getLongVersionCode()!=300382L)throw new IOException("지원 게임 3.3.1 (300382) 필요");
  ArrayList<File> files=new ArrayList<>();files.add(new File(p.applicationInfo.sourceDir));if(p.applicationInfo.splitSourceDirs!=null)for(String s:p.applicationInfo.splitSourceDirs)files.add(new File(s));
  for(File f:files)if(entry(f,LIB))throw new IOException("이미 수정된 APK입니다. 원본 정품이 필요합니다.");
  dst.mkdirs();for(File f:apks(dst))f.delete();int i=0;for(File f:files){copy(f,new File(dst,String.format(java.util.Locale.US,"%03d.apk",i++)));}
  try(FileOutputStream o=new FileOutputStream(new File(dst,"complete"))){o.write(1);}
 }
 static void patch(Context c,File lib)throws Exception{
  backup(c);Release.validate(lib);File dst=output(c);dst.mkdirs();for(File f:apks(dst))f.delete();new File(dst,"complete").delete();
  File tmp=new File(c.getCacheDir(),"work.apk"),dex=new File(c.getCacheDir(),"input.dex"),changed=new File(c.getCacheDir(),"output.dex");boolean found=false,arm=false;
  SigningIdentity key=SigningIdentity.loadOrCreate();
  File[] sources=apks(original(c));int part=0;for(File src:sources){String step="("+ (++part) +"/"+sources.length+") ";Progress.stage(step+"APK 복사 준비");
   copy(src,tmp);
   List<String> dexNames=new ArrayList<>();try(ZipFile z=new ZipFile(src)){Enumeration<? extends ZipEntry> e=z.entries();while(e.hasMoreElements()){String n=e.nextElement().getName();if(n.matches("classes([0-9]+)?\\.dex"))dexNames.add(n);}}
   String hit=null;for(String n:dexNames){Progress.stage(step+"DEX 분석 및 패치: "+n);try(ZipFile z=new ZipFile(src);InputStream in=z.getInputStream(z.getEntry(n));OutputStream out=new FileOutputStream(dex)){in.transferTo(out);}if(patchDex(dex,changed)){if(found)throw new IOException("Unity 클래스 중복");hit=n;found=true;break;}}
   Progress.stage(step+"APK 라이브러리 삽입 및 정렬");boolean nativeSplit=entry(src,"lib/arm64-v8a/libil2cpp.so");
   try(ZipArchive z=new ZipArchive(tmp.toPath())){
    if(hit!=null){z.delete(hit);FullFileSource s=new FullFileSource(changed.toPath(),hit,0);s.align(4);z.add(s);}
    if(nativeSplit){if(arm)throw new IOException("ARM64 라이브러리 중복");arm=true;FullFileSource s=new FullFileSource(lib.toPath(),LIB,0);s.align(16384);z.add(s);}
   }
   Progress.stage(step+"APK 서명 및 검증: "+src.getName());SplitSigner.signAndVerify(tmp,new File(dst,src.getName()),key);
  }
  if(!found||!arm)throw new IOException("Unity Activity 또는 ARM64 split을 찾지 못함");
  try(FileOutputStream o=new FileOutputStream(new File(dst,"complete"))){o.write(1);}
  tmp.delete();dex.delete();changed.delete();
 }
 static boolean patchDex(File input,File output)throws Exception {
  DexFile d=DexFileFactory.loadDexFile(input,Opcodes.forApi(35));List<ClassDef> classes=new ArrayList<>();boolean found=false;
  for(ClassDef cl:d.getClasses()){
   if(!ACT.equals(cl.getType())){classes.add(cl);continue;}found=true;List<Method> methods=new ArrayList<>();boolean injected=false;
   for(Method m:cl.getMethods()){
    if(BOOT.equals(m.getName()))throw new IOException("이미 패치됨");
    if(!"onCreate".equals(m.getName())||m.getParameterTypes().size()!=1||!"Landroid/os/Bundle;".contentEquals(m.getParameterTypes().get(0))||!"V".equals(m.getReturnType())){methods.add(m);continue;}
    if(injected||m.getImplementation()==null)throw new IOException("잘못된 onCreate");injected=true;
    MutableMethodImplementation impl=new MutableMethodImplementation(m.getImplementation());
    impl.addInstruction(0,new BuilderInstruction35c(Opcode.INVOKE_STATIC,0,0,0,0,0,0,new ImmutableMethodReference(ACT,BOOT,Collections.emptyList(),"V")));
    methods.add(new ImmutableMethod(ACT,m.getName(),m.getParameters(),m.getReturnType(),m.getAccessFlags(),m.getAnnotations(),m.getHiddenApiRestrictions(),impl));
   }
   if(!injected)throw new IOException("onCreate 없음");
   ImmutableMethodImplementation body=new ImmutableMethodImplementation(1,Arrays.asList(new ImmutableInstruction21c(Opcode.CONST_STRING,0,new ImmutableStringReference("adofailoader")),new ImmutableInstruction35c(Opcode.INVOKE_STATIC,1,0,0,0,0,0,new ImmutableMethodReference("Ljava/lang/System;","loadLibrary",Collections.singletonList("Ljava/lang/String;"),"V")),new ImmutableInstruction10x(Opcode.RETURN_VOID)),Collections.emptyList(),Collections.emptyList());
   methods.add(new ImmutableMethod(ACT,BOOT,Collections.emptyList(),"V",10,Collections.emptySet(),Collections.emptySet(),body));
   classes.add(new ImmutableClassDef(ACT,cl.getAccessFlags(),cl.getSuperclass(),cl.getInterfaces(),cl.getSourceFile(),cl.getAnnotations(),cl.getFields(),methods));
  }
  if(found)DexFileFactory.writeDexFile(output.getAbsolutePath(),new ImmutableDexFile(d.getOpcodes(),classes));return found;
 }
 static void export(File dir,OutputStream out)throws Exception{if(!new File(dir,"complete").isFile())throw new IOException("완료된 파일 없음");try(ZipOutputStream z=new ZipOutputStream(out)){z.setLevel(0);for(File f:apks(dir)){z.putNextEntry(new ZipEntry(f.getName()));try(InputStream in=new FileInputStream(f)){Progress.transfer(in,z,"백업 내보내기: "+f.getName(),f.length());}z.closeEntry();}}}
 static boolean signatureConflict(Context c,File d)throws Exception{
  File[] files=apks(d);if(files.length==0)throw new IOException("APK 세트 없음");PackageManager pm=c.getPackageManager();
  PackageInfo next=pm.getPackageArchiveInfo(files[0].getAbsolutePath(),PackageManager.GET_SIGNING_CERTIFICATES);
  if(next==null||!PKG.equals(next.packageName)||next.signingInfo==null)throw new IOException("APK 서명을 읽을 수 없습니다.");
  PackageInfo installed;try{installed=pm.getPackageInfo(PKG,PackageManager.GET_SIGNING_CERTIFICATES);}catch(PackageManager.NameNotFoundException e){return false;}
  return installed.signingInfo==null||!Arrays.equals(installed.signingInfo.getApkContentsSigners(),next.signingInfo.getApkContentsSigners());
 }
 static void install(Context c,File d)throws Exception{
  if(!new File(d,"complete").isFile())throw new IOException("완료된 APK 세트 없음");
  if(signatureConflict(c,d))throw new IOException("기존 게임과 서명 충돌: 세이브 백업 후 직접 삭제 필요");
  File[] files=apks(d);long bytes=0;for(File f:files)bytes+=f.length();
  PackageInstaller pi=c.getPackageManager().getPackageInstaller();PackageInstaller.SessionParams params=new PackageInstaller.SessionParams(PackageInstaller.SessionParams.MODE_FULL_INSTALL);params.setAppPackageName(PKG);params.setSize(bytes);params.setRequireUserAction(PackageInstaller.SessionParams.USER_ACTION_REQUIRED);
  int id=pi.createSession(params);try(PackageInstaller.Session session=pi.openSession(id)){int n=0;for(File f:files){String step="설치 파일 전송 ("+(++n)+"/"+files.length+"): "+f.getName();try(InputStream in=new FileInputStream(f);OutputStream out=session.openWrite(f.getName(),0,f.length())){Progress.transfer(in,out,step,f.length());session.fsync(out);}}Progress.stage("Android 설치 승인 요청");android.app.PendingIntent pending=android.app.PendingIntent.getBroadcast(c,id,new Intent(c,InstallReceiver.class),android.app.PendingIntent.FLAG_UPDATE_CURRENT|android.app.PendingIntent.FLAG_MUTABLE);session.commit(pending.getIntentSender());}catch(Exception e){pi.abandonSession(id);throw e;}
 }
}