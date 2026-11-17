package dev.hoonex.adofai.loaderinstaller;
import android.app.PendingIntent;
import android.content.*;
import android.content.pm.*;
import org.json.JSONObject;
import java.io.*;
import java.net.HttpURLConnection;
import java.util.Arrays;
import java.nio.charset.StandardCharsets;

final class SelfUpdate {
 static final String BASE="https://raw.githubusercontent.com/hoonex/adofai/loader-installer/dist/";
 static final String PAGE="https://github.com/hoonex/adofai/tree/loader-installer";
 static final long MAX=32L*1024*1024;
 final long code,size; final String name,filename,hash;
 SelfUpdate(long c,String n,String f,String h,long s){code=c;name=n;filename=f;hash=h;size=s;}
 static long installed(Context c)throws Exception{return c.getPackageManager().getPackageInfo(c.getPackageName(),0).getLongVersionCode();}
 static SelfUpdate latest()throws Exception{
  Progress.stage("GitHub 설치 앱 버전 확인");HttpURLConnection h=Release.connect(BASE+"update.json");h.setUseCaches(false);
  try(InputStream in=h.getInputStream()){
   JSONObject j=new JSONObject(new String(Release.read(in,65536),StandardCharsets.UTF_8));
   long code=j.getLong("versionCode"),size=j.getLong("size");String n=j.getString("versionName"),f=j.getString("file"),sha=j.getString("sha256");
   if(!n.matches("[0-9]+(\\.[0-9]+){2}")||!f.equals("ADOFAI-Loader-Installer-v"+n+".apk")||!sha.matches("[0-9a-fA-F]{64}")||code<1||size<=0||size>MAX||!"dev.hoonex.adofai.loaderinstaller".equals(j.getString("package")))throw new IOException("업데이트 정보 오류");
   return new SelfUpdate(code,n,f,sha,size);
  }finally{h.disconnect();}
 }
 static synchronized File fetch(Context c,SelfUpdate u)throws Exception{
  if(u.code<=installed(c))throw new IOException("이미 최신 버전입니다.");
  File dir=new File(c.getFilesDir(),"installer-update");if(!dir.isDirectory()&&!dir.mkdirs())throw new IOException("저장 공간 오류");
  File f=new File(dir,u.filename);
  if(f.isFile()&&f.length()==u.size&&Release.hash(f).equalsIgnoreCase(u.hash)){verify(c,u,f);return f;}
  File temp=new File(dir,"download.tmp.apk");HttpURLConnection h=Release.connect(BASE+u.filename);h.setUseCaches(false);
  try{try(InputStream in=h.getInputStream();OutputStream out=new FileOutputStream(temp)){
   byte[] b=new byte[65536];long n=0,last=0;int k;Progress.report("설치 앱 다운로드",0,u.size);while((k=in.read(b))!=-1){n+=k;if(n>u.size||n>MAX)throw new IOException("다운로드 크기 오류");out.write(b,0,k);long now=android.os.SystemClock.elapsedRealtime();if(now-last>=200){Progress.report("설치 앱 다운로드",n,u.size);last=now;}}Progress.report("설치 앱 다운로드",n,u.size);
  }finally{h.disconnect();}
  Progress.stage("설치 앱 SHA-256 및 서명 검증");if(temp.length()!=u.size||!Release.hash(temp).equalsIgnoreCase(u.hash))throw new IOException("업데이트 SHA-256 불일치");
  verify(c,u,temp);if(f.exists()&&!f.delete())throw new IOException("이전 파일 삭제 실패");if(!temp.renameTo(f))throw new IOException("저장 실패");
  for(File other:dir.listFiles()==null?new File[0]:dir.listFiles())if(!other.equals(f)&&other.getName().endsWith(".apk"))other.delete();
  return f;
  }finally{temp.delete();}
 }
 static void verify(Context c,SelfUpdate u,File f)throws Exception{
  PackageManager pm=c.getPackageManager();PackageInfo old=pm.getPackageInfo(c.getPackageName(),PackageManager.GET_SIGNING_CERTIFICATES);
  PackageInfo next=pm.getPackageArchiveInfo(f.getAbsolutePath(),PackageManager.GET_SIGNING_CERTIFICATES);
  if(next==null||!c.getPackageName().equals(next.packageName)||next.getLongVersionCode()!=u.code||!u.name.equals(next.versionName)||next.getLongVersionCode()<=old.getLongVersionCode()||old.signingInfo==null||next.signingInfo==null)throw new IOException("업데이트 APK 패키지/버전 오류");
  if(!Arrays.equals(old.signingInfo.getApkContentsSigners(),next.signingInfo.getApkContentsSigners()))throw new IOException("설치 앱 서명 불일치");
 }
 static void install(Context c,SelfUpdate u,File f)throws Exception{
  verify(c,u,f);if(!Release.hash(f).equalsIgnoreCase(u.hash))throw new IOException("APK 무결성 오류");
  PackageInstaller pi=c.getPackageManager().getPackageInstaller();PackageInstaller.SessionParams p=new PackageInstaller.SessionParams(PackageInstaller.SessionParams.MODE_FULL_INSTALL);
  p.setAppPackageName(c.getPackageName());p.setSize(f.length());p.setRequireUserAction(PackageInstaller.SessionParams.USER_ACTION_REQUIRED);
  int id=pi.createSession(p);try(PackageInstaller.Session s=pi.openSession(id)){
   try(InputStream i=new FileInputStream(f);OutputStream o=s.openWrite("base.apk",0,f.length())){Progress.transfer(i,o,"설치 앱 전송",f.length());s.fsync(o);}
   Intent callback=new Intent(c,InstallReceiver.class).putExtra("selfUpdate",true);
   PendingIntent piResult=PendingIntent.getBroadcast(c,id,callback,PendingIntent.FLAG_UPDATE_CURRENT|PendingIntent.FLAG_MUTABLE);
   s.commit(piResult.getIntentSender());
  }catch(Exception e){pi.abandonSession(id);throw e;}
 }
 static String check(Context c,boolean download)throws Exception{
  SelfUpdate u=latest();if(u.code<=installed(c)){c.getSharedPreferences("state",0).edit().remove("appAvailable").remove("appReady").apply();return "설치 앱 최신 버전";}
  c.getSharedPreferences("state",0).edit().putString("appAvailable",u.name).apply();
  if(download){fetch(c,u);c.getSharedPreferences("state",0).edit().putString("appReady",u.name).apply();}
  return "설치 앱 새 버전 "+u.name+(download?" 다운로드 완료":" 있음");
 }
}