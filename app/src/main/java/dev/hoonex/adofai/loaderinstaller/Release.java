package dev.hoonex.adofai.loaderinstaller;
import android.content.*;import org.json.*;import java.io.*;import java.net.*;import java.security.*;
final class Release {
 static final String API="https://api.github.com/repos/Harrot114514/ADOFAI-MobileLevelLoder/releases/latest";
 static final String BASE="https://github.com/Harrot114514/ADOFAI-MobileLevelLoder/releases/download/";
 static final String PIN="9b8ed5c9b2c9fdb61ad6973ea5cd51422024cae3a2c0688c166cf04c0cb15aa6";
 final String tag,url,digest;
 Release(String t,String u,String d){tag=t;url=u;digest=d;}
 static Release latest() throws Exception {
  Progress.stage("GitHub 로더 릴리스 확인");HttpURLConnection c=connect(API); try(InputStream in=c.getInputStream()){
   JSONObject j=new JSONObject(new String(read(in,2*1024*1024),java.nio.charset.StandardCharsets.UTF_8));
   String tag=j.getString("tag_name"); if(!tag.matches("v[0-9]+(\\.[0-9]+){1,4}")||j.optBoolean("draft")||j.optBoolean("prerelease"))throw new IOException("지원하지 않는 릴리스");
   JSONArray a=j.getJSONArray("assets"); for(int i=0;i<a.length();i++){JSONObject x=a.getJSONObject(i);if(!"libadofailoader.so".equals(x.getString("name")))continue;
   String u=x.getString("browser_download_url");if(!u.equals(BASE+tag+"/libadofailoader.so"))throw new IOException("잘못된 릴리스 URL");
   String d=x.optString("digest","");if(d.startsWith("sha256:"))d=d.substring(7); if(tag.equals("v1.1.10.1"))d=PIN;
   return new Release(tag,u,d);}
   throw new IOException("릴리스 라이브러리가 없음");
  }finally{c.disconnect();}
 }
 static HttpURLConnection connect(String url)throws Exception{HttpURLConnection c=(HttpURLConnection)new URL(url).openConnection();c.setConnectTimeout(15000);c.setReadTimeout(30000);c.setRequestProperty("User-Agent","ADOFAI-Local-Installer");return c;}
 static byte[] read(InputStream in,int max)throws IOException{ByteArrayOutputStream b=new ByteArrayOutputStream();byte[] a=new byte[32768];int n;while((n=in.read(a))!=-1){if(b.size()+n>max)throw new IOException("파일 크기 제한");b.write(a,0,n);}return b.toByteArray();}
 static synchronized File fetch(Context ctx,Release r)throws Exception{
  if(!r.digest.matches("[0-9a-fA-F]{64}"))throw new IOException("GitHub SHA-256이 없어 자동 다운로드 중단");
  Progress.stage("기존 로더 파일 확인");File dir=new File(ctx.getFilesDir(),"loader");dir.mkdirs();File out=new File(dir,r.tag+".so"),tmp=new File(dir,r.tag+".tmp");
  if(out.isFile()&&hash(out).equalsIgnoreCase(r.digest)){validate(out);return out;}
  HttpURLConnection c=connect(r.url);try(InputStream in=c.getInputStream();OutputStream o=new FileOutputStream(tmp)){byte[] b=new byte[65536];long total=0,expected=c.getContentLengthLong(),last=0;int n;Progress.report("로더 다운로드",0,expected);while((n=in.read(b))!=-1){total+=n;if(total>64L*1024*1024)throw new IOException("라이브러리가 너무 큼");o.write(b,0,n);long now=android.os.SystemClock.elapsedRealtime();if(now-last>=200){Progress.report("로더 다운로드",total,expected);last=now;}}Progress.report("로더 다운로드",total,expected);}finally{c.disconnect();}
  Progress.stage("로더 SHA-256 및 ARM64 검증");if(!hash(tmp).equalsIgnoreCase(r.digest)){tmp.delete();throw new IOException("SHA-256 불일치");}validate(tmp);
  if(out.exists()&&!out.delete())throw new IOException("이전 파일 삭제 실패");if(!tmp.renameTo(out))throw new IOException("파일 저장 실패");return out;
 }
 static void validate(File f)throws Exception{try(InputStream i=new FileInputStream(f)){byte[] h=new byte[20];new DataInputStream(i).readFully(h);if(h[0]!=127||h[1]!=69||h[2]!=76||h[3]!=70||h[4]!=2||h[5]!=1||(h[18]&255)!=183||h[19]!=0)throw new IOException("ARM64 ELF 파일이 아님");}}
 static String hash(File f)throws Exception{MessageDigest d=MessageDigest.getInstance("SHA-256");try(InputStream i=new FileInputStream(f)){byte[] b=new byte[65536];int n;while((n=i.read(b))!=-1)d.update(b,0,n);}return java.util.HexFormat.of().formatHex(d.digest());}
}