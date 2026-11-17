package dev.hoonex.adofai.loaderinstaller;
import java.io.*;
import java.util.Locale;
final class Progress {
 interface Sink {void update(String stage,long done,long total);}
 static final ThreadLocal<Sink> local=new ThreadLocal<>();
 static void stage(String s){report(s,0,0);}
 static void report(String s,long done,long total){Sink sink=local.get();if(sink!=null)sink.update(s,done,total);}
 static void transfer(InputStream in,OutputStream out,String stage,long total)throws IOException{
  byte[] b=new byte[1024*1024];long n=0,last=0;int k;report(stage,0,total);
  while((k=in.read(b))!=-1){out.write(b,0,k);n+=k;long now=android.os.SystemClock.elapsedRealtime();if(now-last>=200){report(stage,n,total);last=now;}}
  report(stage,n,total);
 }
 static String bytes(long n){return String.format(Locale.US,"%.1f MB",n/1048576.0);}
}