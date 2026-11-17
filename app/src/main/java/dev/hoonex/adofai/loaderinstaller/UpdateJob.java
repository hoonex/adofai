package dev.hoonex.adofai.loaderinstaller;
import android.app.job.*;import android.content.*;
public class UpdateJob extends JobService{
 volatile Thread worker;
 public boolean onStartJob(JobParameters p){Thread t=new Thread(()->{
  boolean retry=false;SharedPreferences s=getSharedPreferences("state",0);
  try{SelfUpdate.check(this,true);String ready=s.getString("appReady","");if(!ready.isEmpty()&&!ready.equals(s.getString("appNotified",""))&&Updater.notify(this,ready,true))s.edit().putString("appNotified",ready).apply();}
  catch(Exception e){android.util.Log.w("LoaderInstaller","app update",e);retry=true;}
  try{Release.recordUpstream(this);}catch(Exception e){android.util.Log.w("LoaderInstaller","upstream",e);}
  try{if(!Thread.currentThread().isInterrupted()){Release r=Release.latest(this);Release.fetch(this,r);s.edit().putString("available",r.tag).apply();if(!r.tag.equals(s.getString("notified","v1.1.10.1"))&&Updater.notify(this,r.tag,false))s.edit().putString("notified",r.tag).apply();}}
  catch(Exception e){android.util.Log.w("LoaderInstaller","loader update",e);retry=true;}
  if(!Thread.currentThread().isInterrupted())jobFinished(p,retry);
 });worker=t;t.start();return true;}
 public boolean onStopJob(JobParameters p){Thread t=worker;if(t!=null)t.interrupt();return true;}
}