package dev.hoonex.adofai.loaderinstaller;
import android.app.*;import android.content.*;import android.os.*;import android.content.pm.PackageManager;
final class Updater {
 static void channel(Context c){NotificationManager nm=c.getSystemService(NotificationManager.class);nm.createNotificationChannel(new NotificationChannel("loader","로더 업데이트",NotificationManager.IMPORTANCE_DEFAULT));nm.createNotificationChannel(new NotificationChannel("app-update","설치 앱 업데이트",NotificationManager.IMPORTANCE_DEFAULT));}
 static boolean notify(Context c,String tag,boolean app){try{
  if(Build.VERSION.SDK_INT>=33&&c.checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS)!=PackageManager.PERMISSION_GRANTED)return false;
  channel(c);PendingIntent p=PendingIntent.getActivity(c,app?11:10,new Intent(c,MainActivity.class),PendingIntent.FLAG_IMMUTABLE|PendingIntent.FLAG_UPDATE_CURRENT);
  c.getSystemService(NotificationManager.class).notify(app?16:15,new Notification.Builder(c,app?"app-update":"loader").setSmallIcon(android.R.drawable.stat_sys_download_done).setContentTitle((app?"새 설치 앱 ":"새 얼불춤 로더 ")+tag).setContentText("다운로드 완료. 앱에서 설치를 승인하세요.").setContentIntent(p).setAutoCancel(true).build());return true;
 }catch(RuntimeException e){android.util.Log.w("LoaderInstaller","notification",e);return false;}}
 static boolean schedule(Context c){try{
  android.app.job.JobScheduler s=c.getSystemService(android.app.job.JobScheduler.class);if(s.getPendingJob(137)==null)s.schedule(new android.app.job.JobInfo.Builder(137,new ComponentName(c,UpdateJob.class)).setRequiredNetworkType(android.app.job.JobInfo.NETWORK_TYPE_ANY).setPeriodic(60*60*1000L).setPersisted(true).build());return s.getPendingJob(137)!=null;
 }catch(RuntimeException e){android.util.Log.w("LoaderInstaller","scheduler",e);return false;}}
}