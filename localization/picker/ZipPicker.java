package dev.hoonex.adofai.bridge;
import android.app.Activity;
import android.app.Fragment;
import android.app.FragmentManager;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import java.io.*;
import java.lang.ref.WeakReference;
import java.util.UUID;
import java.util.zip.ZipFile;

public final class ZipPicker {
 private static final long MAX=4L*1024*1024*1024;
 private static WeakReference<Activity> activity=new WeakReference<>(null);
 private static int state; // 0 idle, 1 choosing, 2 copying, 3 ready, 4 failed, 5 cancelled
 private static String path="",message="";
 private static long done,total;
 private ZipPicker(){}
 public static synchronized void bind(Activity a){activity=new WeakReference<>(a);}
 private static synchronized void update(int s,String p,String m,long d,long t){state=s;path=p;message=m;done=d;total=t;}
 public static synchronized String[] snapshot(){String[] a={Integer.toString(state),path,message,Long.toString(done),Long.toString(total)};if(state>=3)state=0;return a;}
 public static synchronized boolean open(){
  if(state==1||state==2)return false;Activity a=activity.get();if(a==null||a.isFinishing()||a.isDestroyed()){update(4,"","게임 화면을 다시 열어 주세요.",0,0);return true;}
  update(1,"","",0,0);
  a.runOnUiThread(()->{try{
   FragmentManager fm=a.getFragmentManager();Picker f=(Picker)fm.findFragmentByTag("adofai_zip_picker_v1");
   if(f==null){f=new Picker();fm.beginTransaction().add(f,"adofai_zip_picker_v1").commitNowAllowingStateLoss();}
   f.choose();
  }catch(Exception e){update(4,"",error(e),0,0);}});return true;
 }
 private static String error(Throwable e){String m=e.getMessage();return m==null?e.getClass().getSimpleName():m;}
 public static final class Picker extends Fragment{
  private static final int REQUEST=18342;
  public Picker(){}
  void choose(){Intent i=new Intent(Intent.ACTION_OPEN_DOCUMENT);i.addCategory(Intent.CATEGORY_OPENABLE);i.setType("*/*");i.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);startActivityForResult(i,REQUEST);}
  @Override public void onActivityResult(int req,int result,Intent data){super.onActivityResult(req,result,data);if(req!=REQUEST)return;if(result!=Activity.RESULT_OK||data==null||data.getData()==null){update(5,"","",0,0);return;}
   Context c=getActivity().getApplicationContext();Uri uri=data.getData();update(2,"","",0,-1);new Thread(()->copy(c,uri),"adofai-zip-import").start();
  }
 }
 private static void copy(Context c,Uri uri){
  File tmp=null;try{
   String name="map.zip";long size=-1;
   try(Cursor q=c.getContentResolver().query(uri,new String[]{OpenableColumns.DISPLAY_NAME,OpenableColumns.SIZE},null,null,null)){
    if(q!=null&&q.moveToFirst()){int n=q.getColumnIndex(OpenableColumns.DISPLAY_NAME),s=q.getColumnIndex(OpenableColumns.SIZE);if(n>=0&&!q.isNull(n))name=q.getString(n);if(s>=0&&!q.isNull(s))size=q.getLong(s);}
   }catch(Exception ignored){}
   if(name==null||!name.toLowerCase(java.util.Locale.ROOT).endsWith(".zip"))throw new IOException("음악과 .adofai가 들어 있는 ZIP 파일을 선택하세요.");
   name=name.replaceAll("[\\\\/:*?<>|\\p{Cntrl}]","_");if(name.length()>120)name=name.substring(0,110)+".zip";
   if(size>MAX)throw new IOException("ZIP 파일이 4GB를 초과합니다.");
   File base=c.getExternalFilesDir(null);if(base==null)throw new IOException("게임 데이터 폴더에 접근할 수 없습니다.");
   File dir=new File(base,"imports");if(!dir.isDirectory()&&!dir.mkdirs())throw new IOException("가져오기 폴더 생성 실패");
   if(size>0&&size>dir.getUsableSpace())throw new IOException("저장 공간이 부족합니다.");
   String id=UUID.randomUUID().toString();tmp=new File(dir,id+".part");File dest=new File(dir,id.substring(0,8)+"-"+name);
   long count=0,previous=0;update(2,"",name,0,size);
   try(InputStream input=c.getContentResolver().openInputStream(uri);OutputStream out=new FileOutputStream(tmp)){
    if(input==null)throw new IOException("선택한 파일을 읽을 수 없습니다.");
    byte[] buf=new byte[1024*1024];int n;while((n=input.read(buf))!=-1){count+=n;if(count>MAX)throw new IOException("ZIP 파일이 4GB를 초과합니다.");out.write(buf,0,n);long now=android.os.SystemClock.elapsedRealtime();if(now-previous>=200){update(2,"",name,count,size);previous=now;}}
   }
   if(size>=0&&count!=size)throw new IOException("파일 복사가 완료되지 않았습니다.");
   try(ZipFile z=new ZipFile(tmp)){if(z.size()==0)throw new IOException("비어 있는 ZIP 파일입니다.");}
   if(!tmp.renameTo(dest))throw new IOException("ZIP 파일 저장 실패");
   update(3,dest.getAbsolutePath(),name,count,count);
  }catch(Exception e){update(4,"",error(e),0,0);}finally{if(tmp!=null)tmp.delete();}
 }
}