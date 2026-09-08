package dev.hoonex.adofai.companion;

import android.app.Activity;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Clean-room Android gameplay preview for user-authored .adofai charts.
 *
 * This is intentionally not the commercial ADOFAI runtime and contains no game assets.
 * It implements the core orbit/timing loop so the standalone editor can produce a real,
 * playable custom build without modifying, resigning, or injecting into the Play app.
 */
public final class PlayerActivity extends Activity {
    public static final String EXTRA_CHART_PATH = "chart_path";

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_FULLSCREEN |
                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION |
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY);

        String path = getIntent().getStringExtra(EXTRA_CHART_PATH);
        try {
            Level level = Level.load(new File(path));
            setContentView(new GameView(level));
        } catch (Throwable error) {
            setContentView(new ErrorView(error));
        }
    }

    private final class ErrorView extends View {
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final String message;
        ErrorView(Throwable error) {
            super(PlayerActivity.this);
            message = error.getMessage() == null ? error.getClass().getSimpleName() : error.getMessage();
            setBackgroundColor(Color.rgb(12, 12, 16));
        }
        @Override protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            paint.setColor(Color.WHITE); paint.setTextSize(dp(24));
            canvas.drawText("ADOFAI Custom", dp(24), dp(52), paint);
            paint.setColor(Color.rgb(245, 120, 120)); paint.setTextSize(dp(15));
            canvas.drawText("Level load failed: " + message, dp(24), dp(88), paint);
            paint.setColor(Color.LTGRAY); paint.setTextSize(dp(14));
            canvas.drawText("Tap to return", dp(24), dp(122), paint);
        }
        @Override public boolean onTouchEvent(MotionEvent event) {
            if (event.getAction() == MotionEvent.ACTION_DOWN) finish();
            return true;
        }
    }

    private final class GameView extends View {
        private final Level level;
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint line = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final float tileSize = dp(58);
        private final long countdownMs = 900L;

        private boolean running;
        private boolean failed;
        private boolean complete;
        private long startMs;
        private int nextHit = 1;
        private String judgement = "TAP TO START";
        private long judgementUntil;
        private MediaPlayer music;

        GameView(Level level) {
            super(PlayerActivity.this);
            this.level = level;
            setBackgroundColor(Color.rgb(12, 12, 16));
            line.setStrokeWidth(dp(6));
            line.setStrokeCap(Paint.Cap.ROUND);
            line.setStyle(Paint.Style.STROKE);
            setKeepScreenOn(true);
        }

        private void begin() {
            stopMusic();
            running = true; failed = false; complete = false; nextHit = 1;
            startMs = SystemClock.elapsedRealtime() + countdownMs;
            judgement = "READY"; judgementUntil = startMs;
            if (level.songFile != null && level.songFile.isFile()) {
                try {
                    music = new MediaPlayer();
                    music.setDataSource(level.songFile.getAbsolutePath());
                    music.prepare();
                    postDelayed(new Runnable() {
                        @Override public void run() {
                            if (running && music != null) {
                                try { music.start(); } catch (Throwable ignored) {}
                            }
                        }
                    }, countdownMs);
                } catch (Throwable ignored) {
                    stopMusic();
                }
            }
            invalidate();
        }

        @Override public boolean onTouchEvent(MotionEvent event) {
            if (event.getAction() != MotionEvent.ACTION_DOWN) return true;
            if (failed || complete) { begin(); return true; }
            if (!running) { begin(); return true; }
            if (nextHit >= level.tiles.size()) return true;

            long now = SystemClock.elapsedRealtime();
            long target = startMs + level.hitTimesMs[nextHit];
            long delta = now - target;
            long abs = Math.abs(delta);
            if (delta < -190L) {
                judgement = "TOO EARLY"; judgementUntil = now + 240L;
                invalidate();
                return true;
            }
            if (abs <= 190L) {
                if (abs <= 45L) judgement = "PERFECT";
                else if (abs <= 100L) judgement = delta < 0 ? "EARLY" : "LATE";
                else judgement = delta < 0 ? "EARLY!" : "LATE!";
                judgementUntil = now + 360L;
                nextHit++;
                if (nextHit >= level.tiles.size()) {
                    complete = true; running = false; judgement = "COMPLETE";
                    stopMusic();
                }
                invalidate();
            }
            return true;
        }

        @Override protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            long now = SystemClock.elapsedRealtime();
            if (running && !complete && nextHit < level.tiles.size()) {
                long target = startMs + level.hitTimesMs[nextHit];
                if (now > target + 230L) {
                    failed = true; running = false; judgement = "MISS";
                    stopMusic();
                }
            }

            int anchorIndex = Math.max(0, Math.min(nextHit - 1, level.tiles.size() - 1));
            Point anchor = level.tiles.get(anchorIndex);
            float cx = getWidth() * 0.50f;
            float cy = getHeight() * 0.54f;

            line.setColor(Color.rgb(66, 67, 78));
            Path path = new Path();
            for (int i = 0; i < level.tiles.size(); i++) {
                Point p = level.tiles.get(i);
                float sx = cx + (p.x - anchor.x) * tileSize;
                float sy = cy - (p.y - anchor.y) * tileSize;
                if (i == 0) path.moveTo(sx, sy); else path.lineTo(sx, sy);
            }
            canvas.drawPath(path, line);

            for (int i = 0; i < level.tiles.size(); i++) {
                Point p = level.tiles.get(i);
                float sx = cx + (p.x - anchor.x) * tileSize;
                float sy = cy - (p.y - anchor.y) * tileSize;
                if (sx < -tileSize || sx > getWidth() + tileSize || sy < -tileSize || sy > getHeight() + tileSize) continue;
                if (i < nextHit) paint.setColor(Color.rgb(98, 86, 132));
                else if (i == nextHit) paint.setColor(Color.rgb(235, 235, 245));
                else paint.setColor(Color.rgb(92, 93, 108));
                canvas.drawCircle(sx, sy, dp(i == nextHit ? 16 : 13), paint);
            }

            drawPlanets(canvas, now, cx, cy, anchorIndex);
            drawHud(canvas, now);
            if (running) postInvalidateOnAnimation();
        }

        private void drawPlanets(Canvas canvas, long now, float cx, float cy, int anchorIndex) {
            double fraction = 0.0;
            if (running && nextHit < level.tiles.size()) {
                long previous = nextHit <= 1 ? startMs : startMs + level.hitTimesMs[nextHit - 1];
                long target = startMs + level.hitTimesMs[nextHit];
                if (target > previous) fraction = clamp((double) (now - previous) / (double) (target - previous), 0.0, 1.0);
            }
            double startAngle = nextHit < level.entryAngles.length ? level.entryAngles[nextHit] : 0.0;
            double sweep = nextHit < level.travelAngles.length ? level.travelAngles[nextHit] : 180.0;
            double direction = nextHit < level.clockwise.length && level.clockwise[nextHit] ? -1.0 : 1.0;
            double angle = Math.toRadians(startAngle + direction * sweep * fraction);
            float radius = dp(28);
            float px = cx + (float) Math.cos(angle) * radius;
            float py = cy - (float) Math.sin(angle) * radius;

            paint.setColor(Color.rgb(67, 156, 255));
            canvas.drawCircle(cx, cy, dp(13), paint);
            paint.setColor(Color.rgb(255, 90, 102));
            canvas.drawCircle(px, py, dp(13), paint);
        }

        private void drawHud(Canvas canvas, long now) {
            paint.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
            paint.setColor(Color.WHITE); paint.setTextSize(dp(20));
            canvas.drawText("ADOFAI Custom", dp(18), dp(34), paint);
            paint.setTypeface(android.graphics.Typeface.DEFAULT);
            paint.setColor(Color.rgb(170, 174, 190)); paint.setTextSize(dp(12));
            canvas.drawText(level.title + "   " + nextHit + "/" + (level.tiles.size() - 1), dp(18), dp(56), paint);
            canvas.drawText(String.format(Locale.US, "BPM %.1f", level.baseBpm), dp(18), dp(75), paint);
            if (level.unsupportedEvents > 0) {
                canvas.drawText("unsupported visual/events: " + level.unsupportedEvents, dp(18), dp(94), paint);
            }
            if (level.songFile == null || !level.songFile.isFile()) {
                paint.setColor(Color.rgb(235, 176, 90));
                canvas.drawText("NO LOCAL SONG FILE - timing-only playback", dp(18), dp(113), paint);
            }

            String center = judgement;
            if (running && now < startMs) center = String.valueOf(Math.max(1, (startMs - now + 299) / 300));
            if (now > judgementUntil && running && now >= startMs) center = "";
            if (!center.isEmpty()) {
                paint.setColor(failed ? Color.rgb(255, 90, 90) : Color.WHITE);
                paint.setTextSize(dp(failed || complete ? 34 : 25));
                paint.setTextAlign(Paint.Align.CENTER);
                canvas.drawText(center, getWidth() / 2f, dp(55), paint);
                paint.setTextAlign(Paint.Align.LEFT);
            }
            if (failed || complete) {
                paint.setTextAlign(Paint.Align.CENTER); paint.setTextSize(dp(14)); paint.setColor(Color.LTGRAY);
                canvas.drawText("Tap to restart", getWidth() / 2f, getHeight() - dp(24), paint);
                paint.setTextAlign(Paint.Align.LEFT);
            }
        }

        private void stopMusic() {
            if (music != null) {
                try { music.stop(); } catch (Throwable ignored) {}
                try { music.release(); } catch (Throwable ignored) {}
                music = null;
            }
        }

        @Override protected void onDetachedFromWindow() {
            stopMusic();
            super.onDetachedFromWindow();
        }
    }

    static final class Level {
        final List<Point> tiles = new ArrayList<Point>();
        long[] hitTimesMs;
        double[] entryAngles;
        double[] travelAngles;
        boolean[] clockwise;
        double baseBpm = 100.0;
        String title = "Untitled";
        File songFile;
        int unsupportedEvents;

        static Level load(File chart) throws Exception {
            if (chart == null || !chart.isFile()) throw new IllegalArgumentException("Chart file not found");
            JSONObject root = new JSONObject(read(chart));
            JSONObject settings = root.optJSONObject("settings");
            Level level = new Level();
            if (settings != null) {
                level.baseBpm = positive(settings.optDouble("bpm", 100.0), 100.0);
                String artist = settings.optString("artist", "").trim();
                String song = settings.optString("song", "").trim();
                level.title = song.isEmpty() ? chart.getName() : (artist.isEmpty() ? song : artist + " - " + song);
                if (!song.isEmpty()) {
                    File candidate = new File(chart.getParentFile(), song);
                    if (candidate.isFile()) level.songFile = candidate;
                }
            }

            List<Double> dirs = parseDirections(root);
            if (dirs.isEmpty()) throw new IllegalArgumentException("No playable pathData/angleData");
            level.tiles.add(new Point(0.0, 0.0));
            double x = 0.0, y = 0.0;
            for (double direction : dirs) {
                if (direction >= 900.0) continue;
                double rad = Math.toRadians(direction);
                x += Math.cos(rad); y += Math.sin(rad);
                level.tiles.add(new Point(x, y));
            }
            if (level.tiles.size() < 2) throw new IllegalArgumentException("Chart has fewer than two playable floors");

            int count = level.tiles.size();
            level.hitTimesMs = new long[count];
            level.entryAngles = new double[count];
            level.travelAngles = new double[count];
            level.clockwise = new boolean[count];
            boolean twirl = false;
            double bpm = level.baseBpm;
            long elapsed = 0L;
            JSONArray actions = root.optJSONArray("actions");
            Map<Integer, List<JSONObject>> byFloor = new HashMap<Integer, List<JSONObject>>();
            if (actions != null) {
                for (int i = 0; i < actions.length(); i++) {
                    JSONObject action = actions.optJSONObject(i);
                    if (action == null) continue;
                    int floor = Math.max(0, action.optInt("floor", 0));
                    List<JSONObject> list = byFloor.get(floor);
                    if (list == null) { list = new ArrayList<JSONObject>(); byFloor.put(floor, list); }
                    list.add(action);
                }
            }

            double previousDir = dirs.get(0);
            for (int floor = 1; floor < count; floor++) {
                List<JSONObject> atPrevious = byFloor.get(floor - 1);
                double pauseBeats = 0.0;
                if (atPrevious != null) {
                    for (JSONObject action : atPrevious) {
                        String type = action.optString("eventType", "");
                        if ("SetSpeed".equals(type)) {
                            String speedType = action.optString("speedType", "Bpm");
                            if ("Multiplier".equalsIgnoreCase(speedType)) {
                                bpm = positive(bpm * action.optDouble("bpmMultiplier", 1.0), bpm);
                            } else {
                                bpm = positive(action.optDouble("beatsPerMinute", bpm), bpm);
                            }
                        } else if ("Twirl".equals(type)) {
                            twirl = !twirl;
                        } else if ("Pause".equals(type) || "Hold".equals(type)) {
                            pauseBeats += Math.max(0.0, action.optDouble("duration", 0.0));
                        } else if (!isTimingNeutral(type)) {
                            level.unsupportedEvents++;
                        }
                    }
                }

                double currentDir = floor < dirs.size() ? dirs.get(floor) : previousDir;
                double travel = floor == 1 ? 180.0 : mod360(180.0 + previousDir - currentDir);
                if (travel < 0.0001) travel = 360.0;
                if (twirl) travel = 360.0 - travel;
                if (travel < 0.0001) travel = 360.0;
                level.travelAngles[floor] = travel;
                level.clockwise[floor] = twirl;
                level.entryAngles[floor] = previousDir + 180.0;
                double beats = travel / 180.0 + pauseBeats;
                elapsed += Math.max(1L, Math.round(beats * 60000.0 / positive(bpm, level.baseBpm)));
                level.hitTimesMs[floor] = elapsed;
                previousDir = currentDir;
            }
            return level;
        }

        private static boolean isTimingNeutral(String type) {
            return type.isEmpty() || "SetSpeed".equals(type) || "Twirl".equals(type) ||
                    "Pause".equals(type) || "Hold".equals(type) || "Checkpoint".equals(type) ||
                    "EditorComment".equals(type) || "Bookmark".equals(type) || "SetHitsound".equals(type);
        }

        private static List<Double> parseDirections(JSONObject root) throws Exception {
            String path = root.optString("pathData", "");
            if (!path.isEmpty()) return parsePathData(path);
            JSONArray angles = root.optJSONArray("angleData");
            List<Double> out = new ArrayList<Double>();
            if (angles != null) {
                for (int i = 0; i < angles.length(); i++) out.add(angles.optDouble(i, 0.0));
            }
            return out;
        }

        private static List<Double> parsePathData(String path) {
            Map<Character, Double> absolute = new HashMap<Character, Double>();
            String chars = "RpJEToUqGQHWLxNZFVDYBCMA";
            double[] values = {0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345};
            for (int i = 0; i < chars.length(); i++) absolute.put(chars.charAt(i), values[i]);
            Map<Character, Double> relative = new HashMap<Character, Double>();
            relative.put('5',72.0); relative.put('6',-72.0); relative.put('7',52.0); relative.put('8',-52.0);
            relative.put('9',-30.0); relative.put('h',120.0); relative.put('j',-120.0);
            relative.put('t',60.0); relative.put('y',300.0);

            List<Double> out = new ArrayList<Double>();
            double last = 0.0;
            for (int i = 0; i < path.length(); i++) {
                char c = path.charAt(i);
                if (c == '!') { out.add(999.0); continue; }
                Double a = absolute.get(c);
                if (a != null) { last = a; out.add(last); continue; }
                Double delta = relative.get(c);
                if (delta != null) { last = mod360(last + delta); out.add(last); }
            }
            return out;
        }

        private static String read(File file) throws Exception {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            StringBuilder out = new StringBuilder();
            char[] buf = new char[16384]; int n;
            while ((n = reader.read(buf)) >= 0) if (n > 0) out.append(buf, 0, n);
            reader.close();
            if (out.length() > 0 && out.charAt(0) == '\ufeff') out.deleteCharAt(0);
            return out.toString();
        }
    }

    static final class Point {
        final double x, y;
        Point(double x, double y) { this.x = x; this.y = y; }
    }

    private static double mod360(double value) {
        double result = value % 360.0;
        return result < 0.0 ? result + 360.0 : result;
    }
    private static double positive(double value, double fallback) {
        return Double.isNaN(value) || Double.isInfinite(value) || value <= 0.0 ? fallback : value;
    }
    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }
    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
